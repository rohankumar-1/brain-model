"""CQ500 readouts started from the model's own RSNA hemorrhage heads (2026-11-07).

The frozen benchmark fits each CQ500 readout from scratch on ~10 epidural positives. Here the readout starts from what
the checkpoint already learned on RSNA: the shared hemorrhage token, read as each member task, through the shared binary
MLP. Arms, all on the benchmark's own CV folds (brain.downstream.cv.make_folds, 5 x 3, seed 42), pooled out-of-fold AUC
per repeat, mean over repeats (as the benchmark scores):

  zero_shot    the RSNA head unchanged (k = 0)
  init_last    the head's last layer refitted on the fold's scans, L2-penalised towards the trained weights
               (logit = head(z) + dw . h, h the head's hidden layer, standardised; penalty lam * |dw|^2)
  init_stack   the head's logit plus a penalised correction from all six RSNA head logits
  stack6       logistic on the six RSNA head logits, from scratch
  scratch_last logistic on the same hidden layer h, from scratch (no head init): what the init buys
  scratch_pool logistic on the pooled embedding (the benchmark's linear readout, without its MLP / spatial variants)
lam is picked per fold on its validation scans (fit on train, AUC on val, ties to the stronger penalty), then refitted
on train + val. k-curve: P positives + 10 P negatives drawn from each fold's train + val (5 seeds), lam fixed at the
arm's most-picked value, mean per-fold test AUC (the zero-shot head scored the same way for reference).

    python analysis/head_init_cq500.py extract --model BQ-R14F      (GPU)
    python analysis/head_init_cq500.py fit --model BQ-R14F          (CPU)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain.common.paths import resolve  # noqa: E402

OUT = ROOT / "runs/diagnostics/head_init_cq500"
HEADS = ["hemorrhage_any", "hemorrhage_epidural", "hemorrhage_intraparenchymal", "hemorrhage_intraventricular",
         "hemorrhage_subarachnoid", "hemorrhage_subdural"]
# benchmark target -> (task, label or None, head)
TARGETS = [("EDH", "cq500_edh", None, "hemorrhage_epidural"),
           ("ICH", "cq500", "ICH", "hemorrhage_any"),
           ("IPH", "cq500", "IPH", "hemorrhage_intraparenchymal"),
           ("IVH", "cq500", "IVH", "hemorrhage_intraventricular"),
           ("SAH", "cq500", "SAH", "hemorrhage_subarachnoid"),
           ("SDH", "cq500", "SDH", "hemorrhage_subdural")]
LAMS = [0.001, 0.01, 0.1, 1.0, 10.0]


def extract(model_name: str, workers: int) -> None:
    from brain.downstream.models import resolve_model
    from brain.downstream.tasks import DEFAULT_VOLUMES, load_task, load_volumes
    from brain.multitask.data import checkpoint_stats, load_labels, prepare
    from brain.multitask.fewshot import _loader, load_model

    dev = torch.device("cuda")
    amp = dict(device_type="cuda", dtype=torch.bfloat16)
    bench_vols = load_volumes(DEFAULT_VOLUMES)
    ids = sorted({str(v) for t in ("cq500", "cq500_edh") for v in load_task(t, bench_vols).volume_ids})
    ref = resolve_model(model_name)
    model, reg, cfg = load_model(Path(ref.ckpt), dev)
    labels_dir = resolve(cfg["data"]["labels_dir"])
    labels, base = load_labels(labels_dir)
    stats = checkpoint_stats(model.label_stats, labels_dir, base)
    store = resolve(cfg["data"]["store_root"])
    volumes = pd.read_csv(store / cfg["data"].get("index_name", "volumes.csv"), low_memory=False)
    # 48 CQ500 studies are indexed twice under one volume_id (cq500 + its seg_cq500 re-release); the loader skips
    # ids indexed twice, which dropped 6 of the 13 EDH positives. Keep the cq500 copy, as the benchmark does.
    volumes = volumes[~(volumes["volume_id"].duplicated(keep=False) & (volumes["dataset"] == "seg_cq500"))]
    fake =pd.DataFrame({"volume_id": ids, "task": "hemorrhage_epidural", "value": 0.0, "ref": "", "split": "test",
                         "use": "train"})
    loader, _ = _loader(store, fake, volumes, reg, stats, "hemorrhage_epidural", "test", ids, use="train",
                        workers=workers)
    hidx = torch.tensor([reg.index[h] for h in HEADS], device=dev)
    e = HEADS.index("hemorrhage_epidural")
    vids, logits, hid, pooled = [], [], {h: [] for h in HEADS}, []
    lin1, act, lin2 = model.binary
    with torch.no_grad():
        for raw in loader:
            b = prepare(raw, cfg, dev, tuple(model.backbone.patch_xyz), train=False)
            task = hidx.unsqueeze(0).expand(b["x"].shape[0], -1)
            with torch.autocast(**amp):
                enc = model.encode(b["x"], b["keep"], b.get("meta"), task, modality=b["modality"])
                z = model.member(model.slots(enc, task, b["keep"]), task)          # (B, 6, D)
                h = act(lin1(z))
                lg = lin2(h)[..., 0]
                mil = model.mil_logits(enc, z, task, b["keep"])               # pool: mil heads (R17 on)
                if mil is not None:
                    lg = lg + mil.to(lg.dtype)
            vids += [str(v) for v in raw["volume_id"]]
            logits.append(lg.float().cpu())
            for j, name in enumerate(HEADS):
                hid[name].append(h[:, j].float().cpu())
            pooled.append(enc["pooled"].float().cpu())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{model_name}.npz", vids=np.array(vids), logits=torch.cat(logits).numpy(),
                        pooled=torch.cat(pooled).numpy(), heads=np.array(HEADS),
                        **{f"hid_{n}": torch.cat(v).numpy() for n, v in hid.items()})
    print(f"[extract] {model_name}: {len(vids)} of {len(ids)} scans; epidural col {e}", flush=True)


def fit_logit(x, y, offset, lam, steps=200):
    """min mean BCE(offset + x w + b) + lam |w|^2 (L-BFGS). x standardised by the caller."""
    x, y, off = (torch.as_tensor(a, dtype=torch.float64) for a in (x, y, offset))
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(off + x @ w + b, y) + lam * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


class Arm:
    def __init__(self, name, feats, offset):
        self.name, self.feats, self.offset = name, feats, offset      # feats (N, F) or None; offset (N,)

    def predict(self, tr, te, lam):
        if self.feats is None:
            return self.offset[te]
        mu, sd = self.feats[tr].mean(0), self.feats[tr].std(0) + 1e-6
        xs = lambda i: (self.feats[i] - mu) / sd  # noqa: E731
        w, b = fit_logit(xs(tr), Y[tr], self.offset[tr], lam)
        return self.offset[te] + xs(te) @ w + b


def fit(model_name: str) -> None:
    global Y
    from brain.downstream.cv import make_folds
    from brain.downstream.tasks import DEFAULT_VOLUMES, load_task, load_volumes
    from brain.multitask.test_eval import bootstrap

    d = np.load(OUT / f"{model_name}.npz")
    row = {v: i for i, v in enumerate(d["vids"])}
    heads = list(d["heads"])
    bench_vols = load_volumes(DEFAULT_VOLUMES)
    tasks = {t: load_task(t, bench_vols) for t in ("cq500", "cq500_edh")}
    res, curves = [], []
    for label, tname, lab, head in TARGETS:
        task = tasks[tname]
        y_all = np.asarray(task.y, dtype=float)
        y_all = y_all[:, task.label_names.index(lab)] if lab else y_all.reshape(len(y_all), -1)[:, 0]
        vids = [str(v) for v in task.volume_ids]
        have = np.array([v in row for v in vids])
        idx = np.array([row[v] for v, h in zip(vids, have) if h])
        Y = y_all[have]
        vid_h = [v for v, h in zip(vids, have) if h]
        lg = d["logits"][idx]
        own = lg[:, heads.index(head)]
        arms = [Arm("zero_shot", None, own),
                Arm("init_last", d[f"hid_{head}"][idx], own),
                Arm("init_stack", lg, own),
                Arm("stack6", lg, np.zeros(len(idx))),
                Arm("scratch_last", d[f"hid_{head}"][idx], np.zeros(len(idx))),
                Arm("scratch_pool", d["pooled"][idx], np.zeros(len(idx)))]
        pos_of = {v: i for i, v in enumerate(vid_h)}
        folds = make_folds(task, bench_vols, 5, 3, 42)
        print(f"[fit] {model_name} {label}: n={len(Y)} (of {len(vids)}), positives {int(Y.sum())}", flush=True)
        for arm in arms:
            oof, picked = {}, []
            for r, f, splits in folds:
                part = {s: np.array([pos_of[v] for v, ss in splits.items() if ss == s and v in pos_of])
                        for s in ("train", "val", "test")}
                lam = None
                if arm.feats is not None:
                    scores = [(auc(Y[part["val"]], arm.predict(part["train"], part["val"], l)), l) for l in LAMS]
                    best = max(s for s, _ in scores if s == s) if any(s == s for s, _ in scores) else None
                    lam = max(l for s, l in scores if s == best) if best is not None else LAMS[-1]
                    picked.append(lam)
                trv = np.concatenate([part["train"], part["val"]])
                p = arm.predict(trv, part["test"], lam)
                oof.setdefault(r, {}).update(dict(zip(part["test"], p)))
            per_rep = []
            for r, m in oof.items():
                ii = np.array(sorted(m))
                per_rep.append(auc(Y[ii], np.array([m[i] for i in ii])))
            ii = np.array(sorted(oof[0]))
            lo, hi = bootstrap("binary", Y[ii], np.array([oof[0][i] for i in ii]))
            lam_mode = Counter(picked).most_common(1)[0][0] if picked else None
            res.append({"model": model_name, "target": label, "arm": arm.name, "auc": float(np.mean(per_rep)),
                        "lo": lo, "hi": hi, "reps": [round(x, 4) for x in per_rep], "lam_mode": lam_mode,
                        "n": len(Y), "pos": int(Y.sum())})
            print(f"  {arm.name:13s} {res[-1]['auc']:.3f} [{lo:.3f}, {hi:.3f}] reps {res[-1]['reps']} "
                  f"lam {lam_mode}", flush=True)
            # k-curve: P positives + 10 P negatives per fold, lam fixed
            if arm.name in ("zero_shot", "init_last", "init_stack", "stack6", "scratch_last"):
                for P in (1, 2, 4, 8):
                    vals = []
                    for r, f, splits in folds:
                        trv = np.array([pos_of[v] for v, ss in splits.items() if ss != "test" and v in pos_of])
                        te = np.array([pos_of[v] for v, ss in splits.items() if ss == "test" and v in pos_of])
                        pos, neg = trv[Y[trv] == 1], trv[Y[trv] == 0]
                        if len(pos) < P:
                            continue
                        for s in range(5):
                            rng = np.random.default_rng([s, r, f, P])
                            tr = np.concatenate([rng.choice(pos, P, replace=False),
                                                 rng.choice(neg, min(len(neg), 10 * P), replace=False)])
                            vals.append(auc(Y[te], arm.predict(tr, te, lam_mode)))
                    vals = [v for v in vals if v == v]
                    if vals:
                        curves.append({"model": model_name, "target": label, "arm": arm.name, "P": P,
                                       "auc": float(np.mean(vals)), "sd": float(np.std(vals)), "n_fits": len(vals)})
    pd.DataFrame(res).to_csv(OUT / f"{model_name}_cv.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / f"{model_name}_kcurve.csv", index=False)
    c = pd.DataFrame(curves)
    if not c.empty:
        print(c.pivot_table(index=["target", "arm"], columns="P", values="auc").round(3).to_string())
    print(f"[fit] -> {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["extract", "fit"])
    ap.add_argument("--model", nargs="+", default=["BQ-R14F"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for m in a.model:
        extract(m, a.workers) if a.phase == "extract" else fit(m)
