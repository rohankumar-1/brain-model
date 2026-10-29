"""Trained tasks on their held-out test splits: what multitask post-training optimised, measured with the power the
training validation (16 scans per task) lacks.

    python -m brain.multitask.test_eval --ckpts RUN3 RUN2 CTRL --probe-ckpts V3 RUN3 --out runs/diagnostics/test_eval

* Multitask checkpoints (``--ckpts``) answer with their own heads (``run_eval``), each task asked alone.
* Encoders (``--probe-ckpts``, SSL or multitask) get a linear probe (logistic / ridge, ``fewshot.linear_probe``) on the
  pooled embedding, trained on up to ``--max-train`` split-``train`` scans of the same task; scalar tasks only.
  Pooled features are cached per (checkpoint, scan), so tasks sharing scans encode them once.

Every model sees the same scans per task: up to ``--max-test`` (``--max-seg`` for segmentation) split-``test`` rows,
chosen deterministically. Each score has a bootstrap 95% interval; each multitask checkpoint after the first also gets
a paired bootstrap interval on its difference from the first (``delta_vs_ref``). Writes ``<out>/test_eval_<heads |
probe_<encoders>>.csv`` (``--probe-only`` runs the probes alone, so heads and probes can run as parallel jobs).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, roc_auc_score

from brain.common.paths import resolve
from brain.downstream.metrics import balanced_accuracy
from brain.multitask.data import checkpoint_stats, load_labels, prepare
from brain.multitask.evaluate import run_eval
from brain.multitask.fewshot import _loader, linear_probe, load_model

METRIC = {"binary": "auc", "regression": "r2", "multiclass": "bacc", "segmentation": "dice"}


def score(kind: str, y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y)
    y, p = y[ok], p[ok]
    if len(y) < 3:
        return float("nan")
    if kind == "binary":
        return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    if kind == "regression":
        return float(r2_score(y, p))
    if kind == "multiclass":
        return balanced_accuracy(y.astype(int), p.astype(int))
    return float(np.mean(p))                                          # segmentation rows hold per-scan Dice


def bootstrap(kind, y, p, p_ref=None, reps: int = 1000, seed: int = 0, y_ref=None) -> tuple[float, float]:
    """95% interval of the score, or of score(p) - score(p_ref) on the same resampled scans (``y_ref``: the
    reference's targets when its standardisation differs)."""
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    y_ref = y if y_ref is None else y_ref
    for _ in range(reps):
        i = rng.integers(0, n, n)
        v = score(kind, y[i], p[i])
        if p_ref is not None:
            v -= score(kind, y_ref[i], p_ref[i])
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def pick(ids: list[str], n: int) -> list[str]:
    """Deterministic subset (stable hash), the same for every model."""
    return sorted(ids, key=lambda v: hashlib.md5(v.encode()).hexdigest())[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True, help="multitask checkpoints; the first is the reference")
    ap.add_argument("--probe-ckpts", nargs="*", default=[], help="encoders scored by a linear probe on pooled features")
    ap.add_argument("--max-test", type=int, default=400)
    ap.add_argument("--max-seg", type=int, default=100)
    ap.add_argument("--max-train", type=int, default=1000)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--probe-only", action="store_true", help="skip the heads (the first --ckpts still defines tasks)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/diagnostics/test_eval")
    ap.add_argument("--labels-dir", default=None,
                    help="label table (default: the first checkpoint's); one table for every model, e.g. the newest")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"],
                    help="split the heads are scored on (train: how well they fit their own training scans; heads only)")
    a = ap.parse_args()
    if a.split != "test" and a.probe_ckpts:
        raise SystemExit("[test_eval] --split train/val scores heads only (probes fit on the train split)")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

    ref_model, ref_reg, ref_cfg = load_model(Path(a.ckpts[0]), dev)
    store = resolve(ref_cfg["data"]["store_root"])
    labels_dir = resolve(a.labels_dir or ref_cfg["data"]["labels_dir"])
    labels, stats = load_labels(labels_dir)
    volumes = pd.read_csv(store / ref_cfg["data"].get("index_name", "volumes.csv"), low_memory=False)
    tasks = [t.id for t in ref_reg.trained if a.tasks is None or t.id in a.tasks]
    lab = labels[(labels["use"] == "train") & labels["task"].isin(tasks)]
    test_ids, train_ids = {}, {}
    for t in tasks:
        cap = a.max_seg if ref_reg[t].type == "segmentation" else a.max_test
        test_ids[t] = pick(lab.loc[(lab.task == t) & (lab.split == a.split), "volume_id"].unique().tolist(), cap)
        train_ids[t] = pick(lab.loc[(lab.task == t) & (lab.split == "train"), "volume_id"].unique().tolist(), a.max_train)
    n_with = sum(1 for v in test_ids.values() if v)
    print(f"[test_eval] {len(tasks)} tasks, {n_with} with test scans (SynthSeg tasks have no test split); median "
          f"{int(np.median([len(v) for v in test_ids.values() if v] or [0]))} scans", flush=True)
    if n_with == 0:
        raise SystemExit("[test_eval] no test scans selected: check the label table (labels_dir) and its split column")

    rows, preds, pred_rows = [], {}, []
    for i, ck in enumerate([] if a.probe_only else a.ckpts):
        model, reg, cfg = (ref_model, ref_reg, ref_cfg) if i == 0 else load_model(Path(ck), dev)
        name = Path(ck).parent.name + "/" + Path(ck).stem
        patch = tuple(model.backbone.patch_xyz)
        m_stats = checkpoint_stats(model.label_stats, labels_dir, stats)
        for t in tasks:
            if t not in reg.index or not test_ids[t]:
                continue
            loader, _ = _loader(store, labels, volumes, reg, m_stats, t, a.split, ids=test_ids[t], use="train",
                                workers=a.workers)
            _, pr = run_eval(model, loader, cfg, reg, m_stats, dev, patch, amp, return_preds=True)
            r = pr.get(reg.index[t], [])
            if not r:
                continue
            kind = reg[t].type
            # standardised targets differ between models whose stats differ: pair on the raw label value instead
            y, p, vids = np.array([x[0] for x in r], float), np.array([x[1] for x in r], float), [x[2] for x in r]
            pred_rows += [{"model": name, "task": t, "kind": kind, "volume_id": v, "y": yy, "p": pp}
                          for v, yy, pp in zip(vids, y, p)]
            by_vid = dict(zip(vids, zip(y, p)))
            preds[(name, t)] = by_vid
            lo, hi = bootstrap(kind, y, p)
            rec = {"model": name, "kind": "heads", "task": t, "group": reg[t].group, "metric": METRIC[kind],
                   "n": len(y), "score": score(kind, y, p), "ci_lo": lo, "ci_hi": hi}
            ref = preds.get((Path(a.ckpts[0]).parent.name + "/" + Path(a.ckpts[0]).stem, t))
            common = [v for v in vids if ref is not None and v in ref]
            if i > 0 and common:                       # paired by volume id on the scans both models answered
                yr = np.array([ref[v][0] for v in common])
                pr_ref, pr_me = np.array([ref[v][1] for v in common]), np.array([by_vid[v][1] for v in common])
                ym = np.array([by_vid[v][0] for v in common])
                rec["delta_vs_ref"] = score(kind, ym, pr_me) - score(kind, yr, pr_ref)
                rec["delta_lo"], rec["delta_hi"] = bootstrap(kind, yr, pr_ref, p_ref=pr_me, y_ref=ym)
                rec["delta_lo"], rec["delta_hi"] = -rec["delta_hi"], -rec["delta_lo"]     # ref - this -> this - ref
            rows.append(rec)
            print(f"[heads] {name} {t}: {rec['metric']} {rec['score']:.3f} [{lo:.3f}, {hi:.3f}] n={len(y)}", flush=True)
        if i > 0:
            del model
            torch.cuda.empty_cache()

    # linear probes on pooled embeddings (scalar tasks)
    from brain.multitask.explain import encode_backbone, load_backbone

    for ck in a.probe_ckpts:
        name = Path(ck).parent.name + "/" + Path(ck).stem
        enc, _ = load_backbone(Path(ck), dev)
        cache: dict[str, np.ndarray] = {}

        def feats(t, ids, split):
            todo = [v for v in ids if v not in cache]
            if todo:
                loader, _ = _loader(store, labels, volumes, ref_reg, stats, t, split, ids=todo, use="train",
                                    workers=a.workers)
                for raw in loader:
                    b = prepare(raw, ref_cfg, dev, tuple(enc.patch_xyz), train=False)
                    with torch.no_grad(), torch.autocast(**amp):
                        f = encode_backbone(enc, b["x"], b["keep"])["pooled"].float().cpu().numpy()
                    for vid, row in zip(raw["volume_id"], f):
                        cache[str(vid)] = row
            return np.stack([cache[v] for v in ids])

        for t in tasks:
            kind = ref_reg[t].type
            if kind == "segmentation" or not test_ids[t] or len(train_ids[t]) < 10:
                continue
            # targets from the label table; image-computed (``@live``) targets have none there and are skipped
            ys = lab[lab.task == t].drop_duplicates("volume_id").set_index("volume_id")["value"].astype(float)
            tr = [v for v in train_ids[t] if np.isfinite(ys.get(v, np.nan))]
            te = [v for v in test_ids[t] if np.isfinite(ys.get(v, np.nan))]
            if len(tr) < 10 or len(te) < 3:
                continue
            ytr, yte = ys.loc[tr].to_numpy(float), ys.loc[te].to_numpy(float)
            xtr, xte = feats(t, tr, "train"), feats(t, te, "test")
            s = linear_probe(kind, xtr, ytr, xte, yte)
            if s is None:
                continue
            rows.append({"model": name, "kind": "linear_probe", "task": t, "group": ref_reg[t].group,
                         "metric": METRIC[kind], "n": len(yte), "n_train": len(ytr), "score": s})
            print(f"[probe] {name} {t}: {METRIC[kind]} {s:.3f} (train {len(ytr)}, test {len(yte)})", flush=True)
        del enc
        torch.cuda.empty_cache()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    tag = "probe_" + "_".join(Path(c).parent.name for c in a.probe_ckpts) if a.probe_only else "heads"
    if a.split != "test":
        tag += f"_{a.split}"
    df.to_csv(out / f"test_eval_{tag}.csv", index=False)
    if pred_rows:                                          # per-scan predictions: brain.eval.report bootstraps these
        pd.DataFrame(pred_rows).to_parquet(out / f"test_eval_preds{'' if a.split == 'test' else '_' + a.split}.parquet",
                                           index=False)
    wide = df.pivot_table(index=["group", "task", "metric"], columns=["kind", "model"], values="score")
    print(wide.round(3).to_string(), flush=True)
    print("wrote", out / f"test_eval_{tag}.csv", flush=True)


if __name__ == "__main__":
    main()
