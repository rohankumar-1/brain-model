"""Few-shot and transfer evaluation of a multitask checkpoint.

    python -m brain.multitask.fewshot --ckpt runs/multitask/v3init_s42/best.pt --shots 16 64 256 --seeds 3

Held-out tasks (registry ``role: held_out``) have a row in the task-embedding table that training never updated.
Few-shot fitting optimises only that row (and, for multiclass, the task's class vectors) on ``k`` labelled scans of the
split ``train`` rows, everything else frozen, then scores the task on its ``test`` rows. Baseline on the same scans:
a linear probe (logistic / ridge) on the frozen encoder's pooled embedding (classification and regression only).

Transfer: trained tasks with ``use: eval`` cohorts (e.g. age on OASIS / IXI / MS3SEG, whole tumour on MSD) are asked
directly, with no fitting, on those cohorts' test rows.

Results go to ``<ckpt dir>/fewshot_<ckpt stem>.json`` and a printed table; ``preds`` there holds every seed's test
predictions by volume id, so ``brain.eval.report`` can pair two models seed by seed and bootstrap over test subjects.
The k scans are a stratified draw (by class, or by value quantile for regression) seeded identically for every model:
the same seed gives the same scans whenever the label table is the same.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brain.common.paths import resolve
from brain.multitask.data import MultitaskDataset, checkpoint_stats, load_labels, make_loader, prepare
from brain.multitask.evaluate import run_eval
from brain.multitask.losses import slot_losses
from brain.multitask.model import MultitaskModel
from brain.multitask.registry import load_registry

METRIC = {"binary": "auc", "regression": "r2", "multiclass": "bacc", "segmentation": "dice"}


def load_model(ckpt_path: Path, device: torch.device):
    """The full multitask model of a checkpoint (encoder built by the frozen suite's loader), frozen."""
    from brain.downstream.embed3d import load_brain3d

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    reg = load_registry(cfg["data"].get("registry")).subset(list(ck["tasks"]))      # the checkpoint's task order
    enc = load_brain3d(ck, torch.device("cpu"), pretrained=False).model
    model = MultitaskModel(enc, reg, cfg["model"]).to(device)
    # joint checkpoints also hold the SSL predictor, which the multitask model has no use for
    model.load_state_dict({k: v for k, v in ck["multitask"].items() if not k.startswith("backbone.predictor.")})
    model.label_stats = ck.get("label_stats")            # the training normalisation (data.checkpoint_stats)
    return model.eval().requires_grad_(False), reg, cfg


def _loader(store, labels, volumes, reg, stats, task, split, ids=None, use=None, workers=4):
    ds = MultitaskDataset(store, labels, volumes, reg, stats, split=split, tasks=[task], train=False, use=use,
                          volume_ids=ids, max_seg=1)
    return make_loader(ds, batch_size=2, shuffle=False, seed=0, workers=workers, drop_last=False), ds


def fit_embedding(model, reg, cfg, loader, t: int, device, amp, steps: int = 200, lr: float = 3e-3) -> dict:
    """Optimise task ``t``'s embedding row (and its class vectors) on the loader's scans, everything else frozen
    (other rows get zero gradient, so Adam leaves them unchanged). Returns the original values for restoring."""
    saved = {"task": model.task_emb.weight[t].detach().clone(), "cls": model.cls_emb[t].detach().clone()}
    tables = [model.task_emb.weight] + ([model.cls_emb] if reg[t].type == "multiclass" else [])
    row = torch.zeros(len(reg), 1, device=device)
    row[t] = 1
    hooks = []
    for p in tables:
        p.requires_grad_(True)
        hooks.append(p.register_hook(lambda g: g * row.view(-1, *[1] * (g.ndim - 1))))
    opt = torch.optim.Adam(tables, lr=lr)
    batches = list(loader)                          # raw (CPU, stored dtype); prepared per step
    try:
        for i in range(steps):
            b = prepare(batches[i % len(batches)], cfg, device, model.backbone.patch_xyz, train=False)
            with torch.autocast(**amp):
                out = model(b["x"], b["keep"], b["task"], b["seg_pair"], encoder_grad=False, meta=b.get("meta"))
            loss, has = slot_losses(out, b["task"], b["y"], model.task_type, b["seg"], b["seg_pair"])
            opt.zero_grad()
            loss[has].mean().backward()
            opt.step()
    finally:
        for h in hooks:
            h.remove()
        for p in tables:
            p.requires_grad_(False)
    return saved


@torch.no_grad()
def pooled_features(model, cfg, loader, device, amp):
    feats, ys = [], []
    for raw in loader:
        b = prepare(raw, cfg, device, model.backbone.patch_xyz, train=False)
        with torch.autocast(**amp):
            enc = model.encode(b["x"], b["keep"], b.get("meta"))
        feats.append(enc["pooled"].float().cpu())
        ys.append(b["y"][:, 0].cpu())
    return torch.cat(feats).numpy(), torch.cat(ys).numpy()


def draw(pool: pd.DataFrame, k: int, seed: int, kind: str) -> list[str]:
    """``k`` volume ids from ``pool`` (label rows: volume_id, value), stratified by class (by value quintile for
    regression; none for segmentation), so a small draw cannot miss a rare class. Falls back to a plain draw."""
    from sklearn.model_selection import train_test_split

    pool = pool.drop_duplicates("volume_id")
    if k >= len(pool):
        return pool["volume_id"].tolist()
    strata = None
    if kind in ("binary", "multiclass"):
        strata = pool["value"].astype(int).to_numpy()
    elif kind == "regression":
        strata = pd.qcut(pool["value"].astype(float), q=min(5, max(1, k // 4)), labels=False, duplicates="drop").to_numpy()
    try:
        picked, _ = train_test_split(pool, train_size=k, stratify=strata, random_state=seed)
    except ValueError:                           # a stratum too small to split
        picked = pool.sample(k, random_state=seed)
    return picked["volume_id"].tolist()


def linear_probe(kind: str, xtr, ytr, xte, yte) -> float | None:
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    s = StandardScaler().fit(xtr)
    xtr, xte = s.transform(xtr), s.transform(xte)
    if kind == "regression":
        return float(r2_score(yte, RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(xtr, ytr).predict(xte)))
    if len(np.unique(ytr)) < 2:
        return None
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(xtr, ytr.astype(int))
    if kind == "binary":
        return float(roc_auc_score(yte, clf.predict_proba(xte)[:, 1])) if len(np.unique(yte)) == 2 else None
    return float(balanced_accuracy_score(yte.astype(int), clf.predict(xte)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--shots", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tasks", nargs="*", default=None, help="held-out task ids (default: all)")
    ap.add_argument("--no-transfer", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    model, reg, cfg = load_model(a.ckpt, dev)
    d = cfg["data"]
    store = resolve(d["store_root"])
    labels, stats = load_labels(resolve(d["labels_dir"]))
    stats = checkpoint_stats(model.label_stats, resolve(d["labels_dir"]), stats)
    volumes = pd.read_csv(store / d.get("index_name", "volumes.csv"), low_memory=False)
    out: dict = {"ckpt": str(a.ckpt), "fewshot": [], "transfer": [], "preds": []}

    if not a.no_transfer:
        for t in reg.trained:
            rows = labels[(labels["task"] == t.id) & (labels["use"] == "eval") & (labels["split"] == "test")]
            if rows.empty:
                continue
            for cohort, ids in rows.groupby(volumes.set_index("volume_id").loc[rows["volume_id"],
                                                                                "dataset_family"].to_numpy()):
                loader, _ = _loader(store, labels, volumes, reg, stats, t.id, "test", ids["volume_id"].tolist(),
                                    use="eval", workers=a.workers)
                res = run_eval(model, loader, cfg, reg, stats, dev, model.backbone.patch_xyz, amp)
                key = f"{t.id}_{METRIC[t.type]}"
                out["transfer"].append({"task": t.id, "cohort": cohort, "n": len(ids), "metric": METRIC[t.type],
                                        "value": res.get(key)})
                print(f"[transfer] {t.id:24s} {cohort:16s} n={len(ids):4d} {METRIC[t.type]}={res.get(key)}",
                      flush=True)

    held = [t for t in reg.tasks if not t.trained and (a.tasks is None or t.id in a.tasks)]
    for task in held:
        t = reg.index[task.id]
        pool = labels[(labels["task"] == task.id) & (labels["split"] == "train")]
        test_ids = labels[(labels["task"] == task.id) & (labels["split"] == "test")]["volume_id"].tolist()
        if pool.empty or not test_ids:
            continue
        te_loader, _ = _loader(store, labels, volumes, reg, stats, task.id, "test", test_ids, workers=a.workers)
        te_feats = pooled_features(model, cfg, te_loader, dev, amp) if task.type != "segmentation" else None
        for k in a.shots:
            if k > len(pool):
                continue
            for seed in range(a.seeds):
                ids = draw(pool, k, seed, task.type)
                tr_loader, _ = _loader(store, labels, volumes, reg, stats, task.id, "train", ids, workers=a.workers)
                saved = fit_embedding(model, reg, cfg, tr_loader, t, dev, amp, a.steps, a.lr)
                res, pr = run_eval(model, te_loader, cfg, reg, stats, dev, model.backbone.patch_xyz, amp,
                                   return_preds=True)
                rows = pr.get(t, [])
                out["preds"].append({"task": task.id, "k": k, "seed": seed, "metric": METRIC[task.type],
                                     "volume_id": [r[2] for r in rows], "y": [r[0] for r in rows],
                                     "p": [r[1] for r in rows]})
                model.task_emb.weight.data[t] = saved["task"]
                model.cls_emb.data[t] = saved["cls"]
                value = res.get(f"{task.id}_{METRIC[task.type]}")
                probe = None
                if te_feats is not None:
                    xtr, ytr = pooled_features(model, cfg, tr_loader, dev, amp)
                    probe = linear_probe(task.type, xtr, ytr, *te_feats)
                out["fewshot"].append({"task": task.id, "k": k, "seed": seed, "metric": METRIC[task.type],
                                       "value": value, "linear_probe": probe, "n_test": len(test_ids)})
                print(f"[fewshot] {task.id:28s} k={k:4d} seed={seed} {METRIC[task.type]}={value} probe={probe}",
                      flush=True)
    path = a.ckpt.parent / f"fewshot_{a.ckpt.stem}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    if out["fewshot"]:
        f = pd.DataFrame(out["fewshot"])
        print(f.groupby(["task", "k"])[["value", "linear_probe"]].agg(["mean", "std"]).round(3).to_string())
    print(f"[fewshot] -> {path}")


if __name__ == "__main__":
    main()
