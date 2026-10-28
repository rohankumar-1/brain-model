"""Reproduce a published model's own reported results with our adapter (``configs/bench/replicate.yaml``).

The benchmark scores every model under one protocol; this checks the adapter instead: each target copies the paper's
protocol (its test scans where published, its training-set size, its linear probe and metric), so a correctly run
model lands within the paper's interval. Features are the adapter's global feature (``pooled``, e.g. BrainIAC's first
output token), extracted once per volume and cached; the probe is the paper's: dropout 0.2 + one linear layer, Adam
(lr 0.001, batch 16), the best epoch on validation, over ``--seeds`` seeds (training / validation scans resampled per
seed where the paper's are not published). Deviations: features without the paper's training augmentation.

    python -m brain.bench.replicate --model brainiac [--targets mci_oasis1 ...] [--seeds 5]
    python -m brain.bench.replicate --paper brainiac --model BQ-R6     # another model under that paper's protocol

Any model can run a paper's targets (its global feature: CLS for BQ), which compares it with the paper's numbers on the
same test scans. Encoders trained with scan metadata get it as unknown here (it names the contrast: the answer of
sequence classification).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from brain.common.paths import PROCESSED_ROOT, REPO_ROOT, resolve

SPEC = REPO_ROOT / "configs" / "bench" / "replicate.yaml"
OUT = REPO_ROOT / "runs" / "bench" / "replicate"
BOOT = 1000


def pool(target: dict, vols: pd.DataFrame) -> pd.DataFrame:
    """The target's scans: volume_id, subject, y (float; class index for multiclass), modality, path."""
    from brain.downstream.tasks import load_task

    if "task" in target:
        t = load_task(target["task"], vols)
        y = np.asarray(t.y, dtype=float).reshape(len(t.y), -1)[:, 0]
        if target.get("target") == "expm1":
            y = np.expm1(y)
        df = pd.DataFrame({"volume_id": [str(v) for v in t.volume_ids], "y": y})
        if target.get("max_target") is not None:
            df = df[df.y <= float(target["max_target"])]
    else:
        names = list(target["datasets"].values())
        classes = {d: sorted(set(names)).index(c) for d, c in target["datasets"].items()}
        df = vols[vols.dataset.isin(list(classes))][["volume_id", "dataset"]].copy()
        df["y"] = df.dataset.map(classes).astype(float)
        df = df.drop(columns="dataset")
    meta = vols.set_index("volume_id")
    df["subject"] = meta.loc[df.volume_id, "dedup_key"].astype(str).values
    df["modality"] = meta.loc[df.volume_id, "modality"].astype(str).values
    df["path"] = [str(PROCESSED_ROOT / p) for p in meta.loc[df.volume_id, "image_path"].astype(str)]
    return df.reset_index(drop=True)


def features(model, df: pd.DataFrame, cache: Path, device: torch.device) -> np.ndarray:
    """(N, D) pooled features of ``df``'s volumes, cached per volume id in ``cache``."""
    have: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = dict(zip(z["ids"].tolist(), z["x"]))
    todo = [r for r in df.itertuples() if r.volume_id not in have]
    if todo:
        enc = model.load(device)
        enc.meta_off = True                                     # (BQ encoders trained with metadata)
        window = "canonical" if model.is_checkpoint else "native"
        for i, r in enumerate(todo):
            e = enc.encode_volume(Path(r.path), r.volume_id, r.modality, window)
            have[r.volume_id] = e.cls.float().cpu().numpy()
            if (i + 1) % 200 == 0 or i + 1 == len(todo):
                print(f"[replicate] features {i + 1}/{len(todo)}", flush=True)
                ids = sorted(have)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, ids=np.asarray(ids), x=np.stack([have[k] for k in ids]))
    return np.stack([have[v] for v in df.volume_id]).astype(np.float32)


def split(target: dict, df: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean train / val / test masks: the paper's test scans (or held-out subjects, fixed across seeds), then
    ``n_trainval`` of the rest (all when unset), 80:20 train / val by subject, resampled per seed."""
    if target.get("test_ids"):
        ids = pd.read_csv(resolve(target["test_ids"])).iloc[:, 0].astype(str).str.strip()
        key = df.volume_id.str.split(":").str[-1]
        test = key.isin(set(ids)).to_numpy()
        if test.sum() != len(ids):
            raise SystemExit(f"{len(ids) - test.sum()} of the paper's {len(ids)} test scans are not in the pool")
    else:
        subj = np.sort(df.subject.unique())
        held = set(np.random.default_rng(0).choice(subj, int(target["n_test_subjects"]), replace=False))
        test = df.subject.isin(held).to_numpy()
    test_subj = set(df.subject[test])
    rest = np.flatnonzero(~test & ~df.subject.isin(test_subj).to_numpy())
    rng = np.random.default_rng(seed)
    if target.get("n_trainval") and len(rest) > int(target["n_trainval"]):
        rest = rng.choice(rest, int(target["n_trainval"]), replace=False)
    subj = df.subject.to_numpy()
    val_subj = set(rng.permutation(np.unique(subj[rest]))[: max(1, round(0.2 * len(np.unique(subj[rest]))))])
    train, val = np.zeros(len(df), bool), np.zeros(len(df), bool)
    train[[i for i in rest if subj[i] not in val_subj]] = True
    val[[i for i in rest if subj[i] in val_subj]] = True
    return train, val, test


def score(metric: str, y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    from brain.downstream.metrics import balanced_accuracy

    if metric == "auc":
        return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
    if metric == "mae":
        return float(np.abs(y - s).mean())
    return balanced_accuracy(y.astype(int), s.argmax(1))


def probe(metric: str, x: np.ndarray, y: np.ndarray, masks, seed: int, epochs: int = 200) -> np.ndarray:
    """The paper's linear probe; returns test scores of the epoch best on validation."""
    train, val, test = masks
    torch.manual_seed(seed)
    n_out = int(y.max()) + 1 if metric == "balanced_accuracy" else 1
    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(x.shape[1], n_out))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    loss_fn = {"auc": nn.BCEWithLogitsLoss(), "mae": nn.MSELoss()}.get(metric, nn.CrossEntropyLoss())
    xt, yt = torch.from_numpy(x[train]), torch.from_numpy(y[train])
    target = (lambda b: b.long()) if n_out > 1 else (lambda b: b.float()[:, None])
    out = (lambda o: o.softmax(1).numpy()) if n_out > 1 else \
        (lambda o: o[:, 0].sigmoid().numpy() if metric == "auc" else o[:, 0].numpy())
    best, best_state = None, None
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        head.train()
        for idx in torch.randperm(len(xt), generator=g).split(16):
            opt.zero_grad()
            loss_fn(head(xt[idx]), target(yt[idx])).backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            v = score(metric, y[val], out(head(torch.from_numpy(x[val]))))
        if np.isfinite(v) and (best is None or (v < best if metric == "mae" else v > best)):
            best, best_state = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        return out(head(torch.from_numpy(x[test])))


def interval(metric: str, y: np.ndarray, s: np.ndarray, subj: np.ndarray) -> tuple[float, float]:
    """95% subject-bootstrap interval of the test metric."""
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(subj == u) for u in np.unique(subj)]
    vals = []
    for _ in range(BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(score(metric, y[idx], s[idx]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    from brain.downstream.models import resolve_model
    from brain.downstream.tasks import DEFAULT_VOLUMES, load_volumes

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a checkpoint or a configs/bench/models.yaml name")
    ap.add_argument("--paper", default=None, help="whose targets in replicate.yaml (default: the model's own)")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    paper = a.paper or a.model
    spec = (yaml.safe_load(SPEC.read_text()) or {}).get(paper)
    if not spec:
        raise SystemExit(f"no replication targets for {paper} in {SPEC}")
    model = resolve_model(a.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vols = load_volumes(DEFAULT_VOLUMES)
    rows = []
    for name in a.targets or list(spec):
        t = spec[name]
        df = pool(t, vols)
        x = features(model, df, OUT / model.cache_key / "features.npz", device)
        y = df.y.to_numpy(np.float32)
        runs, first = [], None
        for seed in range(a.seeds):
            masks = split(t, df, seed)
            s = probe(t["metric"], x, y, masks, seed)
            runs.append(score(t["metric"], y[masks[2]], s))
            if first is None:
                first = interval(t["metric"], y[masks[2]], s, df.subject.to_numpy()[masks[2]])
                sizes = [int(m.sum()) for m in masks]
        rep, ci = float(t["reported"]), t.get("ci")
        mean = float(np.mean(runs))
        inside = ci is not None and ci[0] <= mean <= ci[1]
        rows.append({"target": name, "metric": t["metric"], "n_train/val/test": "/".join(map(str, sizes)),
                     "paper": rep, "paper_ci": "-" if ci is None else f"[{ci[0]:.3g}, {ci[1]:.3g}]",
                     "ours_mean": mean, "ours_sd": float(np.std(runs)), "ours_seed0_ci": f"[{first[0]:.3g}, {first[1]:.3g}]",
                     "rel_diff": (mean - rep) / rep, "within_paper_ci": inside if ci is not None else None,
                     "source": t.get("source", "")})
        print(f"[replicate] {name}: ours {mean:.4g} ± {np.std(runs):.2g} ({a.seeds} seeds)  paper {rep:.4g}  "
              f"({(mean - rep) / rep:+.1%})", flush=True)
    tab = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / (paper if paper == a.model else f"{paper}__{model.name}")
    tab.to_csv(stem.with_suffix(".csv"), index=False)
    head = f"# {a.model}: published results reproduced with our adapter" if paper == a.model else \
        f"# {model.name} under {paper}'s protocol (the paper column is {paper}'s own result)"
    lines = [head, "",
             "| Target | Metric | n train/val/test | Paper [95% CI] | Ours (mean ± sd over seeds) | Ours, seed 0 [95% CI] "
             "| Difference | Source |", "|---|---|---|---|---|---|---:|---|"]
    for r in rows:
        lines.append(f"| {r['target']} | {r['metric']} | {r['n_train/val/test']} | {r['paper']:.4g} {r['paper_ci']} | "
                     f"{r['ours_mean']:.4g} ± {r['ours_sd']:.2g} | {r['ours_seed0_ci']} | {r['rel_diff']:+.1%} | "
                     f"{r['source']} |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
