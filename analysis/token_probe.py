"""How much does each task token know about each downstream label? A task-token BrainQuery encodes a benchmark task's
scans once (every trained task's token is always in the sequence); each token's output as its head reads it
(``out_norm(tt_proj(token))``) and the CLS token are probed for every label of the task with a cross-validated,
subject-grouped linear probe: logistic (AUC) for binary / multilabel / multiclass labels, ridge (Spearman rho of the
out-of-fold predictions) for regression. E.g. the midline-shift token's AUC for CQ500 ICH.

Every token reads the same patches, so this maps task relatedness (which tokens carry which information), not how much
encoder capacity a task uses: that needs an ablation.

    python runs/diagnostics/token_probe.py --ckpt BQ-12k-tokens --tasks cq500 rsna_ich --out runs/diagnostics/token_probe
    python runs/diagnostics/token_probe.py --ckpt BQ-12k-tokens --tasks bench      # every benchmark transfer task
Writes <out>/<model>_<task>.csv (token x label score), <out>/<model>_summary.csv (per label: CLS, token mean, best /
worst tokens) and prints, per label, CLS and the best and worst tokens."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/home/roku/brain-model")
from brain.common.paths import resolve  # noqa: E402
from brain.downstream.models import resolve_model  # noqa: E402
from brain.downstream.tasks import DEFAULT_VOLUMES, load_task, load_volumes  # noqa: E402
from brain.multitask.data import checkpoint_stats, load_labels, prepare  # noqa: E402
from brain.multitask.fewshot import _loader, load_model  # noqa: E402


@torch.no_grad()
def encode(model, cfg, reg, store, volumes, stats, ids: list[str], dev, workers: int) -> dict[str, np.ndarray]:
    """{volume_id: (T + 1, W)}: the trained tasks' token outputs, then CLS projected the same way."""
    carrier = reg.trained[0].id                         # any trained task: the loader needs one to build rows
    fake = pd.DataFrame({"volume_id": ids, "task": carrier, "value": 0.0, "ref": "", "split": "test", "use": "train"})
    loader, _ = _loader(store, fake, volumes, reg, stats, carrier, "test", ids=ids, use="train", workers=workers)
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    model.eval()
    out = {}
    for raw in loader:
        b = prepare(raw, cfg, dev, tuple(model.backbone.patch_xyz), train=False)
        with torch.autocast(**amp):
            enc = model.encode(b["x"], b["keep"], meta=b.get("meta"))
            tok = model.out_norm(model.tt_proj(enc["task_tokens"]))              # (B, T, W)
            cls = model.out_norm(model.tt_proj(enc["cls"].reshape(tok.shape[0], 1, -1)))
        feats = torch.cat([tok, cls], 1).float().cpu().numpy()
        for v, f in zip(raw["volume_id"], feats):
            out[str(v)] = f
    return out


def probe_auc(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> float:
    """Pooled out-of-fold AUC of a logistic probe (5 folds, subject-grouped, stratified); macro one-vs-rest for more
    than two classes."""
    ok = np.isfinite(y)
    x, y, groups = x[ok], y[ok].astype(int), groups[ok]
    classes = np.unique(y)
    if len(classes) < 2 or np.bincount(y)[classes].min() < 5:
        return np.nan
    p = np.zeros((len(y), len(classes)))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(x, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(x[tr], y[tr])
        p[np.ix_(te, np.searchsorted(classes, clf.classes_))] = clf.predict_proba(x[te])
    return float(roc_auc_score(y, p[:, 1]) if len(classes) == 2 else roc_auc_score(y, p, multi_class="ovr"))


def probe_rho(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Spearman rho of the pooled out-of-fold predictions of a ridge probe (5 subject-grouped folds)."""
    ok = np.isfinite(y)
    x, y, groups = x[ok], y[ok], groups[ok]
    if len(y) < 25:
        return np.nan
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(x, y, groups):
        p[te] = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(x[tr], y[tr]).predict(x[te])
    return float(spearmanr(y, p).statistic)


def probe(x, y, groups, kind: str) -> float:
    return probe_rho(x, y, groups) if kind == "regression" else probe_auc(x, y, groups)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a task-token checkpoint or models.yaml name")
    ap.add_argument("--tasks", nargs="+", default=["cq500", "rsna_ich"],
                    help="benchmark tasks, or 'bench' for every transfer task of configs/bench/benchmark.yaml")
    ap.add_argument("--max-scans", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/diagnostics/token_probe")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = resolve_model(a.ckpt)
    model, reg, cfg = load_model(Path(ref.ckpt), dev)
    if not model.tokens:
        raise SystemExit(f"{a.ckpt}: not a task-token model")
    from brain.bench.spec import load

    labels_dir = resolve(load().raw["labels_dir"])
    _, base_stats = load_labels(labels_dir)
    stats = checkpoint_stats(model.label_stats, labels_dir, base_stats)
    store = resolve(cfg["data"]["store_root"])
    volumes = pd.read_csv(store / cfg["data"].get("index_name", "volumes.csv"), low_memory=False)
    bench = load_volumes(DEFAULT_VOLUMES)
    subject = bench.set_index("volume_id")["dedup_key"].astype(str).to_dict()
    names = [reg[int(i)].id for i in model.tt_write.nonzero()[:, 0].tolist()] + ["CLS"]
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = [t for g in load().raw["transfer"].values() for t in g] if a.tasks == ["bench"] else a.tasks
    summary = []
    for tname in tasks:
        task = load_task(tname, bench)
        ids = [str(v) for v in task.volume_ids]
        keep = np.random.default_rng(0).permutation(len(ids))[: a.max_scans]
        ids = [ids[i] for i in sorted(keep)]
        feats = encode(model, cfg, reg, store, volumes, stats, ids, dev, a.workers)
        ids = [v for v in ids if v in feats]
        y_all = np.asarray(task.y, dtype=np.float64).reshape(len(task.volume_ids), -1)
        row = {str(v): i for i, v in enumerate(task.volume_ids)}
        y = y_all[[row[v] for v in ids]]
        x = np.stack([feats[v] for v in ids])                                    # (N, T + 1, W)
        groups = np.array([subject.get(v, v) for v in ids])
        labels = list(task.label_names) if y.shape[1] == len(task.label_names) else [tname]
        res = pd.DataFrame({lab: [probe(x[:, k], y[:, j], groups, task.task_type) for k in range(len(names))]
                            for j, lab in enumerate(labels)}, index=names)
        res.to_csv(outdir / f"{ref.name}_{tname}.csv")
        metric = "rho" if task.task_type == "regression" else "AUC"
        print(f"\n[{ref.name} / {tname}] n={len(ids)}  probe {metric} per label: CLS, best and worst task tokens")
        for lab in labels:
            col = res[lab].drop("CLS").dropna().sort_values(ascending=False)
            if col.empty:
                continue
            summary.append({"task": tname, "label": lab, "metric": metric, "n": len(ids), "cls": res.at["CLS", lab],
                            "token_mean": col.mean(), "best_token": col.index[0], "best": col.iloc[0],
                            "worst_token": col.index[-1], "worst": col.iloc[-1]})
            print(f"  {lab:24s} CLS {res.at['CLS', lab]:.3f} | best " + ", ".join(f"{k} {v:.3f}" for k, v in col.head(4).items())
                  + " | worst " + ", ".join(f"{k} {v:.3f}" for k, v in col.tail(3).items()))

    pd.DataFrame(summary).to_csv(outdir / f"{ref.name}_summary.csv", index=False)


if __name__ == "__main__":
    main()
