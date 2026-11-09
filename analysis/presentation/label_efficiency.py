"""Label-efficiency curves for frozen encoders: a linear readout on the mean-pooled tokens, fitted on n labelled
training scans (n = 16 ... all, 20 random stratified draws each), scored on the task's fixed test split.
Embeddings come from the frozen benchmark's caches (built if missing).

    python analysis/presentation/label_efficiency.py run --models BQ-FISO brainfm --tasks adni_cn_ad adni_cn_mci --out DIR
    python analysis/presentation/label_efficiency.py plot --out DIR

The readout's regularisation is chosen by 3-fold CV inside the n training scans only (no other labels are used)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
NS = (16, 32, 64, 128, 256, 512)
# task -> (benchmark task, label for a multilabel task or None)
TASKS = {"adni_cn_ad": ("adni_cn_ad", None), "adni_cn_mci": ("adni_cn_mci", None), "adni_mmse": ("adni_mmse", None),
         "cq500_fracture": ("cq500", "CalvarialFracture"), "cq500_ich": ("cq500", "ICH"),
         "ucsf_idh": ("ucsf_idh", None)}


def fit_score(kind, xtr, ytr, xte, yte, seed):
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "regression":
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
        m.fit(xtr, ytr)
        return float(r2_score(yte, m.predict(xte)))
    cv = 3 if min(np.bincount(ytr.astype(int))) >= 3 else 2
    m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-4, 1, 11), cv=cv, max_iter=3000,
                                                             scoring="roc_auc", random_state=seed))
    m.fit(xtr, ytr)
    return float(roc_auc_score(yte, m.predict_proba(xte)[:, 1]))


def run(a) -> None:
    from brain.downstream.embeddings import load_or_build_embeddings
    from brain.downstream.models import resolve_model
    from brain.downstream.run_suite import one_token, pooled_embedding
    from brain.downstream.tasks import DEFAULT_VOLUMES, load_task, load_volumes, split_xy

    volumes = load_volumes(None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tkey in a.tasks:
        bench, label = TASKS[tkey]
        task = load_task(bench, volumes)
        for mname in a.models:
            model = resolve_model(mname)
            cache = ROOT / "runs/downstream/_cache" / ("v7" if model.is_checkpoint else "v6")
            emb = load_or_build_embeddings(model=model, slices_root=DEFAULT_VOLUMES.parent,
                                           image_paths=task.image_paths, volume_ids=task.volume_ids, cache_dir=cache,
                                           task=task.name, window="multi" if model.is_checkpoint else "canonical",
                                           modalities=task.modalities)
            vec, _ = pooled_embedding(emb)
            packs = split_xy(task, one_token(emb, vec))
            tr, te = packs["train"], packs["test"]
            # validation scans join the pool the n training scans are drawn from (no tuning on them)
            if len(packs["val"]["volume_ids"]):
                va = packs["val"]
                tr = {"mus": np.concatenate([tr["mus"], va["mus"]]), "y": np.concatenate([tr["y"], va["y"]])}
            x, xte = tr["mus"][:, 0], te["mus"][:, 0]
            y, yte = np.asarray(tr["y"], float), np.asarray(te["y"], float)
            if label is not None:
                j = task.label_names.index(label)
                y, yte = y[:, j], yte[:, j]
            y, yte = y.reshape(len(y), -1)[:, 0], yte.reshape(len(yte), -1)[:, 0]
            kind = "regression" if task.task_type == "regression" else "binary"
            ok, okt = np.isfinite(y), np.isfinite(yte)
            x, y, xte, yte = x[ok], y[ok], xte[okt], yte[okt]
            ns = [n for n in NS if n < len(y)] + [len(y)]
            for n in ns:
                for r in range(1 if n == len(y) else a.reps):
                    rng = np.random.default_rng(1000 * r + n)
                    if kind == "binary":                         # stratified draw, at least 2 of each class
                        pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
                        k = int(np.clip(round(n * len(pos) / len(y)), 2, min(len(pos), n - 2)))
                        idx = np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, n - k, replace=False)]
                    else:
                        idx = rng.choice(len(y), n, replace=False)
                    s = fit_score(kind, x[idx], y[idx], xte, yte, r)
                    rows.append({"task": tkey, "model": mname, "n": n, "rep": r, "score": s,
                                 "metric": "r2" if kind == "regression" else "auc", "n_test": len(yte)})
                sub = [q["score"] for q in rows if q["task"] == tkey and q["model"] == mname and q["n"] == n]
                print(f"[{tkey}] {mname} n={n}: {np.mean(sub):.3f} ± {np.std(sub):.3f} ({len(sub)} draws)", flush=True)
            pd.DataFrame(rows).to_csv(out / "label_efficiency.csv", index=False)
    print("wrote", out / "label_efficiency.csv")


def plot(a) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED = "#1d1d1b", "#55544f"
    COL = {"BQ-FISO": ("#1c5cab", "BrainQuery (ours)"), "brainfm": ("#eb6834", "BrainFM (published foundation model)"),
           "BQ-R14F": ("#1c5cab", "BrainQuery (ours)")}
    NAME = {"adni_cn_ad": "ADNI · CN vs AD", "adni_cn_mci": "ADNI · CN vs MCI", "adni_mmse": "ADNI · MMSE (R²)",
            "cq500_fracture": "CQ500 · calvarial fracture", "cq500_ich": "CQ500 · any bleed", "ucsf_idh": "UCSF · IDH"}
    d = pd.read_csv(Path(a.out) / "label_efficiency.csv")
    tasks = [t for t in TASKS if t in set(d.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(13.333, 7.5), dpi=200, squeeze=False,
                             gridspec_kw=dict(left=0.06, right=0.98, top=0.74, bottom=0.14, wspace=0.25))
    for ax, t in zip(axes[0], tasks):
        for m in d[d.task == t].model.unique():
            g = d[(d.task == t) & (d.model == m)].groupby("n").score
            mu, sd = g.mean(), g.std().fillna(0)
            c, lab = COL.get(m, ("#8a8983", m))
            ax.plot(mu.index, mu.values, "o-", color=c, lw=2.2, ms=5, label=lab)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ns = sorted(d[d.task == t].n.unique())
        ax.set_xticks(ns, [str(n) for n in ns], fontsize=9)
        ax.minorticks_off()
        ax.set_title(NAME.get(t, t), fontsize=13, color=INK)
        ax.set_xlabel("labelled training scans", fontsize=11, color=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        if d[d.task == t].metric.iloc[0] == "auc":
            ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    axes[0, 0].set_ylabel("test score (AUC, or R²)", fontsize=11, color=MUTED)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.86), ncol=len(lab), frameon=False, fontsize=11.5)
    fig.text(0.06, 0.94, "Fewer labels needed for new tasks", fontsize=19, fontweight="bold", color=INK)
    fig.text(0.06, 0.035, "Frozen encoders, linear readout on mean-pooled features, fitted on n labelled scans only "
             "(regularisation by 3-fold CV inside them); 20 random draws per point (band = ± 1 s.d.); fixed test split.",
             fontsize=9, color=MUTED)
    fig.savefig(Path(a.out) / "label_efficiency.png", facecolor="white")
    print("wrote", Path(a.out) / "label_efficiency.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", default=["BQ-FISO", "brainfm"])
    r.add_argument("--tasks", nargs="+", default=["adni_cn_ad", "adni_cn_mci", "adni_mmse", "cq500_fracture"])
    r.add_argument("--reps", type=int, default=20)
    r.add_argument("--out", required=True)
    p = sub.add_parser("plot")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    (run if a.cmd == "run" else plot)(a)


if __name__ == "__main__":
    main()
