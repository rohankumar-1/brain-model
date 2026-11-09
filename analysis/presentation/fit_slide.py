"""Fit slide (16:9 PNG): each task's own-head score on its training scans vs held-out test scans, against how many
labelled training scans the task has. Data-rich tasks sit on the diagonal (not memorised: capacity / compute is the
limit); small-cohort tasks score near-perfectly on their training scans and drop on test (memorised: data is the
limit). Tasks at chance on both splits or with negative R² (volume targets on 60-scan test sets) are left out and counted.
    python analysis/presentation/fit_slide.py TRAIN_CSV TEST_CSV LABEL_COUNTS_CSV OUT.png"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED = "#1d1d1b", "#55544f"
TYPE = {"auc": ("#2a78d6", "binary (AUC)"), "r2": ("#eb6834", "regression (R²)"),
        "dice": ("#1baf7a", "segmentation (Dice)"), "bacc": ("#6250d6", "multiclass (bal. acc.)")}
MODEL = "full_iso_bank/last"
SHOW = {"glioma_mgmt": "MGMT", "glioma_survival_12m": "12-mo survival", "neuro_diagnosis": "neuro diagnosis",
        "infant_abnormal": "infant abnormal", "mmse_grade": "MMSE grade"}

tr, te = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
tr, te = (d[(d.model == MODEL) & (d.kind == "heads") & (d.metric != "fp")] for d in (tr, te))
n = pd.read_csv(sys.argv[3]).set_index("task")["n_labels"]
m = tr.merge(te, on=["task", "metric"], suffixes=("_train", "_test"))
m = m[m.metric.isin(TYPE)]
m["n_labels"] = m.task.map(n)
keep = (m.score_train >= 0.3) & (m.score_test >= 0.3) & m.n_labels.notna()
dropped, m = int((~keep).sum()), m[keep].copy()
m["gap"] = m.score_train - m.score_test
out = Path(sys.argv[4])
out.parent.mkdir(parents=True, exist_ok=True)
m.to_csv(out.with_suffix(".csv"), index=False)

fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
fig.text(0.06, 0.92, "Data-rich tasks are not memorised; small cohorts are", fontsize=19, fontweight="bold",
         color=INK)
fig.text(0.06, 0.87, "PLACEHOLDER", fontsize=12, color=MUTED)

# (a) train vs test
ax = fig.add_axes([0.06, 0.12, 0.37, 0.66])
for k, (c, lab) in TYPE.items():
    d = m[m.metric == k]
    ax.scatter(d.score_test, d.score_train, s=36, color=c, label=lab, alpha=0.85, ec="white", lw=0.5)
ax.plot([0.3, 1], [0.3, 1], color=INK, lw=1, ls="--")
ax.text(0.97, 0.94, "train = test", rotation=45, ha="right", va="top", fontsize=9, color=MUTED)
ax.set_xlim(0.3, 1.01)
ax.set_ylim(0.3, 1.01)
ax.set_xlabel("held-out test score", fontsize=12, color=INK)
ax.set_ylabel("score on its own training scans", fontsize=12, color=INK)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=10, loc="lower right")

# (b) gap vs labelled training scans
ax2 = fig.add_axes([0.55, 0.12, 0.42, 0.66])
for k, (c, _) in TYPE.items():
    d = m[m.metric == k]
    ax2.scatter(d.n_labels, d.gap, s=36, color=c, alpha=0.85, ec="white", lw=0.5)
ax2.set_xscale("log")
ax2.axhline(0, color=INK, lw=1, ls="--")
for t, lab in SHOW.items():
    r = m[m.task == t]
    if len(r):
        r = r.iloc[0]
        ax2.annotate(lab, (r.n_labels, r.gap), xytext=(5, 4), textcoords="offset points", fontsize=9, color=INK)
big, small = m[m.n_labels >= 5000], m[m.n_labels < 2000]
ax2.annotate("hemorrhage types, anatomy masks,\nCT measures, brain volume:\non the line", (2.2e4, 0.01),
             xytext=(8e4, -0.235), va="bottom", fontsize=10, color=INK, ha="right",
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax2.set_xlabel("labelled training scans for the task (log scale)", fontsize=12, color=INK)
ax2.set_ylabel("train − test score", fontsize=12, color=INK)
ax2.spines[["top", "right"]].set_visible(False)
fig.text(0.06, 0.025, f"{len(m)} tasks shown; {dropped} left out (near chance on both splits, or negative R²: "
         "SynthSeg volume targets on ~60-scan test sets). Training scans scored with the same heads, no augmentation.",
         fontsize=9, color=MUTED)
fig.texts[1].set_text(f"Own heads, training vs held-out test scans. Median train − test gap {big.gap.median():+.2f} "
                      f"with ≥ 5,000 labelled scans (n = {len(big)}), {small.gap.median():+.2f} with < 2,000 "
                      f"(n = {len(small)})")
fig.savefig(out, facecolor="white")
print(out, f"big {big.gap.median():+.3f} n={len(big)} | small {small.gap.median():+.3f} n={len(small)} | dropped {dropped}")
