"""Multitask vs single-task (16:9 PNG): per trained task, the new run's own head against the best
single-task model trained from scratch on that task family (build_focus.SPEC), on held-out test scans, against how
many labelled training scans the task has.
    python analysis/presentation/mt_gain_slide.py LABEL_COUNTS_CSV OUT.png"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
counts, out = pd.read_csv(sys.argv[1]).set_index("task")["n_labels"], Path(sys.argv[2])
sys.argv = [sys.argv[0], str(ROOT / "runs/bench/page/scores.json")]
sys.path.insert(0, str(ROOT / "analysis/bench_page"))
sys.path.insert(0, str(ROOT))
import build_focus as bf  # noqa: E402

INK, MUTED = "#1d1d1b", "#55544f"
TYPE = {"auc": ("#2a78d6", "binary (AUC)"), "r2": ("#eb6834", "regression (R²)"),
        "dice": ("#1baf7a", "segmentation (Dice)"), "bacc": ("#6250d6", "multiclass (bal. acc.)")}
SHOW = {"enhancing_tumor": "enhancing tumour", "tumor_core": "tumour core", "sex": "sex", "age": "age",
        "glioma_survival_12m": "12-mo survival", "neuro_diagnosis": "neuro diagnosis", "tumor_type": "tumour type",
        "ct_brain_density": "CT brain density", "mri_edema": "MRI edema", "hemorrhage_epidural": "epidural",
        "mri_artifact": "MRI artifact"}
SHOW.pop("hemorrhage_epidural")
te = pd.read_csv(ROOT / "runs/diagnostics/te_full_iso/test_eval_heads.csv")
te = te[(te.model == "full_iso_bank/last") & (te.kind == "heads") & (te.metric != "fp")]
rows = []
for r in te.itertuples():
    s = bf.SPEC.get(r.task)
    if s is None or r.metric not in TYPE or r.task not in counts:
        continue
    rows.append({"task": r.task, "metric": r.metric, "mt": r.score, "single": s["v"], "gain": r.score - s["v"],
                 "n_labels": int(counts[r.task])})
d = pd.DataFrame(rows)
d = d[(d.mt > 0) | (d.single > 0)]                 # both negative R² (volume targets on ~60 scans): uninformative
d.to_csv(out.with_suffix(".csv"), index=False)

fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
ax = fig.add_axes([0.07, 0.11, 0.52, 0.7])
lo = 0.2
for k, (c, lab) in TYPE.items():
    x = d[d.metric == k]
    sx = x.single.clip(lower=lo)
    ax.scatter(sx, x.mt.clip(lower=lo), s=18 + 14 * np.log10(x.n_labels), color=c, label=lab, alpha=0.88,
               ec="white", lw=0.6)
ax.plot([lo, 1], [lo, 1], color=INK, lw=1, ls="--")
ax.fill_between([lo, 1], [lo - 0.03, 0.97], [lo + 0.03, 1.03], color="#d9d8d2", alpha=0.5, lw=0)
for t, lab in SHOW.items():
    x = d[d.task == t]
    if len(x):
        px, py_ = max(x.single.iloc[0], lo), max(x.mt.iloc[0], lo)
        off = {"enhancing_tumor": (-8, 9, "right"), "tumor_core": (6, 5, "left"), "neuro_diagnosis": (6, -13, "left"),
               "ct_brain_density": (-8, -14, "right"), "mri_edema": (-8, -14, "right")}
        if t in ("ct_brain_density", "mri_edema"):           # crowded corner: label in the empty space below
            ax.annotate(lab, (px, py_), xytext=(0.99, 0.77 if t == "mri_edema" else 0.72), textcoords="data",
                        fontsize=9.5, color=INK, ha="right", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
            continue
        dx, dy, ha = off.get(t, (6, -3, "left"))
        ax.annotate(lab, (px, py_), xytext=(dx, dy), textcoords="offset points", fontsize=9.5, color=INK, ha=ha)
ax.set_xlim(lo, 1.0)
ax.set_ylim(lo, 1.0)
ax.set_xlabel("single-task model, trained from scratch (test score)", fontsize=12, color=INK)
ax.set_ylabel("one multitask model (test score)", fontsize=12, color=INK)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=10, loc="lower right", markerscale=0.8)
near, ahead, behind = (d.gain.abs() <= 0.03).sum(), (d.gain > 0.03).sum(), (d.gain < -0.03).sum()
fig.text(0.07, 0.93, "One model matches the specialists where they do well, and rescues them where they fail",
         fontsize=17, fontweight="bold", color=INK)
fig.text(0.07, 0.875, f"{len(d)} trained tasks scored by both on held-out test scans; dot size = labelled "
         "training scans", fontsize=11.5, color=MUTED)
fig.text(0.65, 0.74, f"within ±0.03 (grey band): {near} tasks\n   anatomy masks, hemorrhage types\n\n"
         f"multitask ahead by > 0.03: {ahead} tasks\n   tumour sub-regions (+0.28-0.32), sex,\n"
         f"   diagnoses, survival, tumour type, age\n\n"
         f"multitask behind by > 0.03: {behind} tasks\n   CT density / CSF fraction, MRI edema,\n"
         f"   artifact and mass findings, MGMT, epidural",
         fontsize=11.5, color=INK, va="top", linespacing=1.4)
fig.text(0.65, 0.22, "Single-task: per task family, the best of three from-scratch\narchitectures, each trained on that "
         "family's tasks only.\nScores below 0.2 drawn at 0.2 (volume asymmetries whose\nsingle-task R² is negative). "
         "One seed per model;\ntest scans from each task's test split (label tables may\ndiffer slightly between runs).", fontsize=9, color=MUTED, va="top", linespacing=1.4)
fig.savefig(out, facecolor="white")
print(out, f"near {near} ahead {ahead} behind {behind}")
