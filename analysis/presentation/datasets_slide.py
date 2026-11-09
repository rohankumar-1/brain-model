"""Slide version (16:9 PNG) of the dataset map: two butterfly panels on a log scale, training scans left of each
panel's centre column, test scans right (held-out test split of a training cohort: light tint; external cohort:
hatched), rows coloured by task type, with a legend for task types and block types.
    python analysis/presentation/datasets_slide.py OUT.png"""
import sys
from pathlib import Path

import matplotlib

matplotlib.rcParams["hatch.linewidth"] = 1.2

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from datasets_figure import dataset_data  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SHORT = {"CT · intracranial hemorrhage": "CT · hemorrhage", "CT · stroke and vessels": "CT · stroke & vessels",
         "CT · pediatric and synthesis": "CT · pediatric & synthesis", "MRI · tumour": "MRI · tumour",
         "MRI · neurodegeneration": "MRI · neurodegeneration",
         "MRI · development and psychiatry": "MRI · development & psychiatry",
         "MRI · lesions and small vessels": "MRI · lesions & small vessels",
         "MRI · healthy and clinical": "MRI · healthy & clinical"}
INK, INK2, RULE = "#1d1d1b", "#55544f", "#d9d8d2"
D = dataset_data()
groups, rows, tot = D["groups"], D["rows"], D["tot"]
LO, HI = 10, 40000                                     # log axis range (scans)
GAP = 1.55                                             # half-width of the centre label column, in decades


def x_of(v):
    return np.log10(max(v, LO)) - np.log10(LO)         # 0 at 10 scans


SPAN = x_of(HI)


def panel(ax, gidx):
    y = 0
    ax.set_xlim(-(GAP + SPAN), GAP + SPAN)
    for t in (10, 100, 1000, 10000):
        for s in (-1, 1):
            xx = s * (GAP + x_of(t))
            pass
    for gi in gidx:
        c = COLORS[gi]
        lab = SHORT.get(groups[gi], groups[gi])
        ax.text(0, y, lab, ha="center", va="center", fontsize=8.6, fontweight="bold", color=INK,
                bbox=dict(boxstyle="round,pad=0.28", facecolor=c, alpha=0.28, edgecolor="none"))
        y -= 1
        for r in (r for r in rows if r["g"] == gi):
            ax.text(0, y, r["name"], ha="center", va="center", fontsize=7.6, color=INK)
            if r["train"]:
                w = max(x_of(r["train"]), 0.04)
                ax.barh(y, -w, left=-GAP, height=0.68, color=c, edgecolor="white", linewidth=0.5)
            tv = r["test"] or r["ext"]
            if tv:
                w = max(x_of(tv), 0.04)
                if r["test"]:
                    ax.barh(y, w, left=GAP, height=0.68, color=c, alpha=0.42, edgecolor="white", linewidth=0.5)
                else:
                    ax.barh(y, w, left=GAP, height=0.68, facecolor=c, edgecolor="white", linewidth=0.0,
                            hatch="///")
            y -= 1
        y -= 0.35
    ax.set_ylim(y + 0.2, 0.9)
    ticks = [10, 100, 1000, 10000]
    pos = [-(GAP + x_of(t)) for t in ticks] + [GAP + x_of(t) for t in ticks]
    lab = ["10", "100", "1k", "10k"] * 2
    ax.set_xticks(pos, lab, fontsize=7, color=INK2)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-GAP - 0.05, 1.25, "← training scans", ha="right", va="center", fontsize=8, color=INK2, fontweight="bold")
    ax.text(GAP + 0.05, 1.25, "test scans →", ha="left", va="center", fontsize=8, color=INK2, fontweight="bold")


fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
fig.patch.set_facecolor("white")
left = fig.add_axes([0.01, 0.12, 0.48, 0.83])
right = fig.add_axes([0.51, 0.12, 0.48, 0.83])
panel(left, [0, 1, 2, 3])
panel(right, [4, 5, 6, 7])
task_handles = [Patch(facecolor=COLORS[i], label=SHORT.get(g, g)) for i, g in enumerate(groups)]
kind_handles = [Patch(facecolor="#8a8983", label="training scans"),
                Patch(facecolor="#8a8983", alpha=0.42, label="held-out test (same cohort)"),
                Patch(facecolor="#8a8983", hatch="///", edgecolor="white", linewidth=0, label="external test cohort")]
leg = fig.legend(handles=task_handles + kind_handles, loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=6,
                 fontsize=8.6, frameon=False, handlelength=1.4, columnspacing=1.6)
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, facecolor="white")
print(out)
