"""Validation scores of a multitask / joint run across its eval steps: group skill lines + a per-task skill heatmap.

    python runs/diagnostics/plot_a1_tasks.py [runs/joint/a1_warm_v3] [out.png]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from brain.eval.skill import skill  # noqa: E402
from brain.multitask.registry import load_registry  # noqa: E402

run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/joint/a1_warm_v3")
out = Path(sys.argv[2] if len(sys.argv) > 2 else f"runs/diagnostics/{run.name}_task_scores.png")
hist = [r for r in json.loads((run / "metrics.json").read_text())["history"] if "val_score" in r]
steps = [r["step"] for r in hist]
reg = load_registry(None)

# reference palette (dataviz skill, light mode): categorical order, ink, blue sequential ramp
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])

METRIC = (("_auc", "auc"), ("_r2", "r2"), ("_bacc", "bacc"), ("_dice", "dice"))
rows = {}                                                    # task -> (group, metric, [skill per step])
for t in reg.tasks:
    for suf, m in METRIC:
        k = f"val_{t.id}{suf}"
        if any(k in r for r in hist):
            vals = [r.get(k, np.nan) for r in hist]
            n_cls = len(t.classes) if m == "bacc" and t.classes else 2
            rows[t.id] = (t.group, m, [float(skill(m, v, n_cls)) if v == v else np.nan for v in vals])
            break
groups = sorted({g for g, _, _ in rows.values()})
order = sorted(rows, key=lambda t: (groups.index(rows[t][0]), -np.nan_to_num(rows[t][2][-1], nan=-9)))

fig = plt.figure(figsize=(13, 4.2 + 0.2 * len(order)), facecolor=SURFACE)
gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 0.2 * len(order)], hspace=0.12)

# 1. group skill + overall score
ax = fig.add_subplot(gs[0], facecolor=SURFACE)
ends = []
for i, g in enumerate(groups):
    y = [r.get(f"val_group_{g}", np.nan) for r in hist]
    ax.plot(steps, y, color=SERIES[i % len(SERIES)], lw=2, marker="o", ms=4, label=g.replace("_", " "))
    ends.append((y[-1], g.replace("_", " "), SERIES[i % len(SERIES)]))
score = [r["val_score"] for r in hist]
ax.plot(steps, score, color=INK, lw=2.5, ls="--", marker="o", ms=4, label="val_score (mean of groups)")
ends.append((score[-1], f"val_score {score[-1]:.3f}", INK))
ends.sort()
last = -1.0
for y, name, c in ends:                                     # direct labels at the line ends, nudged apart
    y_txt = max(y, last + 0.028)
    ax.annotate(name, (steps[-1], y), (steps[-1] + 0.012 * (steps[-1] - steps[0] + 1), y_txt), color=INK2,
                fontsize=8.5, va="center", annotation_clip=False)
    ax.plot([steps[-1] + 0.004 * (steps[-1] - steps[0] + 1)], [y_txt], marker="s", ms=5, color=c, clip_on=False)
    last = y_txt
ax.set_xlim(steps[0] - 300, steps[-1] + 300)
ax.set_ylabel("skill (0 = chance, 1 = perfect)", color=INK2)
ax.set_title(f"{run.name}: validation skill by task group, steps {steps[0]:,}-{steps[-1]:,}", color=INK, loc="left", pad=34,
             fontsize=12)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=INK2)
ax.set_xticks(steps)
ax.set_xticklabels([f"{s // 1000}k" for s in steps])
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False, ncol=5, labelcolor=INK2)

# 2. per-task heatmap
hx = fig.add_subplot(gs[1], facecolor=SURFACE)
M = np.array([rows[t][2] for t in order])
im = hx.imshow(np.clip(M, 0, 1), aspect="auto", cmap=RAMP, vmin=0, vmax=1, interpolation="nearest")
for i, t in enumerate(order):
    for j, v in enumerate(M[i]):
        if v == v:
            hx.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if v > 0.55 else INK)
hx.set_yticks(range(len(order)))
hx.set_yticklabels([f"{t}  ({rows[t][1]})" for t in order], fontsize=7.5, color=INK2)
hx.set_xticks(range(len(steps)))
hx.set_xticklabels([f"{s // 1000}k" for s in steps], fontsize=8, color=INK2)
prev = None
for i, t in enumerate(order):                                # group separators + group names on the right
    g = rows[t][0]
    if g != prev:
        if prev is not None:
            hx.axhline(i - 0.5, color=SURFACE, lw=3)
        n = sum(1 for u in order if rows[u][0] == g)
        hx.text(len(steps) - 0.3, i - 0.5 + n / 2, g.replace("_", " "), fontsize=8, color=INK, va="center",
                ha="left", rotation=0)
        prev = g
for s in hx.spines.values():
    s.set_visible(False)
hx.tick_params(length=0)
hx.set_title("per-task validation skill (AUC / R² / balanced accuracy rescaled so 0 = chance; Dice as is; "
             "values below 0 shown as 0 in colour)", color=INK2, loc="left", fontsize=9)
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=SURFACE)
print("wrote", out)
