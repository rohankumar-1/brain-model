"""Slide diagram (16:9 PNG) of how BrainQuery is trained: scans -> learned window bank -> patch embedding -> one
transformer with a token per task -> typed heads, with the task groups below. Simplified on purpose.
    python analysis/presentation/training_diagram.py OUT.png"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, INK2, LINE, FILL = "#1d1d1b", "#55544f", "#b9b8b2", "#f4f3ef"
TYPE = {"binary": ("#2a78d6", "Binary", 20), "regression": ("#eb6834", "Regression", 18),
        "multiclass": ("#6250d6", "Multiclass", 4), "segmentation": ("#1baf7a", "Segmentation", 20),
        "dense": ("#8a8983", "Dense / self-supervised", 6)}
GROUPS = [("Anatomy masks", 12, "segmentation"), ("Anatomy volumes", 10, "regression"),
          ("Radiology findings", 9, "binary"), ("Tumour", 7, "segmentation"), ("Hemorrhage", 6, "binary"),
          ("CT measures", 6, "regression"), ("Sequence synthesis", 5, "dense"), ("Brain mask & asymmetry", 4,
                                                                                 "regression"),
          ("Diagnosis", 3, "multiclass"), ("Age & sex", 2, "regression"), ("Clinical scores", 2, "multiclass"),
          ("Lesion masks", 1, "segmentation"), ("Masked reconstruction", 1, "dense")]

fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")


def box(x, y, w, h, title, sub=None, fc=FILL, ec=LINE, tc=INK, fs=12, r=0.18):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=1.2))
    if sub:
        ax.text(x + w / 2, y + h / 2 + 0.17, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=fs - 3, color=INK2,
                linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc)


def arrow(x0, y0, x1, y1, c=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=c, lw=1.4))


# ---- inputs ---------------------------------------------------------------------------------------------------------
Y = 5.75                                                   # centre line of the main flow
box(0.35, Y + 0.25, 2.1, 1.0, "CT scan", "Hounsfield units")
box(0.35, Y - 1.25, 2.1, 1.0, "MRI scan", "T1, T2, FLAIR, T1c, ...")
ax.text(0.35, Y - 1.62, "1 mm grid · native-resolution CT in progress", ha="left", va="center", fontsize=8.5, color=INK2)
arrow(2.5, Y + 0.75, 3.05, Y + 0.12)
arrow(2.5, Y - 0.75, 3.05, Y - 0.12)

# ---- window bank, patch embedding ---------------------------------------------------------------------------------
box(3.0, Y - 0.6, 2.1, 1.2, "Window bank", "6 learned intensity\nwindows, CT + MRI")
arrow(5.15, Y, 5.45, Y)
box(5.5, Y - 0.6, 2.25, 1.2, "Patch embedding", "16 mm patches\n→ image tokens")
arrow(7.8, Y, 8.05, Y)

# ---- transformer with task tokens ---------------------------------------------------------------------------------
tx, tw = 8.1, 4.2
box(tx, Y - 1.35, tw, 2.7, "", fc="white", ec=INK2)
ax.text(tx + tw / 2, Y + 1.0, "Transformer encoder", ha="center", va="center", fontsize=13, fontweight="bold",
        color=INK)
ax.text(tx + tw / 2, Y + 0.62, "one pass answers every task", ha="center", va="center", fontsize=9, color=INK2)
# token row: task tokens (coloured by output type) then image tokens (grey)
order = ["binary", "regression", "multiclass", "segmentation", "dense"]
sz, gap, x0, ty = 0.3, 0.07, tx + 0.35, Y - 0.32
for i, t in enumerate(order):
    ax.add_patch(FancyBboxPatch((x0 + i * (sz + gap), ty), sz, sz, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=TYPE[t][0], ec="white", lw=0.8))
xi = x0 + len(order) * (sz + gap) + 0.35
for i in range(4):
    ax.add_patch(FancyBboxPatch((xi + i * (sz + gap), ty), sz, sz, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="#d6d5cf", ec="white", lw=0.8))
mid_t = x0 + len(order) * (sz + gap) / 2 - gap / 2
mid_i = xi + 4 * (sz + gap) / 2 - gap / 2
ax.text(mid_t, ty - 0.28, "task tokens", ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
ax.text(mid_t, ty - 0.6, "one per task:\ntask + output-type embedding", ha="center", va="center", fontsize=8,
        color=INK2, linespacing=1.2)
ax.text(mid_i, ty - 0.28, "image tokens", ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
ax.text(mid_i, ty - 0.6, "+ position in mm", ha="center", va="center", fontsize=8, color=INK2)
arrow(tx + tw + 0.05, Y, tx + tw + 0.45, Y)

# ---- typed heads --------------------------------------------------------------------------------------------------
hx, hw, hh = 12.75, 2.95, 0.5
ax.text(hx + hw / 2, Y + 1.62, "Heads, shared per output type", ha="center", va="center", fontsize=11.5,
        fontweight="bold", color=INK)
ys = [Y + 1.0, Y + 0.38, Y - 0.24, Y - 0.86, Y - 1.48]
for (k, (c, name, n)), y in zip(TYPE.items(), ys):
    box(hx, y - hh / 2, hw, hh, f"{name}  ({n})", fc=c, ec=c, tc="white", fs=10.5, r=0.12)
    arrow(hx - 0.32, Y, hx - 0.02, y, c=LINE)
ax.text(hx + hw / 2, Y - 2.0, "segmentation: a decoder reads the image\ntokens, conditioned on the task token",
        ha="center", va="center", fontsize=8, color=INK2, linespacing=1.2)

# ---- training tasks -------------------------------------------------------------------------------------------------
ax.plot([0.35, 15.7], [2.25, 2.25], color=LINE, lw=1)
ax.text(0.35, 1.92, "68 training tasks, 13 groups", ha="left", va="center", fontsize=12, fontweight="bold",
        color=INK)
ax.text(15.7, 1.92, "colour = output type  ·  each scan brings only the labels it has; the loss sums over the tasks present",
        ha="right", va="center", fontsize=9, color=INK2)
cols, cw, x, y = 5, 3.07, 0.35, 1.35
for i, (g, n, t) in enumerate(GROUPS):
    cx, cy = x + (i % cols) * cw, y - (i // cols) * 0.5
    ax.add_patch(FancyBboxPatch((cx, cy - 0.15), 0.3, 0.3, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=TYPE[t][0], ec="none"))
    ax.text(cx + 0.42, cy, f"{g}", ha="left", va="center", fontsize=10, color=INK)
    ax.text(cx + cw - 0.25, cy, f"{n}", ha="right", va="center", fontsize=10, color=INK2, fontweight="bold")

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, facecolor="white")
print(out)
