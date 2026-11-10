"""Training-objective slide (16:9 PNG): how tasks are sampled, each output type's loss, the balancing formula, and the
learned per-task loss weights exp(-s_t) of a checkpoint (the uncertainty balancer, clamped to s in [-1, 1]).
    python analysis/presentation/objective_slide.py CKPT OUT.png"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brain.multitask.registry import load_registry  # noqa: E402

INK, MUTED = "#1d1d1b", "#55544f"
TYPE = {"binary": ("#2a78d6", "binary"), "regression": ("#eb6834", "regression"),
        "multiclass": ("#6250d6", "multiclass"), "segmentation": ("#1baf7a", "segmentation"),
        "dense": ("#8a8983", "dense / self-supervised")}
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
cfg = ck["config"]
clamp = float((cfg.get("loss") or {}).get("uncertainty_clamp") or 1.0)
lv = ck["balancer"]["log_var"].float().numpy()
tasks = list(ck["tasks"])
reg = load_registry(cfg["data"].get("registry")).subset(tasks)
trained = set(load_registry(cfg["data"].get("registry")).train_ids(cfg["tasks"]))
w = np.exp(-clamp * np.tanh(lv / clamp))
rows = sorted([(t, reg[t].type, float(x)) for t, x in zip(tasks, w) if t in trained], key=lambda r: r[2])
top = np.exp(clamp)
n_top = sum(1 for r in rows if r[2] > 0.99 * top)

fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
fig.text(0.04, 0.94, "Training objective", fontsize=20, fontweight="bold", color=INK)
tc = cfg["tasks"]
gw = tc.get("group_weights") or {}
blocks = [
    ("1  What each scan is asked",
     f"Each training scan asks 1-{tc['k_max']} of its labelled tasks (all of them with p = {tc['p_all']}).\n"
     f"Task groups are drawn first, then tasks within a group, rare tasks\nup-weighted (∝ n^-{tc['rare_beta']}). "
     f"Cohorts sampled ∝ n^{cfg['data']['balance_alpha']}, each subject once."),
    ("2  One loss per output type",
     "binary: cross-entropy     multiclass: cross-entropy\n"
     "regression: HL-Gauss (80-bin histogram loss, standardised targets)\n"
     "segmentation: BCE + soft Dice, also on lesion-free scans (empty masks)\n"
     "dense (synthesis, masked reconstruction): voxel Huber\n"
     "+ ×0.5 auxiliaries: token occupancy, RSNA slice labels, fastMRI boxes"),
    ("3  Balancing",
     "L  =  Σ_g w_g · mean_{t ∈ g} ( e^{-s_t} L_t + s_t )  /  Σ_g w_g\n"
     f"every task group counts once, whatever its size (w_g = 1; {', '.join(f'{k} {v}' for k, v in gw.items() if v != 1)});\n"
     f"s_t learned per task (Kendall et al. 2018), clamped to [-{clamp:g}, {clamp:g}]"),
]
y = 0.85
for head, body in blocks:
    fig.text(0.04, y, head, fontsize=13, fontweight="bold", color="#1c5cab", va="top")
    fig.text(0.04, y - 0.045, body, fontsize=10.6, color=INK, va="top", linespacing=1.45)
    y -= 0.045 + 0.034 * (body.count("\n") + 1) + 0.05

ax = fig.add_axes([0.62, 0.08, 0.35, 0.72])
ys = np.arange(len(rows))
ax.barh(ys, [r[2] for r in rows], color=[TYPE[r[1]][0] for r in rows], height=0.8)
ax.axvline(top, color=INK, lw=1, ls="--")
ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
ax.text(top, len(rows) + 0.6, f" clamp e^{clamp:g}", fontsize=9, color=INK, ha="right", va="bottom")
ax.set_yticks([])
ax.set_ylim(-1, len(rows) + 2)
ax.set_xlim(0, top * 1.05)
ax.set_xlabel("learned loss weight e^{-s_t}", fontsize=11, color=INK)
for i, (t, typ, x) in enumerate(rows[:12]):
    ax.text(x + 0.04, i, t.replace("_", " "), fontsize=7, va="center", color=INK)
ax.spines[["top", "right", "left"]].set_visible(False)
fig.text(0.62, 0.93, f"Learned weights, {len(rows)} trained tasks (end of training)", fontsize=12.5, color=INK)
present = [k for k in TYPE if any(r[1] == k for r in rows)]
fig.legend(handles=[Patch(fc=TYPE[k][0], label=TYPE[k][1]) for k in present], loc="upper left", frameon=False,
           fontsize=9, ncol=3, bbox_to_anchor=(0.615, 0.915), handlelength=1.2, columnspacing=1.0)
fig.text(0.04, 0.08, f"Reading the weights: {n_top} of {len(rows)} tasks sit at the clamp, where the learned weighting is "
         f"a fixed\n×{top:.1f}; the balance there comes from group averaging and task sampling. The weighting\n"
         "does work on regression: noisy asymmetry and volume targets drop to 0.66-1.2.",
         fontsize=10.5, color=MUTED, va="bottom", linespacing=1.45)
out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor="white")
print(out, f"{n_top}/{len(rows)} at clamp")
