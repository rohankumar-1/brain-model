"""Writes notebooks/run_report.ipynb, and the same code as a script for a headless check (figures to runs/diagnostics/nb).

    python notebooks/make_run_report_nb.py notebooks/run_report.ipynb runs/diagnostics/nb/run_report_check.py
"""
import json
import sys
from pathlib import Path

CELLS = [
("md", """# Run report

Every model against one reference run, on three questions (`brain/eval/report.py`, task lists in `configs/eval/core.yaml`):

| Headline | Question | Source |
|---|---|---|
| **In-domain** | Does it do the tasks it trained on? | its own heads on the test splits, paired with the reference on the same scans |
| **Transfer** | Is the frozen encoder a good foundation? | 21 core downstream tasks in 5 groups (noisy tasks excluded) |
| **Adaptation** | Can it learn a new task from 64 labels? | few-shot on held-out tasks |

All metrics are **skill**: 0 = chance, 1 = perfect (AUROC → `(auc − 0.5) / 0.5`, R² clipped at 0, Dice as is). A headline is the mean over groups of the mean over tasks, on tasks both models have; intervals combine per-task intervals assuming independence (approximate).

Edit `MODELS` / `REF` below and re-run from the top. One run's report alone: `runs/eval/<run>/report.md` (`jobs/submit_report.sh RUN REF`)."""),
("code", """import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# make the repo importable whatever the kernel's working directory or environment
REPO = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "brain" / "__init__.py").exists()),
            Path("/home/roku/brain-model"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brain.eval.report import compare, headlines, val_history

warnings.filterwarnings("ignore")
R = Path("/orcd/scratch/orcd/011/roku/brain-model/runs")
MODELS = {                                    # display name -> checkpoint; order = colour order
    "v3 SSL": R / "v3/snapshots/v3_cls_s42/step_0018000.pt",
    "run2 synthseg": R / "multitask/snapshots/mt_synthseg_s42/step_0017000.pt",
    "run3 general": R / "multitask/v4general_s42/best.pt",
    "run4 enc-lr": R / "multitask/v5enclr_s42/best.pt",
    "run5 uncert": R / "multitask/v6nogenunc_s42/best.pt",
    "run6 no-small": R / "multitask/v7nosmall_s42/best.pt",
    "joint pilot": R / "joint/pilot_joint/best.pt",
    "single hemorrhage": R / "single/hemorrhage/best.pt",
}
REF = "run6 no-small"
VAL_DIRS = {"run2 synthseg": R / "multitask/v3synthseg_s42"}   # training dirs of snapshot copies (their metrics.json)
HEADS_CSV = Path("/home/roku/brain-model/runs/eval/heads_all/test_eval_heads.csv")   # test_eval over all MODELS, ref first
PARTS = {"in_domain": "In-domain", "transfer": "Transfer", "adaptation": "Adaptation"}

# fixed categorical order (validated reference palette); text stays in ink colours, never the series colour
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR = dict(zip(MODELS, SERIES))
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"])
DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])     # worse | same | better
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

rows = compare(MODELS, REF, HEADS_CSV)
H = headlines(rows)
print(f"{len(rows)} task rows; reference = {REF}")"""),
("md", """## 1. Headlines

`score` is the model's skill on the tasks it shares with the reference; `Δ` is model − reference with its approximate 95% interval. Blue = better than the reference, orange = worse; **bold** = interval excludes 0."""),
("code", """def cell(r):
    if not np.isfinite(r.delta):
        return ""
    sig = r.lo > 0 or r.hi < 0
    txt = f"{r.run:.3f}  ({r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}])"
    return f"**{txt}**" if sig else txt

tbl = H.assign(text=H.apply(cell, axis=1)).pivot(index="model", columns="part", values="text")
tbl = tbl.reindex(index=[m for m in MODELS if m in tbl.index], columns=[p for p in PARTS if p in tbl.columns])
tbl.columns = [PARTS[c] for c in tbl.columns]
delta = H.pivot(index="model", columns="part", values="delta").reindex(index=tbl.index, columns=list(PARTS))
delta.columns = [PARTS[c] for c in delta.columns]
(tbl.fillna("—").style
    .apply(lambda col: [f"background-color: {'#d6e6fa' if d > 0.005 else '#fbdccd' if d < -0.005 else ''}"
                        for d in delta[col.name]], axis=0)
    .format(lambda s: s.replace("**", ""))
    .apply(lambda col: ["font-weight: bold" if "**" in str(v) else "" for v in col], axis=0)
    .set_caption(f"skill (Δ vs {REF} [95% CI]); — = not applicable (no heads / no few-shot) or no shared tasks"))"""),
("code", """fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
order = [m for m in MODELS if m != REF]
for ax, (p, name) in zip(axes, PARTS.items()):
    d = H[(H.part == p)].set_index("model").reindex(order)
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        r = d.loc[m]
        if not np.isfinite(r.delta):
            ax.text(0, yi, "n/a", ha="center", va="center", color=INK2, fontsize=8)
            continue
        ax.plot([r.lo, r.hi], [yi, yi], color=COLOR[m], lw=2, solid_capstyle="round")
        ax.plot(r.delta, yi, "o", ms=8, color=COLOR[m], mec=SURFACE, mew=2)
    ax.axvline(0, color=INK2, lw=1)
    if not np.isfinite(d.delta).any():
        ax.set_xlim(-0.1, 0.1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_title(f"{name}: Δ skill vs {REF}", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("← worse      better →")
axes[0].set_yticks(np.arange(len(order))[::-1], order)
fig.tight_layout()
plt.show()"""),
("md", """## 2. Where: group scores

Each model's mean skill per group, on the tasks it has (absolute, not paired). Groups are registry task groups (in-domain), the core-set groups (transfer) and held-out task families (adaptation)."""),
("code", """def heat(part, ax):
    d = rows[rows.part == part].dropna(subset=["run"])
    if d.empty:                                # keep the axis: it carries the shared model labels
        ax.set_ylim(len(MODELS) - 0.5, -0.5)
        ax.set_xticks([])
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        ax.text(0.5, 0.5, "no results yet", transform=ax.transAxes, ha="center", color=INK2)
        ax.set_title(PARTS[part], loc="left", fontsize=10)
        return None
    g = d.groupby(["model", "group"]).run.mean().unstack().reindex(list(MODELS))   # same rows in every panel
    im = ax.imshow(g.to_numpy(), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(g.to_numpy()):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=SURFACE if v > 0.55 else INK)
    ax.set_xticks(range(g.shape[1]), g.columns, rotation=35, ha="right")
    ax.set_yticks(range(g.shape[0]), g.index)
    ax.tick_params(length=0)
    ax.set_title(PARTS[part], loc="left", fontsize=10)
    ax.spines[:].set_visible(False)
    return im

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True, layout="constrained",
                         gridspec_kw={"width_ratios": [7, 5, 3]})
ims = [heat(p, ax) for ax, p in zip(axes, PARTS)]
axes[0].set_yticks(range(len(MODELS)), list(MODELS))
im = next((i for i in ims if i is not None), None)
if im is not None:
    fig.colorbar(im, ax=axes, shrink=0.8, label="skill (0 = chance, 1 = perfect)")
plt.show()"""),
("md", """## 3. Training validation curves

The validation score each run logged during training (recomputed on the skill scale from `metrics.json`, so older runs are comparable). **The joint pilot's curve (dashed) is on a different validation set**: it started before the stratified validation picker (2026-10-24 21:34), so its validation scans came from one cohort per task; the same weights score 0.44 on today's set (run 6: 0.64). Each run is scored on the tasks it trained, so a run with a different task set (run 2: SynthSeg anatomy only) is on a different, easier mix: compare curve shapes across such runs, not levels."""),
("code", """fig, ax = plt.subplots(figsize=(8, 3.6))
for m, ck in MODELS.items():
    if "single" in m:                          # validated on its own task only: not on the shared scale
        continue
    try:
        v = val_history(VAL_DIRS.get(m, ck.parent))
    except FileNotFoundError:                  # snapshots keep no metrics.json
        continue
    v = v[v.group == "score"]
    if v.empty:
        continue
    ax.plot(v.step, v.skill, "--" if "joint" in m else "-", color=COLOR[m], lw=2, label=m)
ax.set_xlabel("step"); ax.set_ylabel("validation skill")
ax.grid(color=GRID, lw=0.8)
ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=2)
fig.tight_layout(); plt.show()"""),
("md", """## 4. Per-task detail

Δ skill vs the reference per task (blue better, orange worse). Use this to see which tasks drive a group's change; single tasks are noisy, so read them in groups."""),
("code", """def task_table(part):
    d = rows[(rows.part == part) & (rows.model != REF)].dropna(subset=["run", "ref"])
    if d.empty:
        return pd.DataFrame()
    t = (d.assign(delta=d.run - d.ref).pivot_table(index=["group", "task"], columns="model", values="delta")
         .reindex(columns=[m for m in MODELS if m != REF and m in set(d.model)]))
    t.insert(0, f"{REF} skill", d.drop_duplicates("task").set_index(["group", "task"]).ref.reindex(t.index))
    return t

for p in PARTS:
    t = task_table(p)
    if t.empty:
        continue
    cols = [c for c in t.columns if c != f"{REF} skill"]
    display(t.style.format("{:+.3f}", subset=cols, na_rep="").format("{:.3f}", subset=[f"{REF} skill"])
            .background_gradient(cmap=DIV, subset=cols, vmin=-0.2, vmax=0.2)
            .set_caption(f"{PARTS[p]}: Δ skill per task vs {REF}"))"""),
("code", """# missing results: what still needs to run before a comparison is complete
miss = rows[rows.run.isna() | rows.ref.isna()]
miss.groupby(["model", "part"]).task.apply(lambda s: ", ".join(s)).to_frame("missing") if len(miss) else "nothing missing\""""),
]


def main(out_nb: Path, out_py: Path) -> None:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "ml-torch", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out_nb.write_text(json.dumps(nb, indent=1) + "\n")
    code = [src.replace("plt.show()", "_show()") for kind, src in CELLS if kind == "code"]
    script = f"""import ast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_n = [0]
def _show():
    _n[0] += 1
    plt.savefig(f"runs/diagnostics/nb/fig{{_n[0]}}.png", dpi=80)
    plt.close("all")
def _out(x):
    print(x.data.to_string() if hasattr(x, "data") else x)
display = _out
g = globals()
for src in {code!r}:                                   # notebook semantics: a final expression is displayed
    tree = ast.parse(src)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, "<cell>", "exec"), g)
    if last is not None:
        _out(eval(compile(ast.Expression(last.value), "<cell>", "eval"), g))
"""
    out_py.write_text(script)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
