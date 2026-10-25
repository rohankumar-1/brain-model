"""Per-task validation trajectories across training runs: what each task learns, when, and whether it later declines.

    python -m brain.multitask.trajectories                      # every run under runs/{multitask,single,joint}
    python -m brain.multitask.trajectories --runs runs/multitask/v4general_s42 runs/joint/pilot_joint

Reads ``metrics.json`` (``history``: one row per evaluation, ``val_<task>_<metric>``) written by every multitask,
single-task and joint run, keeps each task's primary metric (auc / r2 / dice / bacc) and writes to ``--out``:

* ``curves_<group>.png``: one panel per task, one line per run (raw points faint, a 3-point running mean solid);
* ``order_<run>.png``: tasks x evaluation steps, each task scaled to its own min-max in that run -- which tasks learn
  early or late, and which peak and then decline (interference / overfitting);
* ``summary.csv``: per run and task, best score, its step, final score, final - best (``drop``), sorted by drop.

The training validation set is small (``data.val_per_task`` scans per task, 16 in the runs so far), so single points are
noisy: read the smoothed lines and the size of a drop against the curve's own jitter.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RUN_ROOT = Path("/orcd/scratch/orcd/011/roku/brain-model/runs")
PRIMARY = ("auc", "r2", "dice", "bacc")
METRIC_RE = re.compile(r"^val_(.+)_(auc|r2|dice|bacc)$")


def load_runs(dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        f = d / "metrics.json"
        if not f.exists():
            continue
        for r in json.load(open(f)).get("history", []):
            for k, v in r.items():
                m = METRIC_RE.match(k)
                if m and v is not None and np.isfinite(v):
                    rows.append({"run": f"{d.parent.name}/{d.name}", "step": int(r["step"]), "task": m.group(1),
                                 "metric": m.group(2), "value": float(v)})
            if "val_loss" in r:
                rows.append({"run": f"{d.parent.name}/{d.name}", "step": int(r["step"]), "task": "val_loss",
                             "metric": "loss", "value": float(r["val_loss"])})
    return pd.DataFrame(rows)


def discover() -> list[Path]:
    dirs = []
    for sub in ("multitask", "single", "joint"):
        dirs += [p.parent for p in sorted((RUN_ROOT / sub).glob("*/metrics.json")) if "smoke" not in p.parent.name]
    return dirs


def smooth(y: np.ndarray, k: int = 3) -> np.ndarray:
    return pd.Series(y).rolling(k, min_periods=1, center=True).mean().to_numpy()


def summary(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (run, task), g in df[df.task != "val_loss"].groupby(["run", "task"]):
        g = g.sort_values("step")
        y = smooth(g.value.to_numpy())
        i = int(np.argmax(y))
        out.append({"run": run, "task": task, "metric": g.metric.iloc[0], "n_evals": len(g),
                    "best": y[i], "best_step": int(g.step.iloc[i]), "final": y[-1], "drop": y[-1] - y[i],
                    "jitter": float(np.std(np.diff(g.value.to_numpy()))) if len(g) > 2 else np.nan})
    return pd.DataFrame(out).sort_values("drop")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from brain.multitask.registry import load_registry

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/diagnostics/trajectories"))
    a = ap.parse_args()
    dirs = [Path(p) if Path(p).is_absolute() else RUN_ROOT / Path(p).relative_to("runs") if str(p).startswith("runs/")
            else Path(p) for p in a.runs] if a.runs else discover()
    df = load_runs(dirs)
    if df.empty:
        raise SystemExit("no validation history found")
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "history.csv", index=False)
    s = summary(df)
    s.to_csv(a.out / "summary.csv", index=False)
    group_of = {t.id: t.group for t in load_registry().tasks}
    runs = sorted(df.run.unique())
    colors = {r: matplotlib.colormaps["tab10"](i % 10) for i, r in enumerate(runs)}

    # (1) per-task curves, one figure per task group
    tasks = sorted(df.task.unique(), key=lambda t: (group_of.get(t, "zz"), t))
    for group in dict.fromkeys(group_of.get(t, "other") for t in tasks):
        ts = [t for t in tasks if group_of.get(t, "other") == group and t != "val_loss"]
        if group == "other":
            ts = ["val_loss"] + [t for t in ts if t != "val_loss"]
        if not ts:
            continue
        nc = min(4, len(ts))
        nr = int(np.ceil(len(ts) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.0 * nr), squeeze=False)
        for ax, t in zip(axes.ravel(), ts):
            for r in runs:
                g = df[(df.task == t) & (df.run == r)].sort_values("step")
                if g.empty:
                    continue
                ax.plot(g.step, g.value, ".", color=colors[r], alpha=0.25, ms=3)
                ax.plot(g.step, smooth(g.value.to_numpy()), "-", color=colors[r], lw=1.4, label=r)
            ax.set_title(f"{t} ({df[df.task == t].metric.iloc[0]})", fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(ts):]:
            ax.axis("off")
        axes.ravel()[0].legend(fontsize=6)
        fig.suptitle(f"validation trajectories -- {group}")
        fig.tight_layout()
        fig.savefig(a.out / f"curves_{group}.png", dpi=80)
        plt.close(fig)

    # (2) learning-order heatmap per run
    for r in runs:
        g = df[(df.run == r) & (df.task != "val_loss")]
        piv = g.pivot_table(index="task", columns="step", values="value").sort_index(
            key=lambda idx: [(group_of.get(t, "zz"), t) for t in idx])
        if piv.shape[1] < 2:
            continue
        sm = piv.apply(lambda row: pd.Series(smooth(row.to_numpy()), index=row.index), axis=1)
        lo, hi = sm.min(axis=1), sm.max(axis=1)
        norm = sm.sub(lo, axis=0).div((hi - lo).replace(0, np.nan), axis=0)
        fig, ax = plt.subplots(figsize=(min(18, 2 + 0.45 * piv.shape[1]), 0.2 * len(piv) + 1.5))
        ax.imshow(norm.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(piv)), [f"{t} [{group_of.get(t, '?')}]" for t in piv.index], fontsize=6)
        ax.set_xticks(range(piv.shape[1]), [str(c) for c in piv.columns], rotation=90, fontsize=6)
        ax.set_title(f"{r}: each task scaled to its own range (yellow = its best); dark after yellow = decline")
        fig.tight_layout()
        fig.savefig(a.out / f"order_{r.replace('/', '_')}.png", dpi=80)
        plt.close(fig)

    big = s[(s["drop"] < -0.05) & (s["n_evals"] >= 5)]
    print(f"[trajectories] {len(runs)} runs, {df.task.nunique()} tasks -> {a.out}")
    print(f"[trajectories] tasks whose smoothed score ends > 0.05 below its peak: {len(big)}")
    print(big.head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
