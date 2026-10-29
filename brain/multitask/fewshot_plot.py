"""Few-shot and transfer results of every multitask run, as figures.

    python -m brain.multitask.fewshot_plot                     # every fewshot_*.json under runs/{multitask,joint,single}
    python -m brain.multitask.fewshot_plot --runs runs/multitask/v4general_s42 runs/multitask/v6nogenunc_s42

Reads the ``fewshot_<ckpt>.json`` files ``brain.multitask.fewshot`` writes next to each checkpoint and writes to ``--out``:

* ``fewshot.png``: one panel per held-out task, score against k shots (log scale); per run a solid line (task-embedding
  fit, mean over seeds, band = min-max) and a dashed line in the same colour (linear probe on the frozen pooled embedding,
  same scans; classification and regression only);
* ``transfer.png``: trained tasks asked with no fitting on cohorts they never trained on, one bar per run;
* ``fewshot.csv`` / ``transfer.csv``: the numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUN_ROOT = Path("/orcd/scratch/orcd/011/roku/brain-model/runs")


def find(runs: list[str] | None) -> list[Path]:
    if runs:
        dirs = [RUN_ROOT / Path(r).relative_to("runs") if str(r).startswith("runs/") else Path(r) for r in runs]
        return [f for d in dirs for f in sorted(d.glob("fewshot_*.json"))]
    out = []
    for sub in ("multitask", "multitask/snapshots", "joint", "single"):
        out += [f for f in sorted((RUN_ROOT / sub).glob("*/fewshot_*.json")) if "smoke" not in str(f)]
    return out


def name(f: Path) -> str:
    return f"{f.parent.name}/{f.stem.removeprefix('fewshot_')}"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/diagnostics/fewshot"))
    a = ap.parse_args()
    files = find(a.runs)
    if not files:
        raise SystemExit("no fewshot_*.json found")
    fs, tr = [], []
    for f in files:
        j = json.load(open(f))
        fs.append(pd.DataFrame(j.get("fewshot", [])).assign(run=name(f)))
        tr.append(pd.DataFrame(j.get("transfer", [])).assign(run=name(f)))
    fs, tr = pd.concat(fs, ignore_index=True), pd.concat(tr, ignore_index=True)
    a.out.mkdir(parents=True, exist_ok=True)
    fs.to_csv(a.out / "fewshot.csv", index=False)
    tr.to_csv(a.out / "transfer.csv", index=False)
    runs = list(dict.fromkeys(fs["run"]))
    color = {r: matplotlib.colormaps["tab10"](i % 10) for i, r in enumerate(runs)}

    tasks = sorted(fs["task"].unique())
    nc = min(4, len(tasks))
    nr = int(np.ceil(len(tasks) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.3 * nr), squeeze=False)
    for ax, t in zip(axes.ravel(), tasks):
        g = fs[fs["task"] == t]
        for r in runs:
            h = g[g["run"] == r].groupby("k")
            if not len(h):
                continue
            ks = np.array(sorted(g[g["run"] == r]["k"].unique()))
            m, lo, hi = h["value"].mean().loc[ks], h["value"].min().loc[ks], h["value"].max().loc[ks]
            ax.plot(ks, m, "-o", color=color[r], ms=3, lw=1.4, label=r)
            ax.fill_between(ks, lo, hi, color=color[r], alpha=0.12)
            p = h["linear_probe"].mean().loc[ks]
            if p.notna().any():
                ax.plot(ks, p, "--", color=color[r], lw=1.0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(g["k"].unique()), [str(k) for k in sorted(g["k"].unique())])
        ax.set_title(f"{t} ({g['metric'].iloc[0]})", fontsize=9)
        ax.set_xlabel("labelled scans (k)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(tasks):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=6)
    fig.suptitle("Few-shot on held-out tasks: solid = fit the task embedding only; dashed = linear probe on the pooled "
                 "embedding (same scans); band = min-max over seeds")
    fig.tight_layout()
    fig.savefig(a.out / "fewshot.png", dpi=85)
    plt.close(fig)

    if len(tr):
        tr["what"] = tr["task"] + " @ " + tr["cohort"] + " (" + tr["metric"] + ")"
        piv = tr.pivot_table(index="what", columns="run", values="value").reindex(columns=runs)
        fig, ax = plt.subplots(figsize=(1.6 * len(piv) + 3, 4))
        width = 0.8 / max(len(runs), 1)
        for i, r in enumerate(runs):
            ax.bar(np.arange(len(piv)) + i * width, piv[r].clip(lower=-1), width, color=color[r], label=r)
        ax.set_xticks(np.arange(len(piv)) + 0.4 - width / 2, piv.index, rotation=20, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("score (clipped at -1)")
        ax.legend(fontsize=6)
        ax.set_title("Transfer: trained tasks on unseen cohorts, no fitting")
        fig.tight_layout()
        fig.savefig(a.out / "transfer.png", dpi=85)
        plt.close(fig)
    print(f"[fewshot_plot] {len(files)} runs -> {a.out}")
    print(fs.groupby(["task", "k", "run"])[["value", "linear_probe"]].mean().unstack("run").round(3).to_string())


if __name__ == "__main__":
    main()
