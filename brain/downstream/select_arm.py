"""Pick the best prefix-token arm from the step-15k ablation snapshots and print the training command for it.

Score = suite score on the tasks every arm and the random-init floor have: per task, test metric minus the floor
(head picked on validation, ``db.report``), averaged within task groups (classification, regression, segmentation)
and then across groups, so groups with many tasks do not dominate. Ties within ``TIE`` go to the simpler arm
(fewer prefix tokens).

    python -m brain.downstream.select_arm --step 15000 [--json results/selection_15k.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from brain.downstream import results_view as rv

ARMS = {"none_s42": (0, 0), "cls_s42": (1, 0), "cls_reg4_s42": (1, 4)}     # run dir -> (n_cls, n_registers)
SIMPLICITY = ["none_s42", "cls_s42", "cls_reg4_s42"]
TIE = 0.005


def scores(step: int) -> tuple[pd.DataFrame, pd.Series]:
    rep = rv.load(ingest=True)
    label = {arm: rv.model_label(f"{arm}/step_{step:07d}") for arm in ARMS}
    rep = rep[rep.model.isin(label.values()) | rep.is_random]
    have = rep.groupby("task").model.apply(set)
    need = set(label.values())
    tasks = [t for t, ms in have.items() if need <= ms and any(m.startswith(rv.RANDOM_PREFIX) for m in ms)]
    if not tasks:
        raise SystemExit("no task has results for every arm and the random floor")
    sub = rep[rep.task.isin(tasks) & ~rep.is_random]
    per_task = sub.pivot_table(index=["group", "task"], columns="model", values="delta", aggfunc="first")
    suite = per_task.groupby(level="group").mean().mean()
    inv = {v: k for k, v in label.items()}
    return per_task.rename(columns=inv), suite.rename(index=inv)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, default=15000)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    per_task, suite = scores(a.step)
    pd.set_option("display.width", 160)
    print(per_task.round(3).to_string())
    print("\nsuite score (mean over groups of mean gain over random-init):")
    print(suite.round(4).to_string())
    best = suite.max()
    winner = next(arm for arm in SIMPLICITY if arm in suite and suite[arm] >= best - TIE)
    n_cls, n_reg = ARMS[winner]
    print(f"\n[select] winner: {winner} (n_cls={n_cls}, n_registers={n_reg}); ties within {TIE} go to the simpler arm")
    out = {"step": a.step, "winner": winner, "n_cls": n_cls, "n_registers": n_reg,
           "suite": suite.round(5).to_dict(), "tasks": [t for _, t in per_task.index],
           "per_task": {arm: per_task[arm].round(5).droplevel(0).to_dict() for arm in per_task.columns}}
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("winner", "n_cls", "n_registers")}))


if __name__ == "__main__":
    main()
