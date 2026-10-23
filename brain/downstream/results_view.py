"""Readable views of the experiment database (``brain.downstream.db``) for notebooks: one table per task, an overview
grid, gain over the random-init floor, and per-head drill-downs. Used by ``notebooks/results.ipynb``.

Every number is the head picked on validation by ``db.report`` (``picked_on == "test!"`` flags a task whose heads had
no validation score). ``delta`` is test minus the random-init baseline's test score on the same task.
"""

from __future__ import annotations

import re
from fnmatch import fnmatch

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from brain.downstream import db
from brain.downstream.tasks import CLASSIFICATION_TASKS, REGRESSION_TASKS, SEGMENTATION_TASKS

GROUPS = {**{t: "classification" for t in CLASSIFICATION_TASKS}, **{t: "regression" for t in REGRESSION_TASKS},
          **{t: "segmentation" for t in SEGMENTATION_TASKS}}
GROUP_ORDER = ("classification", "regression", "segmentation", "other")
RANDOM_PREFIX = "random"
RED_WHITE_GREEN = LinearSegmentedColormap.from_list("red_white_green", ["#c62828", "#ffffff", "#2e7d32"])
# Models post-trained with labels (brain.multitask) have seen some suite tasks. "labels": the task's own labels
# (train-split subjects) were trained on; "task": the same target was trained on other cohorts (age, hemorrhage,
# tumour), so the frozen features were shaped for it. Neither is a clean transfer number.
SUPERVISED_PREFIXES = ("mt_",)
SEEN_LABELS = frozenset({"rsna_ich", "brats2023_gli", "brats2023_men", "brats2024_gli", "brats2023_ped",
                         "oasis1_mmse", "oasis2_mmse", "abide_ados", "gli_survival_12m", "gli_lgg", "gli_idh"})
SEEN_TASK = frozenset({"cq500", "seg_cq500", "instance", "msd", "oasis1_age", "ixi_age", "pixar_age", "piop1_age",
                       "piop2_age", "id1000_age", "ms3seg_age"})


def seen(model: str, task: str) -> str:
    if not str(model).startswith(SUPERVISED_PREFIXES):
        return ""
    return "labels" if task in SEEN_LABELS else "task" if task in SEEN_TASK else ""


def select(rep: pd.DataFrame, keep: list[str] | None = None, drop: list[str] | None = None) -> pd.DataFrame:
    """Subset models by their labels (``rep.model``) with glob patterns: ``keep`` (any match is kept; None = all),
    then ``drop`` (any match is removed). E.g. ``select(rep, drop=["cls_reg4*", "*@920"])``. Random-init baselines
    are kept unless dropped explicitly (Δ random is computed at load, so it is unaffected either way)."""
    m = rep.model.astype(str)
    ok = pd.Series(True, index=rep.index) if keep is None else m.map(
        lambda x: any(fnmatch(x, k) for k in keep) or x.startswith(RANDOM_PREFIX))
    if drop:
        ok &= ~m.map(lambda x: any(fnmatch(x, d) for d in drop))
    return rep[ok]


def model_label(model_id: str) -> str:
    """``cls_s42/step_0015000@ab12`` -> ``cls_s42 @15k``; ``.../last`` -> ``... (final)``."""
    name = str(model_id).split("@", 1)[0]
    run, _, stem = name.rpartition("/")
    m = re.fullmatch(r"step_0*(\d+)", stem)
    if m:
        step = int(m.group(1))
        return f"{run} @{step // 1000}k" if step % 1000 == 0 else f"{run} @{step}"
    return f"{run} (final)" if stem == "last" else name


def load(ingest: bool = True) -> pd.DataFrame:
    con = db.connect()
    if ingest:
        db.ingest(con)
    rep = db.report(con)
    if rep.empty:
        return rep
    rep["model"] = rep.model_id.map(model_label)
    rep["is_random"] = rep.model_id.str.startswith(RANDOM_PREFIX)
    rep["group"] = rep.task.map(GROUPS).fillna("other")
    floor = rep[rep.is_random].groupby("task").test.max()
    rep["delta"] = rep.test - rep.task.map(floor)
    rep["seen"] = [seen(m, t) for m, t in zip(rep.model, rep.task)]
    return rep.sort_values(["group", "task", "test"], ascending=[True, True, False]).reset_index(drop=True)


def fmt(v: float, lo: float = np.nan, hi: float = np.nan, digits: int = 3) -> str:
    if pd.isna(v):
        return ""
    s = f"{v:.{digits}f}"
    return s + (f"  [{lo:.{digits}f}, {hi:.{digits}f}]" if not (pd.isna(lo) or pd.isna(hi)) else "")


def overview(rep: pd.DataFrame, models: list[str] | None = None, delta: bool = False, hide_seen: str | None = None):
    """Task x model grid of test scores (or gains over random), coloured per task row, grouped by task type.
    ``hide_seen="labels"`` blanks cells where a label-trained model saw the task's own labels; ``"task"`` also those
    where it trained the same target on other cohorts."""
    df = rep if models is None else rep[rep.model.isin(models)]
    if hide_seen:
        df = df[~df.seen.isin(["labels"] if hide_seen == "labels" else ["labels", "task"])]
    val = "delta" if delta else "test"
    grid = df.pivot_table(index=["group", "task"], columns="model", values=val, aggfunc="first")
    grid = grid.reindex(sorted(grid.index, key=lambda k: (GROUP_ORDER.index(k[0]), k[1])))
    cols = sorted(grid.columns, key=lambda c: (not c.startswith(RANDOM_PREFIX), c))
    grid = grid[cols]
    metric = df.groupby("task").metric.first()
    grid.insert(0, "metric", [metric.get(t, "") for _, t in grid.index])
    sty = grid.style.format("{:+.3f}" if delta else "{:.3f}", subset=cols, na_rep="")
    if delta:
        sty = sty.background_gradient(cmap=RED_WHITE_GREEN, subset=cols, vmin=-0.1, vmax=0.1)
        return sty.set_caption("gain over random-init (head picked on val): red -0.1, white 0, green +0.1")
    # fixed scale per metric family, so colours compare across tasks: AUC from chance (0.5), Dice / R2 from 0
    for lo, hi, rows in ((0.5, 1.0, grid.metric.str.startswith("auc")), (0.0, 1.0, ~grid.metric.str.startswith("auc"))):
        if rows.any():
            sty = sty.background_gradient(cmap=RED_WHITE_GREEN, subset=pd.IndexSlice[grid.index[rows], cols],
                                          vmin=lo, vmax=hi)
    return sty.set_caption("test score (head picked on val): AUC red 0.5 / white 0.75 / green 1; "
                           "Dice and R2 red 0 / white 0.5 / green 1 (values outside are clipped)")


def task_table(rep: pd.DataFrame, task: str) -> pd.DataFrame:
    t = rep[rep.task == task].sort_values("test", ascending=False)
    return pd.DataFrame({
        "model": t.model,
        t.metric.iloc[0] if len(t) else "test": [fmt(v, lo, hi) for v, lo, hi in zip(t.test, t.ci_lo, t.ci_hi)],
        "Δ random": [("" if pd.isna(d) else f"{d:+.3f}") for d in t.delta],
        "head": t["head"],
        "picked on": t.picked_on,
        "pretrain overlap": [("" if pd.isna(o) else f"{o:.0%}") for o in t.overlap],
        "label-seen": t.seen,
    }).reset_index(drop=True)


def plot_task(rep: pd.DataFrame, task: str, ax=None):
    import matplotlib.pyplot as plt

    t = rep[rep.task == task].sort_values("test")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.35 * len(t) + 0.8))
    err = np.vstack([(t.test - t.ci_lo).clip(lower=0).fillna(0), (t.ci_hi - t.test).clip(lower=0).fillna(0)])
    colors = ["#9e9e9e" if r else "#2b6cb0" for r in t.is_random]
    ax.barh(t.model, t.test, xerr=err, color=colors, capsize=3)
    ax.set_xlabel(f"{t.metric.iloc[0]} (test, 95% CI)" if len(t) else "")
    ax.set_title(task, loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def all_heads(task: str) -> pd.DataFrame:
    """Every head of every model on one task, with val and test (for checking the head choice)."""
    t = db.report(db.connect(), task=task, all_heads=True)
    if t.empty:
        return t
    t.insert(0, "model", t.model_id.map(model_label))
    return t.drop(columns=["model_id"]).sort_values(["model", "val"], ascending=[True, False]).reset_index(drop=True)


def probes() -> pd.DataFrame:
    t = db.report(db.connect(), probes=True)
    if t.empty:
        return t
    t.insert(0, "model", t.model_id.map(model_label))
    return t.drop(columns=["model_id"])
