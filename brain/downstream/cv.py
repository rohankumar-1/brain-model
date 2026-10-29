"""Repeated, subject-grouped k-fold cross-validation for small frozen-head tasks.

A fixed split leaves small tasks with a handful of test subjects (VALDO EPVS: 4), so their scores are noise. Here every
subject is tested once per repeat: folds are grouped by subject (``dedup_key``: all scans of a person stay together)
and stratified by the label (regression targets by quantile bin). Inside each fold, a random 15% of the training
subjects is the validation set (head choice / early stopping), so nothing about the test fold is used to fit or pick.
A class (multilabel: a label's positives) the random pick left with fewer than two validation subjects gets more of its
training subjects moved there (``_guard_val``; before 2026-10-27 folds without one fitted every head untuned, e.g. 6 of
15 CQ500 fracture folds); the other folds' splits are unchanged.

Validation is cross-fitted (benchmark v2): ``val_halves`` splits each fold's validation subjects in two, stratified by
class, and a head is tuned / early-stopped on one half to score the other (``brain.downstream.heads``), so the reported
validation score, which picks the readout, never comes from the scans a head was tuned on. (Before, tok_conv
early-stopped on the scans it was then scored on: its CV validation AUROC ran 0.17 above test, the linear heads' 0.03.)

Scores are computed on the pooled out-of-fold predictions of each repeat; the reported value is the mean over repeats,
with a bootstrap interval over the pooled predictions of the first repeat and the per-repeat values alongside.

Policy (``--cv auto``): small tasks (< ``max_subjects`` subjects) whose volumes were never in the SSL pool, and every
classification task whose fixed split leaves its rarest class with fewer than ``MIN_TEST_MINORITY`` test or
``MIN_VAL_MINORITY`` validation volumes. For tasks from pretraining families only the fixed
test split was kept out of pretraining, so cross-validating over all subjects tests on images the encoder saw
(unlabelled); a rare-class task gets CV anyway, since its fixed split leaves a few test positives and often none in
validation (before 2026-10-25 CQ500 EDH had 2 test and 0 validation positives, so its head was picked on test). The
record keeps the overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from brain.downstream.tasks import TaskBundle

VAL_FRAC = 0.15
MIN_TEST_MINORITY = 15        # rarest-class test volumes of the fixed split below which a task is cross-validated
MIN_VAL_MINORITY = 3          # ... and validation volumes (head choice needs both classes)


def minority(task: TaskBundle, split: str | None = None) -> int | None:
    """Volumes in the task's rarest class (multilabel: the rarest label's positives), over the whole task or one
    fixed ``split``; None for regression."""
    if task.task_type == "regression":
        return None
    y_all = np.asarray(task.y)
    y = y_all if split is None else y_all[[task.splits.get(str(v)) == split for v in task.volume_ids]]
    if not len(y):
        return 0
    if y.ndim > 1:
        y = y.reshape(len(y), -1) > 0
        return int(min(y.sum(0).min(), (~y).sum(0).min()))
    classes = np.unique(y_all.astype(int))                  # a class absent from the split counts as 0
    return int(min((y.astype(int) == c).sum() for c in classes))


def subjects(task: TaskBundle, volumes: pd.DataFrame | None) -> np.ndarray:
    """Subject key per task volume (``dedup_key``; the volume id when unknown)."""
    key = {}
    if volumes is not None and {"volume_id", "dedup_key"} <= set(volumes.columns):
        key = dict(zip(volumes["volume_id"].astype(str), volumes["dedup_key"].astype(str)))
    return np.asarray([key.get(str(v), str(v)) for v in task.volume_ids])


def strata(task: TaskBundle, bins: int) -> np.ndarray:
    y = np.asarray(task.y)
    if task.task_type == "regression":
        y = y.reshape(len(y), -1)[:, 0].astype(float)
        edges = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)[1:-1]))
        return np.digitize(y, edges)
    if y.ndim > 1:                                   # multilabel: stratify on "any positive"
        return (y.reshape(len(y), -1) > 0).any(1).astype(int)
    return y.astype(int)


def _classes(task: TaskBundle) -> np.ndarray | None:
    """(N, C) bool: volume i is in class c (multilabel: label c positive); None for regression."""
    if task.task_type == "regression":
        return None
    y = np.asarray(task.y)
    if y.ndim > 1:
        return y.reshape(len(y), -1) > 0
    y = y.astype(int)
    return y[:, None] == np.unique(y)[None, :]


def _guard_val(member: np.ndarray | None, groups: np.ndarray, tr: np.ndarray, val_subj: set,
               rng: np.random.Generator) -> int:
    """Give each class at least two validation subjects (one per ``val_halves`` half) by moving training subjects into
    ``val_subj``, never a class's last training subject. Returns how many subjects moved."""
    if member is None:
        return 0
    moved = 0
    for c in range(member.shape[1]):
        pool = np.unique(groups[tr][member[tr, c]])
        have = len(val_subj.intersection(pool))
        need = min(2, len(pool) - 1) - have
        if need <= 0:
            continue
        take = rng.choice(np.setdiff1d(pool, list(val_subj)), size=need, replace=False)
        val_subj.update(take)
        moved += len(take)
    return moved


def val_halves(task: TaskBundle, volumes: pd.DataFrame | None, splits: dict[str, str], seed: int) -> dict[str, int]:
    """Validation volume -> half (0 / 1), by subject and stratified: subjects sorted into strata (their class; the
    rarest positive label for multilabel; the target's quintile for regression) and dealt alternately into the halves,
    so a class with two validation subjects is in both."""
    groups = subjects(task, volumes)
    vids = [str(v) for v in task.volume_ids]
    member = _classes(task)
    if member is None:
        stratum = strata(task, 5)
    else:
        rare = np.argsort(member.sum(0))                      # classes from the rarest
        first = np.full(len(vids), len(rare))
        for rank, c in enumerate(rare):
            first = np.where(member[:, c] & (first == len(rare)), rank, first)
        stratum = first
    subj_stratum: dict[str, int] = {}
    for i, v in enumerate(vids):
        if splits.get(v) == "val":
            subj_stratum[groups[i]] = min(subj_stratum.get(groups[i], 1 << 30), int(stratum[i]))
    rng = np.random.default_rng([seed, 7])
    half, turn = {}, 0
    for s in sorted(set(subj_stratum.values())):
        for g in rng.permutation(sorted(k for k, v in subj_stratum.items() if v == s)):
            half[g], turn = turn, 1 - turn
    return {v: half[groups[i]] for i, v in enumerate(vids) if splits.get(v) == "val"}


def make_folds(task: TaskBundle, volumes: pd.DataFrame | None, k: int = 5, repeats: int = 3,
               seed: int = 42, report: dict | None = None) -> list[tuple[int, int, dict[str, str]]]:
    """``(repeat, fold, splits)`` with ``splits`` mapping every task volume to train / val / test. ``report`` (if
    given) gets ``val_repaired``: the folds whose validation set ``_guard_val`` topped up."""
    groups, labels = subjects(task, volumes), strata(task, k)
    member = _classes(task)
    repaired = 0
    vids = [str(v) for v in task.volume_ids]
    n_sub = len(set(groups))
    counts = np.bincount(labels) if labels.size else np.zeros(0, int)
    k = max(2, min(k, n_sub, int(counts[counts > 0].min()) if counts.size else k))   # every fold gets each class
    out = []
    for r in range(repeats):
        skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + r)
        rng = np.random.default_rng(seed + 1000 + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(vids)), labels, groups)):
            tr_subj = np.unique(groups[tr])
            val_subj = set(rng.choice(tr_subj, size=max(1, int(round(VAL_FRAC * len(tr_subj)))), replace=False))
            # its own stream, so a repair leaves the later folds' draws (and splits) as they were
            repaired += _guard_val(member, groups, tr, val_subj, np.random.default_rng([seed, r, f])) > 0
            splits = {vids[i]: "test" for i in te}
            splits.update({vids[i]: ("val" if groups[i] in val_subj else "train") for i in tr})
            out.append((r, f, splits))
    if report is not None:
        report["val_repaired"] = int(repaired)
    return out


def use_cv(mode: str, task: TaskBundle, volumes: pd.DataFrame | None, overlap: float | None,
           max_subjects: int) -> bool:
    if mode == "on":
        return True
    if mode == "off" or task.task_type == "segmentation":
        return False
    test_min, val_min = minority(task, "test"), minority(task, "val")
    if test_min is not None and (test_min < MIN_TEST_MINORITY or val_min < MIN_VAL_MINORITY):
        return True
    return len(set(subjects(task, volumes))) < max_subjects and not (overlap or 0.0) > 0.0


def pooled(task: TaskBundle, fold_preds: list[tuple[int, list[str], np.ndarray]]):
    """Out-of-fold predictions per repeat: ``{repeat: (volume_ids, y, scores)}`` in task order."""
    y_of = {str(v): task.y[i] for i, v in enumerate(task.volume_ids)}
    by_rep: dict[int, dict[str, np.ndarray]] = {}
    for r, ids, scores in fold_preds:
        by_rep.setdefault(r, {}).update({str(v): s for v, s in zip(ids, scores)})
    out = {}
    for r, d in by_rep.items():
        ids = [str(v) for v in task.volume_ids if str(v) in d]
        out[r] = (ids, np.stack([np.asarray(y_of[v]) for v in ids]), np.stack([np.asarray(d[v]) for v in ids]))
    return out
