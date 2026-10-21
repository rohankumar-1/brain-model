"""Repeated, subject-grouped k-fold cross-validation for small frozen-head tasks.

A fixed split leaves small tasks with a handful of test subjects (VALDO EPVS: 4), so their scores are noise. Here every
subject is tested once per repeat: folds are grouped by subject (``dedup_key``: all scans of a person stay together)
and stratified by the label (regression targets by quantile bin). Inside each fold, a random 15% of the training
subjects is the validation set (head choice / early stopping), so nothing about the test fold is used to fit or pick.

Scores are computed on the pooled out-of-fold predictions of each repeat; the reported value is the mean over repeats,
with a bootstrap interval over the pooled predictions of the first repeat and the per-repeat values alongside.

Policy (``--cv auto``): small tasks (< ``max_subjects`` subjects) whose volumes were never in the SSL pool. For tasks
from pretraining families only the fixed test split was kept out of pretraining, so cross-validating over all subjects
would test on images the encoder saw (unlabelled); ``--cv on`` still allows it and the record keeps the overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from brain.downstream.tasks import TaskBundle

VAL_FRAC = 0.15


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


def make_folds(task: TaskBundle, volumes: pd.DataFrame | None, k: int = 5, repeats: int = 3,
               seed: int = 42) -> list[tuple[int, int, dict[str, str]]]:
    """``(repeat, fold, splits)`` with ``splits`` mapping every task volume to train / val / test."""
    groups, labels = subjects(task, volumes), strata(task, k)
    vids = [str(v) for v in task.volume_ids]
    n_sub = len(set(groups))
    k = max(2, min(k, n_sub, int(np.bincount(labels).max()) if labels.size else k))
    out = []
    for r in range(repeats):
        skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + r)
        rng = np.random.default_rng(seed + 1000 + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(vids)), labels, groups)):
            tr_subj = np.unique(groups[tr])
            val_subj = set(rng.choice(tr_subj, size=max(1, int(round(VAL_FRAC * len(tr_subj)))), replace=False))
            splits = {vids[i]: "test" for i in te}
            splits.update({vids[i]: ("val" if groups[i] in val_subj else "train") for i in tr})
            out.append((r, f, splits))
    return out


def use_cv(mode: str, task: TaskBundle, volumes: pd.DataFrame | None, overlap: float | None,
           max_subjects: int) -> bool:
    if mode == "on":
        return True
    if mode == "off" or task.task_type == "segmentation":
        return False
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
