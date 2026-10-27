"""Classification and segmentation metrics for brain.downstream eval."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray, **kwargs: Any) -> float:
    try:
        y_true = np.asarray(y_true)
        if y_true.ndim == 1 and len(np.unique(y_true)) < 2:
            return float("nan")
        if y_true.ndim == 2:
            # drop labels with a single class
            keep = [i for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) > 1]
            if not keep:
                return float("nan")
            y_true = y_true[:, keep]
            y_score = np.asarray(y_score)[:, keep]
        return float(roc_auc_score(y_true, y_score, **kwargs))
    except ValueError:
        return float("nan")


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean recall over the classes present in ``y_true``: sklearn's ``balanced_accuracy_score``, without its warning
    when a prediction names a class absent from ``y_true`` (a small test set of a multiclass task)."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if not len(y_true):
        return float("nan")
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in np.unique(y_true)]))


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()
    y_pred = (y_score >= threshold).astype(int)
    return {
        "task_type": "binary",
        "auc": _safe_auc(y_true, y_score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "threshold": threshold,
    }


def multilabel_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str] | None = None,
    threshold: float = 0.5,
) -> dict:
    """Multilabel (N, L). OVR AUCs; OVO is N/A (reported as null)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.ndim != 2:
        raise ValueError(f"Expected (N,L) labels, got {y_true.shape}")
    y_pred = (y_score >= threshold).astype(int)
    names = label_names or [f"label_{i}" for i in range(y_true.shape[1])]

    per_label = {}
    for i, name in enumerate(names):
        per_label[name] = {
            "auc": _safe_auc(y_true[:, i], y_score[:, i]),
            "accuracy": float(accuracy_score(y_true[:, i], y_pred[:, i])),
            "n_pos": int(y_true[:, i].sum()),
            "n_neg": int((1 - y_true[:, i]).sum()),
        }

    # element-wise / subset accuracy
    return {
        "task_type": "multilabel",
        "threshold": threshold,
        "auc_ovr_macro": _safe_auc(y_true, y_score, average="macro"),
        "auc_ovr_micro": _safe_auc(y_true, y_score, average="micro"),
        "auc_ovr_weighted": _safe_auc(y_true, y_score, average="weighted"),
        # OVO is defined for multiclass, not multilabel
        "auc_ovo_macro": None,
        "auc_ovo_weighted": None,
        "accuracy_subset": float(accuracy_score(y_true, y_pred)),
        "accuracy_elementwise": float((y_true == y_pred).mean()),
        "per_label": per_label,
        "labels": names,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Age-style targets. ``y_pred`` is in the same units as ``y_true``."""
    from sklearn.metrics import mean_absolute_error, r2_score

    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(yt) < 2 or np.std(yt) == 0 or np.std(yp) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(yt, yp)[0, 1])
    return {
        "task_type": "regression",
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan"),
        "pearson": corr,
        "n": int(len(yt)),
    }


def region_dice(
    gt: np.ndarray,
    pred: np.ndarray,
    regions: dict[str, tuple[int, ...]],
    *,
    eps: float = 1e-6,
) -> dict:
    """Dice per named region. ``gt`` and ``pred`` are integer label volumes."""
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch {gt.shape} vs {pred.shape}")
    per: dict[str, float] = {}
    for name, ids in regions.items():
        t = np.isin(gt, ids).astype(np.float64)
        p = np.isin(pred, ids).astype(np.float64)
        inter = 2.0 * float((t * p).sum())
        denom = float(t.sum() + p.sum())
        per[name] = float((inter + eps) / (denom + eps))
    vals = list(per.values())
    return {
        "task_type": "segmentation",
        "dice_macro": float(np.mean(vals)) if vals else float("nan"),
        "dice_per_class": per,
    }


def summarize_for_table(head_name: str, metrics: dict) -> dict:
    """Flat row for CSV comparison across heads."""
    row = {"head": head_name, "task_type": metrics.get("task_type")}
    for k in (
        "auc",
        "auc_ovr_macro",
        "auc_ovr_micro",
        "auc_ovr_weighted",
        "auc_ovo_macro",
        "auc_ovo_weighted",
        "accuracy",
        "accuracy_subset",
        "accuracy_elementwise",
        "dice_macro",
        "dice_mean",
        "mae",
        "r2",
        "pearson",
    ):
        if k in metrics and metrics[k] is not None and not isinstance(metrics[k], dict):
            row[k] = metrics[k]
    for k, v in metrics.items():
        if k.startswith("dice_") and k not in row and isinstance(v, (int, float)):
            row[k] = v
    return row


HEADLINE = {"multilabel": "auc_ovr_macro", "binary": "auc", "regression": "r2", "segmentation": "dice_macro"}


def bootstrap_ci(stat, n: int, *, reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of ``stat(idx)`` over test items resampled with replacement.

    Resamples where the statistic is undefined (NaN, e.g. one class only) are skipped.
    """
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(reps)], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) < reps // 2:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
