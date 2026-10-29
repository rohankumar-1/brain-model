"""One scale for every metric: skill, 0 = chance and 1 = perfect, so tasks, groups and runs average and compare.

* AUROC (incl. one-vs-rest macro): ``(auc - 0.5) / 0.5``
* balanced accuracy over ``n`` classes: ``(bacc - 1/n) / (1 - 1/n)``
* R2: clipped at 0 (worse than predicting the mean is no skill, not negative skill)
* Dice: as is (an empty prediction scores 0)

Aggregates are means of group means (``score``), so a group with many tasks (e.g. 20 anatomy volumes) counts as one.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def family(metric: str) -> str:
    """``auc_ovr_macro`` -> ``auc``, ``dice_macro`` -> ``dice``; ``auc`` / ``bacc`` / ``r2`` / ``dice`` unchanged."""
    m = metric.split("_")[0]
    if m not in ("auc", "bacc", "r2", "dice"):
        raise ValueError(f"no skill scale for metric {metric!r}")
    return m


def skill(metric: str, value, n_classes: int = 2):
    """``value`` (scalar or array) on the skill scale; NaN stays NaN."""
    v = np.asarray(value, dtype=float)
    m = family(metric)
    if m == "auc":
        s = (v - 0.5) / 0.5
    elif m == "bacc":
        c = 1.0 / max(int(n_classes), 2)
        s = (v - c) / (1.0 - c)
    elif m == "r2":
        s = np.where(np.isnan(v), np.nan, np.maximum(v, 0.0))
    else:
        s = v
    return float(s) if s.ndim == 0 else s


def scale(metric: str, n_classes: int = 2) -> float:
    """d(skill)/d(metric): converts a difference (or interval half-width) in the metric to skill units."""
    m = family(metric)
    if m == "auc":
        return 2.0
    if m == "bacc":
        c = 1.0 / max(int(n_classes), 2)
        return 1.0 / (1.0 - c)
    return 1.0


def summarize(rows) -> dict[str, float]:
    """``rows``: (group, skill) pairs -> ``{"score": mean of group means, "group_<g>": group mean}`` (NaN skipped)."""
    by: dict[str, list[float]] = defaultdict(list)
    for g, s in rows:
        if s is not None and np.isfinite(s):
            by[g].append(float(s))
    out = {f"group_{g}": float(np.mean(v)) for g, v in sorted(by.items())}
    if out:
        out["score"] = float(np.mean(list(out.values())))
    return out
