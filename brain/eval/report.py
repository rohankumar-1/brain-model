"""Scoring shared by the benchmark front end (``brain.bench``: board, card, compare, pin) and its page generators.

* ``suite_heads``: every frozen-suite result of the records database, one row per (model, task, head, variant), with
  its val / test score, protocol and per-scan prediction file; ``fixed_heads`` picks a readout per task on validation.
* ``_suite_preds`` / ``score_ci``: a result's saved per-scan predictions (every CV repeat) and one model's score with
  a subject-level bootstrap interval.
* ``pair_rows`` / ``summarise``: two models paired on the scans both answered, and the paired subject bootstrap of a
  headline on the skill scale (``brain.eval.skill``: 0 = chance, 1 = perfect). A headline is the mean over groups of
  the mean over tasks. Each subject (``dedup_key``) gets a Poisson(1) weight shared by every task and model it appears
  in, so scans of one person, and tasks sharing scans, move together; the headline is recomputed in each replicate and
  its 2.5 / 97.5 percentiles are the interval (per-task intervals come from the same replicates).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import REPO_ROOT
from brain.eval.skill import skill

# readouts of a BrainQuery model's own outputs, not its encoder (records from before 2026-10-27 still have them)
SUPERVISED_READOUTS = ("slots", "taskout", "general")


# ------------------------------------------------------------------------------------------------ subjects
@lru_cache(maxsize=1)
def subject_of() -> dict[str, str]:
    """volume_id -> subject key (``dedup_key``, what the splits group by)."""
    from brain.downstream.tasks import DEFAULT_VOLUMES

    v = pd.read_csv(DEFAULT_VOLUMES, usecols=["volume_id", "dedup_key"], low_memory=False)
    return dict(zip(v.volume_id.astype(str), v.dedup_key.astype(str)))


# ------------------------------------------------------------------------------------------------ weighted metrics
def _auc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted ROC AUC for every row of weights ``w`` (R, n) at once (ties count one half)."""
    y = y.astype(bool)
    if y.all() or not y.any():
        return np.full(w.shape[0], np.nan)
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[:, order]
    _, start = np.unique(s, return_index=True)                  # tie groups of equal scores
    grp = np.searchsorted(start, np.arange(len(s)), side="right") - 1
    wn = np.where(~y, w, 0.0)
    wp = np.where(y, w, 0.0)
    g_neg = np.zeros((w.shape[0], len(start)))
    np.add.at(g_neg.T, grp, wn.T)
    below = np.cumsum(g_neg, axis=1) - g_neg                     # negative weight strictly below each tie group
    pairs = (wp * (below[:, grp] + 0.5 * g_neg[:, grp])).sum(1)
    tot = wp.sum(1) * wn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, pairs / tot, np.nan)


def _r2(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = w.sum(1, keepdims=True)
    mean = (w * y).sum(1, keepdims=True) / np.maximum(sw, 1e-12)
    ss_res = (w * (y - p) ** 2).sum(1)
    ss_tot = (w * (y - mean) ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)


def _bacc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> np.ndarray:
    recalls = []
    for c in np.unique(y):
        m = y == c
        den = w[:, m].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            recalls.append(np.where(den > 0, w[:, m & (p == c)].sum(1) / den, np.nan))
    return np.nanmean(np.stack(recalls), axis=0)


def metric_reps(metric: str, y: np.ndarray, p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The task metric under each row of bootstrap weights ``w`` (R, n). Multilabel targets (n, L): macro AUC over the
    labels that have both classes."""
    fam = metric.split("_")[0]
    if fam == "auc":
        if y.ndim == 1 and p.ndim == 2 and p.shape[1] > 2:          # multiclass, one-vs-rest macro
            y = (y.astype(int)[:, None] == np.arange(p.shape[1])[None]).astype(float)
        if y.ndim == 2 and y.shape[1] > 1:
            if p.ndim == 1:
                return np.full(w.shape[0], np.nan)
            cols = [_auc(y[:, j], p[:, j], w) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
            return np.nanmean(np.stack(cols), axis=0) if cols else np.full(w.shape[0], np.nan)
        y = y.reshape(-1)
        p = p[:, -1] if p.ndim == 2 else p
        return _auc(y, p, w)
    if fam == "r2":
        return _r2(y.reshape(-1).astype(float), (p[:, 0] if p.ndim == 2 else p).astype(float), w)
    if fam == "bacc":
        p = p.argmax(1) if p.ndim == 2 else p
        return _bacc(y.reshape(-1).astype(int), p.astype(int), w)
    if fam == "dice":                                           # per-scan Dice rows: the weighted mean
        w = w * (np.asarray(y).reshape(-1) != 0)                # y 0: a lesion-free scan (test_eval "fp" rows), no Dice
        return (w * p).sum(1) / np.maximum(w.sum(1), 1e-12)
    raise ValueError(f"no metric {metric}")


# ------------------------------------------------------------------------------------------------ inputs
def pair_rows(part: str, group: str, task: str, metric: str, runs: list, refs: list, ncls: int = 2) -> dict | None:
    """One paired task: ``runs`` / ``refs`` are lists (one per seed; heads and transfer have one) of
    (volume_ids, y, p). Restricted to the scans present in every list."""
    common = set(runs[0][0])
    for v, _, _ in runs + refs:
        common &= set(v)
    if len(common) < 3:
        return None
    vids = sorted(common)

    def align(item):
        idx = {v: i for i, v in enumerate(item[0])}
        sel = [idx[v] for v in vids]
        return np.asarray(item[1])[sel], np.asarray(item[2])[sel]

    return {"part": part, "group": group, "task": task, "metric": metric, "ncls": ncls, "vids": vids,
            "run": [align(r) for r in runs], "ref": [align(r) for r in refs]}


def cv_protocol(protocol: str | None) -> str:
    """'' for a fixed split; 'cv-xf' for cross-validation with cross-fitted validation (benchmark v2: a head's
    validation score never comes from the scans it was tuned on); 'cv' for the earlier CV, whose validation scores
    were not cross-fitted (and on some tasks had folds without a class)."""
    c = (json.loads(protocol) if isinstance(protocol, str) and protocol else {}).get("cv")
    if not c:
        return ""
    return "cv-xf" if isinstance(c, dict) and c.get("val") == "cross_fitted" else "cv"


def _reroot(path: str | None) -> str | None:
    """A stored artifact path, moved under this checkout's ``runs/`` when it was written from another checkout (a
    worktree since removed) whose ``runs/`` pointed at the same store."""
    if not path or os.path.exists(path) or "/runs/" not in path:
        return path
    moved = REPO_ROOT / "runs" / path.split("/runs/", 1)[1]
    return str(moved) if moved.exists() else path


@lru_cache(maxsize=1)
def suite_heads() -> pd.DataFrame:
    """Every (model, task, head) of the frozen suite with its val / test score, protocol and prediction file."""
    from brain.downstream import db

    con = db.connect()
    db.ingest(con)
    t = db.table(con)
    t = t[t.kind == "task"].copy()
    # the eval window (canonical / multi / ...) and, in rows from before 2026-11-07, the encoding variant (refinement,
    # fine grid, taps, tiling: since retired) are part of a result's identity: a variant's run is another result
    proto = [(json.loads(p) if isinstance(p, str) and p else {}) for p in t.protocol]
    t["window"] = [p.get("window") or "multi" for p in proto]
    t["variant"] = [(p.get("window") or "multi") + (p.get("encoding") or "") for p in proto]
    t = t.sort_values("finished_utc").drop_duplicates(["model_id", "task", "head", "view", "seed", "variant"],
                                                      keep="last")
    t = t[~t.model_id.str.contains("+", regex=False)].copy()
    # segmentation heads store per-volume Dice (``<head>_volumes``), the others per-scan predictions (``<head>``)
    t["pred"] = [_reroot((json.loads(a).get(h) or json.loads(a).get(f"{h}_volumes")) if isinstance(a, str) and a
                         else None) for a, h in zip(t.artifacts, t["head"])]
    t["cv"] = [cv_protocol(p) for p in t.protocol]
    t["encoder"] = ~t["head"].astype(str).str.startswith(SUPERVISED_READOUTS)
    return t


def fixed_heads(t: pd.DataFrame) -> dict[str, str]:
    """Per task, the encoder readout with the best mean validation score over every evaluated (non-random) model."""
    e = t[t.encoder & t.val.notna() & ~t.model_id.str.startswith("random")]
    means = e.groupby(["task", "head"]).val.mean()
    return {task: g.idxmax()[1] for task, g in means.groupby(level=0)}


def _suite_preds(row) -> list[tuple[list[str], np.ndarray, np.ndarray]] | None:
    """[(volume ids, targets, scores)] of one suite result: one set, or one per CV repeat (results from before
    2026-10-27 kept only the first repeat). Segmentation: per-volume macro Dice as the score (the Dice metric averages
    scores), with unit targets."""
    if not row.pred or not Path(row.pred).exists():
        return None
    if str(row.pred).endswith(".csv"):
        d = pd.read_csv(row.pred)
        return [([str(v) for v in d.volume_id], np.ones(len(d)), d["dice_macro"].to_numpy(float))]
    z = np.load(row.pred, allow_pickle=False)
    out = [([str(v) for v in z["volume_ids"]], np.asarray(z["y_true"]), np.asarray(z["y_score"]))]
    k = 1
    while f"y_score_r{k}" in z.files:
        out.append(([str(v) for v in z[f"volume_ids_r{k}"]], np.asarray(z[f"y_true_r{k}"]),
                    np.asarray(z[f"y_score_r{k}"])))
        k += 1
    return out


def score_ci(metric: str, preds: list, reps: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """One model's score on its own scans, in the metric's units: the mean over prediction sets (CV repeats) with a
    subject-level Poisson bootstrap interval, the same resampling ``summarise`` pairs models with."""
    subj = subject_of()
    keys = sorted({subj.get(x, x) for vids, _, _ in preds for x in vids})
    pos = {k: i for i, k in enumerate(keys)}
    w_subj = np.vstack([np.ones(len(keys)), np.random.default_rng(seed).poisson(1.0, size=(reps, len(keys)))])
    m = np.nanmean(np.stack([metric_reps(metric, np.asarray(y), np.asarray(p),
                                         w_subj[:, [pos[subj.get(x, x)] for x in v]]) for v, y, p in preds]), axis=0)
    return float(m[0]), float(np.nanpercentile(m[1:], 2.5)), float(np.nanpercentile(m[1:], 97.5))


# ------------------------------------------------------------------------------------------------ bootstrap
def summarise(rows: list[dict], reps: int = 1000, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Per-task and headline skill for run and ref, with the paired subject-level Poisson bootstrap."""
    empty = {"run": np.nan, "ref": np.nan, "delta": np.nan, "lo": np.nan, "hi": np.nan, "n_tasks": 0}
    if not rows:
        return pd.DataFrame(), empty
    subj = subject_of()
    keys = sorted({subj.get(v, v) for r in rows for v in r["vids"]})
    pos = {k: i for i, k in enumerate(keys)}
    rng = np.random.default_rng(seed)
    w_subj = np.vstack([np.ones(len(keys)), rng.poisson(1.0, size=(reps, len(keys)))])   # row 0: the point estimate
    per_task, run_s, ref_s = [], [], []
    for r in rows:
        w = w_subj[:, [pos[subj.get(v, v)] for v in r["vids"]]]
        sk = lambda pairs: np.nanmean(np.stack([skill(r["metric"], metric_reps(r["metric"], y, p, w), r["ncls"])  # noqa: E731
                                                for y, p in pairs]), axis=0)
        a, b = sk(r["run"]), sk(r["ref"])
        run_s.append(a)
        ref_s.append(b)
        d = a - b
        per_task.append({"part": r["part"], "group": r["group"], "task": r["task"], "metric": r["metric"],
                         "n": len(r["vids"]), "seeds": len(r["run"]), "head": r.get("head", ""),
                         "run": a[0], "ref": b[0], "delta": d[0],
                         "lo": np.nanpercentile(d[1:], 2.5), "hi": np.nanpercentile(d[1:], 97.5)})
    groups = [r["group"] for r in rows]

    def headline(m: np.ndarray) -> np.ndarray:                  # (tasks, R+1) -> mean of group means per replicate
        g = pd.DataFrame(m.T, columns=pd.MultiIndex.from_arrays([groups, range(len(groups))]))
        return g.T.groupby(level=0).mean().mean(axis=0).to_numpy()

    hr, hf = headline(np.stack(run_s)), headline(np.stack(ref_s))
    hd = hr - hf
    return pd.DataFrame(per_task), {"run": hr[0], "ref": hf[0], "delta": hd[0], "lo": np.nanpercentile(hd[1:], 2.5),
                                    "hi": np.nanpercentile(hd[1:], 97.5), "n_tasks": len(rows)}
