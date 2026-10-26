"""Run evaluation: the skill scale, the report's paired bootstrap, and the evaluation protocol fixes.
python tests/run.py tests/test_eval_report.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score

from brain.eval.report import core_tasks, metric_reps, pair_rows, summarise
from brain.eval.skill import scale, skill, summarize


def test_skill_is_zero_at_chance_and_one_when_perfect():
    assert skill("auc", 0.5) == 0.0 and skill("auc_ovr_macro", 1.0) == 1.0
    assert skill("bacc", 0.25, n_classes=4) == 0.0 and skill("bacc", 1.0, n_classes=4) == 1.0
    assert skill("r2", -0.4) == 0.0 and skill("r2", 0.7) == 0.7
    assert skill("dice_macro", 0.3) == 0.3
    assert np.isnan(skill("r2", float("nan")))
    assert scale("auc") == 2.0 and abs(scale("bacc", 3) - 1.5) < 1e-12


def test_summarize_averages_group_means_so_big_groups_do_not_dominate():
    s = summarize([("anatomy", 0.9)] * 20 + [("hemorrhage", 0.1), ("hemorrhage", float("nan"))])
    assert abs(s["group_anatomy"] - 0.9) < 1e-12 and s["group_hemorrhage"] == 0.1
    assert abs(s["score"] - 0.5) < 1e-12


def test_weighted_metrics_match_sklearn_for_every_bootstrap_row():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 120)
    s = np.round(rng.normal(y * 0.8, 1), 1)                       # rounded: ties
    w = rng.poisson(1.0, (4, 120)).astype(float)
    assert np.allclose(metric_reps("auc", y, s, w), [roc_auc_score(y, s, sample_weight=r) for r in w])
    yr, pr = rng.normal(size=120), rng.normal(size=120)
    assert np.allclose(metric_reps("r2", yr, pr, w), [r2_score(yr, pr, sample_weight=r) for r in w])
    yc = rng.integers(0, 3, 120)
    pc = np.where(rng.random(120) < 0.6, yc, rng.integers(0, 3, 120))
    assert np.allclose(metric_reps("bacc", yc, pc, w), [balanced_accuracy_score(yc, pc, sample_weight=r) for r in w])


def test_pairs_use_only_scans_both_models_answered_and_headline_weights_groups():
    rng = np.random.default_rng(1)
    v = [f"v{i}" for i in range(60)]
    y = rng.integers(0, 2, 60).astype(float)
    good, bad = y + rng.normal(0, 0.3, 60), rng.normal(0, 1, 60)
    r = pair_rows("transfer", "a", "t1", "auc", [(v, y, good)], [(v[10:], y[10:], bad[10:])])
    assert r["vids"] == sorted(v[10:])
    r2 = pair_rows("transfer", "b", "t2", "auc", [(v, y, good)], [(v, y, good)])
    per, h = summarise([r, r2], reps=200)
    assert per.set_index("task").loc["t2", "delta"] == 0.0               # same predictions: no difference
    assert per.set_index("task").loc["t1", "lo"] > 0                     # a clearly better model is detected
    assert abs(h["delta"] - per.delta.mean()) < 1e-12 and h["n_tasks"] == 2   # two groups of one task each


def test_fewshot_draw_is_stratified_and_seeded():
    from brain.multitask.fewshot import draw

    pool = pd.DataFrame({"volume_id": [f"v{i}" for i in range(200)], "value": [1.0] * 10 + [0.0] * 190})
    a = draw(pool, 40, 0, "binary")
    assert a == draw(pool, 40, 0, "binary") and len(set(a)) == 40
    assert sum(pool.set_index("volume_id").loc[a, "value"]) == 2         # 5% positives kept in the draw


def test_rare_class_tasks_are_cross_validated_and_folds_keep_each_class():
    from brain.downstream import cv
    from brain.downstream.tasks import TaskBundle

    ids = [f"v{i}" for i in range(500)]
    y = np.array([1] * 12 + [0] * 488)
    t = TaskBundle(name="rare", task_type="binary", volume_ids=ids, y=y, splits={v: "test" for v in ids},
                   label_names=["pos"])
    vols = pd.DataFrame({"volume_id": ids, "dedup_key": ids})
    assert cv.minority(t) == 12 and cv.use_cv("auto", t, vols, overlap=1.0, max_subjects=200)
    for _, _, splits in cv.make_folds(t, vols, k=5, repeats=1):
        assert sum(1 for v, s in splits.items() if s == "test" and y[ids.index(v)] == 1) >= 2


def test_head_is_never_picked_on_test():
    from brain.downstream import db

    t = pd.DataFrame({"model_id": ["m"] * 2, "task": ["x"] * 2, "head": ["a", "b"], "view": [""] * 2, "seed": [0, 0],
                      "metric": ["auc"] * 2, "val": [np.nan, np.nan], "test": [0.6, 0.9], "ci_lo": [0.5, 0.8],
                      "ci_hi": [0.7, 0.95], "kind": ["task"] * 2, "finished_utc": ["1", "2"], "pretrain_overlap": [0, 0]})
    orig = db.table
    db.table = lambda con: t
    try:
        rep = db.report(None)
    finally:
        db.table = orig
    assert rep.picked_on.iloc[0] == "none"


def test_core_transfer_set_has_no_duplicates():
    ts = core_tasks(with_seg=True)
    assert len(ts) == len(set(ts)) and "cq500" in ts and "rsna_ich" in ts
