"""Subject-grouped repeated k-fold CV for small frozen-head tasks (brain.downstream.cv)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from brain.downstream import cv
from brain.downstream.tasks import TaskBundle


def _task(n_subj: int = 30, per: int = 2, kind: str = "binary") -> tuple[TaskBundle, pd.DataFrame]:
    rng = np.random.default_rng(0)
    vids = [f"d:s{s}_v{k}" for s in range(n_subj) for k in range(per)]
    y_subj = rng.integers(0, 2, n_subj) if kind == "binary" else rng.normal(40, 15, n_subj)
    y = np.repeat(y_subj, per)
    vol = pd.DataFrame({"volume_id": vids, "dedup_key": [v.split("_")[0] for v in vids]})
    return TaskBundle("t", kind, vids, {v: "train" for v in vids}, y, ["y"]), vol


def test_folds_group_subjects_and_test_every_volume_once_per_repeat():
    for kind in ("binary", "regression"):
        task, vol = _task(kind=kind)
        folds = cv.make_folds(task, vol, k=5, repeats=2, seed=0)
        subj = dict(zip(vol.volume_id, vol.dedup_key))
        for r in (0, 1):
            tested = []
            for rr, _, splits in folds:
                if rr != r:
                    continue
                test_s = {subj[v] for v, s in splits.items() if s == "test"}
                other_s = {subj[v] for v, s in splits.items() if s != "test"}
                assert not test_s & other_s                          # a subject never straddles test and train/val
                assert any(s == "val" for s in splits.values())       # an inner validation set exists
                tested += [v for v, s in splits.items() if s == "test"]
            assert sorted(tested) == sorted(task.volume_ids)          # each volume tested exactly once per repeat


def test_auto_mode_for_rare_classes_and_small_tasks_unseen_in_pretraining():
    import dataclasses

    task, vol = _task(n_subj=40)
    assert cv.use_cv("auto", task, vol, overlap=0.0, max_subjects=200)
    assert cv.use_cv("auto", task, vol, overlap=1.0, max_subjects=200)        # rare class: CV whatever the pool
    reg = dataclasses.replace(task, task_type="regression", y=np.arange(len(task.y), dtype=float))
    assert not cv.use_cv("auto", reg, vol, overlap=1.0, max_subjects=200)     # images were in the SSL pool
    assert not cv.use_cv("auto", reg, vol, overlap=0.0, max_subjects=20)      # large enough for a fixed split
    assert cv.use_cv("on", task, vol, overlap=1.0, max_subjects=20)


def test_validation_holds_every_class_and_other_folds_are_unchanged():
    """A rare class the random 15% pick misses gets training subjects moved into validation (else every head is fitted
    untuned in that fold); the repair draws from its own stream, so folds that needed none keep their splits."""
    import dataclasses

    task, vol = _task(n_subj=120, per=1)
    y = np.zeros(len(task.y), int)
    y[:12] = 1                                                     # 10% positives: some folds' val pick misses them
    task = dataclasses.replace(task, y=y)
    info: dict = {}
    folds = cv.make_folds(task, vol, k=5, repeats=3, seed=0, report=info)
    assert info["val_repaired"] > 0
    pos = set(np.asarray(task.volume_ids)[y == 1])
    for _, _, splits in folds:
        assert any(splits[v] == "val" for v in pos)                # both classes in every validation set
        assert any(splits[v] == "train" for v in pos)              # ... and the training set keeps some positives
    # without the repair: the same folds, the same test sets, and identical splits wherever no class was missing
    orig = cv._guard_val
    try:
        cv._guard_val = lambda *a, **k: 0
        plain = cv.make_folds(task, vol, k=5, repeats=3, seed=0)
    finally:
        cv._guard_val = orig
    same = 0
    for (_, _, a), (_, _, b) in zip(folds, plain):
        assert {v for v, s in a.items() if s == "test"} == {v for v, s in b.items() if s == "test"}
        same += a == b
    assert same == len(folds) - info["val_repaired"]


def test_validation_halves_split_subjects_and_hold_every_class():
    task, vol = _task(n_subj=200, per=2)
    for _, _, splits in cv.make_folds(task, vol, k=5, repeats=1, seed=0):
        halves = cv.val_halves(task, vol, splits, seed=0)
        assert set(halves) == {v for v, s in splits.items() if s == "val"}
        subj = dict(zip(vol.volume_id, vol.dedup_key))
        by_subj = {}
        for v, k in halves.items():
            assert by_subj.setdefault(subj[v], k) == k                  # a subject's scans stay in one half
        y = dict(zip(task.volume_ids, np.asarray(task.y)))
        for c in (0, 1):
            assert {k for v, k in halves.items() if y[v] == c} == {0, 1}  # each class in both halves
        n = np.bincount(list(by_subj.values()))
        assert abs(int(n[0]) - int(n[1])) <= 2


def test_linear_head_cross_fits_validation():
    """The grid value that scores a validation half is chosen on the other half; test uses the full-val choice."""
    from brain.downstream.heads import make_logreg_pool

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 1, 8)).astype(np.float32)
    y = (x[:, 0, 0] + rng.normal(scale=1.5, size=200) > 0).astype(int)
    mask = np.ones((200, 1), bool)
    tr, va = slice(0, 140), slice(140, 200)
    half = np.arange(60) % 2
    h = make_logreg_pool(False, 0)
    h.pool = "mean"
    h.fit(x[tr], y[tr], mask[tr], x[va], y[va], mask[va], val_half=half)
    assert h.cross_ is not None and len(h.cross_) == 2
    pv = h.predict_val(x[va], mask[va], half)
    for k in (0, 1):
        assert np.allclose(pv[half == k], h._scores(h.cross_[k], h.inputs(x[va][half == k], mask[va][half == k])))
    full = make_logreg_pool(False, 0)
    full.pool = "mean"
    full.fit(x[tr], y[tr], mask[tr], x[va], y[va], mask[va])
    assert np.allclose(h.predict_scores(x[va], mask[va]), full.predict_scores(x[va], mask[va]))
