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
