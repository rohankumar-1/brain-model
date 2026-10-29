"""The benchmark spec and model registry: task selection by modality and seen labels, model resolution."""

from __future__ import annotations

import pytest

from brain.bench.spec import load
from brain.downstream.models import TRAINING_CLASSES, ModelRef, registry, resolve_model


def _ref(name="m", cls="ssl", modalities=("ct", "mri"), seen=(), ckpt=None) -> ModelRef:
    return ModelRef(name=name, model_id=f"{name}@x", cache_key=name, fingerprint="x", training_class=cls,
                    ckpt=ckpt, spec={"modalities": list(modalities)}, seen_tasks=tuple(seen))


def test_spec_loads_with_a_modality_for_every_task():
    b = load()
    assert b.version >= 1 and len(b.tasks) == len(set(b.tasks)) >= 20
    assert set(b.tasks + b.seg_tasks) <= set(b.modality)
    assert {b.modality[t] for t in b.tasks} == {"ct", "mri"}


def test_a_model_runs_only_tasks_of_its_modalities_and_not_its_seen_ones():
    b = load()
    mri = b.tasks_for(_ref(modalities=("mri",)))
    assert mri and all(b.modality[t] == "mri" for t in mri)
    ct_seen = b.tasks_for(_ref(modalities=("ct",), seen=("rsna_ich",)))
    assert "rsna_ich" not in ct_seen and all(b.modality[t] == "ct" for t in ct_seen)
    # registry models never get segmentation (it reads our token grids)
    assert not set(b.tasks_for(_ref(), with_seg=True)) & set(b.seg_tasks)


def test_our_checkpoints_map_their_trained_tasks_to_seen_benchmark_tasks():
    b = load()
    ours = _ref(cls="supervised", seen=("mmse", "age"), ckpt="x.pt")      # seen_tasks: the registry tasks it trained
    assert {"oasis1_mmse", "oasis2_mmse"} <= b.seen_by(ours)
    assert "oasis1_mmse" not in b.tasks_for(ours)
    assert {"oasis1_cdr", "oasis2_cdr"} <= b.seen_by(ours)                 # CDR stage: MMSE trained on the same cohort
    assert "abide_asd" in b.seen_by(_ref(cls="supervised", seen=("autism_severity",), ckpt="x.pt"))


def test_registry_entries_are_valid_and_resolve():
    reg = registry()
    assert "regional_stats" in reg
    for name, spec in reg.items():
        if spec.get("ckpt"):                                         # BrainQuery checkpoints: class from the file
            assert name.startswith("BQ-"), name
            continue
        assert spec["class"] in TRAINING_CLASSES and ":" in spec["adapter"], name
    m = resolve_model("regional_stats")
    assert m.training_class == "baseline" and m.model_id.startswith("regional_stats@") and not m.is_checkpoint
    with pytest.raises(SystemExit):
        resolve_model("no_such_model")


def test_every_benchmark_task_has_a_pinned_readout():
    import pytest

    b = load()
    if not b.readouts_file.exists():
        pytest.skip(f"benchmark v{b.version} is not pinned yet (python -m brain.bench pin, once its runs are in)")
    pinned = b.readouts()
    assert pinned is not None and set(b.tasks) <= set(pinned)


def test_cv_repeats_saved_and_scored(tmp_path, monkeypatch):
    """Every CV repeat's predictions are saved and read back; a single model's score averages them."""
    from types import SimpleNamespace

    import numpy as np

    from brain.downstream.results import save_head_result
    from brain.eval import report

    (tmp_path / "heads").mkdir()
    (tmp_path / "predictions").mkdir()
    rng = np.random.default_rng(0)
    vids = [f"v{i}" for i in range(60)]
    y = rng.integers(0, 2, 60)
    s0, s1 = y + rng.normal(0, 0.5, 60), y + rng.normal(0, 2.0, 60)
    save_head_result(tmp_path, "h", {}, y_true=y, y_score=s0, volume_ids=vids, repeats=[(vids, y, s1)])
    preds = report._suite_preds(SimpleNamespace(pred=str(tmp_path / "predictions" / "h.npz")))
    assert len(preds) == 2 and preds[1][0] == vids
    monkeypatch.setattr(report, "subject_of", lambda: {})
    point, lo, hi = report.score_ci("auc", preds, reps=200)
    one = report.score_ci("auc", preds[:1], reps=200)[0]
    two = report.score_ci("auc", preds[1:], reps=200)[0]
    assert np.isclose(point, (one + two) / 2) and lo <= point <= hi


def test_index_check_accepts_added_rows_only(tmp_path):
    import pandas as pd

    from brain.bench.spec import index_check

    old = pd.DataFrame({"volume_id": ["a", "b", "b"], "dataset": ["x", "x", "y"], "split": ["train", "test", "test"],
                        "mask_seg": [None, "m", None], "vol_idx": [0, 1, 2]})
    old.to_csv(tmp_path / "old.csv", index=False)
    added = pd.concat([old.iloc[:1], pd.DataFrame({"volume_id": ["c"], "dataset": ["z"], "split": ["val"]}),
                       old.iloc[1:]]).assign(vol_idx=range(4))     # inserted mid-table: row numbers shift
    added.to_csv(tmp_path / "added.csv", index=False)
    old.assign(split=["train", "train", "test"]).to_csv(tmp_path / "moved.csv", index=False)
    assert index_check(tmp_path / "old.csv", tmp_path / "added.csv")
    assert not index_check(tmp_path / "old.csv", tmp_path / "moved.csv")
    assert not index_check(tmp_path / "added.csv", tmp_path / "old.csv")      # a dropped row
    assert load().index_shas[0] == load().index_sha1
