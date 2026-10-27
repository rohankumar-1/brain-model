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


def test_registry_entries_are_valid_and_resolve():
    reg = registry()
    assert "regional_stats" in reg
    for name, spec in reg.items():
        assert spec["class"] in TRAINING_CLASSES and ":" in spec["adapter"], name
    m = resolve_model("regional_stats")
    assert m.training_class == "baseline" and m.model_id.startswith("regional_stats@") and not m.is_checkpoint
    with pytest.raises(SystemExit):
        resolve_model("no_such_model")
