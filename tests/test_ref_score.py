"""train.ref_score: an earlier run's tasks scored on its validation scans only (R19 reports an R18-comparable score)."""
import numpy as np
import pytest
import pandas as pd

from brain.multitask.evaluate import ref_scores
from brain.multitask.registry import Registry, Task


def _reg():
    return Registry([Task("a", "binary", ["?"], [], group="g1"), Task("b", "binary", ["?"], [], group="g2"),
                     Task("new", "binary", ["?"], [], group="g3")])


def test_ref_scores_use_only_reference_tasks_and_scans():
    rng = np.random.default_rng(0)
    perfect = [(float(y), float(y), f"old:{i}") for i, y in enumerate(rng.integers(0, 2, 40))]
    noise = [(float(y), float(rng.random()), f"mr_rate_t1w:{i}") for i, y in enumerate(rng.integers(0, 2, 40))]
    preds = {0: perfect + noise, 1: perfect, 2: noise}           # task "new" is not a reference task
    ref = {"tasks": {"a", "b"}, "volumes": {r[2] for r in perfect}}
    out = ref_scores(preds, _reg(), {}, ref)
    assert out["ref_score"] == 1.0                               # MR-RATE rows and the new task are left out
    assert set(k for k in out if k.startswith("ref_group_")) == {"ref_group_g1", "ref_group_g2"}


def test_reference_val_picks_without_excluded_families(monkeypatch, tmp_path):
    from brain.multitask import data as D

    reg = _reg()
    vols = pd.DataFrame({"volume_id": [f"old:{i}" for i in range(6)] + [f"mr:{i}" for i in range(6)],
                         "dataset_family": ["old"] * 6 + ["mr_rate"] * 6})
    labels = pd.DataFrame({"volume_id": vols.volume_id.tolist() * 2, "task": ["a"] * 12 + ["new"] * 12,
                           "value": [0, 1] * 12, "ref": "", "split": "val", "use": "train"})
    ref_cfg = tmp_path / "ref.yaml"
    ref_cfg.write_text("tasks: {ids: [a]}\n")
    monkeypatch.setattr("brain.common.config.load_config", lambda p: {"tasks": {"ids": ["a"]}})
    cfg = {"train": {"ref_score": {"config": str(ref_cfg), "exclude_families": ["mr_rate"]}}}
    ref = D.reference_val(cfg, labels, vols, reg, ["a", "new"], 64)
    assert ref["tasks"] == {"a"} and ref["volumes"] == {f"old:{i}" for i in range(6)}
    assert D.reference_val({"train": {}}, labels, vols, reg, ["a"], 64) is None


def test_dense_emphasis_weights_bright_target_voxels():
    import torch

    from brain.multitask.losses import emphasis_table, emphasize, slot_losses

    reg = _reg()
    assert emphasis_table({}, reg) is None
    tab = emphasis_table({"dense_emphasis": {"b": {"quantile": 0.9, "weight": 4.0}}}, reg)
    assert tab[1].tolist() == pytest.approx([0.9, 4.0]) and tab[0].tolist() == [0.0, 0.0]
    tgt = torch.zeros(2, 1, 4, 4, 4)
    tgt[:, :, 0, 0, :2] = 1.0                       # 2 bright voxels of 64
    w = torch.ones_like(tgt)
    out = emphasize(tgt, w, torch.tensor([0.9, 0.9]), torch.tensor([4.0, 0.0]))
    assert out[0].max() == 5.0 and out[0].sum() == 64 + 2 * 4 and torch.equal(out[1], w[1])
