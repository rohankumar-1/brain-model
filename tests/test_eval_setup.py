"""CPU tests for CQ500 majority labels, windows, and frozen heads. python -m tests.test_eval_setup"""

from __future__ import annotations

import numpy as np
import torch

from brain.common.windowing import windows_for
from brain.downstream.embeddings import uniform_slice_indices
from brain.downstream.heads import (
    DEFAULT_HEADS, SliceAttnMLP, build_heads, make_logreg_pool, make_ridge, make_xgboost,
)
from brain.downstream.tasks import (
    AISD_REGIONS, CLASSIFICATION_TASKS, CQ500_LABELS, FOCAL_SLICE_TASKS, FROZEN_TASKS,
    ICH_REGIONS, REGRESSION_TASKS, SEGMENTATION_TASKS, TASK_NAMES, cq500_majority,
    load_aisd_task, load_instance_task,
)
from brain.models.vit2d import VisionTransformer


def test_windows_for_counts():
    assert windows_for("canonical", "ct") == [(40.0, 40.0)]
    assert len(windows_for("multi", "ct")) == 3
    assert len(windows_for("shifted", "ct")) == 3
    assert windows_for("canonical", "mri")[0] == (0.5, 1.5)


def test_suite_lists_cover_every_named_task():
    assert TASK_NAMES == CLASSIFICATION_TASKS + REGRESSION_TASKS + SEGMENTATION_TASKS
    assert FROZEN_TASKS == CLASSIFICATION_TASKS + REGRESSION_TASKS
    assert set(TASK_NAMES) == set(CLASSIFICATION_TASKS) | set(REGRESSION_TASKS) | set(SEGMENTATION_TASKS)
    assert len(TASK_NAMES) == len(set(TASK_NAMES))
    assert set(("oasis1_age", "oasis1_cdr", "cq500")) <= set(FROZEN_TASKS)


def test_focal_tasks_include_ich_and_seg():
    assert "cq500" in FOCAL_SLICE_TASKS
    assert "brats2023_gli" in FOCAL_SLICE_TASKS
    assert "aisd" in SEGMENTATION_TASKS and "aisd" in FOCAL_SLICE_TASKS
    assert "instance" in SEGMENTATION_TASKS and "instance" in FOCAL_SLICE_TASKS
    assert AISD_REGIONS["infarct"] == (1, 2, 3, 5)
    assert ICH_REGIONS["ICH"] == (1, 2)
    assert "oasis1_age" not in FOCAL_SLICE_TASKS
    for name in SEGMENTATION_TASKS:
        assert name in FOCAL_SLICE_TASKS
    for name in ("cq500", "rsna", "rsna_ich", "msd"):
        assert name in FOCAL_SLICE_TASKS


def test_uniform_64_covers_the_volume():
    z = uniform_slice_indices(128, 64)
    assert len(z) == 64 and z[0] >= 0 and z[-1] < 128
    assert np.all(np.diff(z) >= 1)


def test_vit_features_are_cls_mean_amax():
    torch.manual_seed(0)
    net = VisionTransformer(size="tiny", img_size=32, patch_size=16)
    y = net.features(torch.randn(2, 1, 32, 32))
    assert y.shape == (2, 3 * net.embed_dim)


def test_logreg_selects_c_on_val():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 4, 6)).astype(np.float32)
    y = (x[:, :, 0].mean(1) > 0).astype(np.int64)
    mask = np.ones((80, 4), dtype=bool)
    head = make_logreg_pool(False, 0)
    head.fit(x[:50], y[:50], mask[:50], x[50:65], y[50:65], mask[50:65])
    assert head.chosen_ is not None
    scores = head.predict_scores(x[65:], mask[65:])
    assert scores.shape == (15,) and np.isfinite(scores).all()


def test_ridge_selects_alpha_on_val():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 4, 6)).astype(np.float32)
    y = x[:, :, 0].mean(1) * 10.0
    mask = np.ones((80, 4), dtype=bool)
    head = make_ridge()
    head.fit(x[:50], y[:50], mask[:50], x[50:65], y[50:65], mask[50:65])
    assert head.name == "ridge" and head.chosen_ is not None
    pred = head.predict_scores(x[65:], mask[65:])
    assert pred.shape == (15,) and np.isfinite(pred).all()


def test_ridge_and_xgboost_use_mean_max():
    assert make_ridge().pool == "mean_max"
    assert make_xgboost(False, False, 0).pool == "mean_max"
    assert make_xgboost(False, True, 0).pool == "mean_max"


def test_slice_attn_scorer_reads_anatomical_z():
    torch.manual_seed(0)
    m = SliceAttnMLP(8, 4, 1)
    x = torch.randn(3, 4, 8)
    mask = torch.ones(3, 4, dtype=torch.bool)
    z = torch.tensor([[0.0, 40.0, 80.0, 127.0]]).expand(3, -1)
    out = m(x, mask, z)
    assert out.shape == (3, 1) and torch.isfinite(out).all()
    assert m.score.in_features == 9


def test_default_heads_are_linear_probes():
    common = dict(latent_dim=8, max_slices=4, n_out=1, multilabel=False, seed=0, which=DEFAULT_HEADS.split(","))
    clf = [h.name for h in build_heads(**common, task_type="binary")]
    reg = [h.name for h in build_heads(**common, task_type="regression")]
    assert clf == ["logreg_pool"]
    assert reg == ["ridge"]


def test_optional_heads_still_build_when_requested():
    names = [h.name for h in build_heads(
        latent_dim=8, max_slices=4, n_out=1, multilabel=False, seed=0,
        which=["ridge", "xgboost_mean", "slice_attn_mlp"], task_type="regression",
    )]
    assert names == ["ridge", "xgboost_mean", "slice_attn_mlp"]


def test_aisd_task_pins_official_test_ids():
    import pandas as pd
    from brain.datasets.download_aisd import splits_for
    from brain.prep.sources import AISD_TEST_IDS

    assert len(AISD_TEST_IDS) == 52 and len(set(AISD_TEST_IDS)) == 52
    s = splits_for(["0073410", "0000001", "0000002"])
    assert s["0073410"] == "test" and set(s.values()) >= {"test"} and s["0000001"] != "test"
    vols = pd.DataFrame({
        "volume_id": ["aisd:0073410"], "dataset": ["aisd"], "split": ["test"],
        "image_path": ["/tmp/x.zip"], "mask_lesion": ["/tmp/x.lesion.zip"], "modality": ["ct"],
    })
    t = load_aisd_task(vols)
    assert t.name == "aisd" and t.splits["aisd:0073410"] == "test" and t.regions["infarct"] == (1, 2, 3, 5)


def test_instance_task_uses_ich_mask_column():
    import pandas as pd

    vols = pd.DataFrame({
        "volume_id": ["instance:001"], "dataset": ["instance"], "split": ["train"],
        "image_path": ["/tmp/x.zip"], "mask_ich": ["/tmp/x.ICH.zip"], "modality": ["ct"],
    })
    t = load_instance_task(vols)
    assert t.name == "instance" and t.task_type == "segmentation"
    assert t.mask_paths["instance:001"] == "/tmp/x.ICH.zip"
    assert t.regions == ICH_REGIONS


def test_cq500_majority_is_binary():
    labels = cq500_majority()
    assert set(CQ500_LABELS) <= set(labels.columns)
    assert labels["name"].nunique() == len(labels)
    assert labels[CQ500_LABELS].isin([0, 1]).all().all()
    # ICH should be common but not everyone
    rate = float(labels["ICH"].mean())
    assert 0.2 < rate < 0.8, rate


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
