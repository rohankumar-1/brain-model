"""Classical brain masks on phantoms. numpy/scipy only."""

from __future__ import annotations

import numpy as np

from brain.prep import brainmask

N = (96, 128, 128)  # (Z, Y, X) at 1 mm


def _radius(rz=45, ry=60, rx=52):
    z, y, x = np.meshgrid(*(np.arange(n) - (n - 1) / 2 for n in N), indexing="ij")
    return np.sqrt((z / rz) ** 2 + (y / ry) ** 2 + (x / rx) ** 2)


def _iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def ct_phantom():
    r = _radius()
    hu = np.full(N, -1000.0, np.float32)
    hu[r < 1.0] = 40.0          # scalp
    hu[r < 0.88] = 900.0        # skull, about 7 mm thick
    hu[r < 0.78] = 30.0         # brain
    hu[r < 0.25] = 5.0          # ventricles (CSF)
    return hu, r < 0.78


def mri_phantom(stripped=False):
    r = _radius()
    z = np.full(N, -6.0, np.float32)
    if not stripped:
        z[r < 1.0] = 2.0        # scalp / fat
        z[r < 0.88] = -4.0      # skull: dark but above the background sentinel
    z[r < 0.78] = 0.5           # brain
    return z, r < 0.78


def test_classical_ct_recovers_the_intracranial_volume():
    hu, truth = ct_phantom()
    m = brainmask.classical_ct(hu)
    assert _iou(m, truth) > 0.85
    assert not m[hu > 300].any()                      # no bone in the mask


def test_classical_ct_ignores_a_table_touching_the_head():
    hu, truth = ct_phantom()
    hu[:, 100:110, 30:100] = 120.0                     # table slab fused to the scalp
    assert _iou(brainmask.classical_ct(hu), truth) > 0.8


def test_classical_mri_erodes_the_head_for_skull_on_scans():
    z, truth = mri_phantom()
    m = brainmask.classical_mri(z, -6.0, stripped=False)
    assert _iou(m, truth) > 0.6 and m.sum() < (z > -5.5).sum()


def test_classical_mri_keeps_the_foreground_of_stripped_scans():
    z, truth = mri_phantom(stripped=True)
    m = brainmask.classical_mri(z, -6.0, stripped=True)
    assert _iou(m, truth) > 0.95


def test_compute_reports_method_and_plausibility_without_synthstrip():
    hu, _ = ct_phantom()
    mask, info = brainmask.compute(hu, hu, "ct", method="classical")
    assert mask.dtype == np.uint8 and info["brain_method"] == "classical" and info["brain_ok"]
    none, info = brainmask.compute(hu, hu, "ct", method="none")
    assert none is None and info["brain_method"] == "none"


def test_auto_falls_back_to_classical_when_synthstrip_is_missing(monkeypatch=None):
    import os

    os.environ.pop("BRAIN_SYNTHSTRIP", None)
    hu, _ = ct_phantom()
    if brainmask.shutil.which("mri_synthstrip"):
        return  # a real install would run it; the fallback path is not what this machine exercises
    _, info = brainmask.compute(hu, hu, "ct", method="auto")
    assert info["brain_method"] == "classical"


def test_synthstrip_pad_stack_batches_variable_shapes():
    import torch
    from brain.prep.synthstrip_runtime import pad_stack

    a = torch.zeros(1, 1, 64, 64, 32)
    b = torch.zeros(1, 1, 48, 80, 16)
    stacked, shapes = pad_stack([a, b])
    assert stacked.shape[0] == 2 and stacked.shape[1] == 1
    assert all(s % 64 == 0 for s in stacked.shape[-3:])
    assert shapes == [(64, 64, 32), (48, 80, 16)]


def test_cli_input_image_carries_the_store_ras_direction():
    import SimpleITK as sitk

    ref = sitk.DICOMOrient(sitk.Image(4, 4, 4, sitk.sitkFloat32), "RAS").GetDirection()
    assert brainmask.ras_image(np.zeros((3, 4, 5), np.float32)).GetDirection() == ref


def test_synthstrip_runtime_hands_the_unet_an_lia_head():
    """Regression: an RAS array labelled with surfa's default (LIA) geometry reached the UNet rotated 90 degrees."""
    import pytest

    pytest.importorskip("surfa")
    pytest.importorskip("torch")
    from brain.prep.synthstrip_runtime import _volume_from_zyx, _zyx_from_volume

    a = np.zeros((40, 50, 60), np.float32)          # (Z, Y, X) = (S, A, R) increasing
    a[35, 5, 5] = 1.0                                # superior, posterior, left
    vol = _volume_from_zyx(a)
    lia = vol.conform(voxsize=1.0, dtype="float32", method="nearest", orientation="LIA")
    i, j, k = np.argwhere(np.asarray(lia.data) > 0.5)[0]
    assert i > lia.shape[0] // 2                     # L axis: a left voxel has a high index
    assert j < lia.shape[1] // 2                     # I axis: a superior voxel has a low index
    assert k < lia.shape[2] // 2                     # A axis: a posterior voxel has a low index
    back = _zyx_from_volume(lia.resample_like(vol, method="nearest"))
    assert np.argwhere(back > 0.5).tolist() == [[35, 5, 5]]


def test_qc_flags_a_mask_that_disagrees_with_the_template():
    _, truth = ct_phantom()
    shifted = np.roll(truth, 40, axis=1)
    fg = _radius() < 1.0
    good, bad = brainmask.qc(truth, truth, fg), brainmask.qc(shifted, truth, fg)
    assert good["brain_ok"] and good["brain_dice_tpl"] > 0.99
    assert not bad["brain_ok"] and bad["brain_dice_tpl"] < brainmask.TEMPLATE_DICE_MIN
    assert "brain_dice_tpl" not in brainmask.qc(shifted)          # no prior: fraction band only


def test_auto_prefers_classical_when_synthstrip_fails_the_template_check(monkeypatch):
    hu, truth = ct_phantom()
    monkeypatch.setattr(brainmask, "synthstrip", lambda raw, modality: np.roll(truth, 40, axis=1))
    mask, info = brainmask.compute(hu, hu, "ct", method="auto", prior=truth, fg=hu > -500)
    assert info["brain_method"] == "classical" and info["brain_fallback"] == "synthstrip_failed_qc"
    assert info["brain_ok"] and _iou(mask.astype(bool), truth) > 0.85


def test_failed_mri_masks_with_a_plausible_extent_still_set_the_z_scale():
    """``pipeline.finish`` standardises MRI on a QC-failed brain mask when its extent is plausible (the rule the
    2026-11-03 ``restat_failed`` pass applied to the store), and on the head otherwise."""
    from brain.prep import intensity
    from brain.prep.pipeline import mri_stats_on_brain

    m = np.ones((2, 2, 2), np.uint8)
    assert not mri_stats_on_brain(None, {"brain_ok": True})
    assert mri_stats_on_brain(m, {"brain_ok": True, "brain_frac": 0.5})
    assert mri_stats_on_brain(m, {"brain_ok": False, "brain_frac": 0.15})
    assert not mri_stats_on_brain(m, {"brain_ok": False, "brain_frac": 0.01})
    assert not mri_stats_on_brain(m, {"brain_ok": False, "brain_frac": 0.45})
    z, truth = mri_phantom()
    rng = np.random.default_rng(0)
    raw = np.where(z > -5.5, 100.0, 0.0).astype(np.float32)
    raw[truth] = 300.0 + 50.0 * rng.standard_normal(int(truth.sum())).astype(np.float32)
    out, st = intensity.mri_native(raw, truth)
    x = out[truth].astype(np.float32)
    assert st["stats_from"] == "brain" and abs(np.median(x)) < 0.05
