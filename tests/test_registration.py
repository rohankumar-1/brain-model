"""Registration on phantoms (SimpleITK, CPU). The first test is the slow one; keep tolerances loose."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from brain.prep import geometry, pipeline, registration
from brain.prep.sources import Case

SP = 2.0
N = 80  # 160 mm FOV at 2 mm


def head(center=(0.0, 0.0, 0.0), scale=1.0, noise=0.0, seed=0):
    """Asymmetric T1-like head: scalp shell, dark skull, brain with two off-axis blobs."""
    rng = np.random.default_rng(seed)
    ax = (np.arange(N) - (N - 1) / 2) * SP
    Z, Y, X = np.meshgrid(ax, ax, ax, indexing="ij")
    cx, cy, cz = center
    r = np.sqrt(((X - cx) / (62 * scale)) ** 2 + ((Y - cy) / (74 * scale)) ** 2 + ((Z - cz) / (62 * scale)) ** 2)
    v = np.zeros_like(X, np.float32)
    v[r < 1.0] = 80.0
    v[r < 0.9] = 15.0
    v[r < 0.8] = 120.0
    v[(((X - cx - 20) / 12) ** 2 + ((Y - cy + 15) / 16) ** 2 + ((Z - cz) / 14) ** 2) < 1] = 200.0
    v[(((X - cx + 18) / 10) ** 2 + ((Y - cy - 25) / 12) ** 2 + ((Z - cz + 10) / 10) ** 2) < 1] = 40.0
    v += rng.normal(0, noise, v.shape).astype(np.float32)
    img = sitk.GetImageFromArray(v)
    img.SetSpacing((SP, SP, SP))
    img.SetOrigin(tuple(-(N - 1) / 2 * SP for _ in range(3)))
    return img


def template() -> registration.Template:
    t = head()
    return registration.Template(t, t, (0.0, 0.0, 0.0), None, "phantom")


def test_affine_registration_recovers_a_shifted_and_slightly_larger_head():
    tpl = template()
    mov = head(center=(14.0, -10.0, 6.0), scale=1.08, noise=3.0, seed=1)
    # start from the (approximate) brain center, as the pipeline does
    res = registration.register(mov, tpl, "mri", init_center_mm=(12.0, -8.0, 4.0), mode="affine")
    assert res.info["reg_ok"], res.info
    mapped = np.array(res.transform.TransformPoint((0.0, 0.0, 0.0)))   # template center in scan space
    assert np.linalg.norm(mapped - np.array([14.0, -10.0, 6.0])) < 5.0, (mapped, res.info)
    assert res.info["reg_rot_deg"] < 6.0 and 0.9 < res.info["reg_scale"] < 1.25, res.info


def test_failed_fit_falls_back_to_the_starting_translation():
    tpl = template()
    blank = sitk.GetImageFromArray(np.zeros((N, N, N), np.float32))
    blank.SetSpacing((SP, SP, SP))
    res = registration.register(blank, tpl, "mri", init_center_mm=(5.0, 0.0, 0.0), mode="affine")
    assert res.info["reg_method"] in ("translation", "rigid", "affine")
    moved = np.array(res.transform.TransformPoint((0.0, 0.0, 0.0)))
    assert np.linalg.norm(moved - np.array([5.0, 0.0, 0.0])) < 30.0     # never flung away by a degenerate fit


def test_pose_summary_reads_rotation_and_scale():
    t = sitk.AffineTransform(3)
    t.SetMatrix((1.2, 0, 0, 0, 1.2, 0, 0, 0, 1.2))
    p = registration.pose_summary(t)
    assert abs(p["scale"] - 1.2) < 1e-6 and p["rot_deg"] < 1e-6 and abs(p["aniso"] - 1) < 1e-6
    rot = sitk.Euler3DTransform()
    rot.SetRotation(0.0, 0.0, np.radians(20))
    assert abs(registration.pose_summary(rot)["rot_deg"] - 20) < 1e-4
    assert registration._sane(rot) and not registration._sane(_big())


def _big():
    t = sitk.AffineTransform(3)
    t.SetMatrix((2.0, 0, 0, 0, 1, 0, 0, 0, 1))
    return t


def test_output_grid_is_centered_on_the_template_center():
    tpl = template()
    g = registration.output_grid(tpl, (64, 64, 32))
    assert g.size == (64, 64, 32) and g.spacing == (1.0, 1.0, 1.0)
    mid = np.array(g.origin) + (np.array(g.size) - 1) / 2 * np.array(g.spacing)
    assert np.allclose(mid, tpl.center_mm, atol=1e-6)


def test_native_info_reports_the_coarsest_axis():
    img = sitk.GetImageFromArray(np.zeros((10, 20, 30), np.float32))
    img.SetSpacing((0.5, 0.5, 5.0))
    info = geometry.native_info(img)
    assert info["thick_axis"] == 2 and info["thick_mm"] == 5.0 and info["aniso"] == 10.0
    assert (info["src_nx"], info["src_ny"], info["src_nz"]) == (30, 20, 10)


def test_process_with_registration_stores_the_head_at_the_template_center(tmp_path):
    td = tmp_path
    tpath, mpath = Path(td) / "template.nii.gz", Path(td) / "scan.nii.gz"
    sitk.WriteImage(head(), str(tpath))
    sitk.WriteImage(head(center=(12.0, -8.0, 5.0), scale=1.05, noise=2.0, seed=2), str(mpath))
    opts = pipeline.Options(registration="rigid", template=str(tpath), brain_mask="classical")
    res = pipeline.process(Case("oasis1", "phantom", "mri", mpath), size=(64, 64, 32), opts=opts)
    assert res.info["reg_method"] in ("rigid", "affine", "translation") and "reg_rot_deg" in res.info
    assert res.vol.shape == (32, 64, 64) and res.brain is not None
    fg = res.vol > -5.5
    zc, yc, xc = (np.argwhere(fg).mean(0) - (np.array(fg.shape) - 1) / 2)
    assert abs(xc) < 6 and abs(yc) < 6 and abs(zc) < 6, (xc, yc, zc, res.info)
    assert res.info["thick_axis"] in (0, 1, 2) and res.info["brain_method"] == "classical"


def test_stripped_scan_registers_to_the_brain_only_template_and_gets_scaled():
    t = head()
    a = sitk.GetArrayFromImage(t)
    brain = sitk.GetImageFromArray((a >= 120).astype(np.float32))      # brain + blobs, no scalp or skull
    brain.CopyInformation(t)
    tpl = registration.Template(t, t, (0.0, 0.0, 0.0), brain, "phantom", registration.brain_only(t, brain))
    mov = head(center=(6.0, -4.0, 2.0), scale=1.12, noise=2.0, seed=3)
    m = sitk.GetArrayFromImage(mov)
    stripped = sitk.GetImageFromArray(np.where(m >= 100, m, 0).astype(np.float32))
    stripped.CopyInformation(mov)
    res = registration.register(stripped, tpl, "mri", init_center_mm=(6.0, -4.0, 2.0), mode="affine", stripped=True)
    assert res.info["reg_template"] == "brain" and res.info["reg_method"] == "affine", res.info
    assert 1.05 < res.info["reg_scale"] < 1.2, res.info
    skull_on = registration.register(mov, tpl, "mri", init_center_mm=(6.0, -4.0, 2.0), mode="affine")
    assert skull_on.info["reg_template"] == "head"


def test_torch_backend_recovers_the_true_pose():
    from brain.prep import registration_torch

    tpl = template()
    mov = head(center=(14.0, -10.0, 6.0), scale=1.08, noise=3.0, seed=1)
    res = registration_torch.register(mov, tpl, "mri", init_center_mm=(12.0, -8.0, 4.0), mode="affine")
    assert res.info["reg_ok"] and res.info["reg_backend"] == "torch", res.info
    mapped = np.array(res.transform.TransformPoint((0.0, 0.0, 0.0)))
    assert np.linalg.norm(mapped - np.array([14.0, -10.0, 6.0])) < 3.0, (mapped, res.info)
    assert res.info["reg_rot_deg"] < 1.0 and 1.07 < res.info["reg_scale"] < 1.09, res.info
    # against the true map (scale 1.08 about the origin, then the shift): sub-millimetre everywhere.
    # (SimpleITK's own error here swings between 0.1 and 7 mm with its random sampling, so it is no reference.)
    probe = [(x, y, z) for x in (-40.0, 0.0, 40.0) for y in (-50.0, 0.0, 50.0) for z in (-40.0, 0.0, 40.0)]
    true = lambda q: 1.08 * np.asarray(q) + np.array([14.0, -10.0, 6.0])  # noqa: E731
    err = [np.linalg.norm(np.subtract(res.transform.TransformPoint(q), true(q))) for q in probe]
    assert max(err) < 1.0 and np.mean(err) < 0.5, err


def test_torch_backend_uses_the_brain_template_for_stripped_scans():
    from brain.prep import registration_torch

    t = head()
    brain = sitk.GetImageFromArray((sitk.GetArrayFromImage(t) >= 120).astype(np.float32))
    brain.CopyInformation(t)
    tpl = registration.Template(t, t, (0.0, 0.0, 0.0), brain, "phantom", registration.brain_only(t, brain))
    mov = head(center=(6.0, -4.0, 2.0), scale=1.12, noise=2.0, seed=3)
    m = sitk.GetArrayFromImage(mov)
    stripped = sitk.GetImageFromArray(np.where(m >= 100, m, 0).astype(np.float32))
    stripped.CopyInformation(mov)
    res = registration_torch.register(stripped, tpl, "mri", init_center_mm=(6.0, -4.0, 2.0), stripped=True)
    assert res.info["reg_template"] == "brain" and res.info["reg_method"] == "affine", res.info
    assert 1.05 < res.info["reg_scale"] < 1.2, res.info


def test_torch_backend_retries_out_of_memory_and_never_falls_back_to_translation():
    import torch
    from brain.prep import registration_torch as T

    calls, real, real_device, real_sleep = [], T._register, T._device, T.time.sleep

    def flaky(*args):                               # first attempt runs out of memory, the retry succeeds (on CPU)
        calls.append(args[-1].type)
        if len(calls) == 1:
            raise torch.OutOfMemoryError("simulated")
        return real(*args[:-1], torch.device("cpu"))

    T._register, T._device, T.time.sleep = flaky, (lambda: torch.device("cuda")), (lambda s: None)
    try:
        res = T.register(head(center=(8.0, 0.0, 0.0), noise=2.0, seed=2), template(), "mri",
                         init_center_mm=(8.0, 0.0, 0.0), mode="rigid")
    finally:
        T._register, T._device, T.time.sleep = real, real_device, real_sleep
    assert calls == ["cuda", "cuda"] and res.info["reg_oom_retries"] == 1, (calls, res.info)
    assert res.info["reg_method"] == "rigid", res.info


def test_stored_masks_map_back_onto_the_source_scan_and_keep_their_true_volume(tmp_path):
    from brain.prep import native, store

    td = tmp_path
    td = Path(td)
    tpath, mpath, lpath = td / "template.nii.gz", td / "scan.nii.gz", td / "lesion.nii.gz"
    sitk.WriteImage(head(), str(tpath))
    scan = head(center=(10.0, -6.0, 4.0), scale=1.10, noise=2.0, seed=5)    # a bigger head than the template
    sitk.WriteImage(scan, str(mpath))
    ax = (np.arange(N) - (N - 1) / 2) * SP
    Z, Y, X = np.meshgrid(ax, ax, ax, indexing="ij")
    truth = (((X - 30) ** 2 + (Y + 4) ** 2 + (Z - 10) ** 2) < 12.0 ** 2).astype(np.uint8)  # 12 mm lesion
    les = sitk.GetImageFromArray(truth)
    les.CopyInformation(scan)
    sitk.WriteImage(les, str(lpath))
    opts = pipeline.Options(registration="affine", template=str(tpath), brain_mask="classical", reg_backend="torch")
    res = pipeline.process(Case("oasis1", "phantom", "mri", mpath, (("seg", lpath),)), size=(96, 96, 96), opts=opts)
    assert res.info["reg_method"] == "affine" and native.has_native(res.info), res.info
    # default: the source grid as prep read it (RAS index order) -> compare with the lesion read the same way
    ras = sitk.GetArrayFromImage(geometry.read(lpath)) > 0.5
    back = sitk.GetArrayFromImage(native.to_native(res.masks["seg"], res.info)) > 0
    assert back.shape == ras.shape
    dice = 2 * (back & ras).sum() / (back.sum() + ras.sum())
    assert dice > 0.85, dice                                   # 2 mm source grid, one nearest-neighbour step
    # like=<original file>: its exact grid and index order, ready to hand over with the scan
    orig = sitk.GetArrayFromImage(native.to_native(res.masks["seg"], res.info, like=sitk.ReadImage(str(lpath)))) > 0
    assert orig.shape == truth.shape and 2 * (orig & (truth > 0)).sum() / (orig.sum() + truth.sum()) > 0.85
    true_ml = float(truth.sum()) * SP ** 3 / 1000
    stored_ml = float((res.masks["seg"] > 0).sum()) / 1000     # 1 mm stored voxels, inflated by the scaling
    assert abs(native.volume_ml(res.masks["seg"], res.info) - true_ml) / true_ml < 0.08
    assert abs(stored_ml - true_ml) / true_ml > 0.15           # why the transform matters: raw counts are off
    store.write(td / "v.zip", "image", res.vol[:, None], res.info)     # lists survive the attribute round trip
    attrs = store.read_attrs(td / "v.zip", "image")
    assert attrs["reg_affine"] == res.info["reg_affine"] and len(attrs["native_direction"]) == 9
    assert attrs["native_source"].endswith("scan.nii.gz")
