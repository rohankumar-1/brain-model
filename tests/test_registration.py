"""Registration on phantoms (SimpleITK, CPU). The first test is the slow one; keep tolerances loose."""

from __future__ import annotations

import tempfile
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


def test_process_with_registration_stores_the_head_at_the_template_center():
    with tempfile.TemporaryDirectory() as td:
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
