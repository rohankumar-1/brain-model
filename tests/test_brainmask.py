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
