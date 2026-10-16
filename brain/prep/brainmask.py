"""Intracranial (brain) masks on the processed 1 mm grid.

The mask serves three purposes downstream: robust MRI intensity statistics taken from parenchyma rather than
scalp fat, the token gate, and *skull dropout* augmentation, which makes skull-on scans (RSNA, CQ500, OASIS)
indistinguishable from skull-stripped ones (BraTS, MSD) so "has a skull" cannot stand in for modality.

Methods, in order of quality:

``synthstrip``  FreeSurfer's SynthStrip CLI (``mri_synthstrip``; contrast- and CT-agnostic). Used when the binary is on
                PATH or named by ``BRAIN_SYNTHSTRIP``. Slow on CPU (minutes per volume), so run it on the cluster.
``classical``   Threshold/morphology for CT; head-mask erosion for skull-on MRI; the foreground itself for scans
                known to be skull-stripped. Approximate by design: good enough for statistics and dropout, not
                for anatomy. Boundary error is a few mm.
``none``        No mask.

``auto`` tries ``synthstrip`` and falls back to ``classical``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

METHODS = ("auto", "synthstrip", "classical", "none")
SKULL_MM = 14.0           # scalp + skull thickness removed from a skull-on MRI head mask
PLAUSIBLE_FRAC = (0.02, 0.30)  # brain fraction of the 256 x 256 x 128 mm grid (a full brain in the slab is ~0.1)


class BrainMaskUnavailable(RuntimeError):
    pass


def _ball(r: int) -> np.ndarray:
    a = np.arange(-r, r + 1)
    return (a[:, None, None] ** 2 + a[None, :, None] ** 2 + a[None, None, :] ** 2) <= r * r


def _largest(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    return mask if n <= 1 else lab == (1 + np.bincount(lab.ravel())[1:].argmax())


def _fill_axial(mask: np.ndarray) -> np.ndarray:
    return np.stack([ndi.binary_fill_holes(s) for s in mask])


def classical_ct(hu: np.ndarray) -> np.ndarray:
    """Soft-tissue window, opened so the scalp/table disconnect across the skull, closed and filled, bone removed."""
    soft = (hu > -5.0) & (hu < 100.0)                     # CSF .. acute blood; excludes air, fat and bone
    core = _largest(ndi.binary_opening(soft, _ball(3)))
    m = _fill_axial(ndi.binary_closing(core, _ball(4)))
    return m & (hu < 300.0)


def classical_mri(vol: np.ndarray, floor: float, stripped: bool, skull_mm: float = SKULL_MM) -> np.ndarray:
    """``vol`` is stored robust-z with background ``floor``. A stripped scan's foreground already is the brain."""
    fg = vol > floor + 0.5
    if stripped:
        return _largest(fg)
    # EDT erosion: keep voxels deeper than the skull+scalp thickness from the head surface
    depth = ndi.distance_transform_edt(_fill_axial(_largest(fg)))
    return _largest(ndi.binary_opening(depth > skull_mm, _ball(2)))


def synthstrip(raw: np.ndarray, modality: str, cmd: str | None = None, threads: int = 1) -> np.ndarray:
    """Run SynthStrip on the resampled (Z, Y, X) grid volume (1 mm isotropic). CT is passed windowed."""
    exe = cmd or os.environ.get("BRAIN_SYNTHSTRIP") or shutil.which("mri_synthstrip")
    if not exe:
        raise BrainMaskUnavailable("mri_synthstrip not found (set BRAIN_SYNTHSTRIP or put it on PATH)")
    import SimpleITK as sitk

    a = np.clip(raw, -200.0, 300.0) if modality == "ct" else raw
    img = sitk.GetImageFromArray(a.astype(np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    with tempfile.TemporaryDirectory(dir=os.environ.get("SLURM_TMPDIR") or None) as td:
        src, out = Path(td) / "in.nii.gz", Path(td) / "mask.nii.gz"
        sitk.WriteImage(img, str(src))
        subprocess.run([exe, "-i", str(src), "-m", str(out), "-t", str(threads)], check=True,
                       capture_output=True, text=True)
        return sitk.GetArrayFromImage(sitk.ReadImage(str(out))) > 0


def compute(raw: np.ndarray, native: np.ndarray, modality: str, method: str = "auto", stripped: bool = False,
            floor: float = -6.0) -> tuple[np.ndarray | None, dict]:
    """(mask uint8 or None, info). ``raw`` is the resampled volume before normalisation; ``native`` is stored units."""
    if method not in METHODS:
        raise ValueError(f"brain mask method must be one of {METHODS}, got {method!r}")
    if method == "none":
        return None, {"brain_method": "none"}
    used, mask = method, None
    if method in ("auto", "synthstrip"):
        try:
            mask, used = synthstrip(raw, modality), "synthstrip"
        except (BrainMaskUnavailable, subprocess.CalledProcessError, FileNotFoundError):
            if method == "synthstrip":
                raise
    if mask is None:
        used = "classical"
        mask = classical_ct(native.astype(np.float32)) if modality == "ct" \
            else classical_mri(native.astype(np.float32), floor, stripped)
    frac = float(mask.mean())
    ok = PLAUSIBLE_FRAC[0] <= frac <= PLAUSIBLE_FRAC[1]
    return mask.astype(np.uint8), {"brain_method": used, "brain_frac": frac, "brain_ok": bool(ok),
                                   "stripped_input": bool(stripped)}
