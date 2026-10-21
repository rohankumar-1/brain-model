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
# brain fraction of the 256 x 256 x 160 mm grid. A full brain is ~0.17; the floor stays low for partial-FOV scans
# (thin CT slabs, limited-coverage DWI), so the template check below is what catches a misplaced mask.
PLAUSIBLE_FRAC = (0.02, 0.30)
# Dice with the registered template brain inside the scan's foreground: correct SynthStrip masks scored 0.92-0.94 on
# IXI / OASIS-2 / CQ500 / BraTS; the misoriented ones 0.66-0.78
TEMPLATE_DICE_MIN = 0.85
MASK_REV = 2  # SynthStrip masks: 2 = input labelled RAS (rev 1 saw a misoriented head)
RAS_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)  # ITK (LPS-world) direction of the RAS store grid


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


def ras_image(a: np.ndarray):
    """(Z, Y, X) store array -> 1 mm SimpleITK image with the store's RAS direction (identity would mean LPS)."""
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(np.asarray(a, dtype=np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetDirection(RAS_DIRECTION)
    return img


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    return float(2 * (a & b).sum() / max(int(a.sum()) + int(b.sum()), 1))


def qc(mask: np.ndarray, prior: np.ndarray | None = None, fg: np.ndarray | None = None) -> dict:
    """Plausibility of a brain mask. ``prior`` is the template brain on the same (registered) grid; ``fg`` the scan's
    foreground, which limits the prior to what the scan covers. Without a prior only the fraction band applies."""
    frac = float(mask.mean())
    out = {"brain_frac": frac}
    ok = PLAUSIBLE_FRAC[0] <= frac <= PLAUSIBLE_FRAC[1]
    if prior is not None:
        p = prior.astype(bool) if fg is None else prior.astype(bool) & fg.astype(bool)
        if p.any():
            out["brain_dice_tpl"] = dice(mask, p)
            ok = ok and out["brain_dice_tpl"] >= TEMPLATE_DICE_MIN
    out["brain_ok"] = bool(ok)
    return out


def synthstrip(raw: np.ndarray, modality: str, cmd: str | None = None, threads: int = 1) -> np.ndarray:
    """Run SynthStrip on the resampled (Z, Y, X) grid volume (1 mm isotropic). CT is passed windowed.

    Prefers an in-process session (model loaded once). Falls back to the CLI for a one-off call
    when the runtime cannot start (missing weights / CUDA).
    """
    inprocess = os.environ.get("SYNTHSTRIP_INPROCESS", "1").strip().lower() not in {"0", "false", "no"}
    if inprocess:
        try:
            from brain.prep.synthstrip_runtime import session
            return session().run(raw, modality)
        except (BrainMaskUnavailable, ImportError, OSError):
            pass
    exe = cmd or os.environ.get("BRAIN_SYNTHSTRIP") or shutil.which("mri_synthstrip")
    if not exe:
        raise BrainMaskUnavailable("mri_synthstrip not found (set BRAIN_SYNTHSTRIP or put it on PATH)")
    import SimpleITK as sitk

    a = np.clip(raw, -200.0, 300.0) if modality == "ct" else raw
    img = ras_image(a)
    with tempfile.TemporaryDirectory(dir=os.environ.get("SLURM_TMPDIR") or None) as td:
        src, out = Path(td) / "in.nii.gz", Path(td) / "mask.nii.gz"
        sitk.WriteImage(img, str(src))
        cmd_line = [exe, "-i", str(src), "-m", str(out), "-t", str(threads)]
        if os.environ.get("SYNTHSTRIP_GPU", "").strip().lower() in {"1", "true", "yes"}:
            cmd_line.append("-g")
        if os.environ.get("SYNTHSTRIP_NOCSF", "").strip().lower() in {"1", "true", "yes"}:
            cmd_line.append("--no-csf")
        subprocess.run(cmd_line, check=True, capture_output=True, text=True, timeout=900)
        return sitk.GetArrayFromImage(sitk.ReadImage(str(out))) > 0


def compute(raw: np.ndarray, native: np.ndarray, modality: str, method: str = "auto", stripped: bool = False,
            floor: float = -6.0, prior: np.ndarray | None = None,
            fg: np.ndarray | None = None, synth: np.ndarray | None = None) -> tuple[np.ndarray | None, dict]:
    """(mask uint8 or None, info). ``raw`` is the resampled volume before normalisation; ``native`` is stored units.

    ``prior`` / ``fg`` feed :func:`qc`. Under ``auto``, a SynthStrip mask that fails the template check is compared
    with the classical one and the better-agreeing mask is kept. ``synth`` is a SynthStrip mask already computed
    elsewhere (the GPU batch in ``brain.prep.run``); it is used instead of running SynthStrip here.
    """
    if method not in METHODS:
        raise ValueError(f"brain mask method must be one of {METHODS}, got {method!r}")
    if method == "none":
        return None, {"brain_method": "none"}

    def classical():
        return classical_ct(native.astype(np.float32)) if modality == "ct" \
            else classical_mri(native.astype(np.float32), floor, stripped)

    used, mask = method, None
    if synth is not None and method in ("auto", "synthstrip"):
        mask, used = synth.astype(bool), "synthstrip"
    elif method in ("auto", "synthstrip"):
        try:
            mask, used = synthstrip(raw, modality), "synthstrip"
        except (BrainMaskUnavailable, subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            if method == "synthstrip":
                raise
    if mask is None:
        used, mask = "classical", classical()
    info = qc(mask, prior, fg)
    if method == "auto" and used == "synthstrip" and not info["brain_ok"] and "brain_dice_tpl" in info:
        alt = classical()
        alt_info = qc(alt, prior, fg)
        if alt_info.get("brain_dice_tpl", 0.0) > info["brain_dice_tpl"]:
            used, mask, info = "classical", alt, alt_info | {"brain_fallback": "synthstrip_failed_qc"}
    rev = {"brain_rev": MASK_REV} if used == "synthstrip" else {}
    return mask.astype(np.uint8), {"brain_method": used, **rev, **info, "stripped_input": bool(stripped)}
