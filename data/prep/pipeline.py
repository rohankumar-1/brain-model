"""Per-volume processing: read -> center -> one resample -> native units."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from data.prep import centering, dicom, geometry, intensity
from data.prep.sources import Case

SIZE = (256, 256, 128)  # (x, y, z)
# raw structural scans; OASIS-1 is gain-field corrected, BraTS is preprocessed, DWI/ADC are not bias-field images
N4_DATASETS = {"oasis2", "abide1", "ixi_t1", "ixi_t2", "atlas", "isles22_flair",
               "pixar", "piop1", "piop2", "id1000", "wmh_flair", "wmh_t1"}


def needs_n4(dataset: str) -> bool:
    """FOMO PT folders are raw skull-on MRI (same class as OASIS-2 / ABIDE). BraTS-GEN is not ``fomo_*``."""
    return dataset in N4_DATASETS or str(dataset).startswith("fomo_")
TMP = os.environ.get("SLURM_TMPDIR") or "/tmp"


@dataclass
class Result:
    vol: np.ndarray                                   # (Z, Y, X) float16 native units
    masks: dict[str, np.ndarray]                      # name -> (Z, Y, X) uint8
    info: dict = field(default_factory=dict)          # QC scalars, stored as array attributes
    timing: dict = field(default_factory=dict)        # seconds per stage
    debug: dict = field(default_factory=dict)         # source image and geometry, for QC figures


def _read(case: Case) -> tuple[sitk.Image, dict]:
    if case.members:
        return dicom.read_study(case.zip_path, case.members, TMP)
    if case.path is not None and Path(case.path).is_dir():
        return dicom.read_series_dir(case.path)
    if case.dataset == "msd_braintumor" and case.path is not None:
        from data.prep.sources import ensure_msd_t1w

        return geometry.read(ensure_msd_t1w(Path(case.path))), {}
    return geometry.read(case.path), {}


def process(case: Case, size=SIZE, z_anchor: str = "auto", keep_debug: bool = False) -> Result:
    """``z_anchor``: "volume" centers the slab on the file's z-center; "brain" on the brain's z-extent;
    "auto" is "brain" for CT (neck/torso in the scan must not shift the slab) and "volume" for MRI."""
    ct, t = case.modality == "ct", {}
    if z_anchor == "auto":
        z_anchor = "brain" if ct else "volume"

    @contextmanager
    def stage(name):
        t0 = time.perf_counter()
        yield
        t[name] = time.perf_counter() - t0

    with stage("read"):
        img, info = _read(case)
        info |= {"spacing_mm": "x".join(f"{s:.2f}" for s in img.GetSpacing()), "z_anchor": z_anchor,
                 "tilt_deg": geometry.tilt_deg(img)}
        if ct:
            img, info["offset_repaired"] = intensity.fix_ct_offset(img)
            img = intensity.clamp_ct(img)
    if needs_n4(case.dataset):
        with stage("n4"):
            img = intensity.n4(img)

    with stage("center"):
        # co-registered sequences of one subject must land on the same grid: the reference decides it
        src = geometry.read(case.ref) if case.ref else img
        small = geometry.coarse(src)
        (cx, cy, cz), (z0, z1), cinfo = centering.estimate(sitk.GetArrayFromImage(small), case.modality,
                                                           geometry.COARSE_MM)
        zc = (z0 + z1) / 2 if z_anchor == "brain" else cz
        xyz = np.array(small.TransformContinuousIndexToPhysicalPoint((cx, cy, zc)))
        center = xyz if z_anchor == "brain" else np.append(xyz[:2], geometry.center_z(src))
        grid = geometry.target_grid(img, center, size)
        shift = xyz[:2] - geometry.center_xy(img)
        b = sorted(small.TransformContinuousIndexToPhysicalPoint((cx, cy, z))[2] for z in (z0, z1))
        lo, hi = geometry.z_range(grid)
        cover = max(0.0, min(hi, b[1]) - max(lo, b[0])) / max(b[1] - b[0], 1e-6)

    with stage("resample"):
        raw = geometry.resample(img, grid, default=intensity.FLOOR["ct"] if ct else 0.0)
        masks = {n: geometry.resample(geometry.read(p), grid, 0.0, nearest=True).astype(np.uint8)
                 for n, p in case.masks}
    with stage("intensity"):
        vol, stats = intensity.ct_native(raw) if ct else intensity.mri_native(raw)

    info |= cinfo | stats | {"shift_x_mm": float(shift[0]), "shift_y_mm": float(shift[1]),
                             "shift_mm": float(np.hypot(*shift)), "brain_cover": float(cover),
                             "modality": case.modality}
    reasons = (["fg_odd"] if stats["fg_odd"] else []) + (
        [cinfo["center_method"]] if cinfo["center_method"] in ("head_disagree", "fallback") else [])
    info |= {"qc_flag": bool(reasons), "qc_reason": ",".join(reasons)}
    debug = {"src": img, "center_xyz": xyz, "grid": grid, "old_xy": geometry.center_xy(img)} if keep_debug else {}
    return Result(vol, masks, info, t, debug)
