"""Per-volume processing: read -> (N4) -> center -> (register to template) -> one resample -> native units -> brain mask."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from brain.prep import brainmask, centering, dicom, geometry, intensity, registration
from brain.prep.sources import Case

SIZE = (256, 256, 128)  # (x, y, z)
# raw structural scans; OASIS-1 is gain-field corrected, BraTS is preprocessed, DWI/ADC are not bias-field images
N4_DATASETS = {"oasis2", "abide1", "ixi_t1", "ixi_t2", "atlas", "isles22_flair",
               "pixar", "piop1", "piop2", "id1000", "wmh_flair", "wmh_t1"}


def needs_n4(dataset: str) -> bool:
    """FOMO PT folders are raw skull-on MRI (same class as OASIS-2 / ABIDE). BraTS-GEN is not ``fomo_*``."""
    return dataset in N4_DATASETS or str(dataset).startswith("fomo_")
TMP = os.environ.get("SLURM_TMPDIR") or "/tmp"


@dataclass(frozen=True)
class Options:
    """What to do beyond reading and resampling. The CLI defaults to affine registration and an automatic brain mask;
    ``process`` itself defaults to neither, so a bare call reproduces head-centering only."""

    registration: str = "none"              # "affine" | "rigid" | "none"
    template: str | None = None             # skull-on T1 template (MNI152 1 mm)
    template_brain: str | None = None       # optional brain mask in the template frame
    brain_mask: str = "none"                # "auto" | "synthstrip" | "classical" | "none"


def add_cli_options(ap) -> None:
    """Flags shared by ``brain.prep.run`` and ``brain.prep.pilot``. Defaults are the current recipe."""
    ap.add_argument("--registration", choices=list(registration.MODES), default="affine",
                    help="register each scan to the template (affine), keep its size (rigid), or only center it (none)")
    ap.add_argument("--template", default=None, help="skull-on T1 template, e.g. MNI152_T1_1mm.nii.gz (or $BRAIN_TEMPLATE)")
    ap.add_argument("--template-brain", default=None, help="brain mask of the template (sets the grid center)")
    ap.add_argument("--brain-mask", choices=list(brainmask.METHODS), default="auto",
                    help="auto = SynthStrip when installed, else classical")


def options_from_args(a) -> Options:
    """Build ``Options`` from parsed flags and fail fast if the template is missing."""
    opts = Options(a.registration, a.template, a.template_brain, a.brain_mask)
    if opts.registration != "none":
        registration.load_template(opts.template or str(registration.default_template_path() or ""),
                                   opts.template_brain)
    return opts


@dataclass
class Result:
    vol: np.ndarray                                   # (Z, Y, X) float16 native units
    masks: dict[str, np.ndarray]                      # name -> (Z, Y, X) uint8
    brain: np.ndarray | None = None                   # (Z, Y, X) uint8 brain mask when one was computed or supplied
    info: dict = field(default_factory=dict)          # QC scalars, stored as array attributes
    timing: dict = field(default_factory=dict)        # seconds per stage
    debug: dict = field(default_factory=dict)         # source image and geometry, for QC figures


def _read(case: Case) -> tuple[sitk.Image, dict]:
    if case.members:
        return dicom.read_study(case.zip_path, case.members, TMP)
    if case.path is not None and Path(case.path).is_dir():
        return dicom.read_series_dir(case.path)
    if case.dataset == "msd_braintumor" and case.path is not None:
        from brain.prep.sources import ensure_msd_t1w

        return geometry.read(ensure_msd_t1w(Path(case.path))), {}
    return geometry.read(case.path), {}


def process(case: Case, size=SIZE, z_anchor: str = "auto", keep_debug: bool = False,
            opts: Options | None = None) -> Result:
    """``z_anchor`` (used when not registering): "volume" centers the slab on the file's z-center; "brain" on the
    brain's z-extent; "auto" is "brain" for CT (neck/torso in the scan must not shift the slab) and "volume" for MRI.
    With registration the slab is fixed in template space and ``z_anchor`` is recorded but unused."""
    opts = opts or Options()
    ct, t = case.modality == "ct", {}
    reg = opts.registration != "none"
    if z_anchor == "auto":
        z_anchor = "brain" if ct else "volume"

    @contextmanager
    def stage(name):
        t0 = time.perf_counter()
        yield
        t[name] = time.perf_counter() - t0

    with stage("read"):
        img, info = _read(case)
        info |= geometry.native_info(img) | {
            "spacing_mm": "x".join(f"{s:.2f}" for s in img.GetSpacing()), "z_anchor": z_anchor,
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
        shift = xyz[:2] - geometry.center_xy(img)
        b = sorted(small.TransformContinuousIndexToPhysicalPoint((cx, cy, z))[2] for z in (z0, z1))

    tf, rinfo = None, {}
    if reg:
        with stage("register"):
            tpl = registration.load_template(opts.template or str(registration.default_template_path() or ""),
                                             opts.template_brain)
            # the starting pose uses the brain center; z follows the brain's extent so a long CT starts well
            start = np.array(small.TransformContinuousIndexToPhysicalPoint((cx, cy, (z0 + z1) / 2)))
            rr = registration.register(src, tpl, case.modality, start, opts.registration)
            tf, rinfo = rr.transform, rr.info
            grid = registration.output_grid(tpl, size)
            cover = registration.fov_cover(img, tf, tpl, grid)
            center = np.array(tf.TransformPoint(tuple(tpl.center_mm)))
    else:
        grid = geometry.target_grid(img, center, size)
        lo, hi = geometry.z_range(grid)
        cover = max(0.0, min(hi, b[1]) - max(lo, b[0])) / max(b[1] - b[0], 1e-6)

    with stage("resample"):
        raw = geometry.resample(img, grid, default=intensity.FLOOR["ct"] if ct else 0.0, tf=tf)
        masks = {n: geometry.resample(geometry.read(p), grid, 0.0, nearest=True, tf=tf).astype(np.uint8)
                 for n, p in case.masks}
    brain = masks.get("brain")                        # a mask shipped with the dataset beats a computed one
    if brain is not None:
        binfo = {"brain_method": "source", "brain_frac": float((brain > 0).mean()), "brain_ok": True}
    elif opts.brain_mask != "none" and ct:
        with stage("brainmask"):                      # CT stats are already in HU: mask first, normalise after
            brain, binfo = brainmask.compute(raw, raw, "ct", opts.brain_mask, case.stripped)
    else:
        binfo = {"brain_method": "none"}
    with stage("intensity"):
        if ct:
            vol, stats = intensity.ct_native(raw)
        else:
            if brain is None and opts.brain_mask != "none":
                # the classical MRI mask reads the head foreground, so build it from a provisional normalisation
                prov, _ = intensity.mri_native(raw)
                with stage("brainmask"):
                    brain, binfo = brainmask.compute(raw, prov.astype(np.float32), "mri", opts.brain_mask,
                                                     case.stripped, intensity.MRI_FLOOR)
            use = brain if brain is not None and binfo.get("brain_ok", True) else None
            vol, stats = intensity.mri_native(raw, use)

    info |= cinfo | stats | rinfo | binfo | {"shift_x_mm": float(shift[0]), "shift_y_mm": float(shift[1]),
                                              "shift_mm": float(np.hypot(*shift)), "brain_cover": float(cover),
                                              "modality": case.modality}
    reasons = (["fg_odd"] if stats["fg_odd"] else []) + (
        [cinfo["center_method"]] if cinfo["center_method"] in ("head_disagree", "fallback") else [])
    info |= {"qc_flag": bool(reasons), "qc_reason": ",".join(reasons)}
    debug = {"src": img, "center_xyz": center, "grid": grid, "old_xy": geometry.center_xy(img)} if keep_debug else {}
    return Result(vol, masks, None if brain is None else (brain > 0).astype(np.uint8), info, t, debug)
