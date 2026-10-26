"""Per-volume processing: read -> (N4) -> center -> (register to template) -> one resample -> native units -> brain mask."""

from __future__ import annotations

import functools
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from brain.prep import brainmask, centering, dicom, geometry, intensity, native, registration
from brain.prep.sources import Case

SIZE = geometry.GRID_XYZ  # (x, y, z)
# raw structural scans; OASIS-1 is distributed gain-field corrected; DWI / ADC (and perfusion maps) are not bias-field
# images. Before 2026-10-25 BraTS, MSD, BrainMetShare, VALDO and MS3SEG were not corrected either (BraTS is
# "preprocessed", but not bias-corrected), so only some tumour cohorts were: they are now (``brain.prep.n4_store``
# corrects the volumes already stored).
LEGACY_N4_DATASETS = {"oasis2", "abide1", "ixi_t1", "ixi_t2", "atlas", "isles22_flair",
                      "pixar", "piop1", "piop2", "id1000", "wmh_flair", "wmh_t1"}
N4_PREFIXES = ("fomo_", "brats", "msd_braintumor", "brainmetshare", "valdo", "ms3seg")
NO_N4_SUFFIXES = ("_dwi", "_adc")


INFANT_FAMILIES = ("fomo_infant",)           # registered through the age-matched NIHPD template (registration.py)


@functools.lru_cache(maxsize=1)
def _infant_ages() -> dict[tuple[str, str], float]:
    """(participant, session) -> age in months for the infant cohort (FOMO participants.tsv)."""
    from brain.downstream.tasks import label_file

    p = pd.read_csv(label_file("fomo260k", "participants.tsv"), sep="\t", low_memory=False)
    p = p[p["dataset"].astype(str).str.startswith("PT023")]
    age = pd.to_numeric(p["age"], errors="coerce")
    return {(str(a), str(b)): float(y) * 12.0 for a, b, y in zip(p["participant_id"], p["session_id"], age)
            if np.isfinite(y)}


def infant_age_months(case) -> float | None:
    """Age in months of an infant-cohort scan (case_id ``sub-001_ses-01_t1w``), or None (not infant / unknown age)."""
    if not str(case.dataset).startswith(INFANT_FAMILIES):
        return None
    parts = str(case.case_id).split("_")
    return _infant_ages().get((parts[0], parts[1] if len(parts) > 1 else ""))


def needs_n4(dataset: str) -> bool:
    """Structural MRI gets N4 (FOMO PT folders are raw skull-on MRI, the same class as OASIS-2 / ABIDE)."""
    ds = str(dataset)
    if ds.endswith(NO_N4_SUFFIXES):
        return False
    return ds in LEGACY_N4_DATASETS or ds.startswith(N4_PREFIXES)
TMP = os.environ.get("SLURM_TMPDIR") or "/tmp"


@dataclass(frozen=True)
class Options:
    """What to do beyond reading and resampling. The CLI defaults to affine registration and an automatic brain mask;
    ``process`` itself defaults to neither, so a bare call reproduces head-centering only."""

    registration: str = "none"              # "affine" | "rigid" | "none"
    template: str | None = None             # skull-on T1 template (MNI152 1 mm)
    template_brain: str | None = None       # optional brain mask in the template frame
    brain_mask: str = "none"                # "auto" | "synthstrip" | "classical" | "none"
    reg_backend: str = "sitk"               # "sitk" (CPU) | "torch" (GPU when available; registration_torch)


def add_cli_options(ap) -> None:
    """Flags shared by ``brain.prep.run`` and ``brain.prep.pilot``. Defaults are the current recipe."""
    ap.add_argument("--registration", choices=list(registration.MODES), default="affine",
                    help="register each scan to the template (affine), keep its size (rigid), or only center it (none)")
    ap.add_argument("--template", default=None, help="skull-on T1 template, e.g. MNI152_T1_1mm.nii.gz (or $BRAIN_TEMPLATE)")
    ap.add_argument("--template-brain", default=None, help="brain mask of the template (sets the grid center)")
    ap.add_argument("--brain-mask", choices=list(brainmask.METHODS), default="auto",
                    help="auto = SynthStrip when installed, else classical")
    ap.add_argument("--reg-backend", choices=["sitk", "torch"], default="sitk",
                    help="registration engine: SimpleITK on the CPU, or the PyTorch port (GPU) of the same fit")


def options_from_args(a) -> Options:
    """Build ``Options`` from parsed flags and fail fast if the template is missing."""
    opts = Options(a.registration, a.template, a.template_brain, a.brain_mask, a.reg_backend)
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


@dataclass
class Partial:
    """A case read, registered and resampled; what is left is the brain mask, intensities and QC (``finish``)."""

    case: Case
    raw: np.ndarray                                   # (Z, Y, X) float32 on the grid, source units
    masks: dict[str, np.ndarray]
    info: dict
    timing: dict
    prior: np.ndarray | None                          # template brain on the grid (registered runs)
    cover: float
    debug: dict = field(default_factory=dict)

    @property
    def needs_synthstrip(self) -> bool:
        return "brain" not in self.masks


def process(case: Case, size=SIZE, z_anchor: str = "auto", keep_debug: bool = False,
            opts: Options | None = None) -> Result:
    """``z_anchor`` (used when not registering): "volume" centers the slab on the file's z-center; "brain" on the
    brain's z-extent; "auto" is "brain" for CT (neck/torso in the scan must not shift the slab) and "volume" for MRI.
    With registration the slab is fixed in template space and ``z_anchor`` is recorded but unused."""
    opts = opts or Options()
    return finish(prepare(case, size, z_anchor, keep_debug, opts), opts)


def prepare(case: Case, size=SIZE, z_anchor: str = "auto", keep_debug: bool = False,
            opts: Options | None = None) -> Partial:
    """Read -> (N4) -> center -> (register) -> one resample. CPU only; see ``process`` for ``z_anchor``."""
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
        info["n4"] = "native"

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

    tf, rinfo, prior = None, {}, None
    if reg:
        with stage("register"):
            tpl = registration.load_template(opts.template or str(registration.default_template_path() or ""),
                                             opts.template_brain)
            # the starting pose uses the brain center; z follows the brain's extent so a long CT starts well
            start = np.array(small.TransformContinuousIndexToPhysicalPoint((cx, cy, (z0 + z1) / 2)))
            age = infant_age_months(case) if opts.registration == "affine" else None
            if age is not None:                  # age-matched infant template, then the fixed template -> adult map
                contrast = "t2w" if case.dataset.endswith("t2w") else "t1w"
                itpl = registration.infant_template(age, contrast)
                rr = registration.register(src, itpl, case.modality, start, opts.registration, stripped=case.stripped)
                b = registration.nihpd_bin(age)
                to_infant = registration.nihpd_to_adult(b, opts.template or str(registration.default_template_path()),
                                                        opts.template_brain)
                rr.transform = registration.compose(rr.transform, to_infant)
                p = registration.pose_summary(rr.transform)
                rr.info |= {"reg_backend": "sitk", "reg_template": f"nihpd_{b}_{contrast}", "infant_age_months": age,
                            "reg_scale_infant": rr.info.get("reg_scale"), "reg_scale": p["scale"],
                            "reg_rot_deg": p["rot_deg"], "reg_aniso": p["aniso"]}
            elif opts.reg_backend == "torch":
                from brain.prep import registration_torch

                rr = registration_torch.register(src, tpl, case.modality, start, opts.registration,
                                                 stripped=case.stripped)
            else:
                rr = registration.register(src, tpl, case.modality, start, opts.registration, stripped=case.stripped)
                rr.info["reg_backend"] = "sitk"
            tf, rinfo = rr.transform, rr.info
            grid = registration.output_grid(tpl, size)
            cover = registration.fov_cover(img, tf, tpl, grid)
            center = np.array(tf.TransformPoint(tuple(tpl.center_mm)))
            prior = registration.template_brain_on_grid(tpl, grid)
    else:
        grid = geometry.target_grid(img, center, size)
        lo, hi = geometry.z_range(grid)
        cover = max(0.0, min(hi, b[1]) - max(lo, b[0])) / max(b[1] - b[0], 1e-6)

    with stage("resample"):
        raw = geometry.resample(img, grid, default=intensity.FLOOR["ct"] if ct else 0.0, tf=tf)
        masks = {n: geometry.resample(geometry.read(p), grid, 0.0, nearest=True, tf=tf).astype(np.uint8)
                 for n, p in case.masks}
    info |= cinfo | rinfo | {"shift_x_mm": float(shift[0]), "shift_y_mm": float(shift[1]),
                             "shift_mm": float(np.hypot(*shift)), "modality": case.modality}
    info |= native.geometry_attrs(img, grid, tf, case)       # map stored arrays back onto the source scan
    debug = {"src": img, "center_xyz": center, "grid": grid, "old_xy": geometry.center_xy(img), "tf": tf} \
        if keep_debug else {}
    return Partial(case, raw, masks, info, t, prior, float(cover), debug)


def finish(p: Partial, opts: Options | None = None, synth: np.ndarray | None = None,
           brain_method: str | None = None) -> Result:
    """Brain mask -> native units -> QC. ``synth`` is a SynthStrip mask computed elsewhere (a GPU batch);
    ``brain_method`` overrides ``opts.brain_mask`` (``classical`` when that batch failed)."""
    opts = opts or Options()
    case, raw, masks, prior, t = p.case, p.raw, p.masks, p.prior, dict(p.timing)
    ct = case.modality == "ct"
    method = brain_method or opts.brain_mask

    @contextmanager
    def stage(name):
        t0 = time.perf_counter()
        yield
        t[name] = t.get(name, 0.0) + time.perf_counter() - t0

    brain = masks.get("brain")                        # a mask shipped with the dataset beats a computed one
    if brain is not None:
        binfo = {"brain_method": "source", "brain_frac": float((brain > 0).mean()), "brain_ok": True}
    elif method != "none" and ct:
        with stage("brainmask"):                      # CT stats are already in HU: mask first, normalise after
            brain, binfo = brainmask.compute(raw, raw, "ct", method, case.stripped, prior=prior,
                                             fg=None if prior is None else intensity.ct_mask(raw), synth=synth)
    else:
        binfo = {"brain_method": "none"}
    with stage("intensity"):
        if ct:
            vol, stats = intensity.ct_native(raw)
        else:
            if brain is None and method != "none":
                # the classical MRI mask reads the head foreground, so build it from a provisional normalisation
                prov, _ = intensity.mri_native(raw)
                with stage("brainmask"):
                    brain, binfo = brainmask.compute(raw, prov.astype(np.float32), "mri", method,
                                                     case.stripped, intensity.MRI_FLOOR, prior=prior,
                                                     fg=None if prior is None else prov > intensity.MRI_LO - 0.5,
                                                     synth=synth)
            use = brain if brain is not None and binfo.get("brain_ok", True) else None
            vol, stats = intensity.mri_native(raw, use)

    info = p.info | stats | binfo | {"brain_cover": p.cover}
    if brain is not None and stats.get("fg_frac"):        # ~1 for skull-stripped input, 0.45-0.7 with a skull
        info["brain_head_ratio"] = float((brain > 0).mean()) / float(stats["fg_frac"])
    cinfo = {"center_method": info.get("center_method")}
    reasons = (["fg_odd"] if stats["fg_odd"] else []) + (
        [cinfo["center_method"]] if cinfo["center_method"] in ("head_disagree", "fallback") else [])
    info |= {"qc_flag": bool(reasons), "qc_reason": ",".join(reasons)}
    return Result(vol, masks, None if brain is None else (brain > 0).astype(np.uint8), info, t, p.debug)
