"""Rigid -> affine registration of every volume to one head template.

After this step a given anatomical point sits at (nearly) the same voxel in every scan, whatever the modality,
sequence, head size or pose, which is the "similar space" the encoder's positional coding assumes. It replaces
the head-centering-only placement of earlier store versions; centering still supplies the starting translation.

Mattes mutual information is used because it needs no intensity relationship between the scan and the template,
so CT and every MRI sequence register to the same T1 template. The template is a skull-on T1 (for example MNI152
1 mm). Resampling happens once, from the source image through the fitted transform onto a fixed 256 x 256 x 160 mm grid
in template space.

Failure handling: a fit is accepted only if the affine stays near rigid-plus-scale (singular values in
[0.65, 1.5], rotation <= 35 deg) and improves the metric over the starting pose. Otherwise the rigid fit, then the
bare centering translation, is used and ``reg_ok`` is False. Nothing is dropped: the manifest records what happened.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from brain.prep import geometry

MODES = ("affine", "rigid", "none")
WORK_MM = 2.0
SCALE_RANGE = (0.65, 1.5)
MAX_ROT_DEG = 35.0


@dataclass(frozen=True)
class Template:
    image: sitk.Image            # RAS float32 at WORK_MM, skull-on T1
    full: sitk.Image             # RAS float32 at native template resolution (sets the output grid's frame)
    center_mm: tuple[float, float, float]  # physical point that becomes the center of the output grid
    brain: sitk.Image | None     # optional brain mask in the same frame as ``full``
    path: str
    image_brain: sitk.Image | None = None  # ``image`` cut to the brain mask: the fixed image for skull-stripped scans


@dataclass
class RegResult:
    transform: sitk.Transform    # maps template (output grid) physical points to scan physical points
    info: dict


def default_template_path() -> Path | None:
    p = os.environ.get("BRAIN_TEMPLATE")
    return Path(p) if p else None


@functools.lru_cache(maxsize=2)
def load_template(path: str, brain_mask: str | None = None) -> Template:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"registration template not found: {p}. Pass --template (or set BRAIN_TEMPLATE) to a skull-on T1 such as "
            "MNI152_T1_1mm.nii.gz, or run with --registration none to keep head-centering only.")
    full = geometry.read(p)
    work = geometry.coarse(full, WORK_MM)
    mask = geometry.read(Path(brain_mask)) if brain_mask else None
    if mask is not None:
        a = sitk.GetArrayViewFromImage(mask) > 0
        zyx = np.argwhere(a).mean(0)
        center = mask.TransformContinuousIndexToPhysicalPoint((float(zyx[2]), float(zyx[1]), float(zyx[0])))
    else:  # centroid of the head foreground
        a = sitk.GetArrayViewFromImage(work)
        zyx = np.argwhere(a > a.mean()).mean(0)
        center = work.TransformContinuousIndexToPhysicalPoint((float(zyx[2]), float(zyx[1]), float(zyx[0])))
    return Template(work, full, tuple(float(c) for c in center), mask, str(p), brain_only(work, mask))


def brain_only(work: sitk.Image, mask: sitk.Image | None) -> sitk.Image | None:
    """The working template multiplied by its brain mask (linear-resampled, so the edge stays soft)."""
    if mask is None:
        return None
    m = sitk.Resample(sitk.Cast(mask > 0, sitk.sitkFloat32), work, sitk.Transform(), sitk.sitkLinear, 0.0)
    return work * m


def _regview(img: sitk.Image, modality: str) -> sitk.Image:
    """Isotropic WORK_MM copy with an intensity range that keeps MI from being dominated by air or bone."""
    out = geometry.coarse(img, WORK_MM)
    if modality == "ct":
        return sitk.Clamp(out, sitk.sitkFloat32, -300.0, 700.0)
    a = sitk.GetArrayViewFromImage(out)
    lo, hi = np.percentile(a, [0.5, 99.5])
    return sitk.Clamp(out, sitk.sitkFloat32, float(lo), float(max(hi, lo + 1e-6)))


def _method(fixed, moving, tf, *, levels: int, lr: float, iters: int, seed: int) -> sitk.ImageRegistrationMethod:
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.25, seed)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=lr, minStep=1e-4, numberOfIterations=iters, relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromPhysicalShift()
    shrink = [4, 2, 1][-levels:]
    R.SetShrinkFactorsPerLevel(shrink)
    R.SetSmoothingSigmasPerLevel([2.0 * s / 2 for s in shrink][:-1] + [0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(tf, inPlace=False)
    return R


def _metric_at(R, fixed, moving) -> float | None:
    try:
        return float(R.MetricEvaluate(fixed, moving))
    except Exception:  # older SimpleITK builds lack MetricEvaluate
        return None


def linear_part(tf: sitk.Transform) -> tuple[np.ndarray, np.ndarray]:
    """(A, t) with ``tf(p) = A p + t``, read from the transform's action so it works on any rigid/affine type
    (SimpleITK returns the generic ``Transform`` from ``Execute``, which has no ``GetMatrix``)."""
    t = np.array(tf.TransformPoint((0.0, 0.0, 0.0)))
    A = np.stack([np.array(tf.TransformPoint(tuple(float(v) for v in e))) - t for e in np.eye(3)], axis=1)
    return A, t


def pose_summary(tf: sitk.Transform) -> dict:
    """Rotation angle (deg), geometric-mean scale, and anisotropy (max/min singular value) of a 3D linear map."""
    A, _ = linear_part(tf)
    U, s, Vt = np.linalg.svd(A)
    R = U @ Vt
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))
    return {"rot_deg": angle, "scale": float(np.cbrt(abs(np.linalg.det(A)))),
            "aniso": float(s.max() / max(s.min(), 1e-9)), "smin": float(s.min()), "smax": float(s.max())}


def as_affine(tf: sitk.Transform, center) -> sitk.AffineTransform:
    """The same map as an AffineTransform whose parameters are expressed about ``center`` (better conditioned)."""
    A, _ = linear_part(tf)
    c = np.asarray(center, float)
    aff = sitk.AffineTransform(3)
    aff.SetCenter(tuple(float(v) for v in c))
    aff.SetMatrix(tuple(float(v) for v in A.ravel()))
    aff.SetTranslation(tuple(float(v) for v in (np.array(tf.TransformPoint(tuple(float(v) for v in c))) - c)))
    return aff


def _sane(tf: sitk.Transform, scale_range: tuple[float, float] = SCALE_RANGE) -> bool:
    p = pose_summary(tf)
    return scale_range[0] <= p["smin"] and p["smax"] <= scale_range[1] and p["rot_deg"] <= MAX_ROT_DEG


def register(moving: sitk.Image, template: Template, modality: str, init_center_mm, mode: str = "affine",
             seed: int = 0, stripped: bool = False, scale_range: tuple[float, float] = SCALE_RANGE) -> RegResult:
    """Fit ``template -> moving``. ``init_center_mm`` is the scan's brain center in physical space.

    A skull-stripped scan is fitted to the brain-only template when the template has a brain mask: against the
    skull-on head its affine is usually rejected, which leaves stripped brains unscaled (larger than skull-on ones).
    """
    if mode not in MODES[:2]:
        raise ValueError(f"registration mode must be one of {MODES[:2]}, got {mode!r}")
    use_brain = stripped and template.image_brain is not None
    fixed, mov = (template.image_brain if use_brain else template.image), _regview(moving, modality)
    shift = np.asarray(init_center_mm, float) - np.asarray(template.center_mm, float)
    start = sitk.Euler3DTransform()
    start.SetCenter(template.center_mm)
    start.SetTranslation(tuple(float(v) for v in shift))
    info: dict = {"reg_mode": mode, "reg_method": "translation", "reg_template": "brain" if use_brain else "head"}
    best: sitk.Transform = start
    try:
        Rr = _method(fixed, mov, start, levels=3, lr=2.0, iters=100, seed=seed)
        mi0 = _metric_at(Rr, fixed, mov)
        rigid = Rr.Execute(fixed, mov)
        mi_r = float(Rr.GetMetricValue())
        info |= {"reg_mi_init": mi0, "reg_mi": mi_r, "reg_iters": int(Rr.GetOptimizerIteration())}
        if _sane(rigid, scale_range) and (mi0 is None or mi_r < mi0):
            best, info["reg_method"] = rigid, "rigid"
            if mode == "affine":
                aff = as_affine(rigid, template.center_mm)
                Ra = _method(fixed, mov, aff, levels=2, lr=1.0, iters=80, seed=seed + 1)
                fit = Ra.Execute(fixed, mov)
                mi_a = float(Ra.GetMetricValue())
                if _sane(fit, scale_range) and mi_a < mi_r:
                    best, info["reg_method"], info["reg_mi"] = fit, "affine", mi_a
    except RuntimeError as e:  # ITK optimizer failure: keep the best transform so far
        info["reg_error"] = str(e)[:120]
    p = pose_summary(best)
    moved = np.array(best.TransformPoint(tuple(template.center_mm))) - np.asarray(init_center_mm, float)
    info |= {"reg_rot_deg": p["rot_deg"], "reg_scale": p["scale"], "reg_aniso": p["aniso"],
             "reg_shift_mm": float(np.linalg.norm(moved)), "reg_ok": info["reg_method"] != "translation"}
    return RegResult(best, info)


# ------------------------------------------------------------------------------------------------ infant templates
# Infants (0-3 years) registered to the adult template fell back to rigid / translation (184 of 1,665) or sat at the
# scale limit: the adult T1 differs in size and, before ~6 months, in contrast. They are registered to the age-matched
# MNI NIHPD Objective-2 asymmetric template (T1w / T2w, 11 bins from birth to 60 months; Fonov et al. 2011) and carried
# into the adult template's grid by a fixed NIHPD -> adult affine per bin, so they land on the same grid as everyone.
NIHPD_BINS = ((0, 2), (2, 5), (5, 8), (8, 11), (11, 14), (14, 17), (17, 21), (21, 27), (27, 33), (33, 44), (44, 60))
TEMPLATE_SCALE_RANGE = (0.4, 1.6)        # a newborn template is ~0.6x the adult one, linearly


def nihpd_dir() -> Path:
    return Path(os.environ.get("BRAIN_INFANT_TEMPLATES", "/home/roku/orcd/scratch/brain-model/templates/nihpd"))


def nihpd_bin(age_months: float) -> str:
    """The NIHPD age bin (``"05-08"``) holding ``age_months``; ages past the last bin use the last."""
    for lo, hi in NIHPD_BINS:
        if age_months < hi:
            return f"{lo:02d}-{hi:02d}"
    lo, hi = NIHPD_BINS[-1]
    return f"{lo:02d}-{hi:02d}"


def infant_template(age_months: float, contrast: str) -> Template:
    b = nihpd_bin(age_months)
    c = "t2w" if contrast == "t2w" else "t1w"
    d = nihpd_dir()
    return load_template(str(d / f"nihpd_asym_{b}_{c}.nii.gz"), str(d / f"nihpd_asym_{b}_mask.nii.gz"))


@functools.lru_cache(maxsize=16)
def nihpd_to_adult(bin_: str, adult_path: str, adult_mask: str | None) -> sitk.AffineTransform:
    """Affine mapping adult-template points to the NIHPD ``bin_`` T1w template's points (fitted once per bin and
    process: the template-to-template registration, scale allowed down to 0.4)."""
    d = nihpd_dir()
    adult = load_template(adult_path, adult_mask)
    infant = geometry.read(d / f"nihpd_asym_{bin_}_t1w.nii.gz")
    inf_tpl = load_template(str(d / f"nihpd_asym_{bin_}_t1w.nii.gz"), str(d / f"nihpd_asym_{bin_}_mask.nii.gz"))
    rr = register(infant, adult, "mri", inf_tpl.center_mm, "affine", scale_range=TEMPLATE_SCALE_RANGE)
    if rr.info["reg_method"] != "affine":
        raise RuntimeError(f"NIHPD {bin_} -> adult template registration fell back to {rr.info['reg_method']}")
    return as_affine(rr.transform, adult.center_mm)


def compose(outer: sitk.Transform, inner: sitk.Transform) -> sitk.AffineTransform:
    """One affine for ``p -> outer(inner(p))`` (both affine)."""
    def mat(tf):
        a, t = linear_part(tf)
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = a, t
        return m

    m = mat(outer) @ mat(inner)
    out = sitk.AffineTransform(3)
    out.SetMatrix(tuple(float(v) for v in m[:3, :3].flatten()))
    out.SetTranslation(tuple(float(v) for v in m[:3, 3]))
    return out


def output_grid(template: Template, size=geometry.GRID_XYZ) -> geometry.Grid:
    """The 1 mm output grid, axis-aligned in template space and centered on the template's brain center."""
    return geometry.target_grid(template.full, np.asarray(template.center_mm), size)


def template_brain_on_grid(template: Template, grid: geometry.Grid) -> np.ndarray | None:
    """The template's brain mask on the output grid (Z, Y, X) bool, or None without a mask."""
    if template.brain is None:
        return None
    return geometry.resample(template.brain, grid, 0.0, nearest=True) > 0.5


def fov_cover(img: sitk.Image, tf: sitk.Transform, template: Template, grid: geometry.Grid) -> float:
    """Fraction of the template's brain (or of the whole grid) that the scan actually covers."""
    ones = sitk.Image(img.GetSize(), sitk.sitkFloat32) + 1.0
    ones.CopyInformation(img)
    valid = geometry.resample(ones, grid, 0.0, nearest=True, tf=tf)
    brain = template_brain_on_grid(template, grid)
    if brain is None:
        return float(valid.mean())
    return float(valid[brain].mean()) if brain.any() else float(valid.mean())
