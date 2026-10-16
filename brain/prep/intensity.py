"""Native-unit intensities: CT in Hounsfield units, MRI as a robust z-score. No windowing here."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

CT_MIN, CT_MAX = -1024.0, 3071.0
MRI_FLOOR, MRI_LO, MRI_HI = -6.0, -5.0, 20.0  # background sentinel, tissue clip range
FLOOR = {"ct": CT_MIN, "mri": MRI_FLOOR}


def largest_cc(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    return lab == (1 + np.bincount(lab.ravel())[1:].argmax())


def mri_mask(vol: np.ndarray) -> np.ndarray:
    """Head/brain foreground: smoothed, low Otsu threshold, largest component, holes filled."""
    sm = ndi.gaussian_filter(vol.astype(np.float32), 1.0)
    sample = sm.ravel()[:: max(1, sm.size // 500_000)]
    bg = np.percentile(sample, 1)                      # background level, not assumed to be zero
    return ndi.binary_fill_holes(largest_cc(sm > bg + 0.5 * (threshold_otsu(sample) - bg)))


def ct_mask(vol: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(largest_cc(vol > -500))


def foreground(vol: np.ndarray, modality: str) -> np.ndarray:
    """Foreground of a stored native-unit volume."""
    return ct_mask(vol) if modality == "ct" else vol > MRI_LO - 0.5


def fix_ct_offset(img: sitk.Image) -> tuple[sitk.Image, bool]:
    """Undo a +1024 shift from unsigned conversions (air near 0 instead of -1000)."""
    a = sitk.GetArrayViewFromImage(img)[::2, ::4, ::4]
    no_air = np.percentile(a, 1) > -200              # true HU volumes contain air (about -1000)
    tissue_mode = ((a > 1000) & (a < 1100)).mean()   # soft tissue sits near 1024 + 40 when shifted
    if no_air and tissue_mode > 0.05:
        return img - 1024.0, True
    return img, False


def clamp_ct(img: sitk.Image) -> sitk.Image:
    """Clamp at the source so FOV padding (-2000, -3024) never blends into resampled tissue."""
    return sitk.Clamp(img, sitk.sitkFloat32, CT_MIN, CT_MAX)


def n4(img: sitk.Image, shrink: int = 4) -> sitk.Image:
    """N4 bias-field correction: field estimated on a shrunk copy inside an Otsu mask, applied at full size."""
    mask = sitk.OtsuThreshold(img, 0, 1, 128)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50])
    small, small_mask = (sitk.Shrink(x, [shrink] * 3) for x in (img, mask))
    corrector.Execute(small, small_mask)
    return img / sitk.Exp(corrector.GetLogBiasFieldAsImage(img))


def ct_native(vol: np.ndarray) -> tuple[np.ndarray, dict]:
    vol = np.clip(vol, CT_MIN, CT_MAX)
    soft = vol[(vol > 0) & (vol < 100)]
    air, fg = float(((vol > -1100) & (vol < -900)).mean()), float(ct_mask(vol).mean())
    stats = {
        "air_frac": air,
        "soft_median_hu": float(np.median(soft)) if soft.size else float("nan"),
        "fg_frac": fg,
        "fg_odd": bool(fg < 0.08 or fg > 0.8 or air < 0.05),  # empty grid, blank volume, or no air at all
    }
    return vol.astype(np.float16), stats


def mri_native(vol: np.ndarray, brain: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Robust z (median / 1.4826 MAD), tissue in [-5, 20], background -6.

    Statistics come from ``brain`` when given (parenchyma; scalp fat would otherwise set the scale of a T1), else from
    the head foreground. The clip and floor apply to the head foreground either way, so the skull stays in the volume.
    """
    m = mri_mask(vol)
    ref = brain.astype(bool) & m if brain is not None else m
    x = vol[ref] if ref.any() else vol[m]
    med = float(np.median(x)) if x.size else 0.0
    mad = float(1.4826 * np.median(np.abs(x - med))) if x.size else 0.0
    z = np.clip((vol - med) / max(mad, 1e-6), MRI_LO, MRI_HI)
    z[~m] = MRI_FLOOR
    fg = float(m.mean())
    return z.astype(np.float16), {"fg_frac": fg, "raw_median": med, "raw_mad": mad,
                                  "stats_from": "brain" if brain is not None and ref.any() else "head",
                                  "fg_odd": bool(fg < 0.02 or fg > 0.8 or mad <= 0)}
