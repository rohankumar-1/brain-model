"""Read, orient, and resample onto a fixed head-centered 1 mm grid with a single interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sitk.ProcessObject.GlobalWarningDisplayOff()  # ITK's per-file "Analyze deprecated" / "non-uniform sampling" noise

ISO_MM = 1.0
COARSE_MM = 4.0
ROI_PAD = 4  # source voxels of margin around the target box


@dataclass(frozen=True)
class Grid:
    size: tuple[int, int, int]      # (x, y, z)
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    direction: tuple[float, ...]


def standardize(img: sitk.Image) -> sitk.Image:
    """float32 in RAS orientation (same physical space)."""
    return sitk.DICOMOrient(sitk.Cast(img, sitk.sitkFloat32), "RAS")


def tilt_deg(img: sitk.Image) -> float:
    """Largest angle (degrees) between an axis of the (oriented) image and its nearest scanner axis."""
    d = np.abs(np.array(img.GetDirection()).reshape(3, 3)).max(0)
    return float(np.degrees(np.arccos(np.clip(d.min(), 0.0, 1.0))))


def read(path: Path) -> sitk.Image:
    """NIfTI / Analyze -> float32 image in RAS orientation (nibabel fallback for bad direction cosines)."""
    try:
        img = sitk.ReadImage(str(path))
    except RuntimeError:
        import nibabel as nib

        nii = nib.as_closest_canonical(nib.load(str(path)))
        data = np.squeeze(np.asanyarray(nii.dataobj, dtype=np.float32))
        img = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
        img.SetSpacing([float(z) for z in nii.header.get_zooms()[:3]])
        img.SetOrigin([float(v) for v in nii.affine[:3, 3]])
    return standardize(img)


def coarse(img: sitk.Image) -> sitk.Image:
    """~4 mm isotropic bin-averaged copy, cheap enough for centering."""
    small = sitk.BinShrink(img, [max(1, int(COARSE_MM / s)) for s in img.GetSpacing()])
    size = [max(1, int(round(n * s / COARSE_MM))) for n, s in zip(small.GetSize(), small.GetSpacing())]
    fill = float(sitk.GetArrayViewFromImage(small).min())
    return sitk.Resample(small, size, sitk.Transform(), sitk.sitkLinear, small.GetOrigin(),
                         (COARSE_MM,) * 3, small.GetDirection(), fill)


def center_xy(img: sitk.Image) -> np.ndarray:
    """Physical (x, y) of the volume center: what the old center crop used."""
    return np.array(img.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in img.GetSize()])[:2])


def center_z(img: sitk.Image) -> float:
    """Physical z of the volume center."""
    return float(img.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in img.GetSize()])[2])


def target_grid(img: sitk.Image, center: np.ndarray, size=(256, 256, 128)) -> Grid:
    """Axis-aligned 1 mm grid centered at physical ``center`` = (x, y) or (x, y, z).

    Without z the slab is centered on the volume, as the original pipeline did.
    """
    d = np.rint(np.array(img.GetDirection()).reshape(3, 3))  # nearest axis-aligned frame
    z = center[2] if len(center) > 2 else center_z(img)
    half = (np.array(size) - 1) / 2 * ISO_MM
    origin = np.array([center[0], center[1], z]) - d @ half
    return Grid(tuple(size), tuple(origin), (ISO_MM,) * 3, tuple(d.ravel()))


def z_range(g: Grid) -> tuple[float, float]:
    """Physical z extent (lo, hi) of the target slab."""
    z = g.origin[2], g.origin[2] + g.direction[8] * (g.size[2] - 1) * ISO_MM
    return min(z), max(z)


def _roi(img: sitk.Image, g: Grid) -> sitk.Image:
    """Source region covering the target box, so smoothing and interpolation skip the rest."""
    corners = [g.origin + np.array(g.direction).reshape(3, 3) @ (np.array([i, j, k]) * (np.array(g.size) - 1) * ISO_MM)
               for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    idx = np.array([img.TransformPhysicalPointToContinuousIndex(tuple(c)) for c in corners])
    lo = np.clip(np.floor(idx.min(0)).astype(int) - ROI_PAD, 0, None)
    hi = np.minimum(np.ceil(idx.max(0)).astype(int) + ROI_PAD + 1, img.GetSize())
    if np.any(hi <= lo):
        return img
    return sitk.RegionOfInterest(img, [int(v) for v in hi - lo], [int(v) for v in lo])


def resample(img: sitk.Image, g: Grid, default: float, nearest: bool = False) -> np.ndarray:
    """Anti-alias axes finer than the target, then interpolate once onto ``g`` -> (Z, Y, X) float32."""
    roi = _roi(img, g)
    if not nearest:
        var = [0.25 if s < 0.9 * ISO_MM else 0.0 for s in roi.GetSpacing()]  # sigma = 0.5 mm
        if any(var):
            roi = sitk.DiscreteGaussian(roi, var)
    out = sitk.Resample(roi, g.size, sitk.Transform(), sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear,
                        g.origin, g.spacing, g.direction, default, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(out)
