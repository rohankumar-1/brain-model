"""Read, orient, and resample onto a fixed head-centered 1 mm grid with a single interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from brain.common.grid import GRID_XYZ

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


def _analyze_orient(path: Path) -> int | None:
    """Analyze 7.5 ``hist.orient`` byte of a .img/.hdr pair (None for NIfTI or anything else)."""
    hdr = Path(path).with_suffix(".hdr")
    if Path(path).suffix != ".img" or not hdr.exists():
        return None
    raw = hdr.read_bytes()
    if len(raw) < 348 or raw[344:347] in (b"ni1", b"n+1"):   # a NIfTI pair carries real orientation
        return None
    return raw[252]


def _read_sagittal_analyze(path: Path) -> sitk.Image:
    """Analyze ``orient = 2`` (sagittal), e.g. raw OASIS-1 MPRAGE (128 x 256 x 256, 1.25 mm slices). ITK ignores the
    orient byte and reads the slice axis as superior-inferior, which no affine registration recovers. Array axes are
    (slice, superior->inferior..., ...): transposing (1, 2, 0) gives superior-up / anterior-up / slice order, checked
    visually and against the MNI template (2026-10-21). Left-right (the slice direction) is not determinable from
    the image; it is taken as stored, so left / right labels of these scans may be mirrored."""
    src = sitk.ReadImage(str(path))
    a = sitk.GetArrayFromImage(src).astype(np.float32)          # (slice, j, i)
    a = a[..., 0] if a.ndim == 4 else a
    sp = src.GetSpacing()                                        # (i, j, slice)
    b = np.ascontiguousarray(np.transpose(a, (1, 2, 0)))        # (S, A, slice) = (z, y, x)
    img = sitk.GetImageFromArray(b)
    img.SetSpacing((float(sp[2]), float(sp[0]), float(sp[1])))   # x = slice, y = i, z = j
    img.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))   # RAS index axes in LPS physical space
    half = [(n - 1) * s / 2 for n, s in zip(img.GetSize(), img.GetSpacing())]
    img.SetOrigin((half[0], half[1], -half[2]))                # centred on the origin
    return img


def read(path: Path) -> sitk.Image:
    """NIfTI / Analyze -> float32 image in RAS orientation (nibabel fallback for bad direction cosines)."""
    if _analyze_orient(path) == 2:
        return standardize(_read_sagittal_analyze(path))
    try:
        img = sitk.ReadImage(str(path))
    except RuntimeError:
        import nibabel as nib

        nii = nib.as_closest_canonical(nib.load(str(path)))
        data = np.squeeze(np.asanyarray(nii.dataobj, dtype=np.float32))
        img = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
        # nibabel's affine is RAS, ITK's geometry LPS: flip x and y, and take direction and spacing from the affine.
        # Before 2026-10-25 the direction was left at ITK's identity (LPS) under RAS-ordered data with a RAS origin,
        # which stored these images rotated 180 degrees about the superior axis (mirrored in x and y).
        a = np.diag([-1.0, -1.0, 1.0]) @ np.asarray(nii.affine, dtype=float)[:3, :4]
        spacing = np.linalg.norm(a[:, :3], axis=0)
        img.SetSpacing([float(s) for s in spacing])
        img.SetDirection([float(v) for v in (a[:, :3] / spacing).flatten()])
        img.SetOrigin([float(v) for v in a[:, 3]])
    return standardize(img)


HEAD_CROP_ABOVE_MM, HEAD_KEEP_MM = 260.0, 240.0


def crop_to_head(img: sitk.Image, above_mm: float = HEAD_CROP_ABOVE_MM,
                 keep_mm: float = HEAD_KEEP_MM) -> tuple[sitk.Image, dict]:
    """A CT longer than ``above_mm`` (head-and-neck CTA from the aortic arch) keeps its top ``keep_mm``: vertex
    through the skull base. Otherwise the neck and shoulders, wider and with more soft tissue than the head, pull the
    centering estimates apart (RSNA-ICA: flagged ``head_disagree`` CTAs had a median 350 mm coverage, the passing ones
    191 mm). The top is the highest slice with tissue (> -300 HU on >= 1% of the widest slice's area) plus 15 mm.
    Physical positions are unchanged (a sub-image), so the stored transform still maps to the source scan."""
    n, sp = img.GetSize(), img.GetSpacing()
    if n[2] * sp[2] <= above_mm:
        return img, {}
    a = sitk.GetArrayViewFromImage(img)
    area = (a[:, ::4, ::4] > -300).mean((1, 2))
    tissue = np.flatnonzero(area >= 0.01 * area.max())
    if not len(tissue):
        return img, {}
    up = np.array(img.GetDirection()).reshape(3, 3)[2, 2] > 0          # slice index runs toward superior
    k, margin = int(round(keep_mm / sp[2])), int(round(15.0 / sp[2]))
    if up:
        hi = min(n[2], int(tissue.max()) + 1 + margin)
        lo = max(0, hi - k)
    else:
        lo = max(0, int(tissue.min()) - margin)
        hi = min(n[2], lo + k)
    return img[:, :, lo:hi], {"head_crop_mm": float(n[2] * sp[2]), "head_crop_slices": [lo, hi]}


def coarse(img: sitk.Image, mm: float = COARSE_MM) -> sitk.Image:
    """Isotropic ``mm`` (default ~4 mm) bin-averaged copy, cheap enough for centering and registration."""
    small = sitk.BinShrink(img, [max(1, int(mm / s)) for s in img.GetSpacing()])
    size = [max(1, int(round(n * s / mm))) for n, s in zip(small.GetSize(), small.GetSpacing())]
    fill = float(sitk.GetArrayViewFromImage(small).min())
    return sitk.Resample(small, size, sitk.Transform(), sitk.sitkLinear, small.GetOrigin(),
                         (mm,) * 3, small.GetDirection(), fill)


def native_info(img: sitk.Image) -> dict:
    """Acquisition geometry of the oriented source, recorded so later stages can reason about native resolution.

    ``thick_axis`` is the grid axis (0 x, 1 y, 2 z) with the coarsest sampling; ``thick_mm`` its spacing. The
    store's grid is axis-aligned RAS, so the axis index carries over to the stored volume.
    """
    sp, size = img.GetSpacing(), img.GetSize()
    ax = int(np.argmax(sp))
    return {"src_dx": float(sp[0]), "src_dy": float(sp[1]), "src_dz": float(sp[2]),
            "src_nx": int(size[0]), "src_ny": int(size[1]), "src_nz": int(size[2]),
            "thick_mm": float(sp[ax]), "thick_axis": ax,
            "aniso": float(max(sp) / max(min(sp), 1e-6))}


def center_xy(img: sitk.Image) -> np.ndarray:
    """Physical (x, y) of the volume center: what the old center crop used."""
    return np.array(img.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in img.GetSize()])[:2])


def center_z(img: sitk.Image) -> float:
    """Physical z of the volume center."""
    return float(img.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in img.GetSize()])[2])


def target_grid(img: sitk.Image, center: np.ndarray, size=GRID_XYZ) -> Grid:
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


def _roi(img: sitk.Image, g: Grid, tf: sitk.Transform | None = None) -> sitk.Image:
    """Source region covering the target box (mapped through ``tf``), so smoothing and interpolation skip the rest."""
    corners = [g.origin + np.array(g.direction).reshape(3, 3) @ (np.array([i, j, k]) * (np.array(g.size) - 1) * ISO_MM)
               for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    if tf is not None:
        corners = [np.array(tf.TransformPoint(tuple(float(v) for v in c))) for c in corners]
    idx = np.array([img.TransformPhysicalPointToContinuousIndex(tuple(c)) for c in corners])
    lo = np.clip(np.floor(idx.min(0)).astype(int) - ROI_PAD, 0, None)
    hi = np.minimum(np.ceil(idx.max(0)).astype(int) + ROI_PAD + 1, img.GetSize())
    if np.any(hi <= lo):
        return img
    return sitk.RegionOfInterest(img, [int(v) for v in hi - lo], [int(v) for v in lo])


def resample(img: sitk.Image, g: Grid, default: float, nearest: bool = False,
             tf: sitk.Transform | None = None) -> np.ndarray:
    """Anti-alias axes finer than the target, then interpolate once onto ``g`` -> (Z, Y, X) float32.

    ``tf`` maps output-grid physical points to ``img`` physical points (a registration result); None is identity.
    """
    roi = _roi(img, g, tf)
    if not nearest:
        var = [0.25 if s < 0.9 * ISO_MM else 0.0 for s in roi.GetSpacing()]  # sigma = 0.5 mm
        if any(var):
            roi = sitk.DiscreteGaussian(roi, var)
    out = sitk.Resample(roi, g.size, tf if tf is not None else sitk.Transform(),
                        sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear,
                        g.origin, g.spacing, g.direction, default, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(out)
