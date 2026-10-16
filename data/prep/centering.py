"""Volume-level head center from a coarse isotropic copy.

Primary: the intracranial blob (CT HU 0..100, opened so tables, straps, wires, FOV rings and noise
disconnect). Cross-check: the core of the head's distance transform (thin attachments have small
radii). One (x, y) per volume: per-slice estimates on the cranial-vault slices, then the median.
The same blob gives the brain's z-extent, used to anchor the slab and to measure its coverage.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from data.prep.intensity import largest_cc, mri_mask

BRAIN_OPEN_MM, HEAD_OPEN_MM = 8.0, 6.0
VAULT_FRAC = 0.6       # brain blob: keep slices with at least this fraction of the max area
EXTENT_FRAC = 0.05     # brain blob: z-extent = slices with at least this fraction of the max area
HEAD_R_FRAC = 0.85     # head: keep slices whose inscribed radius is this close to the max
CORE_FRAC = 0.7        # head: core = pixels with distance >= this fraction of the slice max


def _ball(r: int) -> np.ndarray:
    a = np.arange(-r, r + 1)
    return (a[:, None, None] ** 2 + a[None, :, None] ** 2 + a[None, None, :] ** 2) <= r * r


def _opened(mask: np.ndarray, mm: float, open_mm: float) -> np.ndarray:
    return largest_cc(ndi.binary_opening(mask, _ball(max(1, round(open_mm / mm)))))


def _vault(blob: np.ndarray):
    """(cx, cy, cz, n_slices, z0, z1) from blob centroids on the largest-area slices, or None."""
    area = blob.sum((1, 2))
    zs = np.flatnonzero(area >= VAULT_FRAC * area.max()) if area.max() else []
    if len(zs) < 3:
        return None
    cy, cx = np.array([ndi.center_of_mass(blob[z]) for z in zs]).T
    ext = np.flatnonzero(area >= EXTENT_FRAC * area.max())
    return float(np.median(cx)), float(np.median(cy)), float(np.median(zs)), len(zs), int(ext[0]), int(ext[-1])


def _head(mask: np.ndarray, mm: float):
    """Same tuple from the head's distance-transform core; z-extent is the widest (cranial) slices."""
    m = _opened(mask, mm, HEAD_OPEN_MM)
    dt = [ndi.distance_transform_edt(s) for s in m]
    rmax = np.array([d.max() for d in dt])
    zs = np.flatnonzero(rmax >= HEAD_R_FRAC * rmax.max()) if rmax.max() else []
    if len(zs) < 3:
        return None
    cy, cx = np.array([ndi.center_of_mass(dt[z] >= CORE_FRAC * rmax[z]) for z in zs]).T
    return float(np.median(cx)), float(np.median(cy)), float(np.median(zs)), len(zs), int(zs[0]), int(zs[-1])


def estimate(vol: np.ndarray, modality: str, mm: float, agree_mm: float = 15.0):
    """``vol`` is (Z, Y, X), isotropic ``mm``. Returns ((cx, cy, cz), (z0, z1), info), all in voxels."""
    if modality == "ct":
        head = _head(vol > -300, mm)
        brain = _vault(_opened((vol >= 0) & (vol <= 100), mm, BRAIN_OPEN_MM))
    else:
        head, brain = _head(mri_mask(vol), mm), None
    gap = float(np.hypot(brain[0] - head[0], brain[1] - head[1]) * mm) if brain and head else None
    if brain and (head is None or gap <= agree_mm):
        method, c = "brain", brain
    elif head:
        method, c = ("head" if brain is None else "head_disagree"), head
    else:
        n = vol.shape
        method, c = "fallback", ((n[2] - 1) / 2, (n[1] - 1) / 2, (n[0] - 1) / 2, 0, 0, n[0] - 1)
    return c[:3], c[4:6], {"center_method": method, "center_gap_mm": gap, "center_slices": c[3]}
