"""Binary-mask helpers shared by centering, brain masks, SynthSeg post-processing and mask QC."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def largest_cc(mask: np.ndarray) -> np.ndarray:
    """The largest connected component of ``mask`` (the mask itself when it has at most one)."""
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    return lab == (1 + np.bincount(lab.ravel())[1:].argmax())


def ball(r: int) -> np.ndarray:
    """3D ball structuring element of radius ``r`` voxels."""
    a = np.arange(-r, r + 1)
    return (a[:, None, None] ** 2 + a[None, :, None] ** 2 + a[None, None, :] ** 2) <= r * r


def dice(a: np.ndarray, b: np.ndarray, empty: float = 0.0) -> float:
    """Dice overlap of two binary masks; ``empty`` when both are empty (0 for QC, where an empty mask is a failure;
    1 when two empty masks agree)."""
    a, b = a.astype(bool), b.astype(bool)
    s = int(a.sum()) + int(b.sum())
    return float(2 * (a & b).sum() / s) if s else float(empty)
