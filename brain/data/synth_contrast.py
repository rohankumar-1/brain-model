"""Synthetic tissue contrast from a SynthSeg label map (2026-11-03; the generator SynthSeg itself trains on, Billot et
al. 2023, reduced to what our store holds): every brain label is painted with a random mean and noise level, left
and right of a structure alike, lightly blurred for partial volume, and the brain re-standardised to the store's MRI
scale (z-scored tissue, median 0). Outside the brain the real scan stays (skull, scalp, the -6 background), so the
synthetic brain sits in a real head. Bias field, slice thickness, noise and blur come afterwards from the usual
appearance augmentation.

Only contrast-invariant targets may be asked of such an image (anatomy masks and volumes, brain mask); the dataset
drops every other drawn task and marks the scan's metadata contrast unknown (``MultitaskDataset``, ``aug.synth_contrast``)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from brain.prep.synthseg import LABELS

# FreeSurfer left / right homologues among the SynthSeg labels: one tissue, one random intensity
LR_PAIRS = ((2, 41), (3, 42), (4, 43), (5, 44), (7, 46), (8, 47), (10, 49), (11, 50), (12, 51), (13, 52), (17, 53),
            (18, 54), (26, 58), (28, 60))


def synth_image(img: np.ndarray, lab: np.ndarray, rng: np.random.Generator, mean=(-2.0, 3.0), std=(0.05, 0.6),
                blur=(0.3, 1.0)) -> np.ndarray:
    """``img`` (Z, Y, X) stored MRI, ``lab`` its SynthSeg mask (indices into ``LABELS``, 0 outside the brain) ->
    the scan with a randomly re-rendered brain. ``mean`` / ``std``: per-label intensity ranges (tissue z units),
    ``blur``: Gaussian sigma range in voxels."""
    n = len(LABELS)
    mu = rng.uniform(*mean, size=n).astype(np.float32)
    sd = rng.uniform(*std, size=n).astype(np.float32)
    pos = {int(f): i for i, f in enumerate(LABELS)}
    for left, right in LR_PAIRS:
        mu[pos[right]], sd[pos[right]] = mu[pos[left]], sd[pos[left]]
    lab = np.asarray(lab).astype(np.intp, copy=False)
    out = mu[lab] + sd[lab] * rng.standard_normal(lab.shape, dtype=np.float32)
    out = gaussian_filter(out, float(rng.uniform(*blur)))
    brain = lab > 0
    if not brain.any():
        return img
    sample = out[brain][:: max(1, int(brain.sum()) // 200_000)]           # robust scale from a subsample
    med, spread = np.median(sample), sample.std()
    out = (out - med) / max(float(spread), 1e-3)
    return np.where(brain, out, img).astype(img.dtype, copy=False)
