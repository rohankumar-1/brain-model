"""Baselines that need no training: the floor every learned encoder has to beat.

``RegionalStats``: intensity statistics of the brain, as a whole and in coarse blocks. On the store's registered 1 mm
grid, block position is a crude anatomical position, so these features carry brain size, atrophy (CSF fraction), CT
density and gross asymmetry; anything a frozen encoder adds beyond them is learned representation.
"""

from __future__ import annotations

import numpy as np

from brain.downstream.models import ExternalEncoder

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _stats(v: np.ndarray) -> np.ndarray:
    """Count fraction, mean, std and quantiles of the voxel values ``v`` (zeros when empty)."""
    if v.size == 0:
        return np.zeros(3 + len(QUANTILES), np.float32)
    return np.concatenate([[v.size, v.mean(), v.std()], np.quantile(v, QUANTILES)]).astype(np.float32)


class RegionalStats(ExternalEncoder):
    """Whole-brain and per-block intensity statistics inside the stored brain mask.

    ``block_mm`` (default 32) tiles the 256 x 256 x 160 grid into 8 x 8 x 5 blocks; each block contributes the
    fraction of its voxels inside the brain and their mean and std. CT values are clipped to [-100, 200] HU (the
    tissue range) and MRI to [-5, 5]; features are returned as one global vector."""

    needs_brain = True

    def __init__(self, spec: dict, device):
        super().__init__(spec, device)
        self.block = int((spec.get("kwargs") or {}).get("block_mm", 32))
        nx, ny, nz = (-(-256 // self.block), -(-256 // self.block), -(-160 // self.block))
        self.dim = 3 + len(QUANTILES) + 3 * nx * ny * nz

    def encode(self, vol_xyz: np.ndarray, modality: str, brain: np.ndarray | None, attrs: dict) -> dict:
        lo, hi = (-100.0, 200.0) if str(modality).lower().startswith("ct") else (-5.0, 5.0)
        v = np.clip(vol_xyz, lo, hi)
        m = brain if brain is not None else (vol_xyz > (-500.0 if str(modality).lower().startswith("ct") else -5.5))
        whole = _stats(v[m])
        whole[0] /= m.size                                       # brain fraction of the grid
        b = self.block
        feats = [whole]
        for x in range(0, v.shape[0], b):
            for y in range(0, v.shape[1], b):
                for z in range(0, v.shape[2], b):
                    mb, vb = m[x:x + b, y:y + b, z:z + b], v[x:x + b, y:y + b, z:z + b]
                    inside = vb[mb]
                    feats.append(np.asarray([mb.mean(), inside.mean() if inside.size else 0.0,
                                             inside.std() if inside.size else 0.0], np.float32))
        return {"pooled": np.concatenate(feats)}
