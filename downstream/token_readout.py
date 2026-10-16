"""Spatial readouts for a native 16×16×16 token grid.

The encoder returns one vector per 16×16×8 mm patch, flattened (Z, Y, X).
A single mean over all 4096 tokens erases which plane the signal came from.
These views keep that structure after a 16-component whitened PCA fit on the
training volumes only:

- plane: mean and max inside each of the 16 axial slabs, shape (N, 16, 32)
- grid: mean inside each cell of a 4×4×4 binning, flattened to (N, 1, 1024)
- global: mean and max over every parenchyma token, shape (N, 1, 32)
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

# Conv flatten is (Z, Y, X) with X fastest. 256/16 × 256/16 × 128/8.
GRID_ZYX = (16, 16, 16)
NATIVE_Z = 128
PCA_DIM = 16
GRID_BINS = 4


def _masked_mean(x: np.ndarray, mask: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    m = mask.astype(np.float32)
    num = (x * m[..., None]).sum(axis=axes)
    den = np.clip(m.sum(axis=axes), 1, None)[..., None]
    return num / den


def _masked_max(x: np.ndarray, mask: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    filled = np.where(mask[..., None], x, -np.inf)
    hi = filled.max(axis=axes)
    return np.where(np.isfinite(hi), hi, 0.0).astype(np.float32)


def _fit_pca(train_x: np.ndarray, train_mask: np.ndarray, k: int, seed: int, max_rows: int = 32768) -> PCA:
    flat = np.asarray(train_x, dtype=np.float32).reshape(-1, train_x.shape[-1])
    keep = np.flatnonzero(np.asarray(train_mask).reshape(-1))
    if keep.size == 0:
        raise ValueError("training mask has no parenchyma tokens")
    rng = np.random.default_rng(seed)
    if keep.size > max_rows:
        keep = rng.choice(keep, size=max_rows, replace=False)
    comps = min(int(k), flat.shape[-1], int(keep.size))
    # Do not whiten. On this encoder the first component is most of the variance;
    # whitening would amplify the trailing noise directions up to that scale.
    pca = PCA(n_components=comps, whiten=False, random_state=seed)
    pca.fit(flat[keep])
    return pca


def _project(pca: PCA, x: np.ndarray) -> np.ndarray:
    n, t, _ = x.shape
    y = pca.transform(np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1]))
    return y.reshape(n, t, -1).astype(np.float32)


def _planes(x: np.ndarray, mask: np.ndarray, grid_zyx: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, nz, ny, nx = x.shape[0], *grid_zyx
    vol = x.reshape(n, nz, ny * nx, -1)
    m = mask.reshape(n, nz, ny * nx)
    mean = _masked_mean(vol, m, axes=(2,))
    hi = _masked_max(vol, m, axes=(2,))
    feat = np.concatenate([mean, hi], axis=-1)
    plane_mask = m.any(axis=2)
    # Slab centers in native z voxels. Slice attention divides by depth - 1.
    centers = (np.arange(nz, dtype=np.float32) + 0.5) * (NATIVE_Z / nz)
    z = np.broadcast_to(centers, (n, nz)).copy()
    return feat, plane_mask, z


def _bins(x: np.ndarray, mask: np.ndarray, grid_zyx: tuple[int, int, int], bins: int) -> np.ndarray:
    n = x.shape[0]
    nz, ny, nx = grid_zyx
    if any(a % bins for a in grid_zyx):
        raise ValueError(f"grid {grid_zyx} is not divisible by {bins}")
    vol = x.reshape(n, bins, nz // bins, bins, ny // bins, bins, nx // bins, -1)
    m = mask.reshape(n, bins, nz // bins, bins, ny // bins, bins, nx // bins)
    # Average the interior of each bin (axes 2, 4, 6).
    mean = _masked_mean(vol, m, axes=(2, 4, 6))
    return mean.reshape(n, 1, -1).astype(np.float32)


def _global(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mean = _masked_mean(x, mask, axes=(1,))
    hi = _masked_max(x, mask, axes=(1,))
    return np.concatenate([mean, hi], axis=-1)[:, None, :].astype(np.float32)


def _pack(src: dict, mus: np.ndarray, mask: np.ndarray, z: np.ndarray | None = None) -> dict:
    out = {"volume_ids": src["volume_ids"], "mus": mus, "mask": mask, "y": src["y"]}
    if z is not None:
        out["z"] = z
    if "cohort" in src:
        out["cohort"] = src["cohort"]
    return out


def token_readout_views(
    packs: dict, *, k: int = PCA_DIM, seed: int = 0, grid_zyx: tuple[int, int, int] = GRID_ZYX,
) -> dict:
    """Fit PCA on train parenchyma tokens and return plane, grid, and global packs."""
    tr = packs["train"]
    n_tok = tr["mus"].shape[1]
    expect = int(np.prod(grid_zyx))
    if n_tok != expect:
        raise ValueError(f"expected {expect} tokens for grid {grid_zyx}, got {n_tok}")
    pca = _fit_pca(tr["mus"], tr["mask"], k, seed)
    views = {"plane": {}, "grid": {}, "global": {}}
    for split, pack in packs.items():
        if len(pack["volume_ids"]) == 0:
            views["plane"][split] = pack
            views["grid"][split] = pack
            views["global"][split] = pack
            continue
        z = _project(pca, pack["mus"])
        m = np.asarray(pack["mask"]).astype(bool)
        planes, plane_mask, plane_z = _planes(z, m, grid_zyx)
        grid = _bins(z, m, grid_zyx, GRID_BINS)
        glob = _global(z, m)
        ones_g = np.ones((len(pack["volume_ids"]), 1), dtype=bool)
        views["plane"][split] = _pack(pack, planes, plane_mask, plane_z)
        views["grid"][split] = _pack(pack, grid, ones_g)
        views["global"][split] = _pack(pack, glob, ones_g)
    views["explained_variance"] = float(pca.explained_variance_ratio_.sum())
    views["pca_dim"] = int(pca.n_components_)
    return views
