"""Budgeted token refinement: which coarse cells of a view are encoded as 8 half-size tokens instead of one.

A refined cell's children are embedded with the same patch embedding (its kernel resized, ``PatchEmbed3D``), carry
their own size (``ScaleEmbed``) and their centres in mm (RoPE), and their outputs average back into the cell, so
everything downstream of the encoder (losses, predictor, readouts, caches) keeps the coarse grid
(``BrainEncoder.encode_rows`` / ``forward`` with ``refine``).

The budget is a fraction of a view's kept tokens. Cells are picked by detail (the spread of the encoder input inside
the cell: bone edges, lesion borders, small structures) with a share picked at random, so the encoder also sees fine
tokens where the rule would not put them.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def detail_scores(view: torch.Tensor, patch_xyz: Sequence[int]) -> torch.Tensor:
    """(R, N) standard deviation of the input inside each patch; ``view`` is (R, C, Z, Y, X), tokens in (Z, Y, X)
    order as ``PatchEmbed3D`` emits them (channels averaged)."""
    px, py, pz = (int(v) for v in patch_xyz)
    x = view.float().mean(1, keepdim=True)
    k = (pz, py, px)
    var = (F.avg_pool3d(x * x, k) - F.avg_pool3d(x, k) ** 2).clamp_min(0)
    return var.flatten(1).sqrt()


@torch.no_grad()
def choose_cells(scores: torch.Tensor, keep: torch.Tensor, frac: float | torch.Tensor, random_frac: float = 0.5,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """(R, N) cells to refine: ``frac`` (a float, or one per row) of each row's kept cells, the top ``1 -
    random_frac`` of them by ``scores``, the rest drawn at random from the other kept cells."""
    r = keep.shape[0]
    frac = torch.as_tensor(frac, dtype=torch.float32, device=keep.device).expand(r)
    k = (frac * keep.sum(1).float()).floor().long()
    n_rand = (k.float() * float(random_frac)).round().long()
    n_top = k - n_rand
    neg = torch.finfo(torch.float32).min
    s = torch.where(keep, scores.float(), torch.full_like(scores, neg, dtype=torch.float32))
    rank = s.argsort(1, descending=True).argsort(1)
    top = keep & (rank < n_top[:, None])
    u = torch.rand(keep.shape, device=keep.device, generator=generator)
    u = torch.where(keep & ~top, u, torch.full_like(u, -1.0))
    rank_r = u.argsort(1, descending=True).argsort(1)
    return top | (keep & ~top & (rank_r < n_rand[:, None]))


def child_index(grid_zyx: Sequence[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """For a coarse grid (nz, ny, nx): the (N, 8) indices of each cell's children in the half-size grid (both in
    (Z, Y, X) order, X fastest), and their (8, 3) offsets in (x, y, z) half-steps (each -1 or +1)."""
    nz, ny, nx = (int(v) for v in grid_zyx)
    z, y, x = torch.meshgrid(*(torch.arange(n, device=device) for n in (nz, ny, nx)), indexing="ij")
    bits = torch.tensor([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], device=device)   # (8, z y x)
    fz = 2 * z.reshape(-1, 1) + bits[:, 0]
    fy = 2 * y.reshape(-1, 1) + bits[:, 1]
    fx = 2 * x.reshape(-1, 1) + bits[:, 2]
    idx = (fz * (2 * ny) + fy) * (2 * nx) + fx
    off = (2 * bits.flip(1) - 1).float()                                                             # (8, x y z)
    return idx, off


def axis_signs(coords_mm: torch.Tensor, grid_zyx: Sequence[int]) -> torch.Tensor:
    """(R, 3) +1 / -1: the direction coords move along each grid axis (x, y, z); +1 for an axis of one token."""
    nz, ny, nx = (int(v) for v in grid_zyx)
    c = coords_mm.reshape(coords_mm.shape[0], nz, ny, nx, 3)
    sign = torch.ones(coords_mm.shape[0], 3, device=coords_mm.device)
    if nx > 1:
        sign[:, 0] = torch.sign(c[:, 0, 0, 1, 0] - c[:, 0, 0, 0, 0])
    if ny > 1:
        sign[:, 1] = torch.sign(c[:, 0, 1, 0, 1] - c[:, 0, 0, 0, 1])
    if nz > 1:
        sign[:, 2] = torch.sign(c[:, 1, 0, 0, 2] - c[:, 0, 0, 0, 2])
    return torch.where(sign == 0, torch.ones_like(sign), sign)
