"""Physical-coordinate bookkeeping. Axis order is always (x, y, z) in mm.

A voxel index i on an axis has center ``origin + (i + 0.5) * spacing``.
Crops are half-open in voxel indices and in mm. After a resize, token centers
are linear in the crop bounding box, not in the parent voxel grid.
"""

from __future__ import annotations

from typing import Sequence

import torch


def as_vec3(x: Sequence[float] | torch.Tensor, *, device=None, dtype=None) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x.to(device=device or x.device, dtype=dtype or torch.float32)
    else:
        t = torch.as_tensor(x, device=device, dtype=dtype or torch.float32)
    if t.numel() != 3:
        raise ValueError(f"expected 3-vector, got {tuple(t.shape)}")
    return t.reshape(3)


def crop_bbox_mm(
    start_xyz: Sequence[int],
    size_xyz: Sequence[int],
    spacing_mm: Sequence[float] | torch.Tensor,
    origin_mm: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Half-open physical box (x0, y0, z0, x1, y1, z1) of a voxel crop."""
    sp = as_vec3(spacing_mm)
    orig = as_vec3(origin_mm, device=sp.device, dtype=sp.dtype)
    start = as_vec3(start_xyz, device=sp.device, dtype=sp.dtype)
    size = as_vec3(size_xyz, device=sp.device, dtype=sp.dtype)
    lo = orig + start * sp
    hi = orig + (start + size) * sp
    return torch.cat([lo, hi], dim=0)


def token_grid_xyz(out_xyz: Sequence[int], patch_xyz: Sequence[int]) -> tuple[int, int, int]:
    out = tuple(int(v) for v in out_xyz)
    patch = tuple(int(v) for v in patch_xyz)
    if any(o % p for o, p in zip(out, patch)):
        raise ValueError(f"out_xyz {out} is not divisible by patch_xyz {patch}")
    nx, ny, nz = (o // p for o, p in zip(out, patch))
    if min(nx, ny, nz) < 1:
        raise ValueError(f"empty token grid from out={out} patch={patch}")
    return nx, ny, nz


def token_centers_mm(
    bbox_mm: torch.Tensor,
    out_xyz: Sequence[int],
    patch_xyz: Sequence[int],
) -> torch.Tensor:
    """Token-center coordinates (N, 3) in flatten order (Z, Y, X)."""
    nx, ny, nz = token_grid_xyz(out_xyz, patch_xyz)
    bbox = bbox_mm.reshape(6).to(dtype=torch.float32)
    device, dtype = bbox.device, bbox.dtype
    extent = bbox[3:] - bbox[:3]
    n = torch.tensor([nx, ny, nz], dtype=dtype, device=device)
    patch_mm = extent / n
    ix = torch.arange(nx, dtype=dtype, device=device) + 0.5
    iy = torch.arange(ny, dtype=dtype, device=device) + 0.5
    iz = torch.arange(nz, dtype=dtype, device=device) + 0.5
    x = bbox[0] + ix * patch_mm[0]
    y = bbox[1] + iy * patch_mm[1]
    z = bbox[2] + iz * patch_mm[2]
    # Conv3d output is (Z, Y, X); flatten in that order.
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)


def patch_size_mm(bbox_mm: torch.Tensor, out_xyz: Sequence[int], patch_xyz: Sequence[int]) -> torch.Tensor:
    nx, ny, nz = token_grid_xyz(out_xyz, patch_xyz)
    extent = bbox_mm.reshape(6).to(dtype=torch.float32)[3:] - bbox_mm.reshape(6)[:3]
    n = torch.tensor([nx, ny, nz], dtype=extent.dtype, device=extent.device)
    return extent / n


