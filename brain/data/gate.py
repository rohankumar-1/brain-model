"""Foreground gate on the native grid.

A token is kept when the voxel at its center passes a threshold (``center``), or when at least ``min_frac`` of its
voxels do (``frac``). CT keeps everything above the air floor, including bone. MRI keeps everything above the background
sentinel. There is no connected-component step and no blur.
"""

from __future__ import annotations

from typing import Sequence

import torch


def token_keep(
    volume: torch.Tensor,
    modality: torch.Tensor,
    patch_xyz: Sequence[int],
    *,
    ct_hu: tuple[float, float] = (-500.0, 200.0),
    mri_lo: float = -5.5,
    mode: str = "center",
    min_frac: float = 0.25,
) -> torch.Tensor:
    """Foreground tokens as (B, Z, Y, X) from a (B, 1, X, Y, Z) volume. Only ``ct_hu[0]`` is used.

    ``mode="center"`` (original) tests each patch's centre voxel, so edge tokens holding cortex around an air centre are
    dropped (1.2% of brain voxels on average, 5% on WMH-challenge FLAIRs; 2026-11-01, 85 scans of 7 datasets);
    ``mode="frac"`` keeps it when at least ``min_frac`` of its voxels do: at 0.25, 0.14% of brain voxels lost for 14%
    more tokens than ``center``."""
    if volume.ndim != 5:
        raise ValueError(f"expected (B,1,X,Y,Z), got {tuple(volume.shape)}")
    px, py, pz = (int(v) for v in patch_xyz)
    if mode == "frac":
        b, _, x, y, z = volume.shape
        v = volume[:, :, : x // px * px, : y // py * py, : z // pz * pz].float()
        is_ct = (modality.reshape(-1) == 0).view(-1, 1, 1, 1, 1)
        fg = torch.where(is_ct, v > float(ct_hu[0]), v > float(mri_lo)).float()
        keep = torch.nn.functional.avg_pool3d(fg, (px, py, pz))[:, 0] >= float(min_frac)
        return keep.permute(0, 3, 2, 1).contiguous()
    if mode != "center":
        raise ValueError(f"gate mode {mode!r}: center | frac")
    centers = volume[:, 0, px // 2 :: px, py // 2 :: py, pz // 2 :: pz]
    is_ct = (modality.reshape(-1) == 0).view(-1, 1, 1, 1)
    keep = torch.where(is_ct, centers > float(ct_hu[0]), centers > float(mri_lo))
    return keep.permute(0, 3, 2, 1).contiguous()
