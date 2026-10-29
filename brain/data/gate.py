"""Foreground gate on the native grid.

A token is kept when the voxel at its center passes a threshold, the same kind of test the 2D loader
uses. CT keeps everything above the air floor, including bone. MRI keeps everything above the background
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
) -> torch.Tensor:
    """Foreground tokens as (B, Z, Y, X) from a (B, 1, X, Y, Z) volume. Only ``ct_hu[0]`` is used.

    ``mode="center"`` (original) tests each patch's centre voxel, so edge tokens holding cortex around an air centre are
    dropped; ``mode="max"`` keeps a token when any of its voxels passes the threshold."""
    if volume.ndim != 5:
        raise ValueError(f"expected (B,1,X,Y,Z), got {tuple(volume.shape)}")
    px, py, pz = (int(v) for v in patch_xyz)
    if mode == "max":
        b, _, x, y, z = volume.shape
        v = volume[:, :, : x // px * px, : y // py * py, : z // pz * pz]
        centers = torch.nn.functional.max_pool3d(v.float(), (px, py, pz))[:, 0]
    elif mode == "center":
        centers = volume[:, 0, px // 2 :: px, py // 2 :: py, pz // 2 :: pz]
    else:
        raise ValueError(f"gate mode {mode!r}: center | max")
    is_ct = (modality.reshape(-1) == 0).view(-1, 1, 1, 1)
    keep = torch.where(is_ct, centers > float(ct_hu[0]), centers > float(mri_lo))
    return keep.permute(0, 3, 2, 1).contiguous()
