"""Anisotropic 3D patch embed plus a log-mm scale conditioner."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """Strided Conv3d. Public tensors are (B, C, X, Y, Z); conv runs as (B, C, Z, Y, X)."""

    def __init__(self, in_chans: int, dim: int, patch_xyz: Sequence[int]):
        super().__init__()
        px, py, pz = (int(v) for v in patch_xyz)
        self.patch_xyz = (px, py, pz)
        self.proj = nn.Conv3d(in_chans, dim, kernel_size=(pz, py, px), stride=(pz, py, px))

    def _weight_for(self, patch_zyx: tuple[int, int, int]) -> torch.Tensor:
        w = self.proj.weight
        if patch_zyx == (self.patch_xyz[2], self.patch_xyz[1], self.patch_xyz[0]):
            return w
        # FlexiViT: resize the kernel; scale so a constant patch keeps similar activation magnitude.
        resized = torch.nn.functional.interpolate(w, size=patch_zyx, mode="trilinear", align_corners=False)
        src = float(w.shape[-1] * w.shape[-2] * w.shape[-3])
        dst = float(patch_zyx[0] * patch_zyx[1] * patch_zyx[2])
        return resized * (src / dst)

    def forward(
        self, x: torch.Tensor, patch_xyz: Sequence[int] | None = None, zyx: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """``x`` is (B, C, X, Y, Z), or (B, C, Z, Y, X) when ``zyx`` is set."""
        if x.ndim != 5:
            raise ValueError(f"expected 5 dims, got {tuple(x.shape)}")
        px, py, pz = self.patch_xyz if patch_xyz is None else tuple(int(v) for v in patch_xyz)
        vol = x if zyx else x.permute(0, 1, 4, 3, 2)
        if (px, py, pz) == self.patch_xyz:
            y = self.proj(vol)
        else:
            y = torch.nn.functional.conv3d(
                vol, self._weight_for((pz, py, px)), self.proj.bias, stride=(pz, py, px)
            )
        nz, ny, nx = y.shape[2:]
        tokens = y.flatten(2).transpose(1, 2)
        return tokens, (nz, ny, nx)


class ScaleEmbed(nn.Module):
    """Adds a function of log(patch_size_mm) so the net need not infer scale from content."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, patch_size_mm: torch.Tensor) -> torch.Tensor:
        if patch_size_mm.shape[-1] != 3:
            raise ValueError(f"patch_size_mm last dim must be 3, got {tuple(patch_size_mm.shape)}")
        return self.net(torch.log(patch_size_mm.float().clamp_min(1e-3)).to(dtype=self.net[0].weight.dtype))
