"""Anisotropic 3D patch embed plus a log-mm scale conditioner."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


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


class ConvStemEmbed3D(PatchEmbed3D):
    """Two strided convolution stages before the token grid (2026-11-03): each halves every axis (3^3 convolution,
    GroupNorm, GELU, then a residual block), and ``proj`` patchifies the quarter-resolution features into tokens. A
    token then summarises features that already saw their 3^3 neighbourhoods at 1/2 and 1/4 resolution, instead of
    a single linear map of 2048 raw voxels. Same interface as ``PatchEmbed3D``: ``proj`` resizes for other patch
    sizes (FlexiViT, the frozen suite's half patches), which must stay divisible by 4."""

    def __init__(self, in_chans: int, dim: int, patch_xyz: Sequence[int], widths: Sequence[int] = (32, 64),
                 checkpoint: bool = True):
        nn.Module.__init__(self)
        from brain.models.conv3d import Res, conv3

        px, py, pz = (int(v) for v in patch_xyz)
        if px % 4 or py % 4 or pz % 4:
            raise ValueError(f"patch {tuple(patch_xyz)}: a two-stage stem needs every side divisible by 4")
        self.patch_xyz = (px, py, pz)
        w1, w2 = (int(w) for w in widths)
        self.stem = nn.Sequential(conv3(in_chans, w1, stride=2), Res(w1, w1), conv3(w1, w2, stride=2), Res(w2, w2))
        self.proj = nn.Conv3d(w2, dim, kernel_size=(pz // 4, py // 4, px // 4), stride=(pz // 4, py // 4, px // 4))
        self.checkpoint = bool(checkpoint)

    def forward(self, x: torch.Tensor, patch_xyz: Sequence[int] | None = None, zyx: bool = False):
        if x.ndim != 5:
            raise ValueError(f"expected 5 dims, got {tuple(x.shape)}")
        px, py, pz = self.patch_xyz if patch_xyz is None else tuple(int(v) for v in patch_xyz)
        if px % 4 or py % 4 or pz % 4:
            raise ValueError(f"patch {(px, py, pz)}: the stem needs every side divisible by 4")
        vol = x if zyx else x.permute(0, 1, 4, 3, 2)
        if self.checkpoint and self.training:        # half-resolution activations: recomputed in backward
            feats = checkpoint(self.stem, vol, use_reentrant=False)
        else:
            feats = self.stem(vol)
        k = (pz // 4, py // 4, px // 4)
        w = self.proj.weight if (px, py, pz) == self.patch_xyz else self._weight_for(k)
        y = torch.nn.functional.conv3d(feats, w, self.proj.bias, stride=k)
        nz, ny, nx = y.shape[2:]
        return y.flatten(2).transpose(1, 2), (nz, ny, nx)


class NgeoStemEmbed(nn.Module):
    """Token embedding for native-geometry CT (``<case>.ngeo.zip``, 2026-11-07): slices along the acquisition's slice
    normal at their native spacing, 0.5 mm in-plane. Everything is in-plane: two stride-(1, 2, 2) stages of (1, 3, 3)
    convolutions (0.5 -> 1 -> 2 mm), each with a residual block, then ``proj`` patchifies 1 x 8 x 8 (slice x row x
    col) at 2 mm, so a token is 16 x 16 mm in-plane and one native slice deep; nothing mixes slices before the
    transformer (no through-plane interpolation of a 5 mm acquisition). ``patch_xyz`` is in input voxels (x = col,
    y = row, z = slice), like ``views.patch_xyz``: (32, 32, 1) by default; x and y must be divisible by 4. Public
    tensors are (B, C, X, Y, Z) like the other embeddings, or (B, C, Z, Y, X) = (B, C, S, R, C) with ``zyx``."""

    def __init__(self, in_chans: int, dim: int, patch_xyz: Sequence[int] = (32, 32, 1),
                 widths: Sequence[int] = (32, 64), checkpoint: bool = True):
        super().__init__()
        from brain.models.conv3d import Res, conv3

        px, py, pz = (int(v) for v in patch_xyz)
        if px % 4 or py % 4 or pz < 1:
            raise ValueError(f"ngeo patch {tuple(patch_xyz)}: in-plane sides must be divisible by 4 (two stride-2 stages)")
        self.patch_xyz = (px, py, pz)
        w1, w2 = (int(w) for w in widths)
        k, s = (1, 3, 3), (1, 2, 2)
        self.stem = nn.Sequential(conv3(in_chans, w1, stride=s, kernel=k), Res(w1, w1, kernel=k),
                                  conv3(w1, w2, stride=s, kernel=k), Res(w2, w2, kernel=k))
        self.proj = nn.Conv3d(w2, dim, kernel_size=(pz, py // 4, px // 4), stride=(pz, py // 4, px // 4))
        self.checkpoint = bool(checkpoint)

    def forward(self, x: torch.Tensor, patch_xyz: Sequence[int] | None = None, zyx: bool = False):
        if x.ndim != 5:
            raise ValueError(f"expected 5 dims, got {tuple(x.shape)}")
        if patch_xyz is not None and tuple(int(v) for v in patch_xyz) != self.patch_xyz:
            raise ValueError(f"the ngeo stem has a fixed patch {self.patch_xyz}, got {tuple(patch_xyz)}")
        vol = x if zyx else x.permute(0, 1, 4, 3, 2)
        b, c, s = vol.shape[:3]
        pz = self.patch_xyz[2]
        if s % pz:
            raise ValueError(f"{s} slices are not divisible by the ngeo patch depth {pz}")
        # slabs of pz slices (one slice by default) as separate rows: GroupNorm statistics stay within a slab, so a
        # token never depends on other slices, nor on how many padded slices its batch carries
        rows = vol.reshape(b, c, s // pz, pz, *vol.shape[3:]).transpose(1, 2).reshape(b * (s // pz), c, pz,
                                                                                      *vol.shape[3:])
        if self.checkpoint and self.training:        # full in-plane resolution activations: recomputed in backward
            feats = checkpoint(self.stem, rows, use_reentrant=False)
        else:
            feats = self.stem(rows)
        y = self.proj(feats)                                              # (B * S / pz, D, 1, ny, nx)
        y = y.reshape(b, s // pz, *y.shape[1:]).squeeze(3).transpose(1, 2)   # (B, D, nz, ny, nx)
        nz, ny, nx = y.shape[2:]
        return y.flatten(2).transpose(1, 2), (nz, ny, nx)


class ScaleEmbed(nn.Module):
    """Adds a function of log(patch_size_mm) so the net need not infer scale from content."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, patch_size_mm: torch.Tensor) -> torch.Tensor:
        if patch_size_mm.shape[-1] != 3:
            raise ValueError(f"patch_size_mm last dim must be 3, got {tuple(patch_size_mm.shape)}")
        return self.net(torch.log(patch_size_mm.float().clamp_min(1e-3)).to(dtype=self.net[0].weight.dtype))
