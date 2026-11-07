"""3D ResNet encoder (2026-11-02) for from-scratch single-task baselines: a convolutional counterpart of
``vit3d.BrainEncoder`` that the multitask model (``brain.multitask.model.BrainQuery``, slot readout) can sit on
unchanged.

Basic residual blocks (two 3^3 convolutions, group norm, ReLU), four stages; the strides are anisotropic so the last
stage lands on the ViT's token grid: stem (1, 2, 2) and max-pool 2 halve y / x, then z / y / x, stages 2 and 3 halve all
three, stage 4 keeps its resolution: z / 8, y / 16, x / 16, the (16, 16, 8) patch grid (20 x 16 x 16 tokens on the
160 x 256 x 256 store grid). The last stage's features come back as ``tokens`` (B, N, D) in the encoder's (z, y, x)
order, so the slot readout, the segmentation decoder (no taps) and the token maps work as on the ViT.

Widths (64, 128, 240, 384) with (2, 2, 2, 2) blocks give 22.07M parameters, against the ViT-S encoder's 22.23M
(without its SSL projector heads), and the same 384-wide tokens, so both feed identical heads (``param_count``). Group norm, not batch norm: 8 whole volumes per GPU and DDP without synchronised statistics.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _norm(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(16, c // 4), c)


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: tuple[int, int, int]):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.n1 = _norm(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.n2 = _norm(cout)
        nn.init.zeros_(self.n2.weight)                   # each block starts as the identity (Goyal et al. 2017)
        self.skip = None
        if cin != cout or any(s != 1 for s in stride):
            self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride=stride, bias=False), _norm(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n2(self.c2(F.relu(self.n1(self.c1(x)))))
        return F.relu(h + (x if self.skip is None else self.skip(x)))


class ResNetEncoder(nn.Module):
    """``forward`` takes and returns what ``BrainEncoder.forward`` does (``view`` (B, C, X, Y, Z), coordinates, ``keep``
    ...); the arguments only a ViT uses (``keep``, ``meta``, task tokens, refinement) are accepted and ignored: every
    voxel is convolved, and the readout masks background tokens with ``keep`` itself."""

    arch = "resnet3d"

    def __init__(self, widths: Sequence[int] = (64, 128, 240, 384), blocks: Sequence[int] = (2, 2, 2, 2),
                 in_chans: int = 1, patch_xyz: Sequence[int] = (16, 16, 8), checkpoint_blocks: bool = True):
        super().__init__()
        px, py, pz = (int(v) for v in patch_xyz)
        if px != py or px % 4 or pz % 2 or (px // 4) & (px // 4 - 1) or (pz // 2) & (pz // 2 - 1):
            raise ValueError(f"patch {tuple(patch_xyz)}: x = y, a power of 2 >= 4, and z a power of 2 >= 2")
        self.widths, self.n_blocks = tuple(int(w) for w in widths), tuple(int(b) for b in blocks)
        self.patch_xyz = (px, py, pz)
        self.in_chans = int(in_chans)
        self.embed_dim = self.widths[-1]
        self.depth = 0                 # no transformer blocks: no taps, no layer-wise LR decay
        self.size = "resnet3d"
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.stem = nn.Sequential(nn.Conv3d(in_chans, self.widths[0], (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3),
                                            bias=False), _norm(self.widths[0]), nn.ReLU(inplace=True))
        # after the stem and max-pool (z / 2, y / 4, x / 4), stages 2-4 halve each axis until it reaches the patch
        rz, rxy, strides = pz // 2, px // 4, [(1, 1, 1)]
        for _ in range(len(self.widths) - 1):
            sz, sxy = (2 if rz > 1 else 1), (2 if rxy > 1 else 1)
            rz, rxy = rz // sz, rxy // sxy
            strides.append((sz, sxy, sxy))
        if rz > 1 or rxy > 1:
            raise ValueError(f"patch {tuple(patch_xyz)} needs more stages than {len(self.widths)}")
        stages, cin = [], self.widths[0]
        for w, n, s in zip(self.widths, self.n_blocks, strides):
            stages.append(nn.Sequential(*[Block(cin if i == 0 else w, w, s if i == 0 else (1, 1, 1)) for i in range(n)]))
            cin = w
        self.stages = nn.ModuleList(stages)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(self.stem(x), 3, stride=2, padding=1)

    def forward(self, view: torch.Tensor, coords_mm: torch.Tensor | None = None,
                patch_size_mm: torch.Tensor | None = None, patch_xyz=None, keep: torch.Tensor | None = None,
                taps: Sequence[int] = (), meta=None, task_tokens=None, task_write=None) -> dict:
        x = view.permute(0, 1, 4, 3, 2)                                   # (B, C, Z, Y, X)
        # the full-resolution stem holds the largest activations: recomputed in backward too (GroupNorm: exact)
        x = (checkpoint(self._stem, x, use_reentrant=False) if self.checkpoint_blocks and self.training
             else self._stem(x))
        for stage in self.stages:
            for blk in stage:
                x = checkpoint(blk, x, use_reentrant=False) if self.checkpoint_blocks and self.training else blk(x)
        b, d, nz, ny, nx = x.shape
        tokens = x.flatten(2).transpose(1, 2)                            # (B, N, D), (z, y, x) order
        if keep is not None:
            w = keep.unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * w).sum(1) / w.sum(1).clamp_min(1)
        else:
            pooled = tokens.mean(1)
        return {"tokens": tokens, "taps": [], "cls": None, "registers": None, "task_tokens": None,
                "pooled": pooled, "patch": None, "coords_mm": coords_mm,
                "grid_zyx": torch.tensor((nz, ny, nx), device=view.device)}
