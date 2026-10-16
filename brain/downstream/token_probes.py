"""Heads that read every token of the 16x16x16 grid at full width: no PCA, no plane or cell pooling.

- ``AttnProbe``: a few learned queries cross-attend to all tokens (token + position). The usual probe for
  JEPA-style encoders, so it matches how the 2D suite attends over slices.
- ``ConvProbe``: the tokens as a 3D feature map, three small 3D convolutions, then mean and max pooling.
  Keeps spatial layout with few parameters.

Both take ``(x, mask, xyz)`` like ``TokenAttnMLP`` and train through ``AttnEnsemble`` (early stopping, seeds).
"""

from __future__ import annotations

import torch
from torch import nn

from brain.downstream.heads import AttnEnsemble

GRID_ZYX = (16, 16, 16)


def _mlp(dim: int, hidden: int, n_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, n_out))


class AttnProbe(nn.Module):
    def __init__(self, dim: int, n_out: int, width: int = 128, heads: int = 4, queries: int = 4,
                 hidden: int = 256, dropout: float = 0.3, token_drop: float = 0.5):
        super().__init__()
        self.norm, self.proj = nn.LayerNorm(dim), nn.Linear(dim, width)
        self.pos = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.query = nn.Parameter(torch.randn(queries, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.out = _mlp(queries * width, hidden, n_out, dropout)
        self.token_drop = token_drop

    def forward(self, x: torch.Tensor, mask: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.norm(x)) + self.pos(xyz.to(x.dtype))
        keep = mask
        if self.training and self.token_drop > 0:  # random token dropout is the probe's only augmentation
            keep = mask & (torch.rand(mask.shape, device=mask.device) > self.token_drop)
        keep = torch.where(keep.any(1, keepdim=True), keep, mask)
        keep = torch.where(keep.any(1, keepdim=True), keep, torch.ones_like(keep))
        q = self.query.unsqueeze(0).expand(len(x), -1, -1)
        pooled, _ = self.attn(q, h, h, key_padding_mask=~keep, need_weights=False)
        return self.out(pooled.flatten(1))


class ConvProbe(nn.Module):
    def __init__(self, dim: int, n_out: int, grid_zyx: tuple[int, int, int] = GRID_ZYX, width: int = 64,
                 hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.grid = tuple(grid_zyx)
        self.norm, self.proj = nn.LayerNorm(dim), nn.Linear(dim, width)

        def block(i, o, stride):  # no per-sample norm: it would erase volume-level shifts the probe should read
            return [nn.Conv3d(i, o, 3, stride=stride, padding=1), nn.GELU()]

        self.net = nn.Sequential(*block(width, width, 1), *block(width, 2 * width, 2), *block(2 * width, 2 * width, 2))
        self.out = _mlp(4 * width, hidden, n_out, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        if t != self.grid[0] * self.grid[1] * self.grid[2]:
            raise ValueError(f"{t} tokens do not fill the grid {self.grid}")
        h = self.proj(self.norm(x)) * mask.unsqueeze(-1).to(x.dtype)  # air tokens are zeros, like padding
        h = self.net(h.view(b, *self.grid, -1).permute(0, 4, 1, 2, 3))
        h = h.flatten(2)
        return self.out(torch.cat([h.mean(-1), h.amax(-1)], dim=-1))


def make_token_heads(dim: int, n_out: int, multilabel: bool, seed: int, regression: bool,
                     grid_zyx: tuple[int, int, int] = GRID_ZYX) -> list[AttnEnsemble]:
    """3D conv over the token grid. ``AttnProbe`` remains available as ``tok_attn`` if wired back in."""
    grid = tuple(int(v) for v in grid_zyx)
    n_tok = 1
    for s in grid:
        n_tok *= int(s)
    # 32^3 x multi-window dims cannot sit on the GPU; smaller batches keep Conv3d activations in range.
    bs = 32 if n_tok <= 4096 else 8
    h = AttnEnsemble(dim, 0, n_out, multilabel, seed, regression=regression, spatial=True,
                     model_factory=lambda: ConvProbe(dim, n_out, grid_zyx=grid), half=True,
                     batch_size=bs)
    h.name = "tok_conv"
    return [h]
