"""The token-grid probe (``tok_conv``): the encoder's tokens as a 3D feature map (``brain.common.grid.TOKEN_GRID``,
20 x 16 x 16), at full width (no PCA or pooling first).

``ConvProbe``: three small 3D convolutions, then mean and max pooling; keeps the spatial layout with few parameters.
It takes ``(x, mask, xyz)`` and trains through ``heads.ProbeEnsemble`` (early stopping, seeds).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from brain.common.grid import TOKEN_GRID
from brain.downstream.heads import ProbeEnsemble

GRID_ZYX = TOKEN_GRID


def token_grid_zyx(n_tok: int) -> tuple[int, int, int] | None:
    """The (Z, Y, X) token grid of a flattened token count, or None when the tokens do not fill it (a global-only
    encoder's single token)."""
    return GRID_ZYX if int(n_tok) == int(np.prod(GRID_ZYX)) else None


def _mlp(dim: int, hidden: int, n_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, n_out))


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


def make_token_heads(dim: int, n_out: int, multilabel: bool, seed: int, regression: bool) -> list[ProbeEnsemble]:
    """The 3D conv probe over the token grid, as the head ``tok_conv``."""
    return [ProbeEnsemble(lambda: ConvProbe(dim, n_out), multilabel, seed, regression=regression, half=True)]
