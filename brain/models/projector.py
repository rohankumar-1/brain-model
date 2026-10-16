"""Projector. The last linear is unnormalized so SigReg can set the scale."""

from __future__ import annotations

import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, norm: str = "batch"):
        super().__init__()
        self.norm = str(norm)
        if self.norm == "batch":
            n1, n2 = nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)
        elif self.norm == "layer":
            n1, n2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        else:
            raise ValueError(f"projector norm must be 'batch' or 'layer', got {norm!r}")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), n1, nn.GELU(),
            nn.Linear(hidden, hidden), n2, nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = self.net(x.reshape(-1, shape[-1]))
        return y.view(*shape[:-1], -1)
