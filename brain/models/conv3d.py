"""3D convolution blocks shared by the segmentation decoders (frozen-suite token decoder ``downstream.seg3d``, the
multitask decoder ``multitask.model``)."""

from __future__ import annotations

import torch.nn as nn


def group_norm(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if c % 8 == 0 else 4 if c % 4 == 0 else 1, c)


def conv3(cin: int, cout: int, stride=1) -> nn.Sequential:
    """3^3 convolution (no bias), GroupNorm, GELU."""
    return nn.Sequential(nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False), group_norm(cout), nn.GELU())


def up3(cin: int, cout: int, stride) -> nn.Sequential:
    """Transposed convolution by ``stride`` (kernel = stride), GroupNorm, GELU."""
    return nn.Sequential(nn.ConvTranspose3d(cin, cout, stride, stride=stride, bias=False), group_norm(cout), nn.GELU())


class Res(nn.Module):
    """Two ``conv3`` with a residual (1^3 projection when the width changes)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(conv3(cin, cout), conv3(cout, cout))
        self.skip = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1, bias=False)

    def forward(self, x):
        return self.block(x) + self.skip(x)
