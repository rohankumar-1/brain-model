"""3D convolution blocks shared by the segmentation decoders (frozen-suite token decoder ``downstream.seg3d``, the
multitask decoder ``multitask.model``)."""

from __future__ import annotations

import torch.nn as nn


def group_norm(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if c % 8 == 0 else 4 if c % 4 == 0 else 1, c)


def conv3(cin: int, cout: int, stride=1, kernel=3) -> nn.Sequential:
    """3^3 convolution (no bias), GroupNorm, GELU. ``kernel`` (z, y, x), e.g. (1, 3, 3) for an in-plane-only
    convolution (the native-geometry stem, ``patch_embed.NgeoStemEmbed``); padding keeps the size at stride 1."""
    k = (kernel,) * 3 if isinstance(kernel, int) else tuple(int(v) for v in kernel)
    return nn.Sequential(nn.Conv3d(cin, cout, k, stride=stride, padding=tuple(v // 2 for v in k), bias=False),
                         group_norm(cout), nn.GELU())


def up3(cin: int, cout: int, stride) -> nn.Sequential:
    """Transposed convolution by ``stride`` (kernel = stride), GroupNorm, GELU."""
    return nn.Sequential(nn.ConvTranspose3d(cin, cout, stride, stride=stride, bias=False), group_norm(cout), nn.GELU())


class Res(nn.Module):
    """Two ``conv3`` with a residual (1^3 projection when the width changes); ``kernel`` as in ``conv3``."""

    def __init__(self, cin: int, cout: int, kernel=3):
        super().__init__()
        self.block = nn.Sequential(conv3(cin, cout, kernel=kernel), conv3(cout, cout, kernel=kernel))
        self.skip = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1, bias=False)

    def forward(self, x):
        return self.block(x) + self.skip(x)
