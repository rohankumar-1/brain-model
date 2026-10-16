"""UNETR-style segmentation decoder over a frozen ViT.

Patch tokens from four depths (default blocks 3, 6, 9, 12) are reshaped to
grids. The deepest one seeds the decoder; each earlier one is upsampled to the
matching resolution and concatenated as a skip connection. Early blocks keep
the fine spatial detail that the final class token has discarded.

    tap 3 (16px grid) ── conv ──► up ─┬─► 32 ─► up ─┬─► 64 ─► up ─┬─► 128 ─► up ─► 256
    tap 2 ─ deconv ────────────────────┘             │              │
    tap 1 ─ deconv x2 ───────────────────────────────┘              │
    tap 0 ─ deconv x3 ──────────────────────────────────────────────┘
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.GELU())


def up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 2, stride=2, bias=False), nn.GroupNorm(8, cout), nn.GELU())


def skip_chain(dim: int, cout: int, n_up: int) -> nn.Sequential:
    return nn.Sequential(*[up(dim if i == 0 else cout, cout) for i in range(n_up)])


class SegDecoder(nn.Module):
    """Token grids from ``len(widths)`` depths -> class logits at full resolution."""

    def __init__(self, embed_dim: int, n_classes: int, widths: tuple[int, ...] = (256, 128, 64, 32),
                 final_up: bool = True):
        super().__init__()
        n = self.n_stages = len(widths) - 1
        self.stem = nn.Sequential(conv(embed_dim, widths[0]), conv(widths[0], widths[0]))
        # Stage j lands at 32 * 2^j px and takes the tap that is j + 1 upsamplings away from it.
        self.skips = nn.ModuleList([skip_chain(embed_dim, widths[j + 1], j + 1) for j in range(n)])
        self.ups = nn.ModuleList([up(widths[j], widths[j + 1]) for j in range(n)])
        self.fuse = nn.ModuleList(
            [nn.Sequential(conv(2 * widths[j + 1], widths[j + 1]), conv(widths[j + 1], widths[j + 1])) for j in range(n)]
        )
        last = up if final_up else conv  # final_up=False when the token grid is already 2x finer
        self.final = nn.Sequential(last(widths[-1], widths[-1] // 2), conv(widths[-1] // 2, widths[-1] // 2))
        self.head = nn.Conv2d(widths[-1] // 2, n_classes, 1)

    @staticmethod
    def grid(tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        side = int(n**0.5)
        return tokens.transpose(1, 2).reshape(b, c, side, side)

    def forward(self, taps: list[torch.Tensor]) -> torch.Tensor:
        """``taps``: (B, N, C) patch tokens ordered shallow to deep."""
        grids = [self.grid(t) for t in taps]
        x = self.stem(grids[-1])
        for j in range(self.n_stages):
            skip = self.skips[j](grids[-2 - j])
            x = self.fuse[j](torch.cat([self.ups[j](x), skip], dim=1))
        return self.head(self.final(x))


def region_seg_loss(logits: torch.Tensor, target: torch.Tensor, regions: dict[str, tuple[int, ...]]) -> torch.Tensor:
    """Cross-entropy on the label map plus soft Dice per named region (fp32).

    Labels outside the head's classes are ignored instead of crashing CE.
    """
    logits, target = logits.float(), target.long()
    target = target.masked_fill((target < 0) | (target >= logits.shape[1]), 255)
    ce = F.cross_entropy(logits, target, ignore_index=255)
    probs = logits.softmax(dim=1)
    dice = []
    for ids in regions.values():
        idx = [i for i in ids if i < probs.shape[1]]
        pred = probs[:, idx].sum(1)
        gt = torch.isin(target, torch.tensor(idx, device=target.device)).float()
        dice.append(1 - (2 * (pred * gt).sum() + 1) / (pred.sum() + gt.sum() + 1))
    return ce + torch.stack(dice).mean()
