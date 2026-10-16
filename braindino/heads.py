"""DINO / iBOT projection heads, teacher centering, and KoLeo."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """Three-layer MLP + frozen-norm prototype layer (DINOv2 / BrainDINO)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 2048, bottleneck: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        self.weight_v = nn.Parameter(torch.empty(out_dim, bottleneck))
        self.weight_g = nn.Parameter(torch.ones(out_dim, 1), requires_grad=False)
        nn.init.trunc_normal_(self.weight_v, std=0.02)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.normalize(self.mlp(x).float(), dim=-1).to(dtype=x.dtype)
        w = (self.weight_g * F.normalize(self.weight_v.float(), dim=1)).to(dtype=x.dtype)
        return F.linear(h, w)


def softmax_center(logits: torch.Tensor, center: torch.Tensor, temp: float) -> torch.Tensor:
    return F.softmax((logits - center) / temp, dim=-1)


@torch.no_grad()
def update_center(center: torch.Tensor, teacher_logits: torch.Tensor, momentum: float = 0.9) -> None:
    batch = teacher_logits.float().mean(0)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(batch)
        batch /= dist.get_world_size()
    center.mul_(momentum).add_(batch, alpha=1.0 - momentum)


def koleo_loss(cls: torch.Tensor) -> torch.Tensor:
    """Encourage spread CLS features (KoLeo, weight 0.1 in BrainDINO)."""
    z = F.normalize(cls.float(), dim=-1)
    sim = z @ z.transpose(0, 1)
    sim.fill_diagonal_(-1.0)
    nearest = sim.max(dim=1).values.clamp(max=1.0 - 1e-6)
    return -(1.0 - nearest).clamp_min(1e-6).log().mean()
