"""Small training helpers: LR schedule, distributed info, clean preemption, embedding rank."""

from __future__ import annotations

import math
import os
import signal

import torch
import torch.distributed as dist


def lr_at(step: int, total: int, peak: float, warmup: int, floor: float = 0.01) -> float:
    """Linear warmup then cosine decay to ``floor * peak``."""
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def dist_info() -> tuple[int, int, int]:
    """(rank, world_size, local_rank) from torchrun env, or (0, 1, 0)."""
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


class StopFlag:
    """Set by SIGTERM/SIGINT. ``poll`` agrees across ranks so all stop on the same step.

    The trainer checks it between steps, so a checkpoint is never written mid-update.
    """

    def __init__(self) -> None:
        self.flag = False
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._set)

    def _set(self, *_) -> None:
        self.flag = True

    def poll(self, device: torch.device) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return self.flag
        t = torch.tensor(float(self.flag), device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())


def effective_rank(z: torch.Tensor) -> float:
    """RankMe: exp of the entropy of the normalized singular values of centered features."""
    z = z.float()
    s = torch.linalg.svdvals(z - z.mean(dim=0))
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
