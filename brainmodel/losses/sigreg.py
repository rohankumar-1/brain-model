"""SigReg and the batch gather that feeds it."""

from __future__ import annotations

import torch
import torch.distributed as dist

from models.sigreg import SIGReg, prediction_loss  # noqa: F401  (shared with the 2D recipe)

__all__ = ["SIGReg", "prediction_loss", "gather_batch"]


def gather_batch(z: torch.Tensor, enabled: bool = True) -> torch.Tensor:
    """All-gather (V, B, D) along the batch axis for the statistic. Other ranks are detached.

    Every rank holds the same B (the loader drops the last partial batch), so no counts are exchanged
    and nothing is read back to the host.
    """
    if not enabled or not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return z
    parts = [torch.empty_like(z) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, z.detach().contiguous())
    parts[dist.get_rank()] = z
    return torch.cat(parts, dim=1)
