"""Collapse / health metrics logged during pretraining."""

from __future__ import annotations

import torch

from lib.train_utils import effective_rank


def embedding_health(z: torch.Tensor) -> dict[str, float]:
    """``z`` is (N, D) detached embeddings (view-level or tokens)."""
    z = z.float()
    if z.ndim != 2 or z.size(0) < 2:
        return {"std_mean": 0.0, "std_min": 0.0, "erank": 0.0}
    std = z.std(dim=0)
    return {
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "erank": effective_rank(z),
    }
