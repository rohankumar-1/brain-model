"""Collapse / health metrics logged during pretraining."""

from __future__ import annotations

import torch


def effective_rank(z: torch.Tensor) -> float:
    """RankMe: exp of the entropy of the normalized singular values of centered features."""
    z = z.float()
    s = torch.linalg.svdvals(z - z.mean(dim=0))
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def embedding_health(z: torch.Tensor) -> dict[str, float]:
    """``z`` is (N, D) detached embeddings (view-level or tokens)."""
    z = z.float()
    if z.ndim != 2 or z.size(0) < 2:
        return {"std_mean": 0.0, "std_min": 0.0, "erank": 0.0}
    std = z.std(dim=0)
    return {"std_mean": float(std.mean()), "std_min": float(std.min()), "erank": effective_rank(z)}
