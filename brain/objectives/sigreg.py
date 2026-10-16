"""SIGReg: sketched isotropic-Gaussian penalty from LeJEPA.

Random 1-D projections of a batch are compared to N(0, 1) with the
Epps-Pulley characteristic-function test. The penalty is high when embeddings
collapse to a point or a subspace, and low when they look like N(0, I).

The prediction term matches LeJEPA: every view is pulled toward the mean of
the global views. With only global views, that mean is the mean of all views.
``lamb`` is the tradeoff from the paper's algorithm 2, default 0.02:
loss = (1 - lamb) * prediction + lamb * SIGReg.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class SIGReg(nn.Module):
    """Epps-Pulley statistic on random unit projections.

    ``proj`` is (B, D) or (V, B, D). Quadrature follows the LeJEPA reference:
    integrate t in [0, 3] and double, using the evenness of the integrand.
    """

    def __init__(self, knots: int = 17, n_proj: int = 256, t_max: float = 3.0):
        super().__init__()
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        window = torch.exp(-t.square() / 2.0)
        self.n_proj = int(n_proj)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim == 2:
            proj = proj.unsqueeze(0)
        if proj.ndim != 3:
            raise ValueError(f"proj must be (B, D) or (V, B, D), got {tuple(proj.shape)}")
        proj = proj.float()
        direction = torch.randn(
            proj.size(-1), self.n_proj, device=proj.device, dtype=proj.dtype
        )
        direction = direction / direction.norm(dim=0, keepdim=True).clamp_min(1e-6)
        # (V, B, P, K)
        x_t = (proj @ direction).unsqueeze(-1) * self.t
        # Mean over the batch axis. Each view is tested on its own.
        err = (x_t.cos().mean(dim=1) - self.phi).square() + x_t.sin().mean(dim=1).square()
        statistic = (err @ self.weights) * proj.size(1)
        return statistic.mean()


def prediction_loss(proj: torch.Tensor, n_global: int | None = None) -> torch.Tensor:
    """Pull every view toward the mean embedding of the global views.

    ``proj`` is (V, B, D), global views first. This is LeJEPA equation (7).
    Passing no ``n_global`` treats every view as global, which is the paper's
    setting with zero local crops.
    """
    if proj.ndim != 3 or proj.size(0) < 2:
        raise ValueError(f"need at least two views (V, B, D), got {tuple(proj.shape)}")
    n_global = proj.size(0) if n_global is None else int(n_global)
    if not 1 <= n_global <= proj.size(0):
        raise ValueError(f"n_global={n_global} outside 1..{proj.size(0)}")
    center = proj[:n_global].mean(dim=0)
    return (proj - center).square().mean()


def per_sample_invariance(proj: torch.Tensor, n_global: int | None = None) -> torch.Tensor:
    """(B,) squared distance of each sample's views to its global-view center."""
    n_global = proj.size(0) if n_global is None else int(n_global)
    return (proj - proj[:n_global].mean(dim=0)).square().mean(dim=(0, 2))


def pairwise_cosine(proj: torch.Tensor) -> torch.Tensor:
    """Mean cosine between view 0 and view 1. ``proj`` is (2, B, D)."""
    if proj.ndim != 3 or proj.size(0) < 2:
        raise ValueError(f"need (2, B, D), got {tuple(proj.shape)}")
    a = torch.nn.functional.normalize(proj[0].float(), dim=-1)
    b = torch.nn.functional.normalize(proj[1].float(), dim=-1)
    return (a * b).sum(dim=-1).mean()


def lejepa_loss(
    proj: torch.Tensor,
    sigreg: SIGReg,
    lamb: float = 0.02,
    n_global: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, prediction loss, and SIGReg.

    ``proj`` is (V, B, D) with the global views first. ``n_global`` of them
    form the prediction target. SIGReg is applied to every view.
    """
    inv = prediction_loss(proj, n_global=n_global)
    reg = sigreg(proj)
    total = (1.0 - lamb) * inv + lamb * reg
    return total, inv, reg


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
