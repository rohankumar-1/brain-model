"""DINO (CLS) + iBOT (masked patches) distillation, teacher centering, and KoLeo."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


def sample_masks(batch: int, n_tok: int, device: torch.device, p_apply: float = 0.5,
                 ratio: tuple[float, float] = (0.1, 0.5)) -> torch.Tensor:
    """Per-row Bernoulli masks. Apply-prob 0.5, ratio U[0.1, 0.5] (BrainDINO)."""
    apply = torch.rand(batch, device=device) < p_apply
    r = torch.empty(batch, 1, device=device).uniform_(*ratio)
    return (torch.rand(batch, n_tok, device=device) < r) & apply.unsqueeze(1)


def dino_loss(student: list[torch.Tensor], teacher: list[torch.Tensor],
              center: torch.Tensor, student_temp: float, teacher_temp: float) -> torch.Tensor:
    """Each student view matches every teacher global except the same-index global."""
    n_t = len(teacher)
    qs = [softmax_center(t.detach(), center, teacher_temp) for t in teacher]
    loss, n = student[0].new_zeros(()), 0
    for i, s in enumerate(student):
        log_p = F.log_softmax(s.float() / student_temp, dim=-1)
        for j, q in enumerate(qs):
            if i < n_t and i == j:
                continue
            loss = loss + -(q * log_p).sum(-1).mean()
            n += 1
    return loss / max(n, 1)


def ibot_loss(student_patches: torch.Tensor, teacher_patches: torch.Tensor, mask: torch.Tensor | None,
              center: torch.Tensor, student_temp: float, teacher_temp: float) -> torch.Tensor:
    """Teacher targets only at masked locations. ``mask=None`` means already gathered."""
    if mask is None:
        s, t = student_patches, teacher_patches
    elif not bool(mask.any()):
        return student_patches.sum() * 0
    else:
        s, t = student_patches[mask], teacher_patches[mask]
    if s.numel() == 0:
        return student_patches.sum() * 0
    q = softmax_center(t.detach(), center, teacher_temp)
    return -(q * F.log_softmax(s.float() / student_temp, dim=-1)).sum(-1).mean()


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
