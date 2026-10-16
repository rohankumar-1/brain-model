"""DINO (CLS) + iBOT (masked patches) distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from braindino.heads import softmax_center


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
