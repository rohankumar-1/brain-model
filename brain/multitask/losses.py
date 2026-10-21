"""Typed, masked multitask losses and their balancing.

Every slot with a label gets the loss of its task's type: binary cross-entropy, Huber on the standardised value,
cross-entropy over the task's classes, BCE + soft Dice for a segmentation slot (soft targets at the decoder's stride).
Slots are averaged per task first (a task asked in many slots of the batch is still one task), each task's loss is
then scaled by the balancing mode, averaged over the tasks of its group, and the groups present are averaged (with
the group weights), so a group of many tasks (SynthSeg) weighs as one group:
  uniform      the task loss as is
  ema          each task's loss divided by a running mean of itself (every task contributes ~1, whatever its scale)
  uncertainty  Kendall et al.: exp(-s_t) L_t + s_t with a learned log-variance s_t per task
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.multitask.data import TYPE_ID


def seg_target(seg: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """(P, 1, Z, Y, X) full-resolution {0, 1} (or soft, after rotation) -> fraction per output voxel."""
    return F.avg_pool3d(seg, stride) if stride > 1 else seg


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(P,) BCE + soft Dice per slot. An empty target scores Dice on the background-free prediction (+1 smoothing)."""
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(1, 2, 3, 4))
    p = logits.sigmoid()
    inter = (p * target).sum(dim=(1, 2, 3, 4))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4)) + 1)
    return bce + dice


def slot_losses(out: dict, task: torch.Tensor, y: torch.Tensor, task_type: torch.Tensor, seg: torch.Tensor,
                seg_pair: torch.Tensor, stride: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-slot loss (B, K) and a mask of slots that carry one."""
    tt = torch.where(task >= 0, task_type[task.clamp_min(0)], torch.full_like(task, -1))
    loss = torch.zeros_like(y, dtype=torch.float32)
    has = torch.zeros_like(task, dtype=torch.bool)
    lab = torch.isfinite(y)
    m = (tt == TYPE_ID["binary"]) & lab
    if m.any():
        loss[m] = F.binary_cross_entropy_with_logits(out["binary"][m].float(), y[m], reduction="none")
        has |= m
    m = (tt == TYPE_ID["regression"]) & lab
    if m.any():
        loss[m] = F.smooth_l1_loss(out["regression"][m].float(), y[m], reduction="none")
        has |= m
    m = (tt == TYPE_ID["multiclass"]) & lab
    if m.any():
        loss[m] = F.cross_entropy(out["multiclass"][m].float(), y[m].long(), reduction="none")
        has |= m
    if out.get("seg") is not None and len(seg_pair):
        sl = seg_loss(out["seg"], seg_target(seg, stride))
        loss = loss.index_put((seg_pair[:, 0], seg_pair[:, 1]), sl)
        has = has.index_put((seg_pair[:, 0], seg_pair[:, 1]), torch.ones_like(sl, dtype=torch.bool))
    return loss, has


class Balancer(nn.Module):
    def __init__(self, n_tasks: int, mode: str = "ema", decay: float = 0.99, group_of: list[int] | None = None,
                 group_weight: list[float] | None = None):
        super().__init__()
        self.mode, self.decay = str(mode), float(decay)
        self.register_buffer("ema", torch.zeros(n_tasks))
        g = torch.as_tensor(group_of if group_of is not None else range(n_tasks), dtype=torch.long)
        self.register_buffer("group_of", g, persistent=False)
        n_g = int(g.max()) + 1 if len(g) else 1
        self.register_buffer("group_weight", torch.as_tensor(group_weight if group_weight is not None
                                                             else [1.0] * n_g, dtype=torch.float32),
                             persistent=False)
        self.log_var = nn.Parameter(torch.zeros(n_tasks)) if self.mode == "uncertainty" else None
        if self.mode not in ("uniform", "ema", "uncertainty"):
            raise ValueError(f"loss.balance {mode}")

    def forward(self, loss: torch.Tensor, has: torch.Tensor, task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced scalar loss and the per-task mean losses (n_tasks,), NaN-free (0 where absent)."""
        n = self.ema.numel()
        t = task[has]
        sums = torch.zeros(n, device=loss.device).index_add(0, t, loss[has])
        cnt = torch.zeros(n, device=loss.device).index_add(0, t, torch.ones_like(loss[has]))
        present = cnt > 0
        per_task = sums / cnt.clamp_min(1)
        if not present.any():
            return loss.sum() * 0.0, per_task.detach()
        if self.mode == "uniform":
            scaled = per_task
        elif self.mode == "ema":
            with torch.no_grad():
                cur = per_task.detach()
                fresh = present & (self.ema == 0)
                self.ema[fresh] = cur[fresh]
                upd = present & ~fresh
                self.ema[upd] = self.decay * self.ema[upd] + (1 - self.decay) * cur[upd]
            scaled = per_task / self.ema.clamp_min(1e-3)
        else:
            scaled = torch.exp(-self.log_var) * per_task + self.log_var
        # mean over the present tasks of each group, then the weighted mean over present groups
        n_g = self.group_weight.numel()
        g = self.group_of[present]
        g_sum = torch.zeros(n_g, device=loss.device).index_add(0, g, scaled[present])
        g_cnt = torch.zeros(n_g, device=loss.device).index_add(0, g, torch.ones_like(g, dtype=torch.float32))
        g_on = g_cnt > 0
        w = self.group_weight[g_on]
        total = (w * g_sum[g_on] / g_cnt[g_on]).sum() / w.sum()
        return total, per_task.detach()
