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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from brain.multitask.data import TYPE_ID


def seg_target(seg: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """(P, 1, Z, Y, X) full-resolution {0, 1} (or soft, after rotation) -> fraction per output voxel."""
    return F.avg_pool3d(seg, stride) if stride > 1 else seg


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(P,) BCE + soft Dice per slot. An empty target scores Dice on the background-free prediction (+1 smoothing).
    A 2-channel target is (target, validity): voxels with validity 0 are left out of both terms."""
    logits, target = logits.float(), target.float()
    w = target[:, 1:] if target.shape[1] == 2 else torch.ones_like(target)
    target = target[:, :1]
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * w).sum(dim=(1, 2, 3, 4)) / w.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
    p = logits.sigmoid() * w
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
    """Per-task loss weighting, then a group-first average. ``ema``: each task's loss over its running mean;
    ``uncertainty``: ``exp(-s) L + s`` with a learned log-variance ``s`` per task, soft-clamped to +-``clamp``
    (unclamped, near-solved tasks reach e^2.5 = 12x the weight of hard ones); ``uniform``.

    Under DDP the ``ema`` statistics are all-reduced in ``forward`` and ``sync_grads`` averages the log-variance
    gradients (call it after ``backward``): the balancer lives outside the DDP module, so without these every rank
    learned its own task weights from its own few volumes. ``data_term`` (after ``forward``) is the balanced loss
    without the ``+ s`` regulariser: its magnitude, unlike the total's, does not cross zero."""

    def __init__(self, n_tasks: int, mode: str = "ema", decay: float = 0.99, group_of: list[int] | None = None,
                 group_weight: list[float] | None = None, clamp: float | None = None):
        super().__init__()
        self.mode, self.decay = str(mode), float(decay)
        self.clamp = float(clamp) if clamp else None
        self.data_term: torch.Tensor | None = None
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
        if self.mode == "ema":                                  # before any early return: every rank must all-reduce
            with torch.no_grad():
                g_sums, g_cnt = sums.detach(), cnt
                if _distributed():                              # every rank updates the same running means
                    st = torch.stack([g_sums, g_cnt])
                    dist.all_reduce(st)
                    g_sums, g_cnt = st[0], st[1]
                g_present = g_cnt > 0
                cur = g_sums / g_cnt.clamp_min(1)
                fresh = g_present & (self.ema == 0)
                self.ema[fresh] = cur[fresh]
                upd = g_present & ~fresh
                self.ema[upd] = self.decay * self.ema[upd] + (1 - self.decay) * cur[upd]
        if not present.any():
            self.data_term = loss.sum().detach() * 0.0
            return loss.sum() * 0.0, per_task.detach()
        data = None
        if self.mode == "uniform":
            scaled = per_task
        elif self.mode == "ema":
            scaled = per_task / self.ema.clamp_min(1e-3)
        else:
            s = self.log_var if self.clamp is None else self.clamp * torch.tanh(self.log_var / self.clamp)
            data = torch.exp(-s) * per_task
            scaled = data + s
        total = self._group_mean(scaled, present)
        self.data_term = total.detach() if data is None else self._group_mean(data, present).detach()
        return total, per_task.detach()

    def _group_mean(self, scaled: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """Mean over the present tasks of each group, then the weighted mean over present groups."""
        n_g = self.group_weight.numel()
        g = self.group_of[present]
        g_sum = torch.zeros(n_g, device=scaled.device).index_add(0, g, scaled[present])
        g_cnt = torch.zeros(n_g, device=scaled.device).index_add(0, g, torch.ones_like(g, dtype=torch.float32))
        g_on = g_cnt > 0
        w = self.group_weight[g_on]
        return (w * g_sum[g_on] / g_cnt[g_on]).sum() / w.sum()

    def weights(self) -> torch.Tensor | None:
        """The current per-task loss weights exp(-s) (uncertainty mode), for logging."""
        if self.log_var is None:
            return None
        s = self.log_var if self.clamp is None else self.clamp * torch.tanh(self.log_var / self.clamp)
        return torch.exp(-s.detach())

    def sync_grads(self) -> None:
        """Average the log-variance gradients over DDP ranks (after ``backward``, before the optimizer step)."""
        if self.log_var is not None and _distributed():          # every rank joins, with zeros if it had no task
            if self.log_var.grad is None:
                self.log_var.grad = torch.zeros_like(self.log_var)
            dist.all_reduce(self.log_var.grad, op=dist.ReduceOp.AVG)


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
