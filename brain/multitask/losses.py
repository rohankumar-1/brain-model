"""Typed, masked multitask losses and their balancing.

Every slot with a label gets the loss of its task's type: binary cross-entropy, Huber on the standardised value,
cross-entropy over the task's classes, BCE + soft Dice for a segmentation slot (soft targets at the decoder's stride),
voxel-weighted Huber for a dense slot (``dense_loss``; targets and weights from ``data.prepare``).
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


def sparse_seg_loss(logits: torch.Tensor, target: torch.Tensor, cell_zyx: tuple[int, int, int],
                    pos_weight: float = 20.0) -> torch.Tensor:
    """(P,) loss for a tiny target that many scans lack (registry ``sparse``: aneurysm balls of a few hundred voxels,
    no target on 58% of scans), where ``seg_loss`` collapses to an empty map (R10's aneurysm_location: peak logit
    <= -3.9 on every test scan). Four terms:

    * voxel BCE with ``pos_weight`` on target voxels;
    * soft Dice on scans that have a target only (on an empty one it rewards predicting nothing);
    * cell BCE: logits and target max-pooled to the encoder's token grid (``cell_zyx`` output voxels per token), i.e.
      "is it in this 16 mm cell", the coarse finder signal a find -> refine pass would pick cells from;
    * scan BCE: the map's peak logit against "this scan has a target", so the peak works as a detector."""
    logits, target = logits.float(), target[:, :1].float()
    dims = (1, 2, 3, 4)
    pw = torch.tensor(float(pos_weight), device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw).mean(dims)
    p = logits.sigmoid()
    has = target.amax(dims) > 0.5
    dice = (1 - (2 * (p * target).sum(dims) + 1) / (p.sum(dims) + target.sum(dims) + 1)) * has.float()
    lc = F.max_pool3d(logits, cell_zyx, ceil_mode=True)
    tc = F.max_pool3d(target, cell_zyx, ceil_mode=True)
    cell = F.binary_cross_entropy_with_logits(lc, (tc > 0.5).float(), reduction="none", pos_weight=pw).mean(dims)
    scan = F.binary_cross_entropy_with_logits(logits.amax(dims), has.float(), reduction="none")
    return bce + dice + cell + scan


def dense_loss(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """(D,) Huber between predicted and target intensities (window units), averaged over each slot's weighted voxels;
    a slot with no weight (nothing hidden / an empty target) scores 0."""
    err = F.smooth_l1_loss(pred.float(), target.float(), reduction="none", beta=0.1)
    return (err * w).sum(dim=(1, 2, 3, 4)) / w.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)


def hl_gauss_loss(logits: torch.Tensor, y: torch.Tensor, edges: torch.Tensor, sigma: float) -> torch.Tensor:
    """(N,) HL-Gauss: KL from the Gaussian around ``y`` (clipped into ``edges``), integrated over each bin, to the
    predicted bin distribution. Cross-entropy minus the target's entropy: same gradients, but 0 at a perfect fit, so
    the loss sits on the scale of the other types' losses for the balancer."""
    y = y.clamp(float(edges[0]), float(edges[-1])).unsqueeze(-1)
    cdf = torch.special.ndtr((edges.float() - y) / sigma)
    q = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0)
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
    logp = logits.float().log_softmax(-1)
    return (q * (q.clamp_min(1e-12).log() - logp)).sum(-1)


def token_occupancy(seg: torch.Tensor, patch_zyx: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Segmentation targets (P, C, Z, Y, X) (channel 1, when present, the validity of each voxel) -> per encoder token
    (P, N), (z, y, x) order: whether the mask reaches into it, and whether the token counts (at least half valid)."""
    occ = F.max_pool3d(seg[:, :1].float(), patch_zyx).flatten(1) > 0.5
    if seg.shape[1] > 1:
        valid = F.avg_pool3d(seg[:, 1:2].float(), patch_zyx).flatten(1) >= 0.5
    else:
        valid = torch.ones_like(occ)
    return occ.float(), valid


def token_occupancy_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(P,) class-balanced token BCE: the mean over occupied tokens and the mean over empty ones, averaged (only the
    empty term on a slot without the structure), so a few occupied tokens are not outvoted by thousands of empty ones
    (the collapse R10's aneurysm map showed)."""
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    pos, neg = (target > 0.5) & valid, (target <= 0.5) & valid
    n_pos, n_neg = pos.sum(1), neg.sum(1)
    lp = (bce * pos).sum(1) / n_pos.clamp_min(1)
    ln = (bce * neg).sum(1) / n_neg.clamp_min(1)
    w_pos, w_neg = (n_pos > 0).float(), (n_neg > 0).float()
    return (lp * w_pos + ln * w_neg) / (w_pos + w_neg).clamp_min(1)


def slab_mil_loss(logits: torch.Tensor, lab: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """(Q,) multiple-instance loss from slice labels on the token grid: ``lab`` (Q, N) per token -1 unlabelled, 0 in
    slices without the finding, r >= 1 in the r-th run of consecutive positive slices. Each run is a bag whose most
    confident (foreground, ``keep``) token should fire; every token in negative slices should not. Bag and negative
    terms averaged as in ``token_occupancy_loss``."""
    lg = logits.float()
    pos_sum = torch.zeros(lg.shape[0], device=lg.device)
    n_runs = torch.zeros_like(pos_sum)
    for r in range(1, int(lab.max().clamp_min(0)) + 1):
        m = (lab == r) & keep
        has = m.any(1)
        if not has.any():
            continue
        top = lg.masked_fill(~m, -1e4).amax(1)
        pos_sum += F.binary_cross_entropy_with_logits(top, torch.ones_like(top), reduction="none") * has
        n_runs += has.float()
    neg = (lab == 0) & keep
    n_neg = neg.sum(1)
    ln = (F.binary_cross_entropy_with_logits(lg, torch.zeros_like(lg), reduction="none") * neg).sum(1) \
        / n_neg.clamp_min(1)
    w_pos, w_neg = (n_runs > 0).float(), (n_neg > 0).float()
    return (pos_sum / n_runs.clamp_min(1) * w_pos + ln * w_neg) / (w_pos + w_neg).clamp_min(1)


def slot_losses(out: dict, task: torch.Tensor, y: torch.Tensor, task_type: torch.Tensor, seg: torch.Tensor,
                seg_pair: torch.Tensor, stride: int = 2, dense: dict | None = None, sparse: torch.Tensor | None = None,
                cell_zyx: tuple[int, int, int] | None = None, pos_weight: float = 20.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-slot loss (B, K) and a mask of slots that carry one. ``dense``: the prepared batch (``dense_pair``,
    ``dense_tgt``, ``dense_w``) when it has dense slots. ``sparse`` (n_tasks bool, registry ``sparse``) with
    ``cell_zyx`` (output voxels per encoder token): those segmentation slots take ``sparse_seg_loss``."""
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
        if "regression_hl" in out:
            lg, edges, sigma = out["regression_hl"]
            loss[m] = hl_gauss_loss(lg[m], y[m], edges, sigma)
        else:
            loss[m] = F.smooth_l1_loss(out["regression"][m].float(), y[m], reduction="none")
        has |= m
    m = (tt == TYPE_ID["multiclass"]) & lab
    if m.any():
        loss[m] = F.cross_entropy(out["multiclass"][m].float(), y[m].long(), reduction="none")
        has |= m
    if out.get("seg") is not None and len(seg_pair):
        tgt = seg_target(seg, stride)
        sl = seg_loss(out["seg"], tgt)
        if sparse is not None and cell_zyx is not None:
            sp = sparse.to(task.device)[task[seg_pair[:, 0], seg_pair[:, 1]]]
            if sp.any():
                sl = sl.clone()
                sl[sp] = sparse_seg_loss(out["seg"][sp], tgt[sp], cell_zyx, pos_weight)
        loss = loss.index_put((seg_pair[:, 0], seg_pair[:, 1]), sl)
        has = has.index_put((seg_pair[:, 0], seg_pair[:, 1]), torch.ones_like(sl, dtype=torch.bool))
    if out.get("dense") is not None and dense is not None and len(dense["dense_pair"]):
        dp = dense["dense_pair"]
        dl = dense_loss(out["dense"], dense["dense_tgt"], dense["dense_w"])
        loss = loss.index_put((dp[:, 0], dp[:, 1]), dl)
        has = has.index_put((dp[:, 0], dp[:, 1]), torch.ones_like(dl, dtype=torch.bool))
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
