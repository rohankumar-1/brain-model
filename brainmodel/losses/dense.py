"""Token-level agreement between crops that cover the same place.

Views live on one coarse token grid per volume. A view reports an embedding for each coarse position it
covers (``parent_mean``: a fine-patch crop averages the fine tokens inside a coarse position). Every
position covered by two or more views is pulled to the mean of its views. This is the LeJEPA invariance
term at the token level: nothing is predicted from a pooled summary, so local content has to survive.
"""

from __future__ import annotations

import torch


def parent_mean(tok: torch.Tensor, keep: torch.Tensor, parent: torch.Tensor, n_par: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Average kept tokens that share a parent. ``tok`` (R, N, D), ``parent`` (R, N) long.

    Returns (R, n_par, D) and a (R, n_par) bool that is True where a parent has a kept token.
    """
    r, n, d = tok.shape
    flat = (parent + torch.arange(r, device=tok.device)[:, None] * n_par).reshape(-1)
    w = keep.to(torch.float32)
    s = tok.new_zeros(r * n_par, d, dtype=torch.float32).index_add(0, flat, (tok.float() * w[..., None]).reshape(-1, d))
    c = tok.new_zeros(r * n_par, dtype=torch.float32).index_add(0, flat, w.reshape(-1))
    return (s / c.clamp_min(1.0)[:, None]).view(r, n_par, d), (c > 0).view(r, n_par)


def consensus(emb: list[torch.Tensor], valid: list[torch.Tensor], pos: list[torch.Tensor], n_pos: int,
              stopgrad_mean: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean squared distance of each view's position embedding to the mean over views at that position.

    ``emb[i]`` is (B, P_i, D), ``valid[i]`` (B, P_i) bool, ``pos[i]`` (B, P_i) long in [0, n_pos). Only
    positions covered by at least two views count. Returns the loss and the fraction of entries that were
    covered (a health number, not a loss term). ``stopgrad_mean`` freezes the consensus so each view has
    to match a target it does not move.
    """
    b, d, dev = emb[0].shape[0], emb[0].shape[-1], emb[0].device
    trash = b * n_pos
    rows = torch.arange(b, device=dev)[:, None] * n_pos
    flats = [torch.where(v, rows + p, torch.full_like(p, trash)).reshape(-1) for v, p in zip(valid, pos)]
    s = emb[0].new_zeros(trash + 1, d, dtype=torch.float32)
    c = emb[0].new_zeros(trash + 1, dtype=torch.float32)
    for e, v, f in zip(emb, valid, flats):
        s = s.index_add(0, f, e.float().reshape(-1, d) * v.reshape(-1, 1))
        c = c.index_add(0, f, v.float().reshape(-1))
    mean = s / c.clamp_min(1.0)[:, None]
    if stopgrad_mean:
        mean = mean.detach()
    total, covered, entries = emb[0].new_zeros(()).float(), emb[0].new_zeros(()).float(), emb[0].new_zeros(()).float()
    for e, v, f in zip(emb, valid, flats):
        cov = v.reshape(-1) & (c[f] >= 2)
        diff = (e.float().reshape(-1, d) - mean[f]).square().sum(-1)
        total = total + (diff * cov).sum()
        covered = covered + cov.sum()
        entries = entries + v.sum()
    return total / (covered.clamp_min(1.0) * d), covered / entries.clamp_min(1.0)


def sample_slots(emb: list[torch.Tensor], valid: list[torch.Tensor], m: int) -> torch.Tensor:
    """``m`` random valid position embeddings per volume from each view, as (len(emb) * m, B, D).

    Draws are with replacement, so a crop with fewer than ``m`` valid positions repeats some.
    """
    out = []
    for e, v in zip(emb, valid):
        p = v.float()
        p = torch.where(p.sum(1, keepdim=True) > 0, p, torch.ones_like(p))
        idx = torch.multinomial(p, m, replacement=True)  # (B, m)
        picked = e.gather(1, idx[..., None].expand(-1, -1, e.shape[-1]))
        out.append(picked.transpose(0, 1))
    return torch.cat(out).float()
