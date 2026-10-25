"""Learning-rate schedule and optimizer parameter groups shared by all trainers."""

from __future__ import annotations

import math

import torch

NO_DECAY_SUFFIXES = ("bias", "pos_embed", "cls_token", "register_tokens", "mask_token", "prefix")


def lr_at(step: int, total: int, peak: float, warmup: int, floor: float = 0.01, kind: str = "cosine",
          decay_steps: int | None = None) -> float:
    """Linear warmup, then ``cosine`` decay to ``floor * peak`` over the rest of ``total``, or ``wsd``
    (warmup-stable-decay): constant at ``peak`` until the last ``decay_steps`` (default 20% of ``total``), then down to
    ``floor * peak`` along 1 - sqrt (the cooldown of Hagele et al. 2024). Under WSD a run still in its constant phase
    can be extended by raising ``total``, and any snapshot can be cooled down on its own."""
    if step < warmup:
        return peak * (step + 1) / warmup
    if kind == "wsd":
        d = int(decay_steps) if decay_steps else round(0.2 * total)
        t = min(max(step - (total - d), 0) / max(1, d), 1.0)
        return peak * (floor + (1 - floor) * (1 - math.sqrt(t)))
    if kind != "cosine":
        raise ValueError(f"optim.schedule must be cosine or wsd, not {kind!r}")
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def param_groups(model: torch.nn.Module, wd: float, layer_decay: float | None = None,
                 lr: float = 1.0) -> list[dict]:
    """Decay matrices only; biases, norms and learned tokens/positions get no weight decay.

    With ``layer_decay`` the ViT backbone's blocks get geometrically smaller LR scales (deepest = 1).
    Every group carries ``lr_scale`` so the schedule sets ``group["lr"] = lr_t * lr_scale``.
    """
    n_blocks = getattr(getattr(model, "backbone", None), "depth", 0)
    buckets: dict[tuple[float, float], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        scale = 1.0
        if layer_decay is not None and n_blocks:
            if name.startswith("backbone.blocks."):
                scale = layer_decay ** (n_blocks - 1 - int(name.split(".")[2]))
            elif name.startswith("backbone."):
                scale = layer_decay ** n_blocks
        skip = p.ndim < 2 or name.endswith(NO_DECAY_SUFFIXES)
        buckets.setdefault((scale, 0.0 if skip else wd), []).append(p)
    return [{"params": ps, "lr": lr * s, "weight_decay": d, "lr_scale": s} for (s, d), ps in buckets.items()]


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr * g.get("lr_scale", 1.0)


def finite_params(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(p).all() for p in model.parameters() if p.is_floating_point())
