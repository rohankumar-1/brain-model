"""Turn loader chunks into one training batch on the device: windows, degradations, length buckets."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from brain.common.windowing import floors
from brain.data.augment import appearance
from brain.data.crops3d import raw_views
from brain.models.window_bank import bank_cfg


def length_plan(counts: torch.Tensor, buckets: int) -> tuple[torch.Tensor, list[int], list[int]]:
    """Sort rows by kept-token count and cut them into buckets. ``counts`` is a CPU (R,) tensor.

    Returns the row order, the bucket sizes, and the longest row of each bucket.
    """
    order = counts.argsort(descending=True, stable=True)
    ranked = counts[order]
    r = int(counts.numel())
    n = max(1, min(int(buckets), r))
    base, extra = divmod(r, n)
    sizes = [base + (1 if i < extra else 0) for i in range(n)]
    starts = [sum(sizes[:i]) for i in range(n)]
    return order, sizes, [int(ranked[s]) for s in starts]


def block_mask(rows: int, grid_zyx: tuple[int, int, int], blocks: int, lo: int, hi: int) -> torch.Tensor:
    """(rows, Z*Y*X) bool, True inside the union of ``blocks`` random cuboids per row.

    Each side of a cuboid is drawn from [lo, hi] tokens, clamped to the grid. CPU, so the caller can plan
    length buckets without reading the GPU.
    """
    out = torch.zeros(rows, *grid_zyx, dtype=torch.bool)
    axes = [torch.arange(g) for g in grid_zyx]
    for _ in range(int(blocks)):
        sel = []
        for a, g in zip(axes, grid_zyx):
            side = torch.randint(min(int(lo), g), min(int(hi), g) + 1, (rows,))
            org = (torch.rand(rows) * (g - side + 1)).long().minimum(g - side)
            sel.append((a[None] >= org[:, None]) & (a[None] < (org + side)[:, None]))
        out |= sel[0][:, :, None, None] & sel[1][:, None, :, None] & sel[2][:, None, None, :]
    return out.reshape(rows, -1)


_ROW_KEYS = ("view", "coords", "fg", "patch_mm", "parent", "pos")


def _warp(x: torch.Tensor, grid: torch.Tensor, mode: str) -> torch.Tensor:
    """``grid_sample`` in the input's dtype (float16 on the GPU); float32 if this build lacks the half kernel."""
    try:
        return F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)
    except RuntimeError:
        return F.grid_sample(x.float(), grid.float(), mode=mode, padding_mode="zeros",
                             align_corners=False).to(x.dtype)


def sample_views(chunks: list[dict], groups, device: torch.device) -> dict[str, torch.Tensor]:
    """Geometric-augmentation batches (``aug.geometric``): sample every view from its whole volume on the GPU.

    Each view is a trilinear ``grid_sample`` through its own rotation / scale / flip (``theta``), outside the volume
    filled with the modality's air value. Views with skull dropout are cut to the (nearest-sampled) brain mask."""
    vol = torch.cat([c["volume_full"].to(device, non_blocking=True) for c in chunks])          # (B, 1, Z, Y, X)
    if device.type != "cuda":
        vol = vol.float()                                       # CPU grid_sample has no float16 kernel
    mod = torch.cat([c["modality"].to(device, non_blocking=True) for c in chunks])
    floor = floors(mod).to(vol.dtype).view(-1, 1, 1, 1, 1)
    shifted = vol - floor                                       # zero padding outside the volume == air
    del vol                                                     # only the shifted copy is needed from here
    brain = torch.cat([c["brain_full"].to(device, non_blocking=True) for c in chunks]).to(shifted.dtype) \
        if "brain_full" in chunks[0] else None
    b, out = shifted.shape[0], {}
    for g in groups:
        theta = torch.cat([c[f"{g}_theta"].to(device, non_blocking=True) for c in chunks], dim=1)   # (V, B, 3, 4)
        drop = torch.cat([c[f"{g}_drop"] for c in chunks], dim=1)                                   # (V, B) CPU
        shape = (b, 1, *chunks[0]["vshape"][g])
        views = []
        for v in range(theta.shape[0]):
            grid = F.affine_grid(theta[v], shape, align_corners=False).to(shifted.dtype)
            x = _warp(shifted, grid, "bilinear") + floor
            if brain is not None and bool(drop[v].any()):
                inside = _warp(brain, grid, "nearest") > 0.5
                cut = drop[v].to(device).view(-1, 1, 1, 1, 1)
                x = torch.where(cut & ~inside, floor.expand_as(x), x)
            views.append(x)
        out[f"{g}_view"] = torch.stack(views)
    return out


def native_aug(cfg: dict) -> dict | None:
    """The SSL appearance config when the encoder has a learned window bank (``model.window_bank``), else None.

    With the bank the encoder windows inside the model (``ViewNet`` applies ``encoder.window_bank`` per view), so the
    views stay in native units (CT HU, MRI robust z; one channel) and every appearance augmentation runs there:
    ``window.mode: native`` (``augment._appearance_native``), exactly the multitask path's recipe. The SSL ``aug``
    keys keep their values; ``ct_physical`` / ``kernel`` (CT's physical augmentations, which the canonical-window SSL
    recipe never had) come from ``aug.appearance`` when the SSL block does not set them. The contrast re-rendering
    (``curve``, ``histmatch``) and the window jitter exist only for windowed views, so they are off and
    ``tie_views`` has nothing to tie."""
    if bank_cfg(cfg.get("model")) is None:
        return None
    aug = dict(cfg["aug"])
    for k in ("ct_physical", "kernel"):
        if k not in aug and k in (aug.get("appearance") or {}):
            aug[k] = aug["appearance"][k]
    aug["window"] = {"mode": "native"}
    return aug


def prepare(chunks: list[dict], cfg: dict, device: torch.device, buckets: int = 4) -> dict:
    """Join worker chunks on ``device``, draw appearance there, and plan the length buckets.

    The large crop tensors are never concatenated on the CPU: each chunk is copied to the GPU and joined
    there. Views stay in (Z, Y, X) order, which is what the patch embed convolves in. The window pair is
    one crop that gets two independent appearances.

    With a window bank (``native_aug``) the views are native units for the encoder's bank, and the window pair is
    left out (it asks two windowings of one crop to agree, which a learned bank makes meaningless): a config that
    still weights it (``loss.win_weight`` > 0) is refused.
    """
    first, aug = chunks[0], cfg["aug"]
    nat = native_aug(cfg)
    if nat is not None:
        if float(cfg["loss"].get("win_weight", 1.0)) > 0:
            raise ValueError("model.window_bank: the SSL window pair (loss.win_weight > 0) needs fixed windows; "
                             "set loss.win_weight: 0")
        aug = nat
    d: dict = {k: v for k, v in first.items() if not torch.is_tensor(v)}
    if nat is not None:
        d["groups"] = [g for g in d["groups"] if g != "win"]
    d["modality"] = torch.cat([c["modality"].to(device, non_blocking=True) for c in chunks])
    mod = d["modality"]
    d["thick_mm"] = torch.cat([c["thick_mm"].to(device, non_blocking=True) for c in chunks])
    d["thick_axis"] = torch.cat([c["thick_axis"].to(device, non_blocking=True) for c in chunks])
    thick = (d["thick_mm"], d["thick_axis"])
    b = int(mod.shape[0])
    d["contrast_id"] = torch.cat([c["contrast_id"].to(device, non_blocking=True) for c in chunks])
    d["tissue_q"] = torch.cat([c["tissue_q"].to(device, non_blocking=True) for c in chunks])
    # the scan's modality / contrast for the encoder (model.meta), replaced by "unknown" (0) with probability drop
    d["meta"] = torch.cat([c.get("meta_id", torch.zeros(len(c["modality"]), dtype=torch.long)) for c in chunks]).to(device)
    drop = float(((cfg.get("model") or {}).get("meta") or {}).get("drop", 0.0))
    if drop > 0:
        d["meta"] = torch.where(torch.rand(b, device=device) < drop, torch.zeros_like(d["meta"]), d["meta"])
    cid, vid, tq = d["contrast_id"], torch.arange(b, device=device), d["tissue_q"]
    d["plans"] = {}
    # aug.tie_views: every view of a volume (all groups but the window pair) draws the same contrast, so the SSL
    # agreement terms ask for invariance to crop / degradation, not to contrast (R15, 2026-11-06)
    tie = int(torch.randint(0, 2**30, (1,))) if aug.get("tie_views") and nat is None else None
    if "volume_full" in first:
        d.update(sample_views(chunks, d["groups"], device))
    for g in d["groups"]:
        for k in _ROW_KEYS:
            if k == "view" and f"{g}_view" in d:
                continue
            d[f"{g}_{k}"] = torch.cat([c[f"{g}_{k}"].to(device, non_blocking=True) for c in chunks], dim=1)
        counts = torch.cat([c[f"{g}_fg"] for c in chunks], dim=1).sum(-1)  # (V, B) on the CPU
        view = d[f"{g}_view"]
        if g == "win":
            crop = view[0]
            d["win_view"] = torch.stack([appearance(crop, mod, aug, thick, vid, cid, tq),
                                         appearance(crop, mod, aug, thick, vid, cid, tq)])
            for k in ("coords", "fg", "patch_mm", "parent", "pos"):
                d[f"win_{k}"] = d[f"win_{k}"].expand(2, *d[f"win_{k}"].shape[1:])
            counts = counts.expand(2, -1)
        elif tie is not None:     # aug.tie_views: one contrast per volume across all its views (R15)
            d[f"{g}_view"] = torch.stack([appearance(view[i], mod, aug, thick, vid, cid, tq, tie_seed=tie)
                                          for i in range(view.shape[0])])
        else:
            n = view.shape[0]
            flat = view.reshape(n * b, *view.shape[2:])
            d[f"{g}_view"] = appearance(flat, mod.repeat(n), aug, (thick[0].repeat(n), thick[1].repeat(n)),
                                        vid.repeat(n), cid.repeat(n), tq.repeat(n, 1)).view(n, b, -1, *flat.shape[2:])   # -1: role channels
        d["plans"][g] = length_plan(counts.flatten(), buckets)
    pr = (cfg["loss"].get("predict") or {})
    if float(pr.get("weight", 0.0)) > 0:
        _mask_group(d, chunks, device, buckets, group="global", ctx_key="ctx_keep", tgt_key="tgt_mask",
                    plan="ctx", blocks=int(pr.get("blocks", 3)), block_tokens=pr.get("block_tokens", (3, 6)))
    return d


def _mask_group(d: dict, chunks: list[dict], device: torch.device, buckets: int, *,
                group: str, ctx_key: str, tgt_key: str, plan: str, blocks: int, block_tokens) -> None:
    """Hide ``blocks`` random cuboids of every crop in ``group``. ``ctx_key`` is what the encoder sees, ``tgt_key``
    what is predicted."""
    fg = torch.cat([c[f"{group}_fg"] for c in chunks], dim=1)  # (V, B, N) on the CPU
    v, b, n = fg.shape
    px, py, pz = (int(t) for t in chunks[0]["patch"][group])
    vshape = chunks[0]["vshape"][group] if "vshape" in chunks[0] else chunks[0][f"{group}_view"].shape[-3:]
    zyx = tuple(int(s) // p for s, p in zip(vshape, (pz, py, px)))
    lo, hi = block_tokens
    hidden = block_mask(v * b, zyx, int(blocks), int(lo), int(hi)) if int(blocks) > 0 \
        else torch.zeros(v * b, n, dtype=torch.bool)
    flat = fg.reshape(v * b, n)
    ctx, tgt = flat & ~hidden, flat & hidden
    d[ctx_key], d[tgt_key] = (t.view(v, b, n).to(device, non_blocking=True) for t in (ctx, tgt))
    d["plans"][plan] = length_plan(ctx.sum(-1), buckets)


def lejepa_batch(batch: dict, cfg: dict, device: torch.device, gen: torch.Generator | None = None, buckets: int = 4) -> dict:
    """Crop and window a (B, 1, X, Y, Z) batch. For tests and synthetic runs; training crops in the loader."""
    zyx = dict(batch)
    zyx["volume"] = batch["volume"].permute(0, 1, 4, 3, 2)
    return prepare([raw_views(zyx, cfg, gen)], cfg, device, buckets)
