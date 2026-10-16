"""Turn loader chunks into one training batch on the device: windows, degradations, length buckets."""

from __future__ import annotations

import torch

from brainmodel.data.augment import appearance
from brainmodel.data.crops import raw_views


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


_ROW_KEYS = ("view", "coords", "fg", "patch_mm", "parent", "pos", "tok_pos")


def prepare(chunks: list[dict], cfg: dict, device: torch.device, buckets: int = 4) -> dict:
    """Join worker chunks on ``device``, draw appearance there, and plan the length buckets.

    The large crop tensors are never concatenated on the CPU: each chunk is copied to the GPU and joined
    there. Views stay in (Z, Y, X) order, which is what the patch embed convolves in. The window pair is
    one crop that gets two independent appearances.
    """
    first, aug = chunks[0], cfg["aug"]
    d: dict = {k: v for k, v in first.items() if not torch.is_tensor(v)}
    d["modality"] = torch.cat([c["modality"].to(device, non_blocking=True) for c in chunks])
    mod = d["modality"]
    b = int(mod.shape[0])
    d["plans"] = {}
    for g in first["groups"]:
        for k in _ROW_KEYS:
            d[f"{g}_{k}"] = torch.cat([c[f"{g}_{k}"].to(device, non_blocking=True) for c in chunks], dim=1)
        counts = torch.cat([c[f"{g}_fg"] for c in chunks], dim=1).sum(-1)  # (V, B) on the CPU
        view = d[f"{g}_view"]
        if g == "win":
            crop = view[0]
            d["win_view"] = torch.stack([appearance(crop, mod, aug), appearance(crop, mod, aug)])
            for k in ("coords", "fg", "patch_mm", "parent", "pos"):
                d[f"win_{k}"] = d[f"win_{k}"].expand(2, *d[f"win_{k}"].shape[1:])
            counts = counts.expand(2, -1)
        else:
            n = view.shape[0]
            flat = view.reshape(n * b, *view.shape[2:])
            d[f"{g}_view"] = appearance(flat, mod.repeat(n), aug).view(n, b, *flat.shape[1:])
        d["plans"][g] = length_plan(counts.flatten(), buckets)
    pr = (cfg["loss"].get("predict") or {})
    if float(pr.get("weight", 0.0)) > 0:
        _mask_group(d, chunks, device, buckets, group="global", ctx_key="ctx_keep", tgt_key="tgt_mask",
                    plan="ctx", blocks=int(pr.get("blocks", 3)), block_tokens=pr.get("block_tokens", (3, 6)))
        if float(pr.get("fine_weight", 0.0)) > 0 and "fine" in d["groups"]:
            _mask_group(d, chunks, device, buckets, group="fine", ctx_key="fine_ctx_keep",
                        tgt_key="fine_tgt_mask", plan="fine_ctx",
                        blocks=int(pr.get("fine_blocks", 2)),
                        block_tokens=pr.get("fine_block_tokens", (2, 3)))
    return d


def _mask_group(d: dict, chunks: list[dict], device: torch.device, buckets: int, *,
                group: str, ctx_key: str, tgt_key: str, plan: str, blocks: int, block_tokens) -> None:
    """Hide blocks of every crop in ``group``. ``ctx_key`` is what the encoder sees, ``tgt_key`` what is predicted."""
    fg = torch.cat([c[f"{group}_fg"] for c in chunks], dim=1)  # (V, B, N) on the CPU
    v, b, n = fg.shape
    px, py, pz = (int(t) for t in chunks[0]["patch"][group])
    zyx = tuple(int(s) // p for s, p in zip(chunks[0][f"{group}_view"].shape[-3:], (pz, py, px)))
    lo, hi = block_tokens
    hidden = block_mask(v * b, zyx, int(blocks), int(lo), int(hi))
    flat = fg.reshape(v * b, n)
    ctx, tgt = flat & ~hidden, flat & hidden
    d[ctx_key], d[tgt_key] = (t.view(v, b, n).to(device, non_blocking=True) for t in (ctx, tgt))
    d["plans"][plan] = length_plan(ctx.sum(-1), buckets)


def lejepa_batch(batch: dict, cfg: dict, device: torch.device, gen: torch.Generator | None = None, buckets: int = 4) -> dict:
    """Crop and window a (B, 1, X, Y, Z) batch. For tests and synthetic runs; training crops in the loader."""
    zyx = dict(batch)
    zyx["volume"] = batch["volume"].permute(0, 1, 4, 3, 2)
    return prepare([raw_views(zyx, cfg, gen)], cfg, device, buckets)
