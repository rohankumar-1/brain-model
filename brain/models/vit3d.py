"""3D ViT encoder with physical-coordinate RoPE. Public layout is (B, C, X, Y, Z)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from brain.models.patch_embed import PatchEmbed3D, ScaleEmbed
from brain.models.projector import Projector
from brain.models.rope3d import RoPE3D

VIT_SIZES: dict[str, dict[str, int]] = {
    "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
    "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "test": {"embed_dim": 64, "depth": 2, "num_heads": 4},
}


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, rope: RoPE3D, qk_norm: bool = False):
        super().__init__()
        if dim % heads:
            raise ValueError(f"embed_dim {dim} is not divisible by heads {heads}")
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        head_dim = dim // heads
        self.q_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        coords_mm: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        q, k = self.rope.rotate(q, k, coords_mm)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, rope: RoPE3D, mlp_ratio: float = 4.0, qk_norm: bool = False):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, rope, qk_norm)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, coords_mm: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), coords_mm, attn_mask)
        return x + self.mlp(self.norm2(x))


def _key_mask(keep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive key mask, broadcast over queries. ``keep`` is (B, N) True = attend.

    A row with no kept keys attends everywhere so softmax stays defined.
    """
    allow = keep | ~keep.any(dim=1, keepdim=True)
    fill = torch.zeros(allow.size(0), 1, 1, allow.size(1), device=allow.device, dtype=dtype)
    return fill.masked_fill(~allow[:, None, None, :], torch.finfo(dtype).min)


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over tokens with ``mask`` True. Empty rows fall back to the full mean."""
    w = mask.to(dtype=tokens.dtype)
    if w.shape[:2] != tokens.shape[:2]:
        raise ValueError(f"pool mask {tuple(w.shape)} does not match tokens {tuple(tokens.shape[:2])}")
    empty = w.sum(dim=1) <= 0
    w = torch.where(empty.unsqueeze(1), torch.ones_like(w), w)
    return (tokens * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True)


def _pack_order(keep: torch.Tensor, max_keep: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable front-pack of True entries. Returns index order and a valid-slot mask.

    ``max_keep`` is the longest row. Pass it when the caller knows it, so no GPU sync is needed.
    """
    m = int(keep.sum(dim=1).max().item()) if max_keep is None else int(max_keep)
    m = max(m, 1)
    order = keep.long().argsort(dim=1, descending=True, stable=True)[:, :m]
    return order, keep.gather(1, order)


def pack_kept(
    tokens: torch.Tensor, coords: torch.Tensor, keep: torch.Tensor, max_keep: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack True positions to the front. Returns tokens, coords, pad, and index order.

    Order is increasing token index, stable across a matching ``scatter_kept``.
    """
    if keep.shape[:2] != tokens.shape[:2]:
        raise ValueError(f"keep {tuple(keep.shape)} does not match tokens {tuple(tokens.shape[:2])}")
    b, _, d = tokens.shape
    order, pad = _pack_order(keep, max_keep)
    m = order.shape[1]
    packed = tokens.gather(1, order.unsqueeze(-1).expand(b, m, d))
    packed_c = coords.gather(1, order.unsqueeze(-1).expand(b, m, 3))
    packed = packed * pad.unsqueeze(-1).to(packed.dtype)
    return packed, packed_c, pad, order


def scatter_kept(packed: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Inverse of ``pack_kept`` for the True positions. Dropped positions are zero."""
    b, n = keep.shape
    d = packed.size(-1)
    idx = (keep.long().cumsum(dim=1) - 1).clamp(0, packed.size(1) - 1)
    gathered = packed.gather(1, idx.unsqueeze(-1).expand(b, n, d))
    return torch.where(keep.unsqueeze(-1), gathered, torch.zeros_like(gathered))


class Predictor(nn.Module):
    """Fills in the masked tokens of a crop from the visible ones.

    Input is the encoder output at the visible positions; every other kept position gets one learned mask
    token. Position enters only through RoPE, so mask tokens differ by where they sit. Outputs, at every
    kept position, an embedding in the token-head space and optionally a few cell means of the patch's voxels.
    """

    def __init__(self, dim: int, heads: int, rope: RoPE3D, depth: int, latent_dim: int, pixel_dim: int = 0,
                 qk_norm: bool = False):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, rope, qk_norm=qk_norm) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.latent = nn.Linear(dim, latent_dim)
        self.pixel = nn.Linear(dim, pixel_dim) if pixel_dim else None

    def forward(
        self, ctx_tokens: torch.Tensor, ctx_keep: torch.Tensor, keep: torch.Tensor, coords_mm: torch.Tensor,
        plan: tuple[torch.Tensor, list[int], list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``ctx_tokens`` (R, N, D) is zero off ``ctx_keep``; ``keep`` (R, N) is every kept position and
        ``plan`` its length plan. Returns (R, N, latent_dim) and (R, N, pixel_dim) or None."""
        x = torch.where(ctx_keep.unsqueeze(-1), self.norm_in(ctx_tokens), self.mask_token.to(ctx_tokens.dtype))
        order, sizes, longest = plan
        order = order.to(x.device, non_blocking=True)
        out, start = [], 0
        for size, m in zip(sizes, longest):
            idx = order[start:start + size]
            start += size
            seq, seq_coords, pad, _ = pack_kept(x[idx], coords_mm[idx], keep[idx], max_keep=m)
            mask = None if seq.size(0) == 1 else _key_mask(pad, seq.dtype)
            for block in self.blocks:
                seq = block(seq, seq_coords, mask)
            out.append(scatter_kept(self.norm(seq), keep[idx]))
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        h = torch.cat(out)[inverse]
        return self.latent(h), (self.pixel(h) if self.pixel is not None else None)


class BrainEncoder(nn.Module):
    """3D encoder. Attention runs on the foreground tokens only."""

    def __init__(
        self,
        size: str = "small",
        patch_xyz: Sequence[int] = (16, 16, 8),
        in_chans: int = 1,
        proj_dim: int = 128,
        proj_hidden: int | None = None,
        qk_norm: bool = False,
        rope_base_mm: float = 100.0,
        checkpoint_blocks: bool = True,
        proj_norm: str = "batch",
        dense_dim: int = 0,
        dense_hidden: int = 512,
        predict_depth: int = 0,
        predict_pixel_dim: int = 0,
    ):
        super().__init__()
        if size not in VIT_SIZES:
            raise ValueError(f"unknown ViT size {size}. Choices: {', '.join(VIT_SIZES)}")
        spec = VIT_SIZES[size]
        dim, depth, heads = spec["embed_dim"], spec["depth"], spec["num_heads"]
        self.size, self.embed_dim, self.depth, self.proj_dim = size, dim, depth, int(proj_dim)
        self.patch_xyz = tuple(int(v) for v in patch_xyz)
        self.checkpoint_blocks = bool(checkpoint_blocks)
        head_dim = dim // heads
        self.rope = RoPE3D(head_dim, base_mm=rope_base_mm)
        self.patch_embed = PatchEmbed3D(in_chans, dim, self.patch_xyz)
        self.scale_embed = ScaleEmbed(dim)
        self.blocks = nn.ModuleList(Block(dim, heads, self.rope, qk_norm=qk_norm) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.projector = Projector(dim, proj_hidden or min(2048, dim * 4), self.proj_dim, norm=proj_norm)
        # Per-token head for the dense loss. No batch statistics: tokens of one volume are correlated.
        self.token_head = Projector(dim, int(dense_hidden), int(dense_dim), norm="layer") if dense_dim else None
        # Training-only: predicts masked tokens' embeddings (and voxel cell means). Not used for evaluation.
        if predict_depth and not dense_dim:
            raise ValueError("masked prediction targets the token head, so it needs dense_dim > 0")
        self.predictor = (
            Predictor(dim, heads, self.rope, int(predict_depth), int(dense_dim), int(predict_pixel_dim), qk_norm)
            if predict_depth else None
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _run_block(self, block: Block, x: torch.Tensor, coords: torch.Tensor, attn_mask):
        if self.checkpoint_blocks and self.training:
            return checkpoint(block, x, coords, attn_mask, use_reentrant=False)
        return block(x, coords, attn_mask)

    def _embed(self, view, coords_mm, patch_size_mm, patch_xyz, zyx):
        tokens, grid = self.patch_embed(view, patch_xyz, zyx=zyx)
        b, n, _ = tokens.shape
        if coords_mm.shape[:2] != (b, n) or coords_mm.size(-1) != 3:
            raise ValueError(f"coords_mm {tuple(coords_mm.shape)} does not match tokens {(b, n, 3)}")
        scale = self.scale_embed(patch_size_mm)
        return tokens + (scale.unsqueeze(1) if patch_size_mm.ndim == 2 else scale), grid

    def _run(self, x: torch.Tensor, coords: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        # One row has no pad, so the additive mask is skipped and flash attention stays on.
        mask = None if x.size(0) == 1 else _key_mask(pad, x.dtype)
        for block in self.blocks:
            x = self._run_block(block, x, coords, mask)
        return self.norm(x)

    def forward(
        self,
        view: torch.Tensor,
        coords_mm: torch.Tensor,
        patch_size_mm: torch.Tensor,
        patch_xyz: Sequence[int] | None = None,
        keep: torch.Tensor | None = None,
        taps: Sequence[int] = (),
    ) -> dict[str, torch.Tensor]:
        """Dense token features for evaluation. ``view`` is (B, C, X, Y, Z). Dropped tokens are zero.

        ``taps`` are block indices whose (pre-norm) outputs are also returned, as ``out["taps"]``.
        """
        tokens, grid = self._embed(view, coords_mm, patch_size_mm, patch_xyz, zyx=False)
        b, n, _ = tokens.shape
        if keep is None:
            keep = torch.ones(b, n, dtype=torch.bool, device=tokens.device)
        if keep.shape != (b, n):
            raise ValueError(f"keep {tuple(keep.shape)} does not match tokens {(b, n)}")
        seq, seq_coords, pad, _ = pack_kept(tokens, coords_mm, keep)
        tapped = []
        if taps:
            mask = None if seq.size(0) == 1 else _key_mask(pad, seq.dtype)
            x = seq
            for i, block in enumerate(self.blocks):
                x = self._run_block(block, x, seq_coords, mask)
                if i in taps:
                    tapped.append(scatter_kept(x, keep))
            x = self.norm(x)
        else:
            x = self._run(seq, seq_coords, pad)
        return {
            "tokens": scatter_kept(x, keep),
            "patch": tokens * keep.unsqueeze(-1).to(dtype=tokens.dtype),
            "taps": tapped,
            "coords_mm": coords_mm,
            "grid_zyx": torch.tensor(grid, device=view.device),
        }

    def encode_rows(
        self,
        view: torch.Tensor,
        coords_mm: torch.Tensor,
        patch_size_mm: torch.Tensor,
        keep: torch.Tensor,
        plan: tuple[torch.Tensor, list[int], list[int]],
        patch_xyz: Sequence[int] | None = None,
        tokens: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pooled embedding of each row, and optionally its dense tokens, in input order.

        ``view`` is (R, C, Z, Y, X). Pooled is the mean over kept tokens, (R, D). Dense tokens are
        (R, N, D) with dropped tokens zero. ``plan`` is (order, sizes, longest): rows are sorted by kept
        count and cut into buckets, so a row is padded to its bucket's longest row, not the batch's. Rows
        never interact, so this changes cost, not values. ``longest`` comes from the CPU: no GPU sync.
        """
        emb, _ = self._embed(view, coords_mm, patch_size_mm, patch_xyz, zyx=True)
        order, sizes, longest = plan
        order = order.to(emb.device, non_blocking=True)
        pooled, dense, start = [], [], 0
        for size, m in zip(sizes, longest):
            idx = order[start:start + size]
            start += size
            seq, seq_coords, pad, _ = pack_kept(emb[idx], coords_mm[idx], keep[idx], max_keep=m)
            x = self._run(seq, seq_coords, pad)
            pooled.append(_masked_mean(x, pad))
            if tokens:
                dense.append(scatter_kept(x, keep[idx]))
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        return torch.cat(pooled)[inverse], (torch.cat(dense)[inverse] if tokens else None)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
