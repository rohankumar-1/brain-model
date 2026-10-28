"""3D ViT encoder with physical-coordinate RoPE. Public layout is (B, C, X, Y, Z)."""

from __future__ import annotations

import copy
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
        n_prefix: int = 0,
    ) -> torch.Tensor:
        """``x`` is (B, n_prefix + N, C); ``coords_mm`` (B, N, 3) are the patch tokens' centers. Prefix tokens
        (CLS, registers) have no position: RoPE rotates only the patch tokens."""
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        if n_prefix:
            qr, kr = self.rope.rotate(q[:, :, n_prefix:], k[:, :, n_prefix:], coords_mm)
            q, k = torch.cat([q[:, :, :n_prefix], qr], dim=2), torch.cat([k[:, :, :n_prefix], kr], dim=2)
        else:
            q, k = self.rope.rotate(q, k, coords_mm)
        if attn_mask is not None and attn_mask.dtype not in (torch.bool, q.dtype):
            attn_mask = attn_mask.to(q.dtype)                   # additive mask built in the residual dtype
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class SwiGLU(nn.Module):
    """Gated MLP (Shazeer 2020) as in Primus / LLaMA; hidden width 2/3 of the GELU MLP's for the same parameters."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class MoEMLP(nn.Module):
    """Mixture-of-experts MLP (top-k routing, Switch load balancing) for the top blocks (OMNI, arXiv 2610.12574).

    Built from a block's MLP: every expert starts as a copy of it and the top-k gates sum to 1, so a trained block
    computes exactly what it did; the experts then diverge on the tokens they are sent. ``aux`` is the last forward's
    load-balancing loss, ``n * sum_e f_e P_e`` (1 when tokens and router mass are spread evenly)."""

    def __init__(self, mlp: nn.Module, dim: int, n_experts: int = 5, top_k: int = 2):
        super().__init__()
        self.experts = nn.ModuleList(copy.deepcopy(mlp) for _ in range(int(n_experts)))
        self.router = nn.Linear(dim, int(n_experts), bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.top_k = int(top_k)
        self.aux: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.reshape(-1, x.shape[-1])
        probs = torch.softmax(self.router(h).float(), dim=-1)                # (M, E)
        w, idx = probs.topk(self.top_k, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        out = None
        for e, expert in enumerate(self.experts):
            rows, slot = (idx == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            y = expert(h[rows])
            y = y * w[rows, slot].unsqueeze(-1).to(y.dtype)
            if out is None:
                out = torch.zeros(h.shape[0], y.shape[-1], dtype=y.dtype, device=h.device)
            out = out.index_add(0, rows, y.to(out.dtype))
        if out is None:
            out = torch.zeros_like(h)
        n = len(self.experts)
        frac = torch.zeros(n, device=h.device).index_add(0, idx.flatten(), torch.ones(idx.numel(), device=h.device))
        self.aux = n * (frac / idx.numel() * probs.mean(0)).sum()
        return out.reshape(*x.shape[:-1], out.shape[-1])


def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Stochastic depth: drop a residual branch for whole rows (samples) with probability ``p``."""
    if p <= 0.0 or not training:
        return x
    keep = torch.rand(x.shape[0], *([1] * (x.ndim - 1)), device=x.device, dtype=x.dtype) >= p
    return x * keep / (1.0 - p)


class Block(nn.Module):
    """Pre-norm transformer block. Defaults are the original (GELU MLP, no LayerScale, no drop-path); ``mlp="swiglu"``,
    ``layer_scale`` (initial per-channel residual scale, e.g. 1e-4) and ``drop_path`` give the Primus-style block."""

    def __init__(self, dim: int, heads: int, rope: RoPE3D, mlp_ratio: float = 4.0, qk_norm: bool = False,
                 mlp: str = "gelu", layer_scale: float | None = None, drop_path: float = 0.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, rope, qk_norm)
        hidden = int(dim * mlp_ratio)
        if mlp == "swiglu":
            self.mlp = SwiGLU(dim, int(round(hidden * 2 / 3 / 8)) * 8)
        else:
            self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ls1 = nn.Parameter(torch.full((dim,), float(layer_scale))) if layer_scale else None
        self.ls2 = nn.Parameter(torch.full((dim,), float(layer_scale))) if layer_scale else None
        self.drop_path = float(drop_path)

    def forward(self, x: torch.Tensor, coords_mm: torch.Tensor, attn_mask: torch.Tensor | None = None,
                n_prefix: int = 0) -> torch.Tensor:
        a = self.attn(self.norm1(x), coords_mm, attn_mask, n_prefix)
        x = x + drop_path(a * self.ls1 if self.ls1 is not None else a, self.drop_path, self.training)
        m = self.mlp(self.norm2(x))
        return x + drop_path(m * self.ls2 if self.ls2 is not None else m, self.drop_path, self.training)


def _key_mask(keep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive key mask, broadcast over queries. ``keep`` is (B, N) True = attend.

    A row with no kept keys attends everywhere so softmax stays defined.
    """
    allow = keep | ~keep.any(dim=1, keepdim=True)
    fill = torch.zeros(allow.size(0), 1, 1, allow.size(1), device=allow.device, dtype=dtype)
    return fill.masked_fill(~allow[:, None, None, :], torch.finfo(dtype).min)


def _task_mask(pad: torch.Tensor, n_prefix: int, n_task: int, write: torch.Tensor | None,
               write_bias: float = 0.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Attention mask (B, 1, L, L) for a sequence [prefix | task tokens | patches] (OMNI).

    Every query attends the prefix and the kept patch tokens (``pad``). A task token also attends itself, never another
    task token. The other tokens attend the task tokens marked ``write``; read-only ones (a few-shot task's) change
    nothing else in the sequence. A query left with no key attends everything, so softmax stays defined.
    Boolean (True = attend), or with ``write_bias`` > 0 additive in ``dtype``: the other tokens' logits on the written
    task tokens are lowered by ``write_bias``, so a warm-started encoder can open its write path gradually (the new
    tokens otherwise shift its features ~40% before any training: pre-attention LayerNorm gives them full scale)."""
    b, n = pad.shape
    s, e = n_prefix, n_prefix + n_task
    allow = pad[:, None, :].expand(b, n, n).clone()
    allow[:, s:e, s:e] = torch.eye(n_task, dtype=torch.bool, device=pad.device)
    w = torch.ones(n_task, dtype=torch.bool, device=pad.device) if write is None else write.to(pad.device).bool()
    allow[:, :s, s:e] = w
    allow[:, e:, s:e] = w
    allow |= ~allow.any(dim=-1, keepdim=True)
    if write_bias <= 0:
        return allow[:, None]
    mask = torch.zeros(b, 1, n, n, dtype=dtype, device=pad.device)
    mask[:, 0, :s, s:e] = -float(write_bias)
    mask[:, 0, e:, s:e] = -float(write_bias)
    return mask.masked_fill(~allow[:, None], torch.finfo(dtype).min)


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


def widen_patch_embed(state: dict, in_chans: int) -> dict:
    """A checkpoint's state for an encoder with ``in_chans`` inputs: a one-channel patch embedding is copied into
    channel 1 and the new channels start at zero, so the widened encoder computes exactly what the checkpoint did on
    channel 1 (``windowing.ROLES``: the canonical window) until training uses the others."""
    key = next((k for k in state if k.endswith("patch_embed.proj.weight")), None)
    if key is None or state[key].shape[1] == in_chans:
        return state
    w = state[key]
    wide = w.new_zeros(w.shape[0], in_chans, *w.shape[2:])
    wide[:, :w.shape[1]] = w
    return {**state, key: wide}


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
        n_cls: int = 0,
        n_registers: int = 0,
        mlp: str = "gelu",
        layer_scale: float | None = None,
        drop_path: float = 0.0,
        rope_min_period_mm: float | None = None,
        rope_fp32: bool = False,
        fp32_residual: bool = False,
        n_meta: int = 0,
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
        self.rope = RoPE3D(head_dim, base_mm=rope_base_mm, min_period_mm=rope_min_period_mm, fp32=rope_fp32)
        # fp32 residual stream under autocast (the block outputs stay bf16 and are added to it); without it the stream
        # is bf16 end to end and small updates round away (checkpoints from before 2026-10-25)
        self.fp32_residual = bool(fp32_residual)
        self.patch_embed = PatchEmbed3D(in_chans, dim, self.patch_xyz)
        self.scale_embed = ScaleEmbed(dim)
        # stochastic depth rises linearly with depth, as in DeiT / Primus
        self.blocks = nn.ModuleList(
            Block(dim, heads, self.rope, qk_norm=qk_norm, mlp=mlp, layer_scale=layer_scale,
                  drop_path=float(drop_path) * i / max(depth - 1, 1))
            for i in range(depth))
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
        # Prefix tokens prepended to every sequence: an optional CLS (the pooled embedding when present) and
        # register tokens (attention scratch space, discarded). They carry no position and are never gated out.
        self.n_cls, self.n_registers = int(n_cls), int(n_registers)
        if self.n_cls not in (0, 1):
            raise ValueError(f"n_cls must be 0 or 1, got {n_cls}")
        self.prefix = nn.Parameter(torch.zeros(1, self.n_prefix, dim)) if self.n_prefix else None
        self.apply(self._init_weights)
        if self.prefix is not None:
            nn.init.trunc_normal_(self.prefix, std=0.02)
        self.meta_emb: nn.Embedding | None = None
        self.n_meta = 0
        # logit offset on the other tokens' attention to task tokens (``_task_mask``); the trainer ramps it to 0
        self.task_write_bias = 0.0
        if n_meta:
            self.enable_meta(n_meta)

    def enable_meta(self, n_meta: int) -> None:
        """Scan metadata (``brain.data.volumes.META``: modality / contrast, 0 = unknown) as a learned embedding added
        to every patch token. Zero-initialised, so adding it to a trained encoder changes nothing until it learns."""
        self.n_meta = int(n_meta)
        self.meta_emb = nn.Embedding(self.n_meta, self.embed_dim)
        nn.init.zeros_(self.meta_emb.weight)
        p = next(self.parameters())
        self.meta_emb.to(device=p.device)

    def enable_moe(self, blocks: int, n_experts: int = 5, top_k: int = 2) -> None:
        """Mixture-of-experts MLPs in the last ``blocks`` blocks (``MoEMLP``), each expert a copy of the block's MLP:
        on a trained encoder the outputs are unchanged until the experts diverge. Blocks already converted stay."""
        for blk in self.blocks[len(self.blocks) - int(blocks):]:
            if not isinstance(blk.mlp, MoEMLP):
                p = next(blk.mlp.parameters())
                blk.mlp = MoEMLP(blk.mlp, self.embed_dim, n_experts, top_k).to(device=p.device, dtype=p.dtype)

    @property
    def moe_blocks(self) -> list[MoEMLP]:
        return [blk.mlp for blk in self.blocks if isinstance(blk.mlp, MoEMLP)]

    def moe_aux(self) -> torch.Tensor:
        """Mean load-balancing loss of the MoE blocks over the last forward (0 without MoE)."""
        aux = [m.aux for m in self.moe_blocks if m.aux is not None]
        if not aux:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack(aux).mean()

    @property
    def n_prefix(self) -> int:
        return self.n_cls + self.n_registers

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _run_block(self, block: Block, x: torch.Tensor, coords: torch.Tensor, attn_mask, n_prefix: int = 0):
        if self.checkpoint_blocks and self.training:
            return checkpoint(block, x, coords, attn_mask, n_prefix, use_reentrant=False)
        return block(x, coords, attn_mask, n_prefix)

    def _embed(self, view, coords_mm, patch_size_mm, patch_xyz, zyx, meta: torch.Tensor | None = None):
        tokens, grid = self.patch_embed(view, patch_xyz, zyx=zyx)
        b, n, _ = tokens.shape
        if coords_mm.shape[:2] != (b, n) or coords_mm.size(-1) != 3:
            raise ValueError(f"coords_mm {tuple(coords_mm.shape)} does not match tokens {(b, n, 3)}")
        return self._add_scale_meta(tokens, patch_size_mm, meta), grid

    def _add_scale_meta(self, tokens: torch.Tensor, patch_size_mm: torch.Tensor, meta: torch.Tensor | None):
        scale = self.scale_embed(patch_size_mm)
        if self.fp32_residual:
            tokens, scale = tokens.float(), scale.float()
        tokens = tokens + (scale.unsqueeze(1) if patch_size_mm.ndim == 2 else scale)
        if self.meta_emb is not None and meta is not None:
            tokens = tokens + self.meta_emb(meta.to(tokens.device).long()).unsqueeze(1).to(tokens.dtype)
        return tokens

    def _refined(self, view, coords_mm, patch_size_mm, patch_xyz, keep, refine, zyx, meta=None):
        """The mixed sequence of a view whose ``refine`` cells (R, N) are split into 8 half-size tokens: tokens
        (R, 9N, D) laid out [coarse | 8 children per coarse cell], their coords and keep mask (a refined cell's
        coarse token is dropped, its children kept), and the coarse grid. ``brain.models.refine``."""
        from brain.models.refine import axis_signs, child_index

        coarse, grid = self._embed(view, coords_mm, patch_size_mm, patch_xyz, zyx, meta)
        if patch_size_mm.ndim != 2:
            raise ValueError("refinement needs one patch size per row, (R, 3)")
        px, py, pz = self.patch_xyz if patch_xyz is None else (int(v) for v in patch_xyz)
        if px % 2 or py % 2 or pz % 2:
            raise ValueError(f"patch {(px, py, pz)} cannot be halved")
        fine, _ = self.patch_embed(view, (px // 2, py // 2, pz // 2), zyx=zyx)
        fine = self._add_scale_meta(fine, patch_size_mm / 2, meta)
        r, n, d = coarse.shape
        idx, off = child_index(grid, coarse.device)                                     # (N, 8), (8, 3)
        children = fine[:, idx.reshape(-1)]                                             # (R, 8N, D), parent-major
        step = (patch_size_mm.float() / 4) * axis_signs(coords_mm, grid)                # (R, 3): a quarter patch
        child_coords = coords_mm.unsqueeze(2) + off.view(1, 1, 8, 3) * step.view(r, 1, 1, 3)
        tokens = torch.cat([coarse, children.to(coarse.dtype)], 1)
        coords = torch.cat([coords_mm, child_coords.reshape(r, 8 * n, 3).to(coords_mm.dtype)], 1)
        keep_all = torch.cat([keep & ~refine, (keep & refine).repeat_interleave(8, dim=1)], 1)
        return tokens, coords, keep_all, grid

    @staticmethod
    def _to_coarse(x_all: torch.Tensor, refine: torch.Tensor) -> torch.Tensor:
        """(R, 9N, D) outputs of a mixed sequence -> (R, N, D) on the coarse grid: a refined cell is its children's
        mean."""
        r, m, d = x_all.shape
        n = m // 9
        children = x_all[:, n:].reshape(r, n, 8, d).mean(2)
        return torch.where(refine.unsqueeze(-1), children, x_all[:, :n])

    def _pool_coarse(self, prefix: torch.Tensor | None, coarse: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """``_pool`` for a refined view: the CLS, else the mean over kept coarse cells (each cell once)."""
        return prefix[:, 0] if self.n_cls else _masked_mean(coarse, keep)

    def _run(self, x: torch.Tensor, coords: torch.Tensor, pad: torch.Tensor, taps: Sequence[int] = (),
             task_tokens: torch.Tensor | None = None, task_write: torch.Tensor | None = None,
             ) -> tuple[torch.Tensor | None, torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
        """Blocks over packed patch tokens (with the prefix, then any task tokens, in front). Returns the normed prefix
        outputs (or None), the normed patch tokens, the pre-norm patch tokens after each block in ``taps``, and the
        normed task-token outputs (or None). ``task_tokens`` (T, D) or (B, T, D) attend as in ``_task_mask``."""
        p, b = self.n_prefix, x.size(0)
        t = 0 if task_tokens is None else int(task_tokens.shape[-2])
        front = []
        if p:
            front.append(self.prefix.to(x.dtype).expand(b, -1, -1))
        if t:
            front.append(task_tokens.to(x.dtype).expand(b, -1, -1) if task_tokens.dim() == 2 else
                         task_tokens.to(x.dtype))
        if front:
            x = torch.cat([*front, x], dim=1)
            pad = torch.cat([torch.ones(b, p + t, dtype=torch.bool, device=pad.device), pad], dim=1)
        if t:
            mask = _task_mask(pad, p, t, task_write, self.task_write_bias, x.dtype)
        else:
            # One row has no pad, so the additive mask is skipped and flash attention stays on.
            mask = None if b == 1 else _key_mask(pad, x.dtype)
        tapped = []
        for i, block in enumerate(self.blocks):
            x = self._run_block(block, x, coords, mask, p + t)
            if i in taps:
                tapped.append(x[:, p + t:])
        x = self.norm(x)
        return (x[:, :p] if p else None), x[:, p + t:], tapped, (x[:, p:p + t] if t else None)

    def _pool(self, prefix: torch.Tensor | None, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """The global embedding: the CLS output when there is one, else the mean over kept patch tokens."""
        return prefix[:, 0] if self.n_cls else _masked_mean(x, pad)

    def forward(
        self,
        view: torch.Tensor,
        coords_mm: torch.Tensor,
        patch_size_mm: torch.Tensor,
        patch_xyz: Sequence[int] | None = None,
        keep: torch.Tensor | None = None,
        taps: Sequence[int] = (),
        meta: torch.Tensor | None = None,
        task_tokens: torch.Tensor | None = None,
        task_write: torch.Tensor | None = None,
        refine: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Dense token features for evaluation. ``view`` is (B, C, X, Y, Z). Dropped tokens are zero. ``refine``
        (B, N): cells encoded as 8 half-size tokens (``_refined``); outputs stay on the coarse grid.

        ``taps`` are block indices whose (pre-norm) outputs are also returned, as ``out["taps"]``. ``meta`` (B,) are
        ``META`` ids (ignored by an encoder without ``n_meta``). ``task_tokens`` (T, D) join the sequence after the
        prefix (OMNI-style, ``_task_mask``; ``task_write`` (T,) False = read-only) and come back as
        ``out["task_tokens"]`` (B, T, D).
        """
        if refine is not None and bool(refine.any()):
            return self._forward_refined(view, coords_mm, patch_size_mm, patch_xyz, keep, taps, meta, task_tokens,
                                         task_write, refine)
        tokens, grid = self._embed(view, coords_mm, patch_size_mm, patch_xyz, zyx=False, meta=meta)
        b, n, _ = tokens.shape
        if keep is None:
            keep = torch.ones(b, n, dtype=torch.bool, device=tokens.device)
        if keep.shape != (b, n):
            raise ValueError(f"keep {tuple(keep.shape)} does not match tokens {(b, n)}")
        seq, seq_coords, pad, _ = pack_kept(tokens, coords_mm, keep)
        prefix, x, tapped, task_out = self._run(seq, seq_coords, pad, taps, task_tokens, task_write)
        return {
            "task_tokens": task_out,
            "tokens": scatter_kept(x, keep),
            "cls": prefix[:, 0] if self.n_cls else None,       # (B, D) global embedding when the model has a CLS
            "registers": prefix[:, self.n_cls:] if self.n_registers else None,   # (B, R, D) register outputs
            "pooled": self._pool(prefix, x, pad),
            "patch": tokens * keep.unsqueeze(-1).to(dtype=tokens.dtype),
            "taps": [scatter_kept(t, keep) for t in tapped],
            "coords_mm": coords_mm,
            "grid_zyx": torch.tensor(grid, device=view.device),
        }

    def _forward_refined(self, view, coords_mm, patch_size_mm, patch_xyz, keep, taps, meta, task_tokens, task_write,
                         refine) -> dict[str, torch.Tensor]:
        if keep is None:
            keep = torch.ones_like(refine)
        tokens, coords, keep_all, grid = self._refined(view, coords_mm, patch_size_mm, patch_xyz, keep, refine,
                                                       zyx=False, meta=meta)
        seq, seq_coords, pad, _ = pack_kept(tokens, coords, keep_all)
        prefix, x, tapped, task_out = self._run(seq, seq_coords, pad, taps, task_tokens, task_write)
        coarse = self._to_coarse(scatter_kept(x, keep_all), refine)
        patch = self._to_coarse(tokens * keep_all.unsqueeze(-1).to(tokens.dtype), refine)
        return {
            "task_tokens": task_out,
            "tokens": coarse * keep.unsqueeze(-1).to(coarse.dtype),
            "cls": prefix[:, 0] if self.n_cls else None,
            "registers": prefix[:, self.n_cls:] if self.n_registers else None,
            "pooled": self._pool_coarse(prefix, coarse, keep),
            "patch": patch,
            "taps": [self._to_coarse(scatter_kept(t, keep_all), refine) * keep.unsqueeze(-1).to(t.dtype)
                     for t in tapped],
            "coords_mm": coords_mm,
            "grid_zyx": torch.tensor(grid, device=view.device),
            "n_refined": refine.sum(1),
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
        meta: torch.Tensor | None = None,
        refine: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pooled embedding of each row, and optionally its dense tokens, in input order. ``refine`` (R, N): cells
        encoded as 8 half-size tokens (``_refined``); dense tokens stay on the coarse grid.

        ``view`` is (R, C, Z, Y, X). Pooled is the CLS output, or without a CLS the mean over kept tokens, (R, D). Dense tokens are
        (R, N, D) with dropped tokens zero. ``plan`` is (order, sizes, longest): rows are sorted by kept
        count and cut into buckets, so a row is padded to its bucket's longest row, not the batch's. Rows
        never interact, so this changes cost, not values. ``longest`` comes from the CPU: no GPU sync.
        """
        refined = refine is not None and bool(refine.any())
        if refined:
            emb, coords_mm, keep_all, _ = self._refined(view, coords_mm, patch_size_mm, patch_xyz, keep, refine,
                                                        zyx=True, meta=meta)
        else:
            emb, _ = self._embed(view, coords_mm, patch_size_mm, patch_xyz, zyx=True, meta=meta)
            keep_all = keep
        order, sizes, longest = plan
        order = order.to(emb.device, non_blocking=True)
        pooled, dense, start = [], [], 0
        for size, m in zip(sizes, longest):
            idx = order[start:start + size]
            start += size
            if refined:                       # the plan's lengths are coarse counts: one sync for the mixed ones
                m = int(keep_all[idx].sum(1).max())
            seq, seq_coords, pad, _ = pack_kept(emb[idx], coords_mm[idx], keep_all[idx], max_keep=m)
            prefix, x, _, _ = self._run(seq, seq_coords, pad)
            if refined:
                coarse = self._to_coarse(scatter_kept(x, keep_all[idx]), refine[idx])
                pooled.append(self._pool_coarse(prefix, coarse, keep[idx]))
                if tokens:
                    dense.append(coarse * keep[idx].unsqueeze(-1).to(coarse.dtype))
                continue
            pooled.append(self._pool(prefix, x, pad))
            if tokens:
                dense.append(scatter_kept(x, keep[idx]))
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        return torch.cat(pooled)[inverse], (torch.cat(dense)[inverse] if tokens else None)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
