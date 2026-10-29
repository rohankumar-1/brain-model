"""3D masked autoencoder on the native token grid: the encoder sees only visible foreground tokens, a narrow decoder
reconstructs the voxels of the hidden ones.

The target is the crop's own voxels in windowed units (no per-patch normalisation): absolute tissue contrast is what a
reconstruction objective should preserve, and what the invariance objectives were found to discard. The loss is the
mean squared error over hidden foreground tokens only (air is never a token).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from brain.models.viewnet3d import cell_means
from brain.models.vit3d import Block, BrainEncoder, RoPE3D, _key_mask, pack_kept, scatter_kept


class MAEDecoder(nn.Module):
    """Visible encoder tokens (projected) plus a learned mask token at every hidden position; RoPE carries position."""

    def __init__(self, enc_dim: int, dim: int, depth: int, heads: int, patch_voxels: int, rope_base_mm: float = 100.0,
                 qk_norm: bool = True):
        super().__init__()
        self.inp = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.rope = RoPE3D(dim // heads, base_mm=rope_base_mm)
        self.blocks = nn.ModuleList(Block(dim, heads, self.rope, qk_norm=qk_norm) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, patch_voxels)

    def forward(self, ctx_tokens: torch.Tensor, ctx_keep: torch.Tensor, keep: torch.Tensor, coords_mm: torch.Tensor,
                plan: tuple[torch.Tensor, list[int], list[int]]) -> torch.Tensor:
        """``ctx_tokens`` (R, N, D_enc) zero off ``ctx_keep``; ``keep`` (R, N) every foreground position, ``plan`` its
        length plan. Returns (R, N, patch_voxels) reconstructions (meaningful on ``keep``)."""
        x = torch.where(ctx_keep.unsqueeze(-1), self.inp(ctx_tokens), self.mask_token.to(ctx_tokens.dtype))
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
        return self.out(torch.cat(out)[inverse])


class MAE3D(nn.Module):
    """``target="voxels"``: the hidden patches' voxels in windowed units. ``target="split"``: each hidden patch as its
    per-patch-normalised voxels (fine structure, with no local-mean shortcut) plus its mean and standard deviation as
    two extra outputs (absolute contrast, kept rather than normalised away); the decoder then has
    ``patch_voxels + 2`` outputs. ``lesion_weight`` up-weights hidden tokens flagged as lesion (``<group>_lesion``)."""

    STD_FLOOR = 0.05                                   # windowed units: flat patches are not normalised into noise

    def __init__(self, enc: BrainEncoder, dec: MAEDecoder, target: str = "voxels", lesion_weight: float = 1.0):
        super().__init__()
        self.enc, self.dec = enc, dec
        self.target, self.lesion_weight = str(target), float(lesion_weight)

    def forward(self, v: dict, group: str = "global") -> dict[str, torch.Tensor]:
        b = v["modality"].shape[0]
        n = v[f"{group}_view"].shape[0]
        rows = lambda t: t.contiguous().reshape(n * b, *t.shape[2:])  # noqa: E731  (V, B, ...) -> (V*B, ...)
        view, coords, pmm = rows(v[f"{group}_view"]), rows(v[f"{group}_coords"]), rows(v[f"{group}_patch_mm"])
        keep, ctx, tgt = rows(v[f"{group}_fg"]), rows(v["ctx_keep"]), rows(v["tgt_mask"])
        patch = v["patch"][group]
        _, tokens = self.enc.encode_rows(view, coords, pmm, ctx, v["plans"]["ctx"], patch, tokens=True)
        out = self.dec(tokens, ctx, keep, coords, v["plans"][group]).float()
        target = cell_means(view, patch, patch).float()            # (R, N, voxels per patch), token order (Z, Y, X)
        w = tgt.float()
        if self.lesion_weight != 1.0 and f"{group}_lesion" in v:
            w = w * (1.0 + (self.lesion_weight - 1.0) * rows(v[f"{group}_lesion"]).float())
        n_vox = target.shape[-1]
        if self.target == "split":
            mu, sd = target.mean(-1, keepdim=True), target.std(-1, keepdim=True).clamp_min(self.STD_FLOOR)
            norm_t = (target - mu) / sd
            p_norm, p_mu, p_sd = out[..., :n_vox], out[..., n_vox:n_vox + 1], out[..., n_vox + 1:]
            per_tok = ((p_norm - norm_t).square().mean(-1, keepdim=True)
                       + (p_mu - mu).square() + (p_sd - sd).square())[..., 0]
            recon = p_mu + p_sd.clamp_min(self.STD_FLOOR) * p_norm                # absolute voxels, for R2
        else:
            per_tok = (out - target).square().mean(-1)
            recon = out
        loss = (per_tok * w).sum() / w.sum().clamp_min(1.0)
        # share of the hidden voxels' variance the (absolute) reconstruction explains: 1 perfect, 0 = predicts the mean
        h = tgt.float().unsqueeze(-1)
        nh = h.sum().clamp_min(1.0) * n_vox
        mse = ((recon.detach() - target).square() * h).sum() / nh
        mean = (target * h).sum() / nh
        var = ((target - mean).square() * h).sum() / nh
        return {"loss": loss, "mse": mse, "r2": (1 - mse / var.clamp_min(1e-6)).detach(),
                "hidden": (tgt.sum() / keep.sum().clamp_min(1)).detach().float()}
