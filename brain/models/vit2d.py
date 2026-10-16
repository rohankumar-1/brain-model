"""1-channel Vision Transformers for 256x256 slices.

Sizes are the usual Tiny / Small / Base widths. Optional extras (all off by
default so older checkpoints load): QK-norm, register tokens, a wider projector.

``forward`` is the pretraining pass: globals then locals go through the encoder,
one shared projector returns (V, B, proj_dim). ``forward_features`` exposes the
class token, patch tokens and intermediate block outputs for brain.downstream use.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.models.projector import Projector

VIT_SIZES: dict[str, dict[str, int]] = {
    "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
    "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
}


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, qk_norm: bool = False):
        super().__init__()
        if dim % heads:
            raise ValueError(f"embed_dim {dim} is not divisible by heads {heads}")
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        head_dim = dim // heads
        self.q_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, qk_norm: bool = False):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, qk_norm)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(
        self,
        size: str = "small",
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 1,
        proj_dim: int = 128,
        proj_hidden: int | None = None,
        registers: int = 0,
        qk_norm: bool = False,
    ):
        super().__init__()
        if size not in VIT_SIZES:
            raise ValueError(f"unknown ViT size {size}. Choices: {', '.join(VIT_SIZES)}")
        if img_size % patch_size:
            raise ValueError(f"img_size {img_size} is not divisible by patch_size {patch_size}")
        spec = VIT_SIZES[size]
        dim, depth, heads = spec["embed_dim"], spec["depth"], spec["num_heads"]
        self.size, self.embed_dim, self.depth = size, dim, depth
        self.proj_dim, self.patch_size, self.n_registers = int(proj_dim), int(patch_size), int(registers)
        n_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        if registers:
            self.register_tokens = nn.Parameter(torch.zeros(1, registers, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, qk_norm=qk_norm) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.projector = Projector(dim, proj_hidden or min(2048, dim * 4), self.proj_dim)
        self.apply(self._init_weights)
        for p in (self.pos_embed, self.cls_token, *([self.register_tokens] if registers else [])):
            nn.init.trunc_normal_(p, std=0.02)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _pos_embed(self, side: int) -> torch.Tensor:
        """Positional grid for a ``side`` x ``side`` patch map, bicubic if not native."""
        cls, grid = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        native = int(round(grid.shape[1] ** 0.5))
        if side == native:
            return self.pos_embed
        grid = grid.reshape(1, native, native, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid.float(), size=(side, side), mode="bicubic", align_corners=False)
        grid = grid.to(self.pos_embed.dtype).permute(0, 2, 3, 1).reshape(1, side * side, -1)
        return torch.cat([cls, grid], dim=1)

    def default_taps(self, n: int = 4) -> tuple[int, ...]:
        """Block indices evenly spaced through the depth, ending at the last block."""
        return tuple(self.depth * (i + 1) // n - 1 for i in range(n))

    def forward_features(self, x: torch.Tensor, taps: tuple[int, ...] = ()):
        """Returns (cls (B, C), patches (B, N, C), taps: list of pre-norm (B, N, C))."""
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        side = int(round(tokens.shape[1] ** 0.5))
        if side * side != tokens.shape[1]:
            raise ValueError(f"patch grid is not square: {tokens.shape[1]} tokens from {tuple(x.shape)}")
        pos = self._pos_embed(side)
        b = tokens.size(0)
        prefix = [self.cls_token.expand(b, -1, -1) + pos[:, :1]]
        if self.n_registers:
            prefix.append(self.register_tokens.expand(b, -1, -1))
        z = torch.cat([*prefix, tokens + pos[:, 1:]], dim=1)
        n_pre, tapped = 1 + self.n_registers, []
        for i, block in enumerate(self.blocks):
            z = block(z)
            if i in taps:
                tapped.append(z[:, n_pre:])
        z = self.norm(z)
        return z[:, 0], z[:, n_pre:], tapped

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Class token, (B, embed_dim). Views may be any square size divisible by the patch."""
        return self.forward_features(x)[0]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Downstream embedding: class, mean patch, max patch. (B, 3 * embed_dim)."""
        cls, patches, _ = self.forward_features(x)
        return torch.cat([cls, patches.mean(dim=1), patches.amax(dim=1)], dim=-1)

    def forward(self, global_views: torch.Tensor, local_views: torch.Tensor | None = None):
        """Views are (V, B, C, H, W), globals first. Returns projections (V_g + V_l, B, proj_dim)."""
        b = global_views.size(1)
        embs = [self.encode(global_views.flatten(0, 1))]
        if local_views is not None and local_views.size(0):
            embs.append(self.encode(local_views.flatten(0, 1)))
        return self.projector(torch.cat(embs)).reshape(-1, b, self.proj_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
