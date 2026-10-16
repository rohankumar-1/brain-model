"""3D rotary embeddings driven by physical token-center coordinates (mm)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _pairs_per_axis(head_dim: int) -> tuple[int, int, int]:
    n_pairs = head_dim // 2
    base, rem = divmod(n_pairs, 3)
    return base + (1 if rem > 0 else 0), base + (1 if rem > 1 else 0), base


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Adjacent-pair rotary: (x_even, x_odd) rotated by (cos, sin) on even indices."""
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    c, s = cos[..., 0::2], sin[..., 0::2]
    rot_even = x_even * c - x_odd * s
    rot_odd = x_even * s + x_odd * c
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


class RoPE3D(nn.Module):
    """Rotary PE on (x, y, z) mm. Head-dim pairs are split across the three axes."""

    def __init__(self, head_dim: int, base_mm: float = 100.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = int(head_dim)
        self.base_mm = float(base_mm)
        self.n_pairs = _pairs_per_axis(self.head_dim)
        for name, n in zip("xyz", self.n_pairs):
            if n == 0:
                continue
            k = torch.arange(n, dtype=torch.float32) / max(n, 1)
            self.register_buffer(f"freq_{name}", 1.0 / (self.base_mm ** k), persistent=False)

    def cos_sin(self, coords_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``coords_mm`` is (B, N, 3). Returns cos, sin each (B, 1, N, head_dim)."""
        parts = []
        for i, name in enumerate("xyz"):
            n = self.n_pairs[i]
            if n == 0:
                continue
            freq = getattr(self, f"freq_{name}")
            parts.append(coords_mm[..., i : i + 1] * freq.to(device=coords_mm.device, dtype=coords_mm.dtype))
        ang = torch.cat(parts, dim=-1)
        cos = ang.cos().repeat_interleave(2, dim=-1).unsqueeze(1)
        sin = ang.sin().repeat_interleave(2, dim=-1).unsqueeze(1)
        pad = self.head_dim - cos.size(-1)
        if pad:
            z = cos.new_zeros(*cos.shape[:-1], pad)
            cos = torch.cat([cos, z], dim=-1)
            sin = torch.cat([sin, z], dim=-1)
        return cos, sin

    def rotate(self, q: torch.Tensor, k: torch.Tensor, coords_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate ``q, k`` of shape (B, H, N, Dh) using ``coords_mm`` (B, N, 3)."""
        cos, sin = self.cos_sin(coords_mm.to(dtype=q.dtype))
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
