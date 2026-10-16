"""Train-time slice augmentations."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

MODALITY_TO_ID = {"ct": 0, "mri": 1}
ID_TO_MODALITY = {v: k for k, v in MODALITY_TO_ID.items()}


class SliceAugment:
    """Train-only 2D augs for (1, H, W) float tensors in [0, 1].

    Horizontal flip, small translation, random resized crop, intensity jitter + noise.
    No vertical flip or rotation.
    """

    def __init__(
        self,
        hflip: float = 0.5,
        translate: float = 0.05,
        crop_scale: tuple[float, float] = (0.85, 1.0),
        intensity: float = 0.1,
        noise_std: float = 0.01,
    ):
        self.hflip = float(hflip)
        self.translate = float(translate)
        lo, hi = crop_scale
        self.crop_scale = (float(lo), float(hi))
        self.intensity = float(intensity)
        self.noise_std = float(noise_std)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> SliceAugment | None:
        if not cfg or not cfg.get("enabled", False):
            return None
        crop = cfg.get("crop_scale", [0.85, 1.0])
        if isinstance(crop, (int, float)):
            crop_scale = (float(crop), 1.0)
        else:
            crop_scale = (float(crop[0]), float(crop[1]))
        return cls(
            hflip=float(cfg.get("hflip", 0.5)),
            translate=float(cfg.get("translate", 0.05)),
            crop_scale=crop_scale,
            intensity=float(cfg.get("intensity", 0.1)),
            noise_std=float(cfg.get("noise_std", 0.01)),
        )

    def _random_resized_crop(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        scale = self.crop_scale[0] + torch.rand(1).item() * (self.crop_scale[1] - self.crop_scale[0])
        ch = max(1, int(round(h * scale)))
        cw = max(1, int(round(w * scale)))
        top = int(torch.randint(0, h - ch + 1, (1,)).item()) if ch < h else 0
        left = int(torch.randint(0, w - cw + 1, (1,)).item()) if cw < w else 0
        cropped = x[:, top : top + ch, left : left + cw]
        return F.interpolate(
            cropped.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _translate(self, x: torch.Tensor) -> torch.Tensor:
        if self.translate <= 0:
            return x
        _, h, w = x.shape
        max_dx = self.translate * w
        max_dy = self.translate * h
        dx = (torch.rand(1).item() * 2 - 1) * max_dx
        dy = (torch.rand(1).item() * 2 - 1) * max_dy
        tx = 2.0 * dx / max(w - 1, 1)
        ty = 2.0 * dy / max(h - 1, 1)
        theta = torch.tensor([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=x.dtype)
        grid = F.affine_grid(theta.unsqueeze(0), size=(1, 1, h, w), align_corners=False)
        return F.grid_sample(
            x.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.hflip > 0 and torch.rand(1).item() < self.hflip:
            x = torch.flip(x, dims=[-1])
        if self.crop_scale[0] < 1.0 and torch.rand(1).item() < 0.5:
            x = self._random_resized_crop(x)
        if self.translate > 0:
            x = self._translate(x)
        if self.intensity > 0:
            scale = 1.0 + (torch.rand(1).item() * 2 - 1) * self.intensity
            shift = (torch.rand(1).item() * 2 - 1) * self.intensity
            x = x * scale + shift
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x.clamp(0.0, 1.0)
