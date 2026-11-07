"""HLIP behind the benchmark's adapter interface (``configs/bench/models.yaml``).

``HLIP``: HLIP (Zhao et al., TMLR 2026; huggingface.co/zch0414/clip-vit_base-scan_study-dualdinotxt1568,
CC-BY-NC-SA 4.0). It is a CLIP model trained on ~220k clinical brain-MRI studies paired with their radiology reports.
Its image tower is a ViT-B on 48 axial slices x 224 x 224 per scan with (6, 16, 16) patches (8 x 14 x 14 = 1568 tokens),
plus a CLS and one register token. It alternates scan-level and study-level attention; for one scan both attend to
every token of that scan, so a single scan runs as a plain ViT-B.

The weights are read from the snapshot's ``model.safetensors`` (``visual.trunk.*``) into ``timm`` blocks: none of the
snapshot's own code is imported.

Input, as in the model card's loader:
- the scan's axial slices in display orientation (rows anterior -> posterior, columns patient right -> left);
- intensities scaled to [0, 1] (here: the 0.5 / 99.5 percentiles of the scan's foreground, raw MRI units);
- 224 x 224 in plane, 48 slices through the stored 160 mm box;
- normalised by the scalar ImageNet mean / std.

Features: the final patch tokens (768-d, a 16 x 16 x 20 mm block each), sampled at our token centres. Pooled: their
mean. MRI only: the head-CT weights are not public.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch
import torch.nn as nn

from brain.common.paths import resolve
from brain.downstream.adapters.published import _box_centre
from brain.downstream.models import ExternalEncoder

MEAN, STD = float(np.mean([0.485, 0.456, 0.406])), float(np.mean([0.229, 0.224, 0.225]))


class _Trunk(nn.Module):
    def __init__(self, dim: int = 768, depth: int = 12, heads: int = 12, patch=(6, 16, 16), grid=(8, 14, 14)):
        super().__init__()
        from timm.models.vision_transformer import Block

        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(1, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.reg_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.spatial_posemb = nn.Parameter(torch.zeros(1, *grid, dim), requires_grad=False)
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=norm) for _ in range(depth))
        self.norm = norm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (B, 1, D, H, W) -> patch tokens (B, d, h, w, C) after the final norm."""
        f = self.patch_embed.proj(x).permute(0, 2, 3, 4, 1)                 # (B, d, h, w, C)
        b, d, h, w, c = f.shape
        t = (f + self.spatial_posemb.to(f.dtype)).reshape(b, d * h * w, c)
        t = torch.cat([self.cls_token.expand(b, -1, -1), self.reg_token.expand(b, -1, -1), t], 1).to(f.dtype)
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)[:, 2:].reshape(b, d, h, w, c)


class HLIP(ExternalEncoder):
    SHAPE = (224, 224, 48)                                      # R, A, S input voxels (1 mm in plane)
    FOV_S = 160.0                                               # mm of the stored box along S
    dim = 768

    def __init__(self, spec: dict, device):
        super().__init__(spec, device)
        from safetensors.torch import load_file

        sd = load_file(str(resolve(spec["weights"])))
        trunk = {k[len("visual.trunk."):]: v for k, v in sd.items() if k.startswith("visual.trunk.")}
        self.model = _Trunk()
        self.model.load_state_dict(trunk, strict=True)
        self.model.eval().to(device)

    def _affine(self, attrs: dict) -> np.ndarray:
        """Input voxel index (R, A, S) -> RAS mm, centred on the stored box: 1 mm in plane, FOV_S / 48 mm along S."""
        sp = np.array([1.0, 1.0, self.FOV_S / self.SHAPE[2]])
        a = np.diag([*sp, 1.0])
        a[:3, 3] = _box_centre(attrs) - (np.asarray(self.SHAPE) - 1) / 2 * sp
        return a

    @torch.no_grad()
    def encode(self, vol_xyz, modality, brain, attrs) -> dict:
        aff = self._affine(attrs)
        raw = self.raw_intensity(vol_xyz, modality, attrs).astype(np.float32)
        fg = vol_xyz > -5.5
        lo, hi = (np.percentile(raw[fg], [0.5, 99.5]) if fg.any() else (float(raw.min()), float(raw.max())))
        img = np.clip((raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        img = np.where(fg, img, 0.0).astype(np.float32)                       # background black, as their padding
        x = self.resample(img, attrs, aff, self.SHAPE, fill=0.0)              # (R, A, S)
        x = x.permute(2, 1, 0).flip(1).flip(2)                                # (S, A desc, R desc): display order
        x = ((x - MEAN) / STD)[None, None]
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            f = self.model(x)[0].float()                                      # (8, 14, 14, C) in (S, A desc, R desc)
        f = f.flip(1).flip(2).permute(3, 2, 1, 0)                             # (C, R, A, S)
        stride = np.diag([16.0, 16.0, 6.0, 1.0])                              # one token per patch, in input voxels
        stride[:3, 3] = [7.5, 7.5, 2.5]
        out = self.to_token_grid(f, aff @ stride, vol_xyz, modality, attrs)
        out["pooled"] = f.mean((1, 2, 3)).cpu().numpy()
        return out
