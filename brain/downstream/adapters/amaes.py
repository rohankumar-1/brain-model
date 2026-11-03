"""AMAES on FOMO260K behind the benchmark's adapter interface (``configs/bench/models.yaml``: ``amaes_fomo260k``).

``AMAES``: the ResEnc U-Net B encoder (90M; asparagus ``resenc_unet_b``, gardening_tools ``ResidualUNetEncoder``)
pretrained with AMAES (augmented masked autoencoding, Munk et al. 2024, arXiv 2408.00640) on FOMO260K (Cerri et al.,
arXiv 2506.14432; huggingface.co/FOMO-MRI/AMAES_resenc_b_fomo260k, CC BY-NC-SA 4.0): 261k minimally preprocessed
brain MRIs, the release our SSL pool's ``fomo_*`` families come from. Rebuilt here in plain PyTorch (no asparagus /
gardening_tools); ``weights/amaes/encoder.safetensors`` is the checkpoint's encoder with the duplicate
``all_modules.*`` names dropped (each identical to its ``conv`` / ``norm_op`` twin).

Input as in its pretraining: RAS axes, the scan's own intensities z-scored over the whole volume, background
included (``volume_wise_znorm``), skull on. FOMO260K keeps native spacing (only reoriented) and AMAES trained on
160^3 crops of it; here the whole stored box at 1 mm (256 x 256 x 160 RAS, a multiple of the encoder's 32x
downsampling) in one pass, the stored background (z -6) set to 0 as in a raw scan. Global feature: the bottleneck
(stage 6, 320 channels) average-pooled; tokens: that map (8 x 8 x 5 at 32 mm) sampled at our token centres.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from brain.common.paths import resolve
from brain.downstream.adapters.published import _box_centre
from brain.downstream.models import ExternalEncoder

FEATURES = (32, 64, 128, 256, 320, 320)
BLOCKS = (1, 3, 4, 6, 6, 6)


class ConvNorm(nn.Module):
    """3^3 (or 1^3) convolution, instance norm, optionally LeakyReLU (gardening_tools ``ConvDropoutNormNonlin``;
    the encoder has no dropout)."""

    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, bias: bool = True, act: bool = True):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, k, stride=stride, padding=k // 2, bias=bias)
        self.norm_op = nn.InstanceNorm3d(cout, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True) if act else None

    def forward(self, x):
        x = self.norm_op(self.conv(x))
        return self.act(x) if self.act is not None else x


class ResBlock(nn.Module):
    """conv-norm-act, conv-norm, plus the shortcut (2x average pool when strided, 1^3 conv-norm when the width
    changes), then LeakyReLU (gardening_tools ``ResidualBlock``)."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = ConvNorm(cin, cout, stride=stride)
        self.conv2 = ConvNorm(cout, cout, act=False)
        ops = ([nn.AvgPool3d(stride, stride)] if stride != 1 else []) + \
            ([ConvNorm(cin, cout, k=1, bias=False, act=False)] if cin != cout else [])
        self.skip = nn.Sequential(*ops) if ops else nn.Identity()
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class Stage(nn.Module):
    def __init__(self, cin: int, cout: int, n: int, stride: int):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(cin if i == 0 else cout, cout, stride if i == 0 else 1)
                                      for i in range(n)])

    def forward(self, x):
        return self.blocks(x)


class Stem(nn.Module):
    def __init__(self, cout: int):
        super().__init__()
        self.conv1 = ConvNorm(1, cout)

    def forward(self, x):
        return self.conv1(x)


class ResEncB(nn.Module):
    """ResEnc U-Net B encoder: stem, then six stages of residual blocks, every stage but the first halving the grid."""

    def __init__(self):
        super().__init__()
        self.stem = Stem(FEATURES[0])
        cin, stages = FEATURES[0], []
        for s, (c, n) in enumerate(zip(FEATURES, BLOCKS)):
            stages.append(Stage(cin, c, n, 1 if s == 0 else 2))
            cin = c
        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        return self.stages(self.stem(x))


class AMAES(ExternalEncoder):
    dim = 320
    STRIDE = 32

    def __init__(self, spec: dict, device):
        super().__init__(spec, device)
        from safetensors.torch import load_file

        kw = spec.get("kwargs") or {}
        self.shape = tuple(int(v) for v in kw.get("shape_ras", (256, 256, 160)))
        if any(s % self.STRIDE for s in self.shape):
            raise ValueError(f"AMAES input {self.shape} must be a multiple of {self.STRIDE} on every axis")
        self.model = ResEncB()
        self.model.load_state_dict(load_file(str(resolve(spec["weights"]))), strict=True)
        self.model.eval().to(device)

    def _affine(self, attrs: dict) -> np.ndarray:
        """Input voxel index (R, A, S axes, 1 mm) -> RAS mm, centred on the stored box."""
        a = np.eye(4)
        a[:3, 3] = _box_centre(attrs) - (np.asarray(self.shape) - 1) / 2
        return a

    @torch.no_grad()
    def encode(self, vol_xyz, modality, brain, attrs) -> dict:
        raw = self.raw_intensity(vol_xyz, modality, attrs)
        raw = np.where(vol_xyz > -5.5, raw, 0.0).astype(np.float32)           # stored background (z -6) -> 0
        aff = self._affine(attrs)
        x = self.resample(raw, attrs, aff, self.shape)
        x = (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-8)           # volume_wise_znorm
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            f = self.model(x[None, None])[0].float()                         # (320, R/32, A/32, S/32)
        stride = np.diag([float(self.STRIDE)] * 3 + [1.0])
        stride[:3, 3] = (self.STRIDE - 1) / 2                                 # feature voxel -> centre of its 32^3 block
        out = self.to_token_grid(f, aff @ stride, vol_xyz, modality, attrs)
        out["pooled"] = f.mean((1, 2, 3)).cpu().numpy()
        return out
