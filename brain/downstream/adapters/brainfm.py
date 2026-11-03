"""BrainFM behind the benchmark's adapter interface (``configs/bench/models.yaml``), like ``published``'s encoders.

``BrainFM``: BrainFM (Liu et al. 2025, arXiv 2509.00549; huggingface.co/peirong26/BrainFM, Apache 2.0). A 3D U-Net
(6 levels, 64 -> 2048 channels, GroupNorm + LeakyReLU, 264M) trained multi-task on real and synthetic ("mild-to-severe"
generated) brain scans of any modality to predict CT / T1 / T2 / FLAIR, a 56-label segmentation, scalp / cortical
distance maps, the bias field and an MNI registration field from one input. Input as in its inference code
(``utils/test_utils.prepare_image``): the scan at 1 mm on axes aligned with RAS (``align_volume_to_ref`` to the
identity), CT clamped to HU 0..80, then the volume min-max scaled to [0, 1]; MRI uses the raw (N4-corrected) intensities
here, the background at their minimum. Field of view: 192 x 224 x 160 mm around the stored box's centre (the whole
head; the network is fully convolutional, trained on 160^3 crops, and needs multiples of 32).

Features (``kwargs.feature``): ``bottleneck`` (default) the deepest encoder map, 2048-d at a 32 mm stride, the
representation every task head reads through the decoder; ``final`` the unit-normalised 64-d full-resolution decoder
output the authors call the feature (``evaluate_image(feature_only=True)``), average-pooled to 16 mm blocks. Global
feature: the chosen map averaged over the field of view; tokens: the map sampled at our token centres. Weights: the
checkpoint's ``backbone.*`` tensors, converted once to ``backbone.safetensors`` (the .pth also holds the optimizer).
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F

from brain.common.paths import resolve
from brain.downstream.adapters.published import _box_centre
from brain.downstream.models import ExternalEncoder


class BrainFM(ExternalEncoder):
    SHAPE = (192, 224, 160)                                     # R, A, S voxels at 1 mm

    def __init__(self, spec: dict, device):
        super().__init__(spec, device)
        from safetensors.torch import load_file

        kw = spec.get("kwargs") or {}
        self.feature = str(kw.get("feature", "bottleneck"))
        if self.feature not in ("bottleneck", "final"):
            raise ValueError(f"BrainFM feature {self.feature!r}: bottleneck | final")
        self.dim = 2048 if self.feature == "bottleneck" else 64
        self.shape = tuple(int(v) for v in kw.get("shape_ras", self.SHAPE))
        if any(s % 32 for s in self.shape):
            raise ValueError(f"BrainFM input {self.shape} must be a multiple of 32 on every axis")
        # their U-Net modules only (Trainer/models/unet3d); the package __init__s pull in the training stack
        code = str(resolve(kw.get("code", "weights/brainfm")) / "Trainer" / "models")
        if code not in sys.path:
            sys.path.insert(0, code)
        from unet3d.model import UNet3D

        self.model = UNet3D(1, 64, layer_order="gcl", num_groups=8, num_levels=6, is_unit_vector=True)
        self.model.load_state_dict(load_file(str(resolve(spec["weights"]))), strict=True)
        self.model.eval().to(device)

    def _affine(self, attrs: dict) -> np.ndarray:
        """Input voxel index (R, A, S axes, 1 mm) -> RAS mm, centred on the stored box."""
        a = np.eye(4)
        a[:3, 3] = _box_centre(attrs) - (np.asarray(self.shape) - 1) / 2
        return a

    def _input(self, vol_xyz, modality, attrs, aff) -> torch.Tensor:
        if str(modality).lower().startswith("ct"):
            x = self.resample(vol_xyz, attrs, aff, self.shape, fill=-1024.0).clamp(0.0, 80.0)
        else:
            raw = self.raw_intensity(vol_xyz, modality, attrs)
            lo = float(raw[vol_xyz > -5.5].min()) if (vol_xyz > -5.5).any() else float(raw.min())
            raw = np.where(vol_xyz > -5.5, raw, lo).astype(np.float32)          # stored background -> scan minimum
            x = self.resample(raw, attrs, aff, self.shape, fill=lo)
        x = x - x.min()
        return x / x.max().clamp_min(1e-6)

    @torch.no_grad()
    def encode(self, vol_xyz, modality, brain, attrs) -> dict:
        aff = self._affine(attrs)
        x = self._input(vol_xyz, modality, attrs, aff)[None, None]
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            if self.feature == "bottleneck":
                f = x
                for enc in self.model.encoders:
                    f = enc(f)
                step = 32.0                                     # five 2x max-pools
            else:
                f = F.avg_pool3d(self.model.get_feature(x)[-1].float(), 16)
                step = 16.0
        f = f[0].float()                                        # (C, R/step, A/step, S/step)
        pooled = f.mean((1, 2, 3))
        stride = np.diag([step, step, step, 1.0])
        stride[:3, 3] = (step - 1) / 2                          # feature voxel -> centre of its input block
        out = self.to_token_grid(f, aff @ stride, vol_xyz, modality, attrs)
        out["pooled"] = pooled.cpu().numpy()
        return out
