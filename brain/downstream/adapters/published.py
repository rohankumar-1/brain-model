"""Published open-weight encoders behind the benchmark's adapter interface (``configs/bench/models.yaml``).

Each adapter rebuilds the model's own input from the stored volume: its orientation, grid, field of view and intensity
scaling, resampled from the store's 1 mm MNI152 grid (``ExternalEncoder.resample``), never from our encoder's windows.
Its global feature is the one the authors publish; its spatial features are sampled at our token centres
(``ExternalEncoder.to_token_grid``), so the same readouts (``tok_conv``, plane / grid PCA) apply to every model.
Weights are loaded from safetensors only (no pickles from third parties). Both need MONAI (``pip install --user
--no-deps monai``; the jobs' user site has 1.5.1).

``CTFM``: CT-FM (Pai et al. 2025, arXiv 2501.09001; huggingface.co/project-lighter/ct_fm_feature_extractor, Apache
2.0). A SegResNet encoder (77M) trained with intra-sample SimCLR on 148k CTs of the Imaging Data Commons. Input as in
its pretraining: SPL orientation, 3 x 1 x 1 mm, HU -1024..2048 scaled to [0, 1]; here the whole stored box (64 x 256 x
256 voxels = 192 x 256 x 256 mm, the head and some padding) in one pass. Global feature: the last stage average-pooled
(the model card's recipe), 512-d; tokens: that stage (4 x 16 x 16 at 48 x 16 x 16 mm). ``ct_fm_triage`` (kwargs
``windows``): the input of CT-FM's own head-CT evaluation instead, 5 x 2 x 2 mm, 48 x 128 x 128, four HU windows
stacked along S; tokens average the four copies' maps.

``BrainIAC``: BrainIAC (Tak et al., Nature Neuroscience 2026; github.com/AIM-KannLab/BrainIAC, non-commercial research
licence). A MONAI 3D ViT-B (16^3 patches, 96^3 input, no CLS token) trained with SimCLR on ~32k structural MRIs (T1,
T2, FLAIR, T1c). Input as in its feature-extraction code: the scan rigidly registered to its 1 mm head template
(170 x 206 x 162, LAS), skull-stripped (HD-BET; here our SynthStrip mask), resized to 96^3 (trilinear) and z-scored
over the nonzero voxels. Our registration is affine (head size normalised) where theirs is rigid: a mild shift. Global
feature: the first output token (``features[0][:, 0]`` in their code; with no CLS token it is the corner patch, which
attention makes a summary), 768-d; tokens: the 6^3 patch tokens.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from brain.common.paths import resolve
from brain.downstream.models import ExternalEncoder


def _box_centre(attrs: dict) -> np.ndarray:
    """RAS mm of the stored box's centre."""
    n = np.asarray(attrs["grid_size"], float)
    return (ExternalEncoder.grid_affine(attrs) @ np.r_[(n - 1) / 2, 1.0])[:3]


class CTFM(ExternalEncoder):
    dim = 512

    def __init__(self, spec: dict, device):
        super().__init__(spec, device)
        from monai.networks.nets.segresnet_ds import SegResEncoder
        from safetensors.torch import load_file

        kw = spec.get("kwargs") or {}
        self.spacing = tuple(float(v) for v in kw.get("spacing_spl", (3.0, 1.0, 1.0)))
        self.shape = tuple(int(v) for v in kw.get("shape_spl", (64, 256, 256)))
        if any(s % 16 for s in self.shape):
            raise ValueError(f"CT-FM input {self.shape} must be a multiple of 16 on every axis")
        # HU windows, each scaled to [0, 1], stacked along S as CT-FM's head-CT triage pipeline does
        # (evaluation/cq500.yaml: blood 0-80, subdural -125-175, stroke 28-36, bone -900-2100, ConcatItemsd dim=1);
        # none: its pretraining scaling, HU -1024..2048 in one copy
        self.windows = [tuple(float(v) for v in w) for w in kw.get("windows") or []]
        self.model = SegResEncoder(spatial_dims=3, init_filters=32, in_channels=1, blocks_down=(1, 2, 2, 4, 4))
        self.model.load_state_dict(load_file(str(resolve(spec["weights"]))), strict=True)
        self.model.eval().to(device)

    def _affine(self, attrs: dict) -> np.ndarray:
        """Input voxel index (S, P, L axes) -> RAS mm, centred on the stored box."""
        s0, s1, s2 = self.spacing
        a = np.eye(4)
        a[:3, :3] = np.array([[0.0, 0.0, -s2], [0.0, -s1, 0.0], [s0, 0.0, 0.0]])   # columns: +S, +P (-A), +L (-R)
        a[:3, 3] = _box_centre(attrs) - a[:3, :3] @ ((np.asarray(self.shape) - 1) / 2)
        return a

    def _input(self, hu: torch.Tensor) -> torch.Tensor:
        """The network input from HU on the model grid (S, P, L): one scaled copy, or the windows stacked along S."""
        if not self.windows:
            return ((hu + 1024.0) / 3072.0).clamp(0, 1)
        return torch.cat([((hu - lo) / (hi - lo)).clamp(0, 1) for lo, hi in self.windows], 0)

    @torch.no_grad()
    def encode(self, vol_xyz, modality, brain, attrs) -> dict:
        aff = self._affine(attrs)
        x = self._input(self.resample(vol_xyz, attrs, aff, self.shape, fill=-1024.0))
        f = self.model(x[None, None])[-1][0]                   # (512, copies * S/16, P/16, L/16)
        pooled = f.mean((1, 2, 3))                             # the whole map, as CT-FM's heads pool it
        if self.windows:                                       # the copies cover the same anatomy: average them
            f = f.reshape(f.shape[0], len(self.windows), -1, *f.shape[2:]).mean(1)
        stride = np.diag([16.0, 16.0, 16.0, 1.0])
        stride[:3, 3] = 7.5                                    # feature voxel -> centre of its 16^3 input block
        out = self.to_token_grid(f, aff @ stride, vol_xyz, modality, attrs)
        out["pooled"] = pooled.cpu().numpy()
        return out


class BrainIAC(ExternalEncoder):
    dim = 768
    needs_brain = True
    SIZE = 96

    def __init__(self, spec: dict, device):
        super().__init__(spec, device)
        import nibabel as nib
        from monai.networks.nets import ViT
        from safetensors.torch import load_file

        kw = spec.get("kwargs") or {}
        tpl = nib.load(str(resolve(kw["template"])))
        self.tpl_affine, self.tpl_shape = np.asarray(tpl.affine, float), tuple(int(v) for v in tpl.shape[:3])
        self.model = ViT(in_channels=1, img_size=(self.SIZE,) * 3, patch_size=(16, 16, 16), hidden_size=768,
                         mlp_dim=3072, num_layers=12, num_heads=12)
        r = self.model.load_state_dict(load_file(str(resolve(spec["weights"]))), strict=False)
        # MONAI >= 1.4 builds (unused) cross-attention layers the MONAI 1.3 checkpoint does not have
        if r.unexpected_keys or any("cross_attn" not in k for k in r.missing_keys):
            raise RuntimeError(f"BrainIAC weights do not match the ViT: {r}")
        self.model.eval().to(device)

    @torch.no_grad()
    def encode(self, vol_xyz, modality, brain, attrs) -> dict:
        raw = self.raw_intensity(vol_xyz, modality, attrs)
        # brain voxels the scan has: a skull-stripped source (ATLAS, BraTS, MSD, ...) leaves background inside the
        # SynthStrip mask, which raw_intensity would turn into large negative values (a false rim, z-scored with the brain)
        m = (brain & (vol_xyz > -5.5)) if brain is not None else vol_xyz > -5.5
        img = self.resample(np.where(m, raw, 0.0).astype(np.float32), attrs, self.tpl_affine, self.tpl_shape)
        x = F.interpolate(img[None, None], size=(self.SIZE,) * 3, mode="trilinear", align_corners=False)
        nz = x != 0
        if nz.any():                                           # MONAI NormalizeIntensity(nonzero=True)
            v = x[nz]
            x[nz] = (v - v.mean()) / v.std(unbiased=False).clamp_min(1e-8)
        h, _ = self.model(x)                                   # (1, 216, 768), final norm applied
        g = self.SIZE // 16
        f = h[0].T.reshape(self.dim, g, g, g)
        # patch p -> its centre in the 96^3 input (16 p + 7.5) -> template voxel (Resize, align_corners=False)
        scale = np.asarray(self.tpl_shape, float) / self.SIZE
        a = np.eye(4)
        a[:3, :3] = np.diag(16.0 * scale)
        a[:3, 3] = 8.0 * scale - 0.5
        out = self.to_token_grid(f, self.tpl_affine @ a, vol_xyz, modality, attrs)
        out["pooled"] = h[0, 0].cpu().numpy()
        return out
