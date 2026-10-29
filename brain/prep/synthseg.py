"""SynthSeg-robust 2.0 (Billot et al., PNAS 2023) in PyTorch, on store volumes.

    runs/diagnostics/synthseg_check.py      # port vs the original Keras model (TF reference venv, CPU)

A port of FreeSurfer's ``mri_synthseg --robust`` (fast post-processing, which ``--robust`` forces): three U-Nets in
cascade, each built like neurite's ``unet`` (ELU convolutions, batch norm closing every level, nearest upsampling,
skips taken from each level's last convolution before its batch norm):

    unet   image (1 ch)              -> 5 coarse groups   (24 features, 3^3 kernels)
    l2l    one-hot(argmax) (5 ch)    -> 5 groups          (16 features, 5^3 kernels; no skips at the 2 finest levels)
    unet2  image + one-hot (6 ch)    -> 33 labels         (24 features, 3^3 kernels)

Weights: ``synthseg_robust_2.0.h5`` from FreeSurfer's model annex (SHA-256 7bc30bf5...), read layer by layer.
Input: a store volume (Z, Y, X) on the 1 mm RAS grid -> (X, Y, Z), CT clipped to [0, 80] HU, rescaled to [0, 1]
between its 0.5 / 99.5 percentiles (as ``preprocess``). Output: FreeSurfer labels on the store grid, (Z, Y, X).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOOLS = Path(os.environ.get("BRAIN_SYNTHSEG", "/home/roku/orcd/scratch/brain-model/tools/synthseg"))
WEIGHTS = TOOLS / "synthseg_robust_2.0.h5"
# output channel order: sorted unique labels of synthseg_segmentation_labels_2.0.npy (robust: no flip reordering)
LABELS = np.array([0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28,
                   41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60])
NAMES = {0: "background", 2: "L cerebral white matter", 3: "L cerebral cortex", 4: "L lateral ventricle",
         5: "L inferior lateral ventricle", 7: "L cerebellum white matter", 8: "L cerebellum cortex",
         10: "L thalamus", 11: "L caudate", 12: "L putamen", 13: "L pallidum", 14: "3rd ventricle",
         15: "4th ventricle", 16: "brainstem", 17: "L hippocampus", 18: "L amygdala", 24: "CSF",
         26: "L accumbens", 28: "L ventral DC", 41: "R cerebral white matter", 42: "R cerebral cortex",
         43: "R lateral ventricle", 44: "R inferior lateral ventricle", 46: "R cerebellum white matter",
         47: "R cerebellum cortex", 49: "R thalamus", 50: "R caudate", 51: "R putamen", 52: "R pallidum",
         53: "R hippocampus", 54: "R amygdala", 58: "R accumbens", 60: "R ventral DC"}
BN_EPS = 1e-3                                   # Keras BatchNormalization default


class UNet(nn.Module):
    """neurite ``unet`` with ``nb_levels=5, nb_conv_per_level=2, feat_mult=2, batch_norm=-1``, inference only."""

    def __init__(self, cin: int, feats: int, n_labels: int, k: int, skip_n: int = 0, levels: int = 5):
        super().__init__()
        self.levels, self.skip_n = levels, skip_n
        f = [feats * 2 ** lv for lv in range(levels)]
        pad = k // 2
        self.down = nn.ModuleList()
        self.bn_down = nn.ModuleList()
        c = cin
        for lv in range(levels):
            self.down.append(nn.ModuleList([nn.Conv3d(c, f[lv], k, padding=pad), nn.Conv3d(f[lv], f[lv], k, padding=pad)]))
            self.bn_down.append(nn.BatchNorm3d(f[lv], eps=BN_EPS))
            c = f[lv]
        self.up = nn.ModuleList()
        self.bn_up = nn.ModuleList()
        for i in range(levels - 1):
            lv = levels - 2 - i
            skip = f[lv] if i < levels - skip_n - 1 else 0
            self.up.append(nn.ModuleList([nn.Conv3d(c + skip, f[lv], k, padding=pad),
                                          nn.Conv3d(f[lv], f[lv], k, padding=pad)]))
            self.bn_up.append(nn.BatchNorm3d(f[lv], eps=BN_EPS))
            c = f[lv]
        self.likelihood = nn.Conv3d(c, n_labels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for lv in range(self.levels):
            for conv in self.down[lv]:
                x = F.elu(conv(x))
            skips.append(x)                           # the skip is the pre-batch-norm tensor
            x = self.bn_down[lv](x)
            if lv < self.levels - 1:
                x = F.max_pool3d(x, 2)
        for i in range(self.levels - 1):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if i < self.levels - self.skip_n - 1:
                x = torch.cat([skips[self.levels - 2 - i], x], dim=1)
            for conv in self.up[i]:
                x = F.elu(conv(x))
            x = self.bn_up[i](x)
        return self.likelihood(x).softmax(dim=1)


class SynthSegRobust(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(1, 24, 5, 3)
        self.l2l = UNet(5, 16, 5, 5, skip_n=2)
        self.unet2 = UNet(6, 24, len(LABELS), 3)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """(B, 1, X, Y, Z) normalised image -> (B, 33, X, Y, Z) posteriors."""
        g = F.one_hot(self.unet(img).argmax(1), 5).permute(0, 4, 1, 2, 3).to(img.dtype)
        g = F.one_hot(self.l2l(g).argmax(1), 5).permute(0, 4, 1, 2, 3).to(img.dtype)
        return self.unet2(torch.cat([img, g], dim=1))


def load_weights(model: SynthSegRobust, path: Path = WEIGHTS) -> SynthSegRobust:
    import h5py

    with h5py.File(path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f

        def ds(layer, var):
            return torch.from_numpy(np.asarray(g[layer][layer][f"{var}:0"]))

        for prefix in ("unet", "l2l", "unet2"):
            net = getattr(model, prefix)
            lv_n = net.levels

            def conv(m, name):
                m.weight.data.copy_(ds(name, "kernel").permute(4, 3, 0, 1, 2))   # (kx,ky,kz,in,out)->(out,in,kx,ky,kz)
                m.bias.data.copy_(ds(name, "bias"))

            def bn(m, name):
                m.weight.data.copy_(ds(name, "gamma"))
                m.bias.data.copy_(ds(name, "beta"))
                m.running_mean.copy_(ds(name, "moving_mean"))
                m.running_var.copy_(ds(name, "moving_variance"))

            for lv in range(lv_n):
                for c in range(2):
                    conv(net.down[lv][c], f"{prefix}_conv_downarm_{lv}_{c}")
                bn(net.bn_down[lv], f"{prefix}_bn_down_{lv}")
            for i in range(lv_n - 1):
                for c in range(2):
                    conv(net.up[i][c], f"{prefix}_conv_uparm_{lv_n + i}_{c}")
                bn(net.bn_up[i], f"{prefix}_bn_up_{i}")
            conv(net.likelihood, f"{prefix}_likelihood")
    return model.eval()


def normalise(vol_xyz: np.ndarray, modality: str) -> np.ndarray:
    im = np.asarray(vol_xyz, dtype=np.float32)
    if str(modality).lower().startswith("ct"):
        im = np.clip(im, 0, 80)
    lo, hi = np.percentile(im, [0.5, 99.5])
    return np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import label

    lab, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def postprocess(post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``postprocess(fast=True)``: (33, X, Y, Z) posteriors -> FreeSurfer label map and volumes (ml, per label)."""
    fg = _largest_component(post[1:].sum(0) > 0.25)
    post = post.copy()
    post[1:] *= fg
    post[1:] *= post[1:] > 0.2
    post /= post.sum(0, keepdims=True).clip(min=1e-12)
    seg = LABELS[post.argmax(0)].astype(np.int16)
    return seg, post[1:].sum(axis=(1, 2, 3)) / 1000.0


class Segmenter:
    def __init__(self, device: str | torch.device = "cuda", amp: bool = False):
        # full fp32 convolutions: the cascade takes an argmax between networks, so TF32 rounding flips boundary
        # voxels early and they propagate (TF32 on: hippocampus Dice 0.91 vs the Keras reference on one scan)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device(device)
        self.model = load_weights(SynthSegRobust()).to(self.device)
        self.amp = bool(amp)

    @torch.no_grad()
    def posteriors(self, vols_xyz: list[np.ndarray], modalities: list[str]) -> np.ndarray:
        x = torch.from_numpy(np.stack([normalise(v, m) for v, m in zip(vols_xyz, modalities)])[:, None])
        x = x.to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            p = self.model(x)
        return p.float().cpu().numpy()

    @torch.no_grad()
    def segment_store(self, vols_zyx: list[np.ndarray], modalities: list[str]):
        """Store volumes (Z, Y, X) -> label maps (Z, Y, X) int16 and per-label volumes (ml); not pipelined."""
        x = self.prepare([np.transpose(v, (2, 1, 0)) for v in vols_zyx], modalities)
        return self.finish(*self.launch(x))

    def prepare(self, vols_xyz: list[np.ndarray], modalities: list[str]) -> torch.Tensor:
        """Normalised (B, 1, X, Y, Z) batch in pinned memory (CPU; loader threads call this)."""
        x = torch.from_numpy(np.stack([normalise(v, m) for v, m in zip(vols_xyz, modalities)])[:, None])
        return x.pin_memory() if self.device.type == "cuda" else x

    @torch.no_grad()
    def launch(self, x: torch.Tensor):
        """Queue the forward and the foreground mask (copied to pinned CPU memory, non-blocking); returns the GPU
        posteriors, the mask copies and an event that marks when the masks are on the CPU."""
        x = x.to(self.device, non_blocking=True)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            post = self.model(x).float()
        fg = post[:, 1:].sum(1) > 0.25
        fg_cpu = torch.empty(fg.shape, dtype=torch.bool, pin_memory=self.device.type == "cuda")
        fg_cpu.copy_(fg, non_blocking=True)
        ev = torch.cuda.Event() if self.device.type == "cuda" else None
        if ev is not None:
            ev.record()
        return post, fg_cpu, ev

    @torch.no_grad()
    def finish(self, post: torch.Tensor, fg_cpu: torch.Tensor, ev):
        """``postprocess(fast=True)`` on the GPU; only the largest-connected-component step runs on the CPU (it waits
        for the masks alone, so it overlaps a forward launched after this batch's)."""
        if ev is not None:
            ev.synchronize()
        labels = torch.as_tensor(LABELS, device=self.device)
        out = []
        for p, fg in zip(post, fg_cpu.numpy()):
            p[1:] *= torch.from_numpy(_largest_component(fg)).to(self.device)
            p[1:] *= p[1:] > 0.2
            p /= p.sum(0, keepdim=True).clamp_min(1e-12)
            seg = labels[p.argmax(0)].to(torch.int16).permute(2, 1, 0).contiguous().cpu().numpy()
            out.append((seg, (p[1:].sum(dim=(1, 2, 3)) / 1000.0).cpu().numpy()))
        return out


def main() -> None:
    raise SystemExit("validation: runs/diagnostics/synthseg_check.py (Keras reference in the TF venv)")


if __name__ == "__main__":
    main()
