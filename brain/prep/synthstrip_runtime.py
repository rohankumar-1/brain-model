"""In-process SynthStrip: load the UNet once, infer one volume or a padded batch.

The CLI (`mri_synthstrip`) reloads PyTorch + weights per volume. Remask was spending most of
its 7–8 s/vol on that. Geometry / conform / CC follow the official script so masks match.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from brain.prep.brainmask import BrainMaskUnavailable


def _model_path(no_csf: bool = False) -> Path:
    home = os.environ.get("FREESURFER_HOME")
    if not home:
        raise BrainMaskUnavailable("FREESURFER_HOME is not set")
    name = "synthstrip.nocsf.1.pt" if no_csf else "synthstrip.1.pt"
    path = Path(home) / "models" / name
    if not path.is_file():
        raise BrainMaskUnavailable(f"missing SynthStrip weights {path}")
    return path


class ConvBlock(nn.Module):
    def __init__(self, ndims, in_channels, out_channels, stride=1, activation="leaky"):
        super().__init__()
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.conv = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = None if activation is None else nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv(x)
        return out if self.activation is None else self.activation(out)


class StripModel(nn.Module):
    """Official SynthStrip 3D UNet (Hoopes et al., NeuroImage 2022)."""

    def __init__(self, nb_features=16, nb_levels=7, feat_mult=2, max_features=64,
                 nb_conv_per_level=2, max_pool=2, return_mask=False):
        super().__init__()
        ndims = 3
        feats = np.clip(np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int), 1, max_features)
        enc_nf = np.repeat(feats[:-1], nb_conv_per_level)
        dec_nf = np.repeat(np.flip(feats), nb_conv_per_level)
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1
        max_pool = [max_pool] * self.nb_levels
        MaxPooling = getattr(nn, "MaxPool%dd" % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode="nearest") for s in max_pool]
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if level < (self.nb_levels - 1):
                prev_nf += encoder_nfs[level]
        self.remaining = nn.ModuleList()
        for nf in final_convs:
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
        if return_mask:
            self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
            self.remaining.append(nn.Softmax(dim=1))
        else:
            self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))

    def forward(self, x):
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if level < (self.nb_levels - 1):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)
        for conv in self.remaining:
            x = conv(x)
        return x


def pad_stack(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
    """Pad (1,1,D,H,W) tensors to a common multiple-of-64 shape and stack to (B,1,D,H,W)."""
    shapes = [tuple(t.shape[-3:]) for t in tensors]
    md, mh, mw = (max(s[i] for s in shapes) for i in range(3))
    md, mh, mw = (int(np.ceil(v / 64) * 64) for v in (md, mh, mw))
    out = []
    for t in tensors:
        t = t.float()
        d, h, w = t.shape[-3:]
        out.append(torch.nn.functional.pad(t, (0, mw - w, 0, mh - h, 0, md - d)))
    return torch.cat(out, dim=0), shapes


def _extend_sdt(sdt, border=1):
    import surfa as sf

    if border < int(sdt.max()):
        return sdt
    mask = sdt < 1
    keep = np.nonzero(mask)
    low = np.min(keep, axis=-1)
    upp = np.max(keep, axis=-1)
    gap = int(border + 0.5)
    low = (max(i - gap, 0) for i in low)
    upp = (min(i + gap, d - 1) for i, d in zip(upp, mask.shape))
    ind = tuple(slice(a, b + 1) for a, b in zip(low, upp))
    out = np.full_like(sdt, fill_value=100)
    out[ind] = sf.Volume(mask[ind]).distance()
    out[keep] = sdt[keep]
    return sdt.new(out)


def _volume_from_zyx(raw: np.ndarray):
    """(Z, Y, X) store array -> surfa volume. The store grid is RAS (x -> Right, y -> Anterior, z -> Superior).

    The orientation must be explicit: surfa's default geometry is LIA, which makes ``conform(orientation="LIA")``
    a no-op and hands the UNet a head rotated 90 degrees in the sagittal plane.
    """
    import surfa as sf
    from surfa.transform.geometry import ImageGeometry

    data = np.ascontiguousarray(np.transpose(np.asarray(raw, dtype=np.float32), (2, 1, 0)))
    return sf.Volume(data, geometry=ImageGeometry(shape=data.shape[:3], voxsize=1.0, rotation="RAS"))


def _zyx_from_volume(vol) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(np.asarray(vol.data), (2, 1, 0)))


class SynthStripSession:
    """One UNet on one device. ``run_batch`` pads heterogeneous volumes into a single forward."""

    def __init__(self, *, gpu: bool = True, threads: int = 1, no_csf: bool = False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        if gpu:
            if not torch.cuda.is_available():
                raise BrainMaskUnavailable("SYNTHSTRIP_GPU set but CUDA is not available")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if threads is not None:
            torch.set_num_threads(max(1, threads))
        self.model = StripModel()
        self.model.to(self.device)
        self.model.eval()
        path = _model_path(no_csf)
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.border = 1.0

    def _prepare(self, raw: np.ndarray, modality: str):
        a = np.clip(raw, -200.0, 300.0) if modality == "ct" else raw
        image = _volume_from_zyx(a)
        conformed = image.conform(voxsize=1.0, dtype="float32", method="nearest", orientation="LIA")
        conformed = conformed.crop_to_bbox()
        target_shape = tuple(int(x) for x in np.clip(
            np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64, 192, 320))
        conformed = conformed.reshape(target_shape)
        conformed -= conformed.min()
        denom = float(conformed.percentile(99)) or 1.0
        conformed = (conformed / denom).clip(0, 1)
        arr = np.asarray(conformed.data, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[..., 0]
        inp = torch.from_numpy(np.ascontiguousarray(arr)[np.newaxis, np.newaxis])
        return image, conformed, inp

    def _finish(self, image, conformed, sdt_cpu: torch.Tensor) -> np.ndarray:
        sdt = _extend_sdt(conformed.new(sdt_cpu.numpy()), border=self.border)
        sdt = sdt.resample_like(image, fill=100)
        mask = (sdt < self.border).connected_component_mask(k=1, fill=True)
        return _zyx_from_volume(mask) > 0

    def run(self, raw: np.ndarray, modality: str) -> np.ndarray:
        return self.run_batch([raw], [modality])[0]

    def run_batch(self, volumes: list[np.ndarray], modalities: list[str]) -> list[np.ndarray]:
        if not volumes:
            return []
        try:
            return self._infer_batch(volumes, modalities)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(volumes) == 1:
                raise
            mid = max(1, len(volumes) // 2)
            return self.run_batch(volumes[:mid], modalities[:mid]) + self.run_batch(volumes[mid:], modalities[mid:])

    def _infer_batch(self, volumes, modalities):
        prepared = [self._prepare(v, m) for v, m in zip(volumes, modalities)]
        stacked, shapes = pad_stack([p[2] for p in prepared])
        with torch.no_grad():
            pred = self.model(stacked.to(self.device, non_blocking=True)).cpu()
        out = []
        for i, (image, conformed, _) in enumerate(prepared):
            d, h, w = shapes[i]
            sdt = pred[i, 0, :d, :h, :w].reshape(d, h, w)
            out.append(self._finish(image, conformed, sdt))
        return out


_SESSION: SynthStripSession | None = None


def no_csf() -> bool:
    """``SYNTHSTRIP_NOCSF=1`` selects the model that excludes extra-cerebral CSF (a tighter, BraTS-like boundary)."""
    return os.environ.get("SYNTHSTRIP_NOCSF", "").strip().lower() in {"1", "true", "yes"}


def session(*, gpu: bool | None = None) -> SynthStripSession:
    global _SESSION
    if _SESSION is None:
        if gpu is None:
            gpu = os.environ.get("SYNTHSTRIP_GPU", "").strip().lower() in {"1", "true", "yes"}
        _SESSION = SynthStripSession(gpu=gpu, threads=int(os.environ.get("OMP_NUM_THREADS", 1)), no_csf=no_csf())
    return _SESSION
