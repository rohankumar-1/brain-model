"""GPU-side crop appearance: an asinh window, then scanner-like degradations.

Windows make a crop's contrast arbitrary. The degradations remove cues that identify a scanner, a
protocol or a site (noise level, sharpness, slow intensity drift), so agreeing across views cannot be
done by recognizing the volume's fingerprint. All shapes are (R, 1, Z, Y, X), one draw per row.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lib.windowing import asinh_window, sample_windows


def window_view(x: torch.Tensor, modality: torch.Tensor, aug: dict) -> torch.Tensor:
    win = aug.get("window") or {}
    b = x.shape[0]
    c, w = sample_windows(
        modality.reshape(b),
        str(win.get("mode", "random")),
        mix=str(win.get("mix", "brain")),
    )
    return asinh_window(x.float(), c.view(b, 1, 1, 1, 1), w.view(b, 1, 1, 1, 1))


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(lo, hi)


def _gaussian_blur(x: torch.Tensor, sigma: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Separable per-row Gaussian blur. ``sigma`` (R,) in voxels; a row with sigma 0 is unchanged."""
    r = x.shape[0]
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    w = torch.exp(-k.square()[None] / (2 * sigma.clamp_min(1e-3).square()[:, None]))
    w = w / w.sum(1, keepdim=True)
    y = x.transpose(0, 1)  # (1, R, Z, Y, X): one group per row
    for axis in (2, 3, 4):
        shape = [r, 1, 1, 1, 1]
        shape[axis] = 2 * radius + 1
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - axis): 2 * (4 - axis) + 2] = [radius, radius]
        y = F.conv3d(F.pad(y, pad, mode="replicate"), w.view(shape), groups=r)
    return y.transpose(0, 1)


def degrade(x: torch.Tensor, aug: dict) -> torch.Tensor:
    """Noise, blur and a smooth additive drift. Each is drawn per row with its own probability."""
    r, device = x.shape[0], x.device
    n, bl, dr = (aug.get(k) or {} for k in ("noise", "blur", "drift"))
    if bl and float(bl.get("p", 0)) > 0:
        on = torch.rand(r, device=device) < float(bl["p"])
        lo, hi = bl.get("sigma_vox", (0.5, 1.5))
        x = _gaussian_blur(x, torch.where(on, _uniform(r, lo, hi, device), torch.zeros(r, device=device)))
    if dr and float(dr.get("p", 0)) > 0:
        on = (torch.rand(r, device=device) < float(dr["p"])).view(r, 1, 1, 1, 1)
        field = torch.randn(r, 1, 3, 3, 3, device=device) * float(dr.get("amp", 0.25))
        x = x + on * F.interpolate(field, size=x.shape[2:], mode="trilinear", align_corners=True)
    if n and float(n.get("p", 0)) > 0:
        on = torch.rand(r, device=device) < float(n["p"])
        lo, hi = n.get("sigma", (0.02, 0.15))
        sigma = torch.where(on, _uniform(r, lo, hi, device), torch.zeros(r, device=device)).view(r, 1, 1, 1, 1)
        x = x + sigma * torch.randn_like(x)
    return x


def appearance(x: torch.Tensor, modality: torch.Tensor, aug: dict) -> torch.Tensor:
    """A window and the degradations, one independent draw per row."""
    return degrade(window_view(x, modality, aug), aug)
