"""GPU views for LeJEPA: affine crop, native-unit noise/blur, then an asinh window.

Crop ``area`` is a fraction of the image area (as in DINO / torchvision), not of
the side length. Local crops are centred on foreground pixels because most of a
slice is empty background. ``photometric`` is the v3 [0, 1] jitter still used
by the segmentation trainer.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from brain.common.windowing import asinh_window, floors, foreground_mask, sample_windows


def affine_theta(
    b: int,
    area: tuple[float, float],
    *,
    device: torch.device,
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
    max_rot: float = 10.0,
    hflip: float = 0.5,
    centers: torch.Tensor | None = None,
) -> torch.Tensor:
    """(B, 2, 3) theta for a window covering ``area`` of the image.

    Output coords -> flip -> scale -> rotate -> translate, all in [-1, 1] image coords.
    """
    def uni(lo: float, hi: float) -> torch.Tensor:
        return torch.empty(b, device=device).uniform_(lo, hi)

    a, r = uni(*area), uni(math.log(ratio[0]), math.log(ratio[1])).exp()
    sx, sy = (a * r).sqrt().clamp(max=1.0), (a / r).sqrt().clamp(max=1.0)
    ang = uni(-max_rot, max_rot) * math.pi / 180
    flip = torch.where(torch.rand(b, device=device) < hflip, -1.0, 1.0)
    if centers is None:
        centers = torch.stack([uni(-1, 1), uni(-1, 1)], dim=1)
    cx = centers[:, 0].clamp(min=-(1 - sx), max=1 - sx)
    cy = centers[:, 1].clamp(min=-(1 - sy), max=1 - sy)
    cos, sin = ang.cos(), ang.sin()
    row0 = torch.stack([cos * sx * flip, -sin * sy, cx], dim=1)
    row1 = torch.stack([sin * sx * flip, cos * sy, cy], dim=1)
    return torch.stack([row0, row1], dim=1)


def warp(x: torch.Tensor, theta: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    grid = F.affine_grid(theta, (x.size(0), 1, size, size), align_corners=False)
    return F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)


def warp_floor(x: torch.Tensor, theta: torch.Tensor, size: int, floor: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Warp native units with padding equal to the modality floor, not zero."""
    floor = floor.to(dtype=x.dtype, device=x.device)
    while floor.ndim < x.ndim:
        floor = floor.view(*floor.shape, *([1] * (x.ndim - floor.ndim)))
    return warp(x - floor, theta, size, mode=mode) + floor


def foreground_centers(x: torch.Tensor, thresh: float = 0.02) -> torch.Tensor:
    """One random foreground pixel per image, as (B, 2) (x, y) in [-1, 1]."""
    return _centers_from_mask(x.flatten(1) > thresh, x.shape[-2], x.shape[-1])


def _centers_from_mask(flat: torch.Tensor, h: int, w: int) -> torch.Tensor:
    idx = torch.multinomial(flat.float() + 1e-6, 1).squeeze(1)
    cy, cx = (idx // w).float(), (idx % w).float()
    return torch.stack([(cx + 0.5) / w * 2 - 1, (cy + 0.5) / h * 2 - 1], dim=1)


def photometric(x: torch.Tensor, p_blur: float = 0.5) -> torch.Tensor:
    """Gamma, gain, blur and foreground-only noise. Background zeros stay zero."""
    b = x.size(0)

    def uni(lo: float, hi: float) -> torch.Tensor:
        return torch.empty(b, 1, 1, 1, device=x.device).uniform_(lo, hi)

    fg = (x > 0).to(x.dtype)
    x = x.clamp(0, 1) ** uni(-0.4, 0.4).exp() * uni(0.8, 1.2)
    w = uni(0, 1) * (torch.rand(b, 1, 1, 1, device=x.device) < p_blur)
    x = x + w * (F.avg_pool2d(x, 5, 1, 2) - x)
    return (x + torch.randn_like(x) * uni(0, 0.03) * fg).clamp(0, 1)


def noise_blur(x: torch.Tensor, mod: torch.Tensor, p_blur: float = 0.5) -> torch.Tensor:
    """Native-unit blur (p=0.5) and foreground noise. CT σ in HU, MRI σ in robust z."""
    b = x.size(0)

    def uni(lo: float, hi: float) -> torch.Tensor:
        return torch.empty(b, 1, 1, 1, device=x.device).uniform_(lo, hi)

    is_ct = (mod == 0).view(b, 1, 1, 1)
    sigma = torch.where(is_ct, uni(0.0, 10.0), uni(0.0, 0.15))
    mix = uni(0, 1) * (torch.rand(b, 1, 1, 1, device=x.device) < p_blur)
    x = x + mix * (F.avg_pool2d(x, 5, 1, 2) - x)
    fg = foreground_mask(x, mod).to(dtype=x.dtype)
    return x + torch.randn_like(x) * sigma * fg


def apply_window(x: torch.Tensor, mod: torch.Tensor, mode: str = "random",
                 cw: tuple[torch.Tensor, torch.Tensor] | None = None, mix: str = "v4") -> torch.Tensor:
    c, w = cw if cw is not None else sample_windows(mod, mode, mix=mix)
    return asinh_window(x, c, w)


def multi_crop(
    x: torch.Tensor,
    mod: torch.Tensor,
    n_global: int = 2,
    n_local: int = 6,
    global_area: tuple[float, float] = (0.4, 1.0),
    local_area: tuple[float, float] = (0.05, 0.4),
    local_size: int = 96,
    max_rot: float = 10.0,
    window: str = "random",
    window_mix: str = "v4",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``x`` is (B, 1, H, W) native units, ``mod`` is (B,) 0=CT / 1=MRI.

    ``window`` is ``fixed`` (canonical), ``random`` (independent per view), or
    ``slice`` (one draw shared by every view of a slice).
    """
    b, size = x.size(0), x.size(-1)
    floor = floors(mod).view(b, 1, 1, 1)
    shared = sample_windows(mod, "random", mix=window_mix) if window == "slice" else None
    win_mode = "fixed" if window == "fixed" else "random"
    fg = foreground_mask(x, mod)

    def view(area, out, centers=None):
        theta = affine_theta(b, area, device=x.device, max_rot=max_rot, centers=centers)
        y = noise_blur(warp_floor(x, theta, out, floor), mod)
        return apply_window(y, mod, win_mode, shared, mix=window_mix)

    globals_ = torch.stack([view(global_area, size) for _ in range(n_global)])
    if n_local <= 0:
        return globals_, None
    locals_ = torch.stack(
        [view(local_area, local_size, _centers_from_mask(fg.flatten(1), *x.shape[-2:])) for _ in range(n_local)]
    )
    return globals_, locals_


def same_crop_window_pair(
    x: torch.Tensor,
    mod: torch.Tensor,
    area: tuple[float, float] = (0.4, 1.0),
    max_rot: float = 10.0,
    window_mix: str = "v4",
    size: int | None = None,
) -> torch.Tensor:
    """One warp, two independent asinh windows. Returns (2, B, 1, H, W).

    Isolates photometric invariance: same crop and noise, different (c, w).
    Always draws random windows so the pair is a no-op under ``mode=fixed``.
    """
    b, size = x.size(0), size or x.size(-1)
    floor = floors(mod).view(b, 1, 1, 1)
    theta = affine_theta(b, area, device=x.device, max_rot=max_rot)
    y = noise_blur(warp_floor(x, theta, size, floor), mod)
    return torch.stack([apply_window(y, mod, "random", mix=window_mix) for _ in range(2)])


def same_crop_seq_pair(
    x: torch.Tensor,
    sib: torch.Tensor,
    mod: torch.Tensor,
    area: tuple[float, float] = (0.4, 1.0),
    max_rot: float = 10.0,
    window_mix: str = "v4",
    size: int | None = None,
) -> torch.Tensor:
    """One warp and one window, two co-registered sequences. Returns (2, B, 1, H, W).

    Isolates sequence invariance: same geometry and (c, w), different contrast.
    """
    b, size = x.size(0), size or x.size(-1)
    floor = floors(mod).view(b, 1, 1, 1)
    theta = affine_theta(b, area, device=x.device, max_rot=max_rot)
    cw = sample_windows(mod, "random", mix=window_mix)

    def one(vol):
        y = noise_blur(warp_floor(vol, theta, size, floor), mod)
        return apply_window(y, mod, "random", cw, mix=window_mix)

    return torch.stack([one(x), one(sib)])
