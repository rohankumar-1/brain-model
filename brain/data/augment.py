"""Appearance augmentation for crops, on the GPU, plus the volume-level skull dropout used by the loader.

The goal is a crop whose *contrast* carries no information about which scanner, sequence, modality or dataset
it came from, so that agreement between views can only be reached through anatomy. Per crop row, in order:

1. ``bias``          smooth multiplicative gain (MRI coil shading), applied in native units.
2. ``through_plane`` Gaussian blur along the acquisition's thick axis, applied in native units, so thin-slice scans
                     also look thick-slice. The extra blur is the quadrature difference to a randomly drawn target
                     thickness, so a scan already at or beyond the target is left alone (uses ``thick_mm`` and
                     ``thick_axis`` recorded by prep; without them the axis is random).
3. ``window``        asinh window (``y = asinh((x - c) / w)``), random centre and width per row.
4. ``curve``         random monotone re-rendering of tissue contrast: a piecewise-linear map whose knots sit at quantiles
                     of the crop's own foreground intensities, so each tissue mode can be stretched or squeezed
                     independently. ``flip`` negates it about the median, which swaps the order of tissues as T1 -> T2
                     or CT-like -> MRI-like contrast does. Background stays the darkest value.
5. ``blur``/``drift``/``noise``  isotropic blur, a slow additive drift, and noise, in windowed units.
   With ``noise.kind: rician`` MRI rows instead get Rician noise in native units (before the window): the magnitude of
   the signal plus complex Gaussian noise, which is how MRI noise arises (biased upward and grainy in dark tissue and
   air, Gaussian in bright tissue). CT rows keep Gaussian noise.

Optional ``histmatch`` (MRI only, first, in native units): a row's tissue intensities are remapped to the quantiles of
another volume's row of the same contrast group in the batch (blended), so contrast and dynamic range of a real other
scanner are borrowed and "which scanner" stops being a cheap cue. Not CT: Hounsfield units are physical (blood ~60 HU).

All are drawn independently per row; shapes are (R, 1, Z, Y, X). ``skull`` is separate: with probability ``p`` a whole
volume is cut to its brain mask before cropping, so "has a skull" is not a modality cue.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from brain.common.windowing import MOD_CT, MOD_MRI, asinh_window, fg_thresh, floors, sample_windows

FWHM_TO_SIGMA = 1.0 / 2.3548
QUANTILE_SAMPLES = 16384  # foreground voxels sampled per row to place the curve knots


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(lo, hi)


def _enabled(cfg: dict | None) -> bool:
    return bool(cfg) and float(cfg.get("p", 0.0)) > 0.0


def _gaussian_blur(x: torch.Tensor, sigma: torch.Tensor, dims: tuple[int, ...] = (2, 3, 4)) -> torch.Tensor:
    """Separable per-row Gaussian blur along ``dims``. ``sigma`` (R,) in voxels; a row with sigma 0 is unchanged."""
    r = x.shape[0]
    radius = int(min(10, max(3, math.ceil(3.0 * float(sigma.max())))))
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    w = torch.exp(-k.square()[None] / (2 * sigma.clamp_min(1e-3).square()[:, None]))
    w = w / w.sum(1, keepdim=True)
    y = x.transpose(0, 1)  # (1, R, Z, Y, X): one group per row
    for axis in dims:
        shape = [r, 1, 1, 1, 1]
        shape[axis] = 2 * radius + 1
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - axis): 2 * (4 - axis) + 2] = [radius, radius]
        y = F.conv3d(F.pad(y, pad, mode="replicate"), w.view(shape), groups=r)
    return y.transpose(0, 1)


def _foreground(x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
    return x > fg_thresh(modality).to(x.device).view(-1, 1, 1, 1, 1)


def bias_field(x: torch.Tensor, modality: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Multiplicative shading on MRI rows, applied about the background floor so air stays air.

    For robust-z volumes (tissue median 0, floor -6) ``floor + (x - floor) * gain`` is a multiplicative field on a
    signal whose offset-to-spread ratio is about 6, which is a typical T1/T2 tissue signal-to-spread. ``amp`` is the
    std of the log-gain on a 4 x 4 x 4 control grid, upsampled trilinearly.
    """
    r = x.shape[0]
    on = (torch.rand(r, device=x.device) < float(cfg["p"])) & (modality.to(x.device) == MOD_MRI)
    idx = on.nonzero(as_tuple=True)[0]
    if idx.numel():
        low = torch.randn(idx.numel(), 1, 4, 4, 4, device=x.device) * float(cfg.get("amp", 0.3))
        gain = F.interpolate(low, size=x.shape[2:], mode="trilinear", align_corners=True).exp()
        fl = floors(modality.to(x.device))[idx].view(-1, 1, 1, 1, 1)
        x[idx] = fl + (x[idx] - fl) * gain
    return x


def through_plane(x: torch.Tensor, cfg: dict, thick: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
    """Blur along one axis so the effective slice thickness reaches a random target in ``fwhm_vox`` (mm = voxels)."""
    r, dev = x.shape[0], x.device
    lo, hi = cfg.get("fwhm_vox", (2.0, 6.0))
    if thick is None:
        native, axis = torch.ones(r, device=dev), torch.full((r,), -1, device=dev, dtype=torch.long)
    else:
        native, axis = thick[0].to(dev).float(), thick[1].to(dev).long()
    # volumes without a recorded axis: mostly axial stacks, so lean toward z
    guess = torch.where(torch.rand(r, device=dev) < 0.6, torch.full((r,), 2, device=dev), torch.randint(0, 3, (r,), device=dev))
    axis = torch.where(axis >= 0, axis, guess)
    extra = (_uniform(r, lo, hi, dev).square() - native.square()).clamp_min(0).sqrt()
    sigma = extra * FWHM_TO_SIGMA
    on = (torch.rand(r, device=dev) < float(cfg["p"])) & (sigma > 0.25)
    for ax in (0, 1, 2):  # grid axis -> tensor dim: x is dim 4, y dim 3, z dim 2
        idx = (on & (axis == ax)).nonzero(as_tuple=True)[0]
        if idx.numel():
            x[idx] = _gaussian_blur(x[idx], sigma[idx], dims=(4 - ax,))
    return x


def window_view(x: torch.Tensor, modality: torch.Tensor, aug: dict) -> torch.Tensor:
    win = aug.get("window") or {}
    b = x.shape[0]
    c, w = sample_windows(modality.reshape(b), str(win.get("mode", "random")), mix=str(win.get("mix", "brain")))
    return asinh_window(x.float(), c.view(b, 1, 1, 1, 1), w.view(b, 1, 1, 1, 1))


def contrast_curve(y: torch.Tensor, fg: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Random monotone tissue-contrast curve (optionally inverted) on windowed values ``y`` (R, 1, Z, Y, X).

    Knots ``t_0 < ... < t_{K-1}`` are foreground quantiles of each row. Segment ``k`` is scaled by a log-normal slope
    ``s_k`` (std ``strength`` in log space): ``out = y + sum_k (s_k - 1) * clamp(y - t_k, 0, t_{k+1} - t_k)``, which is
    strictly increasing and the identity outside ``[t_0, t_{K-1}]``. With probability ``flip`` the result is negated
    about its value at the median. The row's background is then set below its darkest foreground value.
    """
    r, dev = y.shape[0], y.device
    k = max(3, int(cfg.get("knots", 5)))
    strength, flip_p = float(cfg.get("strength", 0.6)), float(cfg.get("flip", 0.3))
    on = torch.rand(r, device=dev) < float(cfg["p"])
    flat_fg = fg.flatten(1)
    on &= flat_fg.sum(1) >= 16  # a nearly empty crop has no tissue to re-render
    idx = on.nonzero(as_tuple=True)[0]
    if not idx.numel():
        return y
    yy, ff = y[idx], fg[idx]
    stride = max(1, yy[0].numel() // QUANTILE_SAMPLES)
    s, f = yy.flatten(1)[:, ::stride], ff.flatten(1)[:, ::stride]
    srt = torch.where(f, s, torch.full_like(s, float("inf"))).sort(1).values
    n_fg = f.sum(1)
    ok = n_fg >= 8                                           # sampled foreground must be enough to place knots
    cnt = n_fg.clamp_min(2)
    q = torch.linspace(0.05, 0.95, k, device=dev)
    pos = (q[None] * (cnt[:, None] - 1).float()).long().clamp_max(srt.shape[1] - 1)
    t = torch.where(ok[:, None], srt.gather(1, pos), torch.zeros_like(pos, dtype=srt.dtype))  # (m, K)
    seg = (t[:, 1:] - t[:, :-1]).clamp_min(0)                # (m, K-1)
    slope = torch.where(ok[:, None], torch.exp(strength * torch.randn(idx.numel(), k - 1, device=dev)),
                        torch.ones(idx.numel(), k - 1, device=dev))
    out = yy.clone()
    for j in range(k - 1):
        rise = (yy - t[:, j].view(-1, 1, 1, 1, 1)).clamp_min(0)
        out += (slope[:, j] - 1).view(-1, 1, 1, 1, 1) * torch.minimum(rise, seg[:, j].view(-1, 1, 1, 1, 1))
    mid = t[:, k // 2]
    out_mid = mid + ((slope - 1) * (mid[:, None] - t[:, :-1]).clamp_min(0).minimum(seg)).sum(1)
    sign = torch.where((torch.rand(idx.numel(), device=dev) < flip_p) & ok, -1.0, 1.0).view(-1, 1, 1, 1, 1)
    out = sign * (out - out_mid.view(-1, 1, 1, 1, 1)) + mid.view(-1, 1, 1, 1, 1)
    dark = torch.where(ff, out, torch.full_like(out, float("inf"))).amin(dim=(1, 2, 3, 4)) - 0.25
    new = torch.where(ff, out, dark.view(-1, 1, 1, 1, 1))
    y[idx] = torch.where(ok.view(-1, 1, 1, 1, 1), new, yy)
    return y


def rician_noise(x: torch.Tensor, modality: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Rician noise on MRI rows in native robust-z units. Returns the volume and the rows that got it.

    The stored MRI is a robust z-score with air at the floor (-6), so ``a = x - floor`` is a non-negative
    magnitude-like signal whose tissue sits near 6. With ``snr`` drawn per row, ``sigma = 6 / snr`` and the result is
    ``floor + |a + n1 + i n2|`` for independent Gaussian ``n1, n2`` of that sigma: air becomes Rayleigh noise above
    the floor and dark tissue is lifted, as on a real magnitude image."""
    r, dev = x.shape[0], x.device
    on = (torch.rand(r, device=dev) < float(cfg["p"])) & (modality.to(dev) == MOD_MRI)
    idx = on.nonzero(as_tuple=True)[0]
    if idx.numel():
        lo, hi = cfg.get("snr", (15.0, 100.0))
        snr = torch.exp(_uniform(idx.numel(), math.log(float(lo)), math.log(float(hi)), dev))
        sigma = (6.0 / snr).view(-1, 1, 1, 1, 1)
        fl = floors(modality.to(dev))[idx].view(-1, 1, 1, 1, 1)
        a = (x[idx] - fl).clamp_min(0)
        x[idx] = fl + torch.sqrt((a + sigma * torch.randn_like(a)).square() + (sigma * torch.randn_like(a)).square())
    return x, on


def histogram_match(x: torch.Tensor, modality: torch.Tensor, fg: torch.Tensor, vol: torch.Tensor,
                    contrast: torch.Tensor, quant: torch.Tensor, cfg: dict) -> torch.Tensor:
    """MRI rows: remap intensities from their volume's tissue distribution onto another volume's (same contrast
    group), blended. ``quant`` (R, Q) are whole-volume tissue quantiles per row (``volumes.tissue_quantiles``), so the
    map compares scanners, not the anatomy two crops happen to contain. The map is piecewise linear between quantiles,
    each segment's slope clamped to [1/max_slope, max_slope] so sparse intensity ranges (partial-volume CSF) are not
    stretched into speckle, and shifted (slope 1) beyond the tissue range. Air stays air; rows without a valid partner
    are left alone. ``alpha`` ~ U(alpha) blends matched and original values."""
    r, dev = x.shape[0], x.device
    mri = modality.to(dev) == MOD_MRI
    quant = quant.to(dev).float()
    valid = mri & torch.isfinite(quant).all(1)
    on = (torch.rand(r, device=dev) < float(cfg["p"])) & valid
    if not bool(on.any()):
        return x
    vol, contrast = vol.to(dev), contrast.to(dev)
    ok_pair = (contrast[:, None] == contrast[None, :]) & (vol[:, None] != vol[None, :]) & valid[None, :] \
        & (contrast[:, None] > 0)
    score = torch.rand(r, r, device=dev).masked_fill(~ok_pair, -1.0)
    best, partner = score.max(1)
    idx = (on & (best >= 0)).nonzero(as_tuple=True)[0]
    if not idx.numel():
        return x
    q = quant.shape[1]
    src = torch.cummax(quant[idx], 1).values + torch.arange(q, device=dev) * 1e-5       # strictly increasing knots
    ref = torch.cummax(quant[partner[idx]], 1).values
    max_slope = float(cfg.get("max_slope", 3.0))
    slope = ((ref[:, 1:] - ref[:, :-1]) / (src[:, 1:] - src[:, :-1])).clamp(1.0 / max_slope, max_slope)
    ref = torch.cat([ref[:, :1], ref[:, :1] + torch.cumsum(slope * (src[:, 1:] - src[:, :-1]), 1)], 1)
    rows = x[idx].flatten(1)
    k = torch.searchsorted(src.contiguous(), rows.contiguous()).clamp(1, q - 1)
    x0, y0 = src.gather(1, k - 1), ref.gather(1, k - 1)
    mapped = y0 + (rows - x0) * slope.gather(1, k - 1)
    mapped = torch.where(rows < src[:, :1], rows - src[:, :1] + ref[:, :1], mapped)
    mapped = torch.where(rows > src[:, -1:], rows - src[:, -1:] + ref[:, -1:], mapped)
    lo, hi = cfg.get("alpha", (0.5, 1.0))
    a = _uniform(idx.numel(), float(lo), float(hi), dev)[:, None]
    out = (1 - a) * rows + a * mapped
    x[idx] = torch.where(fg[idx].flatten(1), out, rows).view_as(x[idx])               # air stays air
    return x


def degrade(x: torch.Tensor, aug: dict, skip_noise: torch.Tensor | None = None) -> torch.Tensor:
    """Isotropic blur, a smooth additive drift, and noise, in windowed units. Each is drawn per row.
    ``skip_noise`` marks rows that already got (Rician) noise in native units."""
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
        if skip_noise is not None:
            on &= ~skip_noise.to(device)
        lo, hi = n.get("sigma", (0.02, 0.15))
        sigma = torch.where(on, _uniform(r, lo, hi, device), torch.zeros(r, device=device)).view(r, 1, 1, 1, 1)
        x = x + sigma * torch.randn_like(x)
    return x


def appearance(x: torch.Tensor, modality: torch.Tensor, aug: dict,
               thick: tuple[torch.Tensor, torch.Tensor] | None = None,
               vol: torch.Tensor | None = None, contrast: torch.Tensor | None = None,
               quant: torch.Tensor | None = None) -> torch.Tensor:
    """Full appearance pipeline, one independent draw per row. ``x`` is never modified in place.
    ``vol`` / ``contrast`` / ``quant`` (per row) enable histogram matching against another volume of the same contrast
    group, using whole-volume tissue quantiles."""
    x = x.float().clone() if x.dtype == torch.float32 else x.float()
    fg = _foreground(x, modality)
    if _enabled(aug.get("histmatch")) and vol is not None and contrast is not None and quant is not None:
        x = histogram_match(x, modality, fg, vol, contrast, quant, aug["histmatch"])
    if _enabled(aug.get("bias")):
        x = bias_field(x, modality, aug["bias"])
    if _enabled(aug.get("through_plane")):
        x = through_plane(x, aug["through_plane"], thick)
    noisy = None
    noise = aug.get("noise") or {}
    if _enabled(noise) and str(noise.get("kind", "gaussian")) == "rician":
        x, noisy = rician_noise(x, modality, noise)
    y = window_view(x, modality, aug)
    if _enabled(aug.get("curve")):
        y = contrast_curve(y, fg, aug["curve"])
    return degrade(y, aug, skip_noise=noisy)


def skull_dropout(volume: torch.Tensor, brain: torch.Tensor, modality: int, p: float,
                  gen: torch.Generator | None = None) -> torch.Tensor:
    """With probability ``p``, set everything outside ``brain`` to the background floor. ``volume`` is (Z, Y, X)."""
    if p <= 0 or float(torch.rand(1, generator=gen)) >= p:
        return volume
    floor = -1024.0 if modality == MOD_CT else -6.0
    return torch.where(brain.bool(), volume, volume.new_tensor(floor))
