"""CPU tests for the appearance augmentation (contrast curve, bias, blur, Rician noise, histogram matching, tied
views) and skull dropout."""

from __future__ import annotations

import math

import torch

from brain.common.windowing import MOD_CT, MOD_MRI
from brain.data import augment
from brain.data.augment import _foreground, appearance, histogram_match, rician_noise
from brain.data.volumes import tissue_quantiles


def _mri_rows(r=4, n=16):
    """MRI-like crops: background -6, a bright block of ramped tissue values."""
    x = torch.full((r, 1, n, n, n), -6.0)
    ramp = torch.linspace(-1.0, 3.0, 8 * 8 * 8).view(1, 1, 8, 8, 8).expand(r, -1, -1, -1, -1)
    x[:, :, 4:12, 4:12, 4:12] = ramp
    return x, torch.full((r,), MOD_MRI, dtype=torch.long)


def test_curve_is_monotone_without_flip_and_decreasing_with_flip():
    torch.manual_seed(0)
    y = torch.linspace(-2, 2, 4000).view(1, 1, 10, 20, 20).repeat(6, 1, 1, 1, 1)
    fg = torch.ones_like(y, dtype=torch.bool)
    up = augment.contrast_curve(y.clone(), fg, {"p": 1.0, "knots": 5, "strength": 0.8, "flip": 0.0})
    flat_in, flat_up = y[0].flatten(), up.flatten(1)
    assert (flat_up[:, 1:] >= flat_up[:, :-1] - 1e-5).all() and not torch.allclose(up, y)
    down = augment.contrast_curve(y.clone(), fg, {"p": 1.0, "knots": 5, "strength": 0.8, "flip": 1.0})
    assert (down.flatten(1)[:, 1:] <= down.flatten(1)[:, :-1] + 1e-5).all()
    # the median of the foreground keeps its value in both cases
    mid = flat_in.median()
    assert torch.allclose(up.flatten(1).median(1).values, mid.expand(6), atol=0.1)
    assert torch.allclose(down.flatten(1).median(1).values, mid.expand(6), atol=0.1)


def test_curve_leaves_undrawn_rows_alone_and_keeps_background_darkest():
    torch.manual_seed(1)
    x, mod = _mri_rows()
    y = torch.where(x > -5.5, x, torch.full_like(x, -4.0))  # windowed-like: background sits at -4
    fg = x > -5.5
    same = augment.contrast_curve(y.clone(), fg, {"p": 0.0})
    assert torch.equal(same, y)
    out = augment.contrast_curve(y.clone(), fg, {"p": 1.0, "knots": 5, "strength": 0.6, "flip": 0.5})
    for r in range(out.shape[0]):
        assert out[r][~fg[r]].max() < out[r][fg[r]].min()


def test_curve_ignores_crops_without_tissue():
    y = torch.full((2, 1, 8, 8, 8), -4.0)
    fg = torch.zeros_like(y, dtype=torch.bool)
    assert torch.equal(augment.contrast_curve(y.clone(), fg, {"p": 1.0}), y)


def test_bias_field_only_touches_mri_foreground_and_is_smooth():
    torch.manual_seed(2)
    x, _ = _mri_rows(r=2)
    mod = torch.tensor([MOD_CT, MOD_MRI])
    ct = x.clone()  # values are irrelevant: CT rows must be untouched
    out = augment.bias_field(x.clone(), mod, {"p": 1.0, "amp": 0.4})
    assert torch.equal(out[0], ct[0])                       # CT row unchanged
    assert not torch.allclose(out[1], x[1])                 # MRI row shaded
    assert torch.equal(out[1][x[1] == -6.0], x[1][x[1] == -6.0])  # background stays at the floor
    assert torch.isfinite(out).all()


def test_through_plane_blurs_only_the_thick_axis_and_skips_already_thick_scans():
    torch.manual_seed(3)
    tex = torch.randn(3, 1, 16, 16, 16)
    cfg = {"p": 1.0, "fwhm_vox": (6.0, 6.0)}
    thin = (torch.ones(3), torch.full((3,), 2, dtype=torch.long))            # 1 mm, z is the thick axis
    out = augment.through_plane(tex.clone(), cfg, thin)
    dz = (out[:, :, 1:] - out[:, :, :-1]).abs().mean()
    dy = (out[:, :, :, 1:] - out[:, :, :, :-1]).abs().mean()
    assert dz < 0.5 * dy and torch.isfinite(out).all()
    thick = (torch.full((3,), 8.0), torch.full((3,), 2, dtype=torch.long))   # already 8 mm >= target
    assert torch.equal(augment.through_plane(tex.clone(), cfg, thick), tex)


def test_appearance_does_not_modify_its_input_and_is_finite():
    torch.manual_seed(4)
    x, mod = _mri_rows()
    x0 = x.clone()
    aug = {"window": {"mode": "roles", "channels": 1}, "curve": {"p": 1.0, "flip": 0.5}, "bias": {"p": 1.0},
           "through_plane": {"p": 1.0}, "noise": {"p": 1.0, "sigma": (0.05, 0.05)}, "blur": {"p": 1.0},
           "drift": {"p": 1.0, "amp": 0.2}}
    a, b = augment.appearance(x, mod, aug), augment.appearance(x, mod, aug)
    assert torch.equal(x, x0)                                # the window pair relies on this
    assert torch.isfinite(a).all() and not torch.allclose(a, b)


def test_appearance_with_everything_off_is_just_the_window():
    x, mod = _mri_rows()
    torch.manual_seed(5)
    a = augment.appearance(x, mod, {"window": {"mode": "fixed"}})
    torch.manual_seed(5)
    assert torch.allclose(a, augment.window_view(x, mod))


def test_skull_dropout_cuts_to_the_brain_only_when_drawn():
    vol = torch.full((8, 8, 8), 30.0)
    brain = torch.zeros(8, 8, 8, dtype=torch.bool)
    brain[2:6, 2:6, 2:6] = True
    out = augment.skull_dropout(vol, brain, MOD_CT, p=1.0)
    assert (out[brain] == 30.0).all() and (out[~brain] == -1024.0).all()
    assert (augment.skull_dropout(vol, brain, MOD_MRI, p=1.0)[~brain] == -6.0).all()
    assert torch.equal(augment.skull_dropout(vol, brain, MOD_CT, p=0.0), vol)


AUG = {"window": {"mode": "roles", "jitter": 0.1}, "curve": {"p": 1.0, "knots": 5, "strength": 0.8, "flip": 0.5}}


def _vol(b=3):
    torch.manual_seed(0)
    return torch.randn(b, 1, 16, 24, 24) * 1.5, torch.ones(b, dtype=torch.long)      # MRI rows


def test_same_seed_same_contrast_on_crops_of_any_size():
    x, mod = _vol()
    full = appearance(x, mod, AUG, tie_seed=123)
    crop = appearance(x[..., 4:12, 6:18, 6:18].contiguous(), mod, AUG, tie_seed=123)
    # the curve is fitted on the row's foreground quantiles, so compare where both see the same tissue: rank order
    a, b = full[..., 4:12, 6:18, 6:18].flatten(1), crop.flatten(1)
    for r in range(x.shape[0]):
        ca = torch.corrcoef(torch.stack([a[r], b[r]]))[0, 1]
        assert ca > 0.95, ca                     # same curve (incl. flip) on every row


def test_untied_views_differ_and_global_stream_is_untouched():
    x, mod = _vol()
    torch.manual_seed(5)
    appearance(x, mod, AUG, tie_seed=7)
    after_tied = torch.rand(1)
    torch.manual_seed(5)
    appearance(x, mod, AUG, tie_seed=8)
    assert torch.equal(after_tied, torch.rand(1))  # the seeded draws do not advance the global stream differently
    u1, u2 = appearance(x, mod, AUG), appearance(x, mod, AUG)
    assert not torch.allclose(u1, u2)


def test_rician_is_rayleigh_in_air_and_gaussian_in_bright_tissue():
    torch.manual_seed(0)
    x = torch.full((2, 1, 16, 32, 32), -6.0)
    x[:, :, :, :16] = 4.0                                     # bright tissue: a = 10, far above the noise
    mod = torch.tensor([1, 0])                                # row 0 MRI, row 1 CT
    y, on = rician_noise(x.clone(), mod, {"p": 1.0, "snr": (20.0, 20.0)})
    sigma = 6.0 / 20.0
    air, tissue = y[0, 0, :, 16:], y[0, 0, :, :16]
    assert on.tolist() == [True, False]
    assert abs(float((air + 6.0).mean()) - sigma * math.sqrt(math.pi / 2)) < 0.02     # Rayleigh mean, biased up
    assert float((air + 6.0).min()) >= 0                                               # magnitude: never below air
    assert abs(float(tissue.std()) - sigma) < 0.02 and abs(float(tissue.mean()) - 4.0) < 0.02
    assert torch.equal(y[1], x[1])                                                     # CT untouched


def test_histogram_match_moves_tissue_quantiles_and_respects_groups():
    torch.manual_seed(0)
    x = torch.full((4, 1, 16, 32, 32), -6.0)
    x[0, :, :, 8:24, 8:24] = torch.randn(16, 16, 16) * 0.5         # T1, narrow
    x[1, :, :, 8:24, 8:24] = torch.randn(16, 16, 16) * 1.0 + 1.0   # T1 of another volume, wider and brighter
    x[2, :, :, 8:24, 8:24] = torch.randn(16, 16, 16) * 3.0         # FLAIR: no partner in its group
    x[3, :, :, 8:24, 8:24] = torch.randn(16, 16, 16) * 30 + 40     # CT
    mod, vol, con = torch.tensor([1, 1, 1, 0]), torch.arange(4), torch.tensor([1, 1, 4, 0])
    quant = torch.stack([tissue_quantiles(x[i, 0], int(mod[i])) for i in range(4)])
    fg = _foreground(x, mod)
    y = histogram_match(x.clone(), mod, fg, vol, con, quant, {"p": 1.0, "alpha": (1.0, 1.0)})
    q = torch.tensor([0.1, 0.5, 0.9])
    assert torch.allclose(torch.quantile(y[0][fg[0]], q), torch.quantile(x[1][fg[1]], q), atol=0.15)
    assert torch.equal(y[2], x[2]) and torch.equal(y[3], x[3])                        # other group / CT unchanged
    assert torch.equal(y[0][~fg[0]], x[0][~fg[0]])                                     # air stays air


def test_histogram_match_slopes_are_bounded():
    torch.manual_seed(0)
    x = torch.full((2, 1, 8, 32, 32), -6.0)
    x[:, :, :, 4:28, 4:28] = torch.randn(2, 1, 8, 24, 24)
    quant = torch.stack([torch.linspace(-2, 2, 32), torch.linspace(-2, 2, 32) * 20])  # partner 20x wider
    y = histogram_match(x.clone(), torch.tensor([1, 1]), _foreground(x, torch.tensor([1, 1])), torch.arange(2),
                        torch.tensor([1, 1]), quant, {"p": 1.0, "alpha": (1.0, 1.0), "max_slope": 3.0})
    fg = _foreground(x, torch.tensor([1, 1]))[0]
    ratio = (y[0][fg].std() / x[0][fg].std()).item()           # tissue only: air is not remapped
    assert 2.5 <= ratio <= 3.05                                # stretched by the 3x cap, not 20x


def test_appearance_runs_with_both():
    torch.manual_seed(1)
    x = torch.randn(4, 1, 16, 32, 32)
    mod = torch.tensor([1, 1, 0, 1])
    aug = {"window": {"mode": "roles", "channels": 1}, "noise": {"p": 1.0, "kind": "rician"},
           "histmatch": {"p": 1.0}}
    quant = torch.stack([tissue_quantiles(x[i, 0], int(mod[i])) for i in range(4)])
    y = appearance(x, mod, aug, None, torch.arange(4), torch.tensor([1, 1, 0, 1]), quant)
    assert y.shape == x.shape and torch.isfinite(y).all()
