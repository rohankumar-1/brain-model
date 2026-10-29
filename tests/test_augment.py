"""CPU tests for the appearance augmentation and skull dropout."""

from __future__ import annotations

import torch

from brain.data import augment
from brain.common.windowing import MOD_CT, MOD_MRI


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
    aug = {"window": {"mode": "random", "mix": "brain"}, "curve": {"p": 1.0, "flip": 0.5}, "bias": {"p": 1.0},
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
    assert torch.allclose(a, augment.window_view(x, mod, {"window": {"mode": "fixed"}}))


def test_skull_dropout_cuts_to_the_brain_only_when_drawn():
    vol = torch.full((8, 8, 8), 30.0)
    brain = torch.zeros(8, 8, 8, dtype=torch.bool)
    brain[2:6, 2:6, 2:6] = True
    out = augment.skull_dropout(vol, brain, MOD_CT, p=1.0)
    assert (out[brain] == 30.0).all() and (out[~brain] == -1024.0).all()
    assert (augment.skull_dropout(vol, brain, MOD_MRI, p=1.0)[~brain] == -6.0).all()
    assert torch.equal(augment.skull_dropout(vol, brain, MOD_CT, p=0.0), vol)
