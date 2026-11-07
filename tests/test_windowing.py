"""CPU tests for v4 asinh windows and floor-padded warps. python -m tests.test_windowing"""

from __future__ import annotations

import torch

from brain.common.windowing import asinh_window, canonical_windows, foreground_mask
from brain.objectives.sigreg import prediction_loss


def test_asinh_is_monotone_and_finite():
    x = torch.linspace(-1024, 3071, 256)
    y = asinh_window(x, 40.0, 40.0)
    assert torch.isfinite(y).all()
    assert torch.diff(y).min() > 0


def test_asinh_inverse_recovers_x():
    x = torch.linspace(-200, 800, 64)
    c, w = 40.0, 80.0
    y = asinh_window(x, c, w)
    rec = y.sinh() * w + c
    assert torch.allclose(rec, x, atol=1e-4)


def test_floor_maps_to_a_constant():
    ct = torch.full((4, 1, 8, 8), -1024.0)
    mri = torch.full((4, 1, 8, 8), -6.0)
    y_ct = asinh_window(ct, 40.0, 40.0)
    y_mri = asinh_window(mri, 0.5, 1.5)
    assert torch.allclose(y_ct, y_ct.reshape(4, -1)[:, :1].view(4, 1, 1, 1))
    assert torch.allclose(y_mri, y_mri.reshape(4, -1)[:, :1].view(4, 1, 1, 1))


def test_canonical_windows_by_modality():
    c, w = canonical_windows(torch.tensor([0, 0, 1, 1]))
    assert c.shape == (4, 1, 1, 1) and w.shape == (4, 1, 1, 1)
    assert torch.equal(c.view(-1), torch.tensor([40.0, 40.0, 0.5, 0.5]))
    assert torch.equal(w.view(-1), torch.tensor([40.0, 40.0, 1.5, 1.5]))


def test_window_pair_loss_is_zero_when_views_match():
    z = torch.randn(2, 8, 16)
    z[1] = z[0]
    assert prediction_loss(z).item() < 1e-6



def test_foreground_mask_by_modality():
    x = torch.tensor([-900.0, 40.0, -6.0, 0.5]).view(4, 1, 1, 1)
    mod = torch.tensor([0, 0, 1, 1])
    fg = foreground_mask(x, mod).view(-1)
    assert torch.equal(fg, torch.tensor([False, True, False, True]))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
