"""CPU tests for v4 asinh windows and floor-padded warps. python -m tests.test_windowing"""

from __future__ import annotations

import torch

from brain.common.windowing import WINDOW_MIXES, asinh_window, floors, foreground_mask, sample_windows
from brain.data.crops2d import same_crop_seq_pair, same_crop_window_pair, warp_floor
from brain.objectives.sigreg import pairwise_cosine, prediction_loss


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


def test_sample_windows_shapes_and_positive_w():
    mod = torch.tensor([0, 0, 1, 1])
    c, w = sample_windows(mod, "random")
    assert c.shape == (4, 1, 1, 1) and w.shape == (4, 1, 1, 1)
    assert (w > 0).all()
    cf, wf = sample_windows(mod, "fixed")
    assert torch.equal(cf.view(-1)[:2], torch.tensor([40.0, 40.0]))
    assert torch.equal(wf.view(-1)[2:], torch.tensor([1.5, 1.5]))


def test_tissue_mix_is_brain_heavy():
    ct = sum(WINDOW_MIXES["tissue"]["ct_p"])
    v4 = sum(WINDOW_MIXES["v4"]["ct_p"])
    assert abs(ct - 1.0) < 1e-9 and abs(v4 - 1.0) < 1e-9
    assert WINDOW_MIXES["tissue"]["ct_p"][0] > WINDOW_MIXES["v4"]["ct_p"][0]
    assert WINDOW_MIXES["tissue"]["ct_p"][4] < WINDOW_MIXES["v4"]["ct_p"][4]
    assert WINDOW_MIXES["tissue"]["mri_tissue_p"] > WINDOW_MIXES["v4"]["mri_tissue_p"]
    assert WINDOW_MIXES["tissue"]["mri_tissue_w_log"][0] > WINDOW_MIXES["v4"]["mri_tissue_w_log"][0]
    torch.manual_seed(0)
    c, w = sample_windows(torch.zeros(8, dtype=torch.long), "random", mix="tissue")
    assert (w > 0).all() and c.shape == (8, 1, 1, 1)


def test_warp_floor_padding_is_the_modality_floor():
    x = torch.full((2, 1, 16, 16), 40.0)
    mod = torch.tensor([0, 1])
    x[1] = 1.0
    floor = floors(mod).view(2, 1, 1, 1)
    theta = torch.zeros(2, 2, 3)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = 1.5  # shift out so most of the crop is padding
    y = warp_floor(x, theta, 16, floor)
    pad = (y[0] - (-1024.0)).abs() < 1e-3
    assert pad.float().mean() > 0.4
    assert ((y[1] + 6.0).abs() < 1e-3).float().mean() > 0.4


def test_same_crop_window_pair_disagrees_photometrically():
    torch.manual_seed(0)
    x = torch.zeros(4, 1, 32, 32)
    x[:, :, 8:24, 8:24] = 80.0
    mod = torch.zeros(4, dtype=torch.long)
    pair = same_crop_window_pair(x, mod, area=(1.0, 1.0), max_rot=0.0)
    assert pair.shape == (2, 4, 1, 32, 32)
    assert torch.isfinite(pair).all()
    assert not torch.allclose(pair[0], pair[1])


def test_window_pair_loss_is_zero_when_views_match():
    z = torch.randn(2, 8, 16)
    z[1] = z[0]
    assert prediction_loss(z).item() < 1e-6
    assert pairwise_cosine(z).item() > 0.999


def test_same_crop_seq_pair_shares_geometry():
    torch.manual_seed(0)
    x = torch.zeros(2, 1, 32, 32)
    sib = torch.zeros(2, 1, 32, 32)
    x[:, :, 8:24, 8:24] = 2.0
    sib[:, :, 8:24, 8:24] = -1.0
    mod = torch.ones(2, dtype=torch.long)
    pair = same_crop_seq_pair(x, sib, mod, area=(1.0, 1.0), max_rot=0.0)
    assert pair.shape == (2, 2, 1, 32, 32)
    assert torch.isfinite(pair).all()
    assert not torch.allclose(pair[0], pair[1])


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
