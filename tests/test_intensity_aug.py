"""Rician noise and MRI histogram matching (brain.data.augment)."""

from __future__ import annotations

import math

import torch

from brain.data.augment import _foreground, appearance, histogram_match, rician_noise
from brain.data.volumes import tissue_quantiles


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
    aug = {"window": {"mode": "random", "mix": "brain_bone"}, "noise": {"p": 1.0, "kind": "rician"},
           "histmatch": {"p": 1.0}}
    quant = torch.stack([tissue_quantiles(x[i, 0], int(mod[i])) for i in range(4)])
    y = appearance(x, mod, aug, None, torch.arange(4), torch.tensor([1, 1, 0, 1]), quant)
    assert y.shape == x.shape and torch.isfinite(y).all()
