"""Intensity windowing shared by preprocessing QC, pretraining views and downstream.

``y = asinh((x - c) / w)`` is linear within about +/- w of the center ``c`` and logarithmic beyond,
so no intensity is clipped. ``x`` is in native units (HU for CT, robust z for MRI).
"""

from __future__ import annotations

import math

import numpy as np
import torch

# (center, width) presets; w is the linear half-width of the knee.
PRESETS = {
    "ct": {"brain": (40.0, 40.0), "subdural": (75.0, 100.0), "stroke": (32.0, 20.0),
           "soft": (40.0, 200.0), "bone": (500.0, 400.0)},
    "mri": {"tissue": (0.5, 1.5), "wide": (2.0, 4.0)},
}
CANONICAL = {"ct": PRESETS["ct"]["brain"], "mri": PRESETS["mri"]["tissue"]}
# CT multi-window concat (spec 3.9). Shifted pair for robustness.
MULTI = {
    "ct": [PRESETS["ct"]["brain"], PRESETS["ct"]["subdural"], PRESETS["ct"]["bone"]],
    "mri": [PRESETS["mri"]["tissue"], PRESETS["mri"]["wide"]],
}
SHIFTED = {
    "ct": [(40.0, 40.0), (60.0, 25.0), (40.0, 150.0)],
    "mri": [PRESETS["mri"]["tissue"], (0.5, 0.6), (0.5, 4.0)],
}
FLOOR = {"ct": -1024.0, "mri": -6.0}
# Derived foreground: CT air/tissue split, MRI sentinel is -6.
FG_THRESH = {"ct": -500.0, "mri": -5.5}
MOD_CT, MOD_MRI = 0, 1

_CT_CW = torch.tensor([
    [40.0, 40.0], [75.0, 100.0], [32.0, 20.0], [40.0, 200.0], [500.0, 400.0],
])
# last bin is the uniform tail, not a preset
WINDOW_MIXES = {
    "v4": {
        "ct_p": (0.35, 0.15, 0.15, 0.15, 0.10, 0.10),
        "mri_tissue_p": 0.70,
        "mri_tissue_c": (0.3, 0.5),
        "mri_tissue_w_log": (math.log(1.5), 0.3),
    },
    # More mass on the eval/tissue windows; bone and the wild tail stay rare.
    "tissue": {
        "ct_p": (0.50, 0.18, 0.12, 0.12, 0.05, 0.03),
        "mri_tissue_p": 0.80,
        "mri_tissue_c": (0.3, 0.5),
        "mri_tissue_w_log": (math.log(2.2), 0.25),
    },
    # Brain, subdural, stroke, soft tissue. No bone preset and no wild tail:
    # a bone window flattens the tissue the brain mask keeps.
    "brain": {
        "ct_p": (0.55, 0.25, 0.15, 0.05, 0.0, 0.0),
        "mri_tissue_p": 0.85,
        "mri_tissue_c": (0.3, 0.5),
        "mri_tissue_w_log": (math.log(1.5), 0.25),
    },
}


def asinh_window(x, c, w):
    """Works on torch tensors and numpy arrays."""
    z = (x - c) / w
    return z.asinh() if hasattr(z, "asinh") else np.arcsinh(z)


def windows_for(mode: str, modality: str) -> list[tuple[float, float]]:
    """Named window lists for frozen-encoder eval. ``modality`` is ``ct`` or ``mri``."""
    m = "ct" if str(modality).lower().startswith("ct") else "mri"
    mode = (mode or "canonical").lower()
    if mode == "multi":
        return list(MULTI[m])
    if mode == "shifted":
        return list(SHIFTED[m])
    return [CANONICAL[m]]


def canonical_c_w(modality: str) -> tuple[float, float]:
    m = "ct" if str(modality).lower().startswith("ct") else "mri"
    return CANONICAL[m]


def floors(mod: torch.Tensor) -> torch.Tensor:
    """Per-sample background floor, broadcastable to (B, 1, 1, 1)."""
    return torch.where(mod == MOD_CT, FLOOR["ct"], FLOOR["mri"]).to(dtype=torch.float32)


def fg_thresh(mod: torch.Tensor) -> torch.Tensor:
    return torch.where(mod == MOD_CT, FG_THRESH["ct"], FG_THRESH["mri"]).to(dtype=torch.float32)


def foreground_mask(x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
    """``x`` is (B, 1, H, W) native units; ``mod`` is (B,) 0=CT, 1=MRI."""
    t = fg_thresh(mod).view(-1, *([1] * (x.ndim - 1)))
    return x > t


def _mix(name: str) -> dict:
    key = (name or "v4").lower()
    if key not in WINDOW_MIXES:
        raise KeyError(f"Unknown window mix {name!r}. Choices: {', '.join(WINDOW_MIXES)}")
    return WINDOW_MIXES[key]


def sample_windows(mod: torch.Tensor, mode: str = "random", mix: str = "v4", generator=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent (c, w) per sample. Returns (B, 1, 1, 1) tensors.

    ``mode``: ``fixed`` uses the canonical window; ``random`` draws from ``mix``
    (CT radiology presets plus a tail; MRI tissue vs wide).
    """
    b, device = mod.shape[0], mod.device
    if mode == "fixed":
        c = torch.where(mod == MOD_CT, 40.0, 0.5)
        w = torch.where(mod == MOD_CT, 40.0, 1.5)
        return c.view(b, 1, 1, 1), w.view(b, 1, 1, 1)

    spec = _mix(mix)
    ct_p = torch.tensor(spec["ct_p"], device=device, dtype=torch.float32)
    choice = torch.multinomial(ct_p.expand(b, -1), 1, generator=generator).squeeze(1)
    cw = _CT_CW.to(device=device, dtype=torch.float32)[choice.clamp(max=4)]
    c_ct = cw[:, 0] + torch.randn(b, device=device, generator=generator) * 10.0
    w_ct = cw[:, 1] * torch.exp(torch.randn(b, device=device, generator=generator) * 0.25)
    tail = choice == 5
    c_ct = torch.where(tail, torch.empty(b, device=device).uniform_(-200.0, 600.0, generator=generator), c_ct)
    w_ct = torch.where(
        tail,
        torch.empty(b, device=device).uniform_(math.log(15.0), math.log(400.0), generator=generator).exp(),
        w_ct,
    )

    c_mu, c_std = spec["mri_tissue_c"]
    w_mu, w_std = spec["mri_tissue_w_log"]
    tissue = torch.rand(b, device=device, generator=generator) < float(spec["mri_tissue_p"])
    c_mri = torch.where(
        tissue,
        torch.randn(b, device=device, generator=generator) * c_std + c_mu,
        torch.empty(b, device=device).uniform_(-2.0, 6.0, generator=generator),
    )
    w_mri = torch.where(
        tissue,
        torch.exp(torch.randn(b, device=device, generator=generator) * w_std + w_mu),
        torch.empty(b, device=device).uniform_(math.log(0.5), math.log(4.0), generator=generator).exp(),
    )
    is_ct = mod == MOD_CT
    c = torch.where(is_ct, c_ct, c_mri)
    w = torch.where(is_ct, w_ct, w_mri).clamp_min(1e-3)
    return c.view(b, 1, 1, 1), w.view(b, 1, 1, 1)
