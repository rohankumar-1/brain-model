"""Intensity windowing shared by preprocessing QC, pretraining views and brain.downstream.

``y = asinh((x - c) / w)`` is linear within about +/- w of the center ``c`` and logarithmic beyond,
so no intensity is clipped. ``x`` is in native units (HU for CT, robust z for MRI).
"""

from __future__ import annotations

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
# Three input channels by role, aligned across modalities so a channel means the same thing for CT and MRI:
# 1 narrow tissue contrast (each modality's canonical window), 2 a wider pathology range (CT subdural; MRI from CSF to
# lesions), 3 the bright extreme (CT bone; MRI hyperintensities: enhancement, FLAIR lesions, fat). MRI values are robust
# z-scores, CT Hounsfield units.
ROLES = {"ct": [(40.0, 40.0), (75.0, 100.0), (500.0, 400.0)], "mri": [(0.5, 1.5), (2.0, 4.0), (4.0, 2.0)]}

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


def canonical_windows(mod: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's canonical (c, w) (``CANONICAL``) by its modality, as (B, 1, 1, 1) tensors."""
    b = mod.shape[0]
    c = torch.where(mod == MOD_CT, CANONICAL["ct"][0], CANONICAL["mri"][0])
    w = torch.where(mod == MOD_CT, CANONICAL["ct"][1], CANONICAL["mri"][1])
    return c.float().view(b, 1, 1, 1), w.float().view(b, 1, 1, 1)


def role_windows(x: torch.Tensor, modality: torch.Tensor, jitter: float = 0.0) -> torch.Tensor:
    """Native-unit volumes ``x`` (B, 1, ...) -> the three role channels (B, 3, ...) (``ROLES``), asinh-windowed per row
    by its modality. ``jitter`` > 0 moves each channel's centre by N(0, jitter) widths and scales its width by
    exp(N(0, jitter)), independently per row and channel (training)."""
    b = x.shape[0]
    cw = torch.tensor([ROLES["ct"], ROLES["mri"]], device=x.device, dtype=torch.float32)[modality.reshape(b).long()]
    c, w = cw[..., 0], cw[..., 1]                                  # (B, 3)
    if jitter > 0:
        c = c + w * jitter * torch.randn_like(c)
        w = w * torch.exp(jitter * torch.randn_like(w))
    shape = (b, 3) + (1,) * (x.dim() - 2)
    return asinh_window(x.float().expand(b, 3, *x.shape[2:]), c.view(shape), w.view(shape))


def canonical_window(x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
    """Each row of native-unit ``x`` (B, 1, ...) in its modality's canonical window (CT brain / MRI tissue)."""
    b = x.shape[0]
    cw = torch.tensor([CANONICAL["ct"], CANONICAL["mri"]], device=x.device, dtype=torch.float32)[mod.reshape(b).long()]
    shape = (b,) + (1,) * (x.dim() - 1)
    return asinh_window(x, cw[:, 0].view(shape), cw[:, 1].view(shape))


def encoder_input(x: torch.Tensor, mod: torch.Tensor, in_chans: int = 1, bank=None) -> torch.Tensor:
    """THE unaugmented encoder input of native-unit volumes ``x`` (B, 1, ...) with per-row modality ``mod`` (B,):
    the encoder's learned window bank (``brain.models.window_bank``, K channels) when it has one, else the three role
    windows for a three-channel encoder, else the canonical window. Training (``multitask.data.prepare``), evaluation,
    few-shot, frozen embedding (``downstream.embed3d``) and the explain tools all window through here."""
    if bank is not None:
        return bank(x, mod)
    return role_windows(x, mod) if int(in_chans) == 3 else canonical_window(x, mod)
