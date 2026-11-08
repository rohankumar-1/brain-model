"""Shared learned window bank: every scan, every modality, through one module (runs/diagnostics/ngeo/CONTRACT.md 4).

1. fixed calibration to a common "tissue unit" (no parameters): CT ``u = (HU - 40) / 40``, MRI ``u = (z - 0.5) / 1.5``
   (each modality's canonical window, ``windowing.CANONICAL``, becomes ``(0, 1)`` in ``u``)
2. ``K`` learnable asinh windows ``y_k = asinh((u - c_k) / w_k)``, the centre bounded (``c = lo + (hi - lo) sigmoid(a)``)
   and the width a bounded softplus (``w = min(w_lo + softplus(b), w_hi)``)
3. ``K`` output channels: the encoder's patch embed / conv stem takes ``in_chans = K``

Because ``(u - c) / w = (x - (m + s c)) / (s w)`` for the calibration ``u = (x - m) / s``, window ``(c, w)`` in ``u`` is
exactly ``windowing.asinh_window(x, m + s c, s w)`` in native units: the init windows reproduce the presets they
correspond to (tissue = CT brain / MRI tissue, narrow tissue = CT stroke, very bright = CT bone).

The parameters are shared by every modality and every task (not task-conditioned); ``per_modality: true`` keeps the
same module with a separate (c, w) table per modality (fallback only). The bank lives on the encoder
(``encoder.window_bank``), so its learned windows are in the encoder's state dict and frozen evaluation
(``brain.downstream.embed3d``) windows with them automatically. Config: ``model.window_bank`` (``bank_cfg``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MOD_CT, MOD_MRI = 0, 1
# u = (x - m) / s per modality (row 0 CT in HU, row 1 MRI in robust z)
CALIBRATION = ((40.0, 40.0), (0.5, 1.5))
# initial (c, w) in u: narrow tissue, tissue, wide, dark, bright, very bright
INITS = {
    "contract6": ((-0.2, 0.5), (0.0, 1.0), (1.0, 2.5), (-3.0, 2.0), (3.0, 3.0), (11.5, 10.0)),
}
C_RANGE = (-8.0, 16.0)          # u: CT -280 .. 680 HU, MRI z -11.5 .. 24.5
W_RANGE = (0.05, 25.0)          # u: CT 2 .. 1000 HU, MRI z 0.075 .. 37.5
DEFAULTS = {"enabled": False, "k": 6, "init": "contract6", "per_modality": False, "lr_mult": 0.1, "decoder_grad": False}


def bank_cfg(model_cfg: dict | None) -> dict | None:
    """``model.window_bank`` with defaults filled in, or None when the bank is off (absent or ``enabled: false``)."""
    wb = (model_cfg or {}).get("window_bank")
    if not wb or not wb.get("enabled"):
        return None
    return {**DEFAULTS, **wb}


def encoder_in_chans(model_cfg: dict | None) -> int:
    """The encoder's input channels: K with the bank, else ``model.in_chans`` (1, or 3 for the role windows)."""
    wb = bank_cfg(model_cfg)
    return int(wb["k"]) if wb else int((model_cfg or {}).get("in_chans", 1))


def calibrate(x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
    """Native units ``x`` (B, 1, ...) (HU for CT rows, robust z for MRI rows) -> tissue units ``u``."""
    b = x.shape[0]
    cal = torch.tensor(CALIBRATION, device=x.device, dtype=torch.float32)[modality.reshape(b).long().to(x.device)]
    shape = (b,) + (1,) * (x.dim() - 1)
    return (x.float() - cal[:, 0].view(shape)) / cal[:, 1].view(shape)


def init_windows(init, k: int) -> list[tuple[float, float]]:
    """``init`` (a name in ``INITS`` or a list of [c, w] pairs in u) -> exactly ``k`` (c, w) pairs. A named init
    shorter than ``k`` is extended with windows spread evenly over its centre range (width 2)."""
    pairs = [tuple(map(float, p)) for p in (INITS[init] if isinstance(init, str) else init)]
    if isinstance(init, str) and len(pairs) < k:
        lo, hi = min(c for c, _ in pairs), max(c for c, _ in pairs)
        extra = k - len(pairs)
        pairs += [(lo + (hi - lo) * (i + 1) / (extra + 1), 2.0) for i in range(extra)]
    if len(pairs) < k:
        raise ValueError(f"window_bank init has {len(pairs)} windows, k = {k}")
    return pairs[:k]


def _windows(u: torch.Tensor, c: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.asinh((u - c) / w)


class WindowBank(nn.Module):
    """``forward(x, modality)``: native-unit volumes (B, 1, ...) and per-row modality (B,) 0 = CT, 1 = MRI ->
    (B, K, ...) float32 asinh windows. Computed in fp32 (also under autocast)."""

    def __init__(self, k: int = 6, init="contract6", per_modality: bool = False,
                 c_range=C_RANGE, w_range=W_RANGE, recompute: bool = True):
        super().__init__()
        if int(k) < 2:
            raise ValueError("window_bank.k must be >= 2 (one channel is indistinguishable from a windowed input)")
        self.k, self.per_modality = int(k), bool(per_modality)
        self.c_lo, self.c_hi = (float(v) for v in c_range)
        self.w_lo, self.w_hi = (float(v) for v in w_range)
        self.recompute = bool(recompute)       # training: recompute the (B, K, ...) asinh argument in the backward
        pairs = init_windows(init, self.k)
        c0 = torch.tensor([p[0] for p in pairs])
        w0 = torch.tensor([p[1] for p in pairs])
        if not ((c0 > self.c_lo) & (c0 < self.c_hi)).all() or not ((w0 > self.w_lo) & (w0 < self.w_hi)).all():
            raise ValueError(f"window_bank init {pairs} outside c {c_range} / w {w_range}")
        frac = (c0 - self.c_lo) / (self.c_hi - self.c_lo)
        a = torch.log(frac) - torch.log1p(-frac)                       # logit
        d = (w0 - self.w_lo).double()
        b = (d + torch.log(-torch.expm1(-d))).float()                  # softplus^-1, stable for large d
        rows = 2 if self.per_modality else 1
        self.c_raw = nn.Parameter(a.repeat(rows, 1))
        self.w_raw = nn.Parameter(b.repeat(rows, 1))

    def centres_widths(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(rows, K) centres and widths in tissue units (rows = 2 with ``per_modality``, else 1)."""
        c = self.c_lo + (self.c_hi - self.c_lo) * torch.sigmoid(self.c_raw.float())
        w = (self.w_lo + F.softplus(self.w_raw.float())).clamp(max=self.w_hi)
        return c, w

    def native_windows(self) -> dict[str, list[tuple[float, float]]]:
        """The current windows as native-unit (centre, width) presets per modality (for logs and reports)."""
        c, w = (t.detach().cpu() for t in self.centres_widths())
        out = {}
        for m, name in ((MOD_CT, "ct"), (MOD_MRI, "mri")):
            r = m if self.per_modality else 0
            mu, s = CALIBRATION[m]
            out[name] = [(mu + s * float(c[r, i]), s * float(w[r, i])) for i in range(self.k)]
        return out

    def forward(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 1:
            raise ValueError(f"window_bank takes native-unit volumes (B, 1, ...), got {tuple(x.shape)}")
        b = x.shape[0]
        with torch.autocast(device_type=x.device.type, enabled=False):
            u = calibrate(x, modality)
            c, w = self.centres_widths()
            if self.per_modality:
                idx = modality.reshape(b).long().to(x.device)
                c, w = c[idx], w[idx]                                   # (B, K)
            else:
                c, w = c.expand(b, -1), w.expand(b, -1)
            shape = (b, self.k) + (1,) * (x.dim() - 2)
            c, w = c.reshape(shape), w.reshape(shape)
            if self.recompute and torch.is_grad_enabled() and self.training:
                return checkpoint(_windows, u, c, w, use_reentrant=False)
            return _windows(u, c, w)


def build_bank(model_cfg: dict | None) -> WindowBank | None:
    wb = bank_cfg(model_cfg)
    if wb is None:
        return None
    kw = {k: wb[k] for k in ("c_range", "w_range") if k in wb}
    return WindowBank(k=int(wb["k"]), init=wb["init"], per_modality=bool(wb["per_modality"]), **kw)


def attach_bank(encoder: nn.Module, model_cfg: dict | None) -> nn.Module:
    """Give ``encoder`` its window bank (``encoder.window_bank``) when the config enables one; its ``in_chans`` must be
    K. Always sets the attribute (None when off), so callers can test ``encoder.window_bank``."""
    bank = build_bank(model_cfg)
    if bank is not None:
        have = int(getattr(encoder, "in_chans", 0))
        if have != bank.k:
            raise ValueError(f"window_bank k = {bank.k} but the encoder takes {have} input channels")
        p = next(encoder.parameters(), None)
        if p is not None:
            bank = bank.to(p.device)
    encoder.window_bank = bank
    return encoder


def bank_of(encoder: nn.Module | None) -> WindowBank | None:
    """The encoder's window bank, or None (encoders built before the bank have no attribute)."""
    return getattr(encoder, "window_bank", None) if encoder is not None else None


def bank_input(bank: WindowBank | None, x: torch.Tensor, modality: torch.Tensor | None) -> torch.Tensor:
    """The encoder input of ``x`` for an encoder with window bank ``bank``: native-unit (B, 1, ...) goes through the
    bank; an input that already has K channels (windowed by the caller) passes. Without a bank, ``x`` passes."""
    if bank is None or x.shape[1] == bank.k:
        return x
    if modality is None:
        raise ValueError("an encoder with a window bank needs the modality of its native-unit input")
    return bank(x, modality)


def tissue_channel(bank: WindowBank) -> int:
    """The bank channel closest to the canonical window ((0, 1) in tissue units: CT brain 40 / 40 HU, MRI tissue
    0.5 / 1.5 z; the first row when ``per_modality``): the channel a 1-channel encoder's weights move to."""
    c, w = (t.detach().float()[0] for t in bank.centres_widths())
    return int(((c - 0.0).abs() + (w - 1.0).abs()).argmin())


def widen_input(w: torch.Tensor, k: int, ch: int) -> torch.Tensor:
    """A 1-input-channel convolution weight (out, 1, ...) -> (out, k, ...) with the old kernel in channel ``ch`` and
    zeros elsewhere: on the bank's output it computes exactly what the old one did on the canonical window."""
    out = w.new_zeros((w.shape[0], int(k), *w.shape[2:]))
    out[:, ch] = w[:, 0]
    return out


def warm_start_state(state: dict, target: dict, ch: int) -> dict:
    """A state dict of a model without a bank (1 input channel: the canonical window) for a model with one: every
    5-d convolution weight whose input is 1 channel in ``state`` and K > 1 in ``target`` (otherwise equal shape: the
    patch embedding / conv stem input, the segmentation decoder's image pyramid) is widened (``widen_input``); the
    bank's own parameters are absent (they start at their init, whose ``ch`` window is the canonical one)."""
    out = {}
    for key, v in state.items():
        t = target.get(key)
        if (torch.is_tensor(v) and t is not None and v.ndim == 5 and t.ndim == 5 and v.shape[1] == 1 and t.shape[1] > 1
                and v.shape[0] == t.shape[0] and v.shape[2:] == t.shape[2:]):
            v = widen_input(v, t.shape[1], ch)
        out[key] = v
    return out


def add_bank(encoder: nn.Module, model_cfg: dict) -> nn.Module:
    """Give a trained 1-channel encoder (no bank) the configured bank, its input convolution widened to K channels so
    that the encoder's outputs are unchanged at the bank's init (``warm_start_state`` on its own weights)."""
    bank = build_bank(model_cfg)
    if bank is None or bank_of(encoder) is not None:
        return encoder
    if int(getattr(encoder, "in_chans", 1)) != 1:
        raise ValueError("window_bank warm start: the encoder must take 1 input channel (the canonical window)")
    ch = tissue_channel(bank)
    pe = encoder.patch_embed
    conv = pe.stem[0][0] if hasattr(pe, "stem") else pe.proj
    new = nn.Conv3d(bank.k, conv.out_channels, conv.kernel_size, stride=conv.stride, padding=conv.padding,
                    bias=conv.bias is not None).to(conv.weight.device)
    with torch.no_grad():
        new.weight.copy_(widen_input(conv.weight, bank.k, ch))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    if hasattr(pe, "stem"):
        pe.stem[0][0] = new
    else:
        pe.proj = new
    encoder.in_chans = bank.k
    return attach_bank(encoder, model_cfg)


def is_bank_param(name: str) -> bool:
    """Whether a parameter name belongs to a window bank (its own optimizer group, ``lr_mult``, no weight decay)."""
    return ".window_bank." in f".{name}"


def describe(bank: WindowBank | None) -> str:
    if bank is None:
        return "off"
    nw = bank.native_windows()
    fmt = lambda ws: " ".join(f"({c:.4g},{w:.4g})" for c, w in ws)              # noqa: E731
    return f"k={bank.k} per_modality={bank.per_modality} ct={fmt(nw['ct'])} mri={fmt(nw['mri'])}"


__all__ = ["WindowBank", "tissue_channel", "widen_input", "warm_start_state", "add_bank", "bank_cfg", "encoder_in_chans", "calibrate", "init_windows", "build_bank", "attach_bank",
           "bank_of", "bank_input", "is_bank_param", "describe", "CALIBRATION", "INITS"]
