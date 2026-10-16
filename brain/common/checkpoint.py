"""Atomic checkpoint writes and resume discovery, shared by every trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(obj: Any, path: Path) -> None:
    """Write to ``<path>.tmp`` then rename, so a preemption never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(path: Path, map_location="cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def state_is_finite(state: dict[str, Any]) -> bool:
    """False if any floating-point tensor in a state dict holds NaN/Inf."""
    return all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v) and v.is_floating_point())


def find_resume(out_dir: Path, resume_from: Path | None = None) -> Path | None:
    """``<out_dir>/last.pt`` if present, else the configured ``output.resume_from`` (which may not exist)."""
    last = Path(out_dir) / "last.pt"
    if last.exists():
        return last
    if resume_from is not None and Path(resume_from).exists():
        return Path(resume_from)
    return None


def strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    return {k.removeprefix("module."): v for k, v in state.items()}
