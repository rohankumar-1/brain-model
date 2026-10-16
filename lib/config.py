"""Load experiment configs from YAML or JSON."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data": {
        "slices_root": "data/processed/slices_zarr",
        "augment": {
            "enabled": True,
            "hflip": 0.5,
            "translate": 0.05,
            "crop_scale": [0.85, 1.0],
            "intensity": 0.1,
            "noise_std": 0.01,
        },
    },
    "model": {
        "latent_dim": 64,
    },
    "optim": {
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "beta": 1.0e-3,
        "batch_size": 64,
        "num_workers": 8,
        "prefetch_factor": 2,
        # ReduceLROnPlateau on val recon MSE; set enabled=false to disable.
        "scheduler": {
            "enabled": False,
            "factor": 0.5,
            "patience": 3,
            "min_lr": 1.0e-6,
        },
    },
    "train": {
        "max_epochs": 1000,
        "patience": 8,
        "min_delta": 0.0,
    },
    "output": {
        "out_dir": "runs/vae_256",
        "save_recon_grid": True,
    },
}


def load_config(path: Path | str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load YAML/JSON config and merge onto defaults."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        user = yaml.safe_load(text) or {}
    elif suffix == ".json":
        user = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix} (use .yaml/.yml/.json)")
    if not isinstance(user, dict):
        raise ValueError(f"Config root must be a mapping, got {type(user)}")
    cfg = _deep_update(DEFAULT_CONFIG, user)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def save_config(cfg: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    else:
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
