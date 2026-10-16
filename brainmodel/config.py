"""YAML config for BrainModel. Does not inherit VAE defaults from lib.config."""

from __future__ import annotations

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
        "store_root": "data/processed/slices_zarr",
        "index_name": "volumes.csv",
        "default_spacing_mm": [1.0, 1.0, 1.0],
        "default_origin_mm": [0.0, 0.0, 0.0],
        "volume_size_xyz": [256, 256, 128],
        "num_workers": 10,
        "balance_alpha": 1.0,  # family sampling mass ~ n**alpha: 1 natural mix, 0 uniform over families
    },
    "model": {
        "size": "tiny",
        "in_chans": 1,
        "proj_dim": 128,
        "proj_hidden": 2048,
        "qk_norm": True,
        "rope_base_mm": 100.0,
        "checkpoint": True,
        "proj_norm": "batch",
        "length_buckets": 4,
        # Token head for the dense loss: 0 builds none (a plain pooled-only model).
        "dense_dim": 128,
        "dense_hidden": 512,
    },
    # Crops live on the volume's token grid, so two crops share a token exactly where they overlap.
    "views": {
        "patch_xyz": [16, 16, 8],
        "global": {"n": 2, "size_xyz": [128, 128, 64]},
        "local": {"n": 2, "size_xyz": [64, 64, 32]},
        # Finer patches on a small crop. Each fine token maps to the coarse token that contains it.
        "fine": {"n": 2, "size_xyz": [32, 32, 16], "patch_xyz": [8, 8, 4]},
        "nested": True,  # later globals are centered inside the first; local and fine crops lie inside a global
        "fg_min": 0.15,
        "retries": 8,
        "gate": {"ct_hu": [-500.0, 200.0], "mri_lo": -5.5},
    },
    "aug": {
        "window": {"mode": "random", "mix": "brain"},
        # Applied per crop after windowing, in windowed (asinh) units. p is the chance a crop gets it.
        "noise": {"p": 0.5, "sigma": [0.02, 0.15]},
        "blur": {"p": 0.3, "sigma_vox": [0.5, 1.5]},
        "drift": {"p": 0.3, "amp": 0.25},
    },
    "loss": {
        "lamb": 0.02,
        "win_weight": 1.0,
        "local_pooled": False,  # pool-and-project local crops toward the global mean, as in the 2D recipe
        "n_proj": 256,
        "gather_ddp": True,
        "dense": {"weight": 1.0, "tokens_per_view": 4, "fine_weight": 0.0},
        # Masked prediction on the global crops (needs the dense term). weight 0 builds no predictor and no extra pass.
        # Hidden blocks: ``blocks`` cuboids with sides in ``block_tokens``. The predictor regresses each hidden token's
        # own full-crop embedding and, with pixel_weight > 0, the voxel cell means (pixel_cells = x, y, z cells per
        # patch) of every kept patch. The weight ramps up over ramp_steps.
        # fine_weight > 0: the same predictor also fills hidden blocks of the fine-patch crops (fine_blocks,
        # fine_block_tokens). dense.fine_weight > 0 makes those native fine tokens agree with each other.
        "predict": {"weight": 0.0, "pixel_weight": 0.5, "pixel_cells": [2, 2, 2], "blocks": 3,
                    "block_tokens": [3, 6], "depth": 4, "ramp_steps": 2000,
                    "fine_weight": 0.0, "fine_blocks": 2, "fine_block_tokens": [2, 3]},
    },
    "optim": {
        "lr": 2.5e-4,
        "weight_decay": 0.05,
        "warmup_steps": 2000,
        "clip_grad": 1.0,
        "batch_size": 4,
    },
    "train": {
        "max_steps": 45000,
        "log_every": 50,
        "eval_every": 1000,
        "ckpt_every": 500,
        "best_metric": "loss",  # val key; lower writes best.pt. last.pt is always the latest step.
        "keep_every": 5000,
        "stop_poll": 20,
    },
    "output": {
        "out_dir": "runs/brainmodel_tiny_lejepa",
    },
}


def load_config(path: Path | str | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        path = Path(path)
        user = yaml.safe_load(path.read_text()) or {}
        if not isinstance(user, dict):
            raise ValueError(f"Config root must be a mapping, got {type(user)}")
        cfg = _deep_update(cfg, user)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def save_config(cfg: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))


def smoke_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """A tiny model on 32 x 32 x 16 phantoms, for tests and ``--synthetic``."""
    out = deepcopy(cfg)
    out["data"]["volume_size_xyz"] = [32, 32, 16]
    out["model"].update(size="test", checkpoint=False, proj_hidden=32, proj_dim=16, dense_dim=16, dense_hidden=32)
    v = out["views"]
    v["patch_xyz"] = [8, 8, 4]
    v["global"] = {"n": 2, "size_xyz": [16, 16, 8]}
    v["local"] = {"n": 2, "size_xyz": [16, 16, 8]}
    v["fine"] = {"n": 2, "size_xyz": [16, 16, 8], "patch_xyz": [4, 4, 2]}
    v["fg_min"] = 0.05
    v["gate"] = {"ct_hu": [-5.0, 200.0], "mri_lo": -1.0}
    out["optim"]["batch_size"] = 2
    return out
