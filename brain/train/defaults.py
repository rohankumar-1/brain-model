"""Default configs for every method. A YAML file only needs the keys it changes.

``METHODS`` maps a method name to the module holding its ``Trainer``; the modules import lazily so that
``--help`` and config handling work without torch-heavy imports.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

METHODS = {
    "lejepa2d": "brain.train.lejepa2d",
    "dino2d": "brain.train.dino2d",
    "lejepa3d": "brain.train.lejepa3d",
}

_TRAIN_2D = {
    "max_steps": 60000, "log_every": 100, "eval_every": 2000, "eval_batches": 8,
    "ckpt_every": 1000, "keep_every": 10000, "stop_poll": 20, "best_metric": "loss",
}
_DATA_2D = {
    "slices_root": "data/processed/slices_zarr", "slices_per_read": 2, "balance_alpha": 0.5,
    "priority_beta": 0.0, "priority_clip": 3.0,
}
_VIEWS_2D = {"window": {"mode": "random", "mix": "v4", "inv_weight": 0.0, "seq_weight": 0.0}}

LEJEPA2D: dict[str, Any] = {
    "seed": 42,
    "data": _DATA_2D,
    "model": {"size": "tiny", "img_size": 256, "patch_size": 16, "proj_dim": 128, "proj_hidden": 2048,
              "registers": 4, "qk_norm": True},
    "views": _VIEWS_2D,
    "optim": {"lr": 5.0e-4, "weight_decay": 0.05, "warmup_steps": 2000, "clip_grad": 1.0, "batch_size": 384,
              "compile": True, "num_workers": 7, "prefetch_factor": 2,
              "lamb": 0.02, "n_global": 2, "n_local": 6, "global_area": [0.4, 1.0],
              "local_area": [0.05, 0.4], "local_size": 96, "max_rot": 10.0},
    "train": dict(_TRAIN_2D),
    "output": {"out_dir": "runs/lejepa2d", "resume_from": None},
}

DINO2D: dict[str, Any] = {
    "seed": 42,
    "data": _DATA_2D,
    "model": {"size": "small", "img_size": 256, "patch_size": 16, "proj_dim": 128, "proj_hidden": 2048,
              "registers": 4, "qk_norm": True, "proto_dim": 65536, "head_hidden": 2048, "head_bottleneck": 256},
    "loss": {"student_temp": 0.1, "teacher_temp": 0.04, "ema": 0.994, "ibot_weight": 1.0, "koleo_weight": 0.1,
             "mask_prob": 0.5, "mask_ratio": [0.1, 0.5], "center_momentum": 0.9},
    "views": {"window": {"mode": "random", "mix": "v4"}},
    "optim": {"lr": 1.0e-4, "weight_decay": 0.04, "warmup_steps": 2000, "clip_grad": 3.0, "layer_decay": 0.9,
              "batch_size": 64, "compile": False, "num_workers": 7, "prefetch_factor": 2,
              "n_global": 2, "n_local": 8, "global_area": [0.32, 1.0], "local_area": [0.05, 0.32],
              "local_size": 112, "max_rot": 10.0},
    "train": {**_TRAIN_2D, "max_steps": 40000, "log_every": 50},
    "output": {"out_dir": "runs/dino2d", "resume_from": None},
}

LEJEPA3D: dict[str, Any] = {
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
        "size": "tiny", "in_chans": 1, "proj_dim": 128, "proj_hidden": 2048, "qk_norm": True,
        "rope_base_mm": 100.0, "checkpoint": True, "proj_norm": "batch", "length_buckets": 4,
        # Token head for the dense loss: 0 builds none (a plain pooled-only model).
        "dense_dim": 128, "dense_hidden": 512,
    },
    # Crops live on the volume's token grid, so two crops share a token exactly where they overlap.
    "views": {
        "patch_xyz": [16, 16, 8],
        "global": {"n": 2, "size_xyz": [128, 128, 64]},
        "local": {"n": 2, "size_xyz": [64, 64, 32]},
        # Finer patches on a small crop. Each fine token maps to the coarse token that contains it.
        "fine": {"n": 2, "size_xyz": [32, 32, 16], "patch_xyz": [8, 8, 4]},
        "nested": True,  # later globals sit inside the first; local and fine crops lie inside a global
        "fg_min": 0.15,
        "retries": 8,
        "gate": {"ct_hu": [-500.0, 200.0], "mri_lo": -5.5},
    },
    # Intensity re-rendering, applied per crop. See brain/data/augment.py.
    "aug": {
        "window": {"mode": "random", "mix": "brain"},
        # Monotone tissue-contrast curve: random knots over the window's output range, so soft tissue can be
        # stretched, compressed or (with ``flip``) inverted, as T1 <-> T2 <-> CT do.
        "curve": {"p": 0.8, "knots": 5, "strength": 0.6, "flip": 0.3},
        # Smooth multiplicative bias field (MRI coil shading). Skipped for CT.
        "bias": {"p": 0.5, "amp": 0.3},
        "noise": {"p": 0.5, "sigma": [0.02, 0.15]},
        "blur": {"p": 0.3, "sigma_vox": [0.5, 1.5]},
        # Anisotropic blur along one axis, as in thick-slice acquisitions resampled to 1 mm.
        "through_plane": {"p": 0.5, "fwhm_vox": [2.0, 6.0]},
        "drift": {"p": 0.3, "amp": 0.25},
        # With probability p a whole volume is cut to its brain mask (needs a store built with brain masks).
        "skull": {"p": 0.5},
    },
    "loss": {
        "lamb": 0.02,
        "win_weight": 1.0,
        "local_pooled": False,
        "n_proj": 256,
        "gather_ddp": True,
        "dense": {"weight": 1.0, "tokens_per_view": 4, "fine_weight": 0.0, "stopgrad_mean": False},
        # Masked prediction on the global crops (needs the dense term); weight 0 builds no predictor.
        # Hidden blocks: ``blocks`` cuboids with sides in ``block_tokens``. The predictor regresses each
        # hidden token's full-crop embedding and, with pixel_weight > 0, the voxel cell means of kept patches.
        "predict": {"weight": 0.0, "pixel_weight": 0.5, "pixel_cells": [2, 2, 2], "blocks": 3,
                    "block_tokens": [3, 6], "depth": 4, "ramp_steps": 2000,
                    "fine_weight": 0.0, "fine_blocks": 2, "fine_block_tokens": [2, 3]},
    },
    "optim": {"lr": 2.5e-4, "weight_decay": 0.05, "warmup_steps": 2000, "clip_grad": 1.0, "batch_size": 4},
    "train": {"max_steps": 45000, "log_every": 50, "eval_every": 1000, "ckpt_every": 500, "keep_every": 5000,
              "stop_poll": 20, "best_metric": "loss"},
    # resume_from: checkpoint used only when <out_dir>/last.pt is absent (weights + optimizer + step).
    # init_from: weights only, fresh optimizer and schedule (heads the source lacks, e.g. the predictor, start random).
    "output": {"out_dir": "runs/lejepa3d", "resume_from": None, "init_from": None},
}

_DEFAULTS = {"lejepa2d": LEJEPA2D, "dino2d": DINO2D, "lejepa3d": LEJEPA3D}


def default_config(method: str) -> dict[str, Any]:
    return deepcopy(_DEFAULTS[method])


def smoke_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """A tiny 3D model on 32 x 32 x 16 phantoms, for tests and ``--synthetic``."""
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
