"""Default configs for every method. A YAML file only needs the keys it changes.

``METHODS`` maps a method name to the module holding its ``Trainer``; the modules import lazily so that
``--help`` and config handling work without torch-heavy imports.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from brain.common.grid import GRID_XYZ

METHODS = {
    "lejepa2d": "brain.train.lejepa2d",
    "dino2d": "brain.train.dino2d",
    "lejepa3d": "brain.train.lejepa3d",
    "mae3d": "brain.train.mae3d",
    "multitask": "brain.train.multitask",
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
        "volume_size_xyz": list(GRID_XYZ),
        "num_workers": 10,
        "balance_alpha": 1.0,  # family sampling mass ~ n**alpha: 1 natural mix, 0 uniform over families
    },
    "model": {
        "size": "tiny", "in_chans": 1, "proj_dim": 128, "proj_hidden": 2048, "qk_norm": True,
        "rope_base_mm": 100.0, "checkpoint": True, "proj_norm": "batch", "length_buckets": 4,
        # Token head for the dense loss: 0 builds none (a plain pooled-only model).
        "dense_dim": 128, "dense_hidden": 512,
        # Prefix tokens: n_cls=1 adds a CLS that becomes the pooled (global) embedding; registers are extra
        # position-free tokens attention can use as scratch space. 0/0 = mean-pooled patch tokens.
        "n_cls": 0, "n_registers": 0,
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
        "dense": {"weight": 1.0, "tokens_per_view": 32, "fine_weight": 0.0, "stopgrad_mean": False},
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

def _mae3d_defaults() -> dict[str, Any]:
    """3D MAE: Primus-style ViT (SwiGLU, LayerScale, drop-path) on 8 x 8 x 4 mm tokens; 75% of each crop's foreground
    tokens hidden; a narrow decoder reconstructs their voxels in windowed units (no per-patch normalisation)."""
    c = deepcopy(LEJEPA3D)
    c["model"].update(size="base", mlp="swiglu", layer_scale=0.1, drop_path=0.1, n_cls=0, n_registers=0, dense_dim=0,
                      decoder={"dim": 384, "depth": 4, "heads": 6})
    c["views"].update(patch_xyz=[8, 8, 4], nested=False, win=False,
                      local={"n": 0, "size_xyz": [64, 64, 32]}, fine={"n": 0, "size_xyz": [32, 32, 16]})
    c["aug"].update(curve={"p": 0.0}, skull={"p": 0.3, "per_view": True})
    c["loss"] = {"predict": {"weight": 1.0, "mask_ratio": 0.75, "blocks": 0, "block_tokens": [3, 6]}}
    c["optim"].update(lr=1.5e-4, weight_decay=0.05, warmup_steps=2000, batch_size=24)
    c["train"].update(max_steps=30000, keep_every=3000)
    c["output"]["out_dir"] = "runs/mae3d"
    return c


MAE3D: dict[str, Any] = _mae3d_defaults()


def _multitask_defaults() -> dict[str, Any]:
    """Task-conditioned multitask post-training (brain.multitask): an SSL encoder on whole volumes, learned task
    embeddings as slot queries, typed shared heads. The encoder's architecture comes from ``model.init_from``."""
    return {
        "seed": 42,
        "data": {"store_root": "data/processed/slices_zarr", "index_name": "volumes.csv",
                 "labels_dir": "data/processed/slices_zarr/labels/multitask", "registry": None,
                 "volume_size_xyz": list(GRID_XYZ), "default_spacing_mm": [1.0, 1.0, 1.0],
                 "num_workers": 10, "balance_alpha": 0.5, "balance_groups": None,
                 "val_per_task": 24, "val_max_seg": 4},
        "model": {"init_from": None, "checkpoint": True, "taps": [3, 7],
                  "slot": {"dim": 256, "layers": 3, "heads": 8, "width": None, "mode": "slots",
                           "cross_task": True, "embedding": "learned"},
                  "seg": {"widths": [128, 64, 32, 16]}},
        "views": {"patch_xyz": [16, 16, 8], "gate": {"ct_hu": [-500.0, 200.0], "mri_lo": -5.5}},
        "tasks": {"k_max": 8, "p_all": 0.2, "rare_beta": 0.5, "max_seg": 2, "group_weights": {}},
        "aug": {"flip_lr": 0.5, "affine": {"p": 0.8, "rot_deg": [8, 8, 12], "shift_mm": 8.0}},
        "loss": {"balance": "ema", "ema_decay": 0.99, "seg_stride": 2},
        "optim": {"lr": 3.0e-4, "enc_lr_mult": 0.25, "layer_decay": 0.8, "weight_decay": 0.05,
                  "warmup_steps": 1000, "freeze_steps": 1000, "clip_grad": 1.0, "batch_size": 8},
        "train": {"max_steps": 20000, "log_every": 50, "eval_every": 1000, "ckpt_every": 500,
                  "keep_every": 5000, "stop_poll": 20, "best_metric": "loss"},
        "output": {"out_dir": "runs/multitask", "resume_from": None},
    }


MULTITASK: dict[str, Any] = _multitask_defaults()

_DEFAULTS = {"lejepa2d": LEJEPA2D, "dino2d": DINO2D, "lejepa3d": LEJEPA3D, "mae3d": MAE3D, "multitask": MULTITASK}


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
