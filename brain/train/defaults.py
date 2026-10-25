"""Default configs for every method. A YAML file only needs the keys it changes.

``METHODS`` maps a method name to the module holding its ``Trainer``; the modules import lazily so that
``--help`` and config handling work without torch-heavy imports.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from brain.common.grid import GRID_XYZ

METHODS = {
    "lejepa3d": "brain.train.lejepa3d",
    "mae3d": "brain.train.mae3d",
    "multitask": "brain.train.multitask",
    "joint": "brain.train.joint",
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
        "balance_groups": None,  # {group: [family, ...]}: two-level balance (volumes.family_weights)
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
        # kind gaussian (sigma in window units) or rician (snr range, magnitude MRI noise; CT stays gaussian)
        "noise": {"p": 0.5, "sigma": [0.02, 0.15], "kind": "gaussian", "snr": [15.0, 100.0]},
        "blur": {"p": 0.3, "sigma_vox": [0.5, 1.5]},
        # Anisotropic blur along one axis, as in thick-slice acquisitions resampled to 1 mm.
        "through_plane": {"p": 0.5, "fwhm_vox": [2.0, 6.0]},
        "drift": {"p": 0.3, "amp": 0.25},
        # With probability p a whole volume is cut to its brain mask (needs a store built with brain masks);
        # per_view draws it per crop instead of per volume.
        "skull": {"p": 0.5, "per_view": False},
        # Per-view rotation / scale / left-right flip on the GPU (crops3d.draw_geometry); p 0 = off.
        "geometric": {"p": 0.0, "rot_deg": [10.0, 10.0, 15.0], "scale": [0.85, 1.15], "flip_lr": 0.5},
        # MRI histogram matching to another volume of the same contrast group (augment.histogram_match); p 0 = off.
        "histmatch": {"p": 0.0, "alpha": [0.3, 0.7]},
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
                    "fine_weight": 0.0, "fine_blocks": 2, "fine_block_tokens": [2, 3],
                    # MAE (mae3d): voxels | split target; lesion = {visible, weight} lesion-aware masking (None = off)
                    "target": "voxels", "lesion": None},
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
    c["loss"] = {"predict": {"weight": 1.0, "mask_ratio": 0.75, "blocks": 0, "block_tokens": [3, 6],
                             "target": "voxels", "lesion": None}}
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
                 "val_per_task": 64, "val_max_seg": 4},
        "model": {"init_from": None, "checkpoint": True, "taps": [3, 7],
                  "slot": {"dim": 256, "layers": 3, "heads": 8, "width": None, "mode": "slots",
                           "cross_task": True, "embedding": "learned"},
                  "seg": {"widths": [128, 64, 32, 16]},
                  # task-agnostic general-embedding slot (trained by loss.general); off keeps old checkpoints loadable
                  "general": {"enabled": False, "proj_dim": 128}},
        "views": {"patch_xyz": [16, 16, 8], "gate": {"ct_hu": [-500.0, 200.0], "mri_lo": -5.5}},
        # groups: train only these task groups (None = every trained task; e.g. the no-human-label control)
        "tasks": {"k_max": 8, "p_all": 0.2, "rare_beta": 0.5, "max_seg": 2, "group_weights": {}, "groups": None,
                  "ids": None},
        "aug": {"flip_lr": 0.5, "affine": {"p": 0.8, "rot_deg": [8, 8, 12], "shift_mm": 8.0},
                # brain.data.augment.appearance with the canonical window (CT density targets live in it); p 0 = off
                "appearance": {"bias": {"p": 0.0, "amp": 0.2}, "through_plane": {"p": 0.0, "fwhm_vox": [2.0, 6.0]},
                               "noise": {"p": 0.0, "kind": "gaussian", "snr": [15.0, 100.0], "sigma": [0.02, 0.1]},
                               "blur": {"p": 0.0, "sigma_vox": [0.5, 1.0]}, "drift": {"p": 0.0, "amp": 0.1}},
                # blank a random inferior slab (truncated coverage) on rows asking no whole-brain target; p 0 = off
                "fov_cut": {"p": 0.0, "mm": [10, 40]}},
        # general: LeJEPA (prediction + SIGReg, lamb) on the general slot of two global views; the second view of the
        # first ``frac`` rows is a small rotation / shift plus a box crop (each axis keeps >= crop_min); weight 0 = off
        "loss": {"balance": "ema", "ema_decay": 0.99, "seg_stride": 2,
                 "general": {"weight": 0.0, "frac": 0.5, "lamb": 0.02, "n_proj": 256, "crop_min": 0.7,
                             "rot_deg": [8, 8, 12], "shift_mm": 8.0}},
        "optim": {"lr": 3.0e-4, "enc_lr_mult": 0.25, "layer_decay": 0.8, "weight_decay": 0.05,
                  "warmup_steps": 1000, "freeze_steps": 1000, "clip_grad": 1.0, "batch_size": 8},
        "train": {"max_steps": 20000, "log_every": 50, "eval_every": 1000, "ckpt_every": 500,
                  "keep_every": 5000, "stop_poll": 20, "best_metric": "loss"},
        "output": {"out_dir": "runs/multitask", "resume_from": None},
    }


MULTITASK: dict[str, Any] = _multitask_defaults()

def _merge(a: dict, b: dict) -> dict:
    """Deep merge, ``b`` over ``a``."""
    out = deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else deepcopy(v)
    return out


def _joint_defaults() -> dict[str, Any]:
    """Joint SSL + multitask training from scratch (brain.train.joint): the 3D LeJEPA config plus the multitask
    sections (their keys do not collide: LeJEPA ``aug`` / ``loss`` / ``model`` keys and the multitask ones are disjoint),
    the encoder fixes as options and the curriculum under ``joint``."""
    c = _merge(MULTITASK, LEJEPA3D)
    c["model"].update(init_from=None, n_cls=1,
                      # RoPE frequencies from a 32 mm shortest period (None: the original 6.3 mm, ~38% aliased pairs)
                      rope_min_period_mm=None)
    c["views"].setdefault("gate", {}).update(mode="center")        # "max": any voxel of the patch counts
    c["joint"] = {"ssl_floor": 0.5, "warmup_frac": 0.25, "task_ramp_frac": 0.15, "taper_end_frac": 0.6,
                  "task_batch_size": 4, "task_workers": 6, "normalize": True}
    c["output"]["out_dir"] = "runs/joint"
    return c


JOINT: dict[str, Any] = _joint_defaults()

_DEFAULTS = {"lejepa3d": LEJEPA3D, "mae3d": MAE3D, "multitask": MULTITASK,
             "joint": JOINT}


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
