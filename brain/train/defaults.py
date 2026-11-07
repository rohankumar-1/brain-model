"""Default configs for every method. A YAML file only needs the keys it changes.

``METHODS`` maps a method name to the module holding its ``Trainer``; the modules import lazily so that
``--help`` and config handling work without torch-heavy imports.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from brain.common.config import deep_update
from brain.common.grid import GRID_XYZ

METHODS = {
    "lejepa3d": "brain.train.lejepa3d",
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
        "subject_beta": 0.0,   # a subject's volumes share n_subject ** -beta of the mass (1: a subject counts once)
        "balance_groups": None,  # {group: [family, ...]}: two-level balance (volumes.family_weights)
        # datasets / families (fnmatch patterns) this pretraining round leaves out of the SSL pool (volumes.excluded)
        "ssl_exclude": [],
    },
    "model": {
        "size": "tiny", "in_chans": 1, "proj_dim": 128, "proj_hidden": 2048, "qk_norm": True, "conv_stem": None,
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
        # mode fixed: the canonical asinh window (bias, through-plane, noise, blur, drift only); roles: the three role
        # windows as channels (model.in_chans 3, windowing.ROLES), each jittered by ``jitter`` widths; roles with
        # channels 1: the same pipeline with one role window per crop (model.in_chans 1; the v4 recipe)
        "window": {"mode": "fixed", "jitter": 0.0, "channels": 3},
        # (roles) Monotone tissue-contrast curve on MRI rows: random knots over the window's output range, so soft tissue can
        # be stretched, compressed or (with ``flip``) inverted, as T1 <-> T2 do. Never CT (HU are physical).
        "curve": {"p": 0.8, "knots": 5, "strength": 0.6, "flip": 0.3},
        # Smooth multiplicative bias field (MRI coil shading). Skipped for CT.
        "bias": {"p": 0.5, "amp": 0.3},
        # kind gaussian (sigma in window units) or rician (snr range, magnitude MRI noise; CT stays gaussian)
        "noise": {"p": 0.5, "sigma": [0.02, 0.15], "kind": "gaussian", "snr": [15.0, 100.0],
                  "sigma_hu": [2.0, 15.0]},                      # sigma_hu: CT noise in HU (window mode roles)
        "blur": {"p": 0.3, "sigma_vox": [0.5, 1.5]},
        # Anisotropic blur along one axis, as in thick-slice acquisitions resampled to 1 mm.
        "through_plane": {"p": 0.5, "fwhm_vox": [2.0, 6.0]},
        "drift": {"p": 0.3, "amp": 0.25},
        # With probability p a whole volume is cut to its brain mask (needs a store built with brain masks);
        # per_view draws it per crop instead of per volume.
        "skull": {"p": 0.5, "per_view": False},
        # Per-view rotation / scale / left-right flip on the GPU (crops3d.draw_geometry); p 0 = off.
        "geometric": {"p": 0.0, "rot_deg": [10.0, 10.0, 15.0], "scale": [0.85, 1.15], "flip_lr": 0.5},
        # (roles) MRI histogram matching to another volume of the same contrast group (augment.histogram_match); p 0 = off.
        "histmatch": {"p": 0.0, "alpha": [0.3, 0.7]},
        # one contrast draw per volume shared by all its views (roles mode; batch3d.prepare): the agreement terms then
        # teach invariance to crop and degradation, not to contrast (R15); False = an independent draw per view
        "tie_views": False,
    },
    "loss": {
        "lamb": 0.02,
        "win_weight": 1.0,
        "n_proj": 256,
        "gather_ddp": True,
        "dense": {"weight": 1.0, "tokens_per_view": 32},
        # Masked prediction on the global crops (needs the dense term); weight 0 builds no predictor.
        # Hidden blocks: ``blocks`` cuboids with sides in ``block_tokens``. The predictor regresses each
        # hidden token's full-crop embedding and, with pixel_weight > 0, the voxel cell means of kept patches.
        "predict": {"weight": 0.0, "pixel_weight": 0.5, "pixel_cells": [2, 2, 2], "blocks": 3,
                    "block_tokens": [3, 6], "depth": 4, "ramp_steps": 2000},
    },
    # lr_floor: the schedule's final LR as a fraction of the peak (engine.lr_of)
    "optim": {"lr": 2.5e-4, "weight_decay": 0.05, "warmup_steps": 2000, "clip_grad": 1.0, "batch_size": 4,
              "lr_floor": 0.01},
    "train": {"max_steps": 45000, "log_every": 50, "eval_every": 1000, "ckpt_every": 500, "keep_every": 5000,
              "stop_poll": 20, "best_metric": "loss"},
    # resume_from: checkpoint used only when <out_dir>/last.pt is absent (weights + optimizer + step).
    "output": {"out_dir": "runs/lejepa3d", "resume_from": None},
}

def _multitask_defaults() -> dict[str, Any]:
    """Task-conditioned multitask post-training (brain.multitask): an SSL encoder on whole volumes, one learned task
    token per trained task inside the encoder, typed shared heads. The encoder's architecture comes from
    ``model.init_from``."""
    return {
        "seed": 42,
        "data": {"store_root": "data/processed/slices_zarr", "index_name": "volumes.csv",
                 "labels_dir": "data/processed/slices_zarr/labels/multitask_v2", "registry": None,
                 "volume_size_xyz": list(GRID_XYZ), "default_spacing_mm": [1.0, 1.0, 1.0],
                 "num_workers": 10, "balance_alpha": 0.5, "subject_beta": 0.0,
                 # RSNA slice labels (brain.prep.rsna_slices parquet) for loss.slice_weight
                 "rsna_slices": None,
                 # dataset-name prefixes whose scans leave training and validation entirely (multitask.data.drop_datasets)
                 "exclude_datasets": [],
                 "exclude_volumes": None,       # a CSV of volume_ids out of every task (and the joint SSL pool)
                 # fastMRI+ box tables (brain.prep.fastmri_labels parquet) for loss.box_weight
                 "fastmri_boxes": None,
                 # the BraTS re-releases count as one family for sampling
                 "balance_groups": {"brats": ["brats_gli", "brats_men", "brats_ped", "brats_goat"]},
                 "val_per_task": 64, "val_max_seg": 4},
        "model": {"init_from": None, "checkpoint": True, "taps": [3, 7],
                  # task tokens in the encoder (OMNI); the patches' attention to them opens over write_ramp_steps
                  # from write_bias logits so a warm-started encoder is not shifted at step 0
                  "task_tokens": {"enabled": True, "write_bias": 10.0, "write_ramp_steps": 2000, "share": False},
                  # study-level input (R14-F): {after_block, heads}: a study's co-registered scans fused inside the encoder
                  "fusion": None,
                  # regression head: ``scalar`` (smooth L1) or ``hlgauss`` (bins over the standardised range, Gaussian
                  # targets of sigma_ratio bin widths; switching a resumed run adds the head and resets the optimizer)
                  "regression": {"kind": "scalar", "bins": 80, "range": [-5.0, 5.0], "sigma_ratio": 0.75},
                  # a logit per encoder token for a slot (BrainQuery.token_logits), for loss.token_weight / slice_weight
                  "token_map": {"enabled": False, "dim": 64},
                  # encoder: the ViT (init_from an SSL checkpoint, or ``size`` from scratch) or ``resnet3d``
                  # (brain.models.resnet3d, from scratch; ``resnet`` its widths / blocks)
                  "arch": "vit", "resnet": {"widths": [64, 128, 240, 384], "blocks": [2, 2, 2, 2]},
                  # ViT size when training from scratch (no init_from); None = the init_from checkpoint's
                  "size": None,
                  # from-scratch ViT only: widths of a two-stage convolution stem before the token grid
                  # (patch_embed.ConvStemEmbed3D, e.g. [32, 64]); None = the plain linear patch embedding
                  "conv_stem": None,
                  # ``dim``: the heads' width. layers / heads: the slot readout (task_tokens.enabled: false)
                  "slot": {"dim": 256, "layers": 3, "heads": 8},
                  # shared: decode once per volume with a 1x1 output per task (SharedSegDecoder; tasks.all_seg asks
                  # every mask), instead of once per segmentation slot
                  # head_hidden (shared only): a per-task C -> head_hidden -> 1 MLP per voxel instead of a 1x1 readout
                  "seg": {"widths": [128, 64, 32, 16], "shared": False, "head_hidden": 0}},
        # gate mode: "center" (a token's centre voxel) or "frac" (>= min_frac of its voxels; data.gate.token_keep)
        "views": {"patch_xyz": [16, 16, 8], "gate": {"ct_hu": [-500.0, 200.0], "mri_lo": -5.5, "mode": "center",
                                                     "min_frac": 0.25}},
        # groups: train only these task groups (None = every trained task; e.g. the no-human-label control)
        # exclude: small clinical tasks (55-350 training scans) that overfit and never learned (run 6, 2026-10-25);
        # they stay in the registry as held-out few-shot tasks
        # ssl: also train the self-supervised registry tasks (masked_recon, synth_*); mask_p: share of samples (with a
        # reconstruction row) that are masked and ask only self-supervised tasks; mask_ratio: their hidden foreground share
        # max_dense: decoder slots per sample for dense tasks on their own (None: they share max_seg)
        "tasks": {"ssl": False, "mask_p": 0.0, "mask_ratio": 0.6, "max_dense": None,
                  "k_max": 8, "p_all": 0.2, "rare_beta": 0.5, "max_seg": 2, "all_seg": False, "lesion_ignore": False, "study": None, "group_weights": {},
                  "groups": None,
                  "ids": None, "exclude": ["mmse", "autism_severity", "adhd", "parkinsons", "glioma_survival_12m",
                                           "glioma_lower_grade", "glioma_idh"]},
        "aug": {"flip_lr": 0.5, "affine": {"p": 0.8, "rot_deg": [8, 8, 12], "shift_mm": 8.0, "shear_deg": 0.0},   # shear: gantry tilt
                # brain.data.augment.appearance with the canonical window (CT density targets live in it); p 0 = off
                # bias is MRI only and mild (intensity asymmetry is a target); noise is Rician on MRI, Gaussian on CT
                "appearance": {"bias": {"p": 0.3, "amp": 0.2}, "through_plane": {"p": 0.3, "fwhm_vox": [2.0, 6.0]},
                               "noise": {"p": 0.5, "kind": "rician", "snr": [15.0, 100.0], "sigma": [0.02, 0.1]},
                               "blur": {"p": 0.2, "sigma_vox": [0.5, 1.0]}, "drift": {"p": 0.2, "amp": 0.1}},
                # blank a random inferior slab (truncated coverage) on rows asking no whole-brain target; p 0 = off
                "fov_cut": {"p": 0.2, "mm": [10, 40]},
                # skull-strip skull-on MRI with its SynthStrip mask (never rows asking a needs_skull task): the tumour
                # and stroke cohorts arrive stripped, so "no skull" would otherwise mean "lesion cohort"
                # match: tasks whose positives arrive skull-stripped (BraTS); a strippable scan asked one is always
                # stripped, so "no skull" cannot stand for "tumour" once healthy scans carry empty masks (labels v11)
                "skull_strip": {"p": 0.3, "match": []},
                # synthetic tissue contrast (brain.data.synth_contrast): with chance p an MRI scan is re-rendered from
                # its SynthSeg map; only contrast-invariant tasks (groups / tasks minus exclude) are asked of it
                "synth_contrast": {"p": 0.0, "groups": ["anatomy_seg", "anatomy_volume"],
                                   "tasks": ["brain_mask", "brain_volume"], "exclude": ["vol_wmh", "wmh_grade"],
                                   "mean": [-2.0, 3.0], "std": [0.05, 0.6], "blur": [0.3, 1.0]},
                # rows asking a registry ``fine`` task (small targets) get no blur, field-of-view cut or stripping
                "task_aware": False},
        # negative_tasks: segmentation tasks with true negatives (labels v11 empty masks): losses.negative_seg_loss.
        # sparse_pos_weight: target-voxel / target-cell weight of losses.sparse_seg_loss (registry ``sparse`` tasks).
        # token_weight / slice_weight (need model.token_map): token-map losses added to a slot's loss, on mask tasks'
        # token occupancy (losses.token_occupancy_loss) and on RSNA slice labels (losses.slab_mil_loss)
        "loss": {"seg_stride": 2, "sparse_pos_weight": 20.0,
                 "negative_tasks": [], "token_weight": 0.0, "slice_weight": 0.0, "box_weight": 0.0},
        "optim": {"lr": 3.0e-4, "enc_lr_mult": 0.25, "layer_decay": 0.8, "weight_decay": 0.05,
                  "warmup_steps": 1000, "clip_grad": 1.0, "batch_size": 8},
        # ema_decay: weight average (brain.train.engine.WeightEMA), evaluated beside the live weights; None = off
        "train": {"max_steps": 20000, "log_every": 50, "eval_every": 1000, "ckpt_every": 500,
                  "keep_every": 5000, "stop_poll": 20, "best_metric": "score",   # validation skill (brain.eval.skill)
                  "ema_decay": None},
        "output": {"out_dir": "runs/multitask", "resume_from": None},
    }


MULTITASK: dict[str, Any] = _multitask_defaults()

def _joint_defaults() -> dict[str, Any]:
    """Joint SSL + multitask training from scratch (brain.train.joint): the 3D LeJEPA config plus the multitask
    sections (their keys do not collide: LeJEPA ``aug`` / ``loss`` / ``model`` keys and the multitask ones are disjoint),
    the encoder fixes as options and the curriculum under ``joint``."""
    c = deep_update(MULTITASK, LEJEPA3D)
    c["model"].update(init_from=None, n_cls=1, drop_path=0.0,
                      # RoPE frequencies from a 32 mm shortest period (None: the original 6.3 mm, ~38% aliased pairs)
                      rope_min_period_mm=None)
    c["views"].setdefault("gate", {}).update(mode="center")
    # the curriculum in fractions of max_steps, or in steps (warmup_steps / task_ramp_steps / taper_end_steps win when
    # set: a run that may be extended keeps its curriculum fixed instead of stretching it)
    # ssl_start: the SSL weight while it tapers from the start (1.0 = equal to the tasks' full weight); R15 4 -> floor 1
    c["joint"] = {"ssl_start": 1.0, "ssl_floor": 0.5, "warmup_frac": 0.25, "task_ramp_frac": 0.15, "taper_end_frac": 0.6,
                  "warmup_steps": None, "task_ramp_steps": None, "taper_end_steps": None,
                  "ssl_batch_size": 4, "normalize": True}
    # the task side is brain.train.multitask's trainer: optim.batch_size = task volumes per GPU; from scratch the encoder
    # trains at the full LR (no multitask post-training discount, no layer-wise decay)
    c["optim"].update(batch_size=4, enc_lr_mult=1.0, layer_decay=1.0)
    c["train"]["best_metric"] = "score"                             # validation skill; none until tasks ramp in
    c["output"]["out_dir"] = "runs/joint"
    return c


JOINT: dict[str, Any] = _joint_defaults()

_DEFAULTS = {"lejepa3d": LEJEPA3D, "multitask": MULTITASK, "joint": JOINT}
for _name, _c in _DEFAULTS.items():
    # optim.schedule: cosine | wsd (warmup-stable-decay; decay_steps = length of the final decay, None = 20% of the
    # run). train.keep_full: keep_every snapshots carry optimizer state too (to branch a WSD cooldown from them).
    _c["optim"].setdefault("schedule", "cosine")
    _c["optim"].setdefault("decay_steps", None)
    _c["optim"].setdefault("betas", [0.9, 0.999] if _name == "multitask" else [0.9, 0.95])
    _c["train"].setdefault("keep_full", False)
    # encoder numerics fixed 2026-10-25, on for new runs: RoPE angles in fp32 (under bf16 autocast they were bf16, up
    # to ~0.5 rad wrong for the high frequencies) and an fp32 residual stream. Checkpoints whose config lacks the keys
    # load with the old behaviour (False), so existing models evaluate as they were trained; multitask post-training
    # inherits them from its init_from encoder.
    _c["model"].setdefault("rope_fp32", True)
    _c["model"].setdefault("fp32_residual", True)
    # scan metadata (modality / contrast, brain.data.volumes.META) as an encoder input, replaced by "unknown" with
    # probability drop; checkpoints without the key have none. Off by default since 2026-10-28: the sequence label is a
    # shortcut against a modality-agnostic encoder (runs saved with it on keep it)
    _c["model"].setdefault("meta", {"enabled": False, "drop": 0.2})
    if _name in ("multitask", "joint"):
        # uncertainty balancing: log-variances soft-clamped to +-uncertainty_clamp (task weights within e^+-clamp)
        _c["loss"].setdefault("uncertainty_clamp", 1.0)


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
