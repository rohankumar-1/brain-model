# BrainModel Stage 1 — implementation plan

This is the working plan for [brainmodel_spec.md](brainmodel_spec.md). The existing 2D slice ViT (`train_vit.py`, `models/vit.py`) stays untouched so current Small/Tiny runs remain reproducible.

## Decisions (defaults; change in config, not code)

| Topic | Choice | Why |
|---|---|---|
| Package | New `brainmodel/` tree, entry `train_brainmodel.py` | Spec layout; 2D jobs keep working |
| Config | Plain YAML + `brainmodel/config.py` (no Hydra, no VAE defaults) | Matches this repo |
| Volume source | v4 zarr store (RAS, 1 mm iso, 256×256×128, native HU / MRI-z) | Already processed; do not re-interpolate z |
| Spacing / origin | Read from zip attrs when present; else config `default_spacing_mm` (v4 is 1 mm) and origin at the (0,0,0) voxel corner | Spec: never hard-code; v4 resampled to 1 mm so attrs are often missing |
| IDs | `patient_id` = existing `dedup_key`; `timepoint` parsed from OASIS/BraTS ids; `site` / `scanner` optional columns (null until labeled) | Stage 3 needs index columns now; they are not training inputs |
| Intensity | View-time **v4 asinh** (`y=asinh((x-c)/w)`), independent `(c,w)` per view from mix `v4`. Native-unit noise/blur *before* the window. `clip` remains as an ablation. Eval can set `aug.window.mode=fixed` | Same photometric recipe as the 2D Tiny/Small runs; HU stay in the store, not in the encoder input |
| Local crops | Discrete sizes that are multiples of the local patch, default **64×64×32** (512 tokens). Size jitter is a config list, not a continuous range | 96×96×48 native with 8×8×4 patches is 1728 tokens, not ~512 |
| First train recipe | 2 global + 2 local, ViT-S, activation checkpointing, `lambda=0.05`, modest dense/token weights | Spec §11 memory warning |
| Attention | Full attention in Stage 1 training. `window_size` is in the encoder API; windowed attention is used when set | Full-volume 32³ tokens needs windows later |
| SigReg | Reuse `models.sigreg.SIGReg` (Epps–Pulley). Token-level applies the same module to the pooled foreground tokens. DDP: all-gather embeddings **without** gradients for the statistic; document that grad-accum does not grow the SigReg sample count | Do not fork the tested kernel |
| Tests | `tests/test_brainmodel.py` (repo convention), not a second pytest root | `python -m tests.test_brainmodel` |

## Reuse vs new

**Reuse**

- `models.sigreg.SIGReg` and view-level `prediction_loss`
- `data.prep.store` zip I/O; `data.prep.intensity` largest-CC / MRI Otsu helpers
- `data.prep.index` columns (`dedup_key`, modality, contrast, split, held-out)
- `lib.train_utils` (`lr_at`, `dist_info`, `StopFlag`, `effective_rank`)
- `lib.checkpoint.atomic_torch_save`
- v4 volume geometry (do not rebuild the NIfTI pipeline)

**Do not reuse**

- 2D `VisionTransformer` learned absolute `pos_embed`
- `lib/views.py` 2D affine crops / asinh as the default 3D view path
- `lib/slabs.py` slice sampling
- `lib.config.DEFAULT_CONFIG` (VAE keys)

## Coordinate convention

- **Storage** (v4): array `(Z, Y, X)` or `(Z, 1, Y, X)`.
- **Encoder API** (spec): `view` is `(B, C, X, Y, Z)`.
- **PyTorch Conv3d**: internally permute to `(N, C, D, H, W) = (N, C, Z, Y, X)`.
- **Token flatten order:** `(Z, Y, X)` C-order, matching Conv3d output.
- **Physical:** axis order `(x, y, z)` mm. Voxel-center of index `i` on an axis is `origin + (i + 0.5) * spacing`.
- **Token center** after a crop/resize: `bbox_min + (patch_index + 0.5) * (bbox_extent / n_patches)`.
- **BBox:** half-open `(x0, y0, z0, x1, y1, z1)` in mm.

## Module map

| Spec | Path |
|---|---|
| Index + IDs | `brainmodel/data/index.py` |
| FG mask cache | `brainmodel/data/preprocess.py` |
| Volume dataset | `brainmodel/data/dataset.py` |
| Physical coords | `brainmodel/data/coords.py` |
| Multi-crop | `brainmodel/data/views.py` |
| Augment | `brainmodel/data/augment.py` |
| Patch + scale | `brainmodel/models/patch_embed.py` |
| 3D RoPE | `brainmodel/models/rope3d.py` |
| Encoder | `brainmodel/models/vit3d.py` |
| Projector | `brainmodel/models/projector.py` |
| SigReg wrap | `brainmodel/losses/sigreg.py` |
| View inv | `brainmodel/losses/invariance.py` |
| Dense + `footprint_pool` | `brainmodel/losses/dense.py` |
| Combine | `brainmodel/losses/ssl.py` |
| Train | `brainmodel/train/train_ssl.py`, `train_brainmodel.py` |
| Health / probes / viz | `brainmodel/eval/{health,probes,viz}.py` |
| Config | `configs/brainmodel.yaml`, `brainmodel/config.py` |

## What Stage 1 code does *not* include

Per spec **[Later]**: reasoning module, latent dynamics, language, atlas coordinates as extra PE, cross-scan positives as a loss (index columns are present so this can be added later).

Dense frozen segmentation against nnU-Net is a probe *interface* plus a linear token decoder; a full nnU-Net baseline is not trained in this drop.

## Train/infer contract for Stage 2

`BrainEncoder.forward` returns tokens **and** `coords_mm`. Background tokens remain in the sequence by default (`drop_background: false`) so attention sees skull/air edges; losses use `token_mask` / fg fraction. A later reasoning module can prune with the same mask and request a native local crop through the same forward.
