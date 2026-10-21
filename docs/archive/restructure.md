# Restructure notes

Structural and implementation changes only. Baseline is commit `d488fbe`.

## Size

| | Before | After |
|---|---|---|
| All tracked text files (lines) | 22,854 | ~20,200 (incl. this doc) |
| Python (lines) | 19,019 | 16,829 |

Line count dropped while adding registration, brain masking, the new augmentations and five new test modules. The savings come from deleting the VAE stack and merging three trainers.

## Layout

One package, `brain/`, with a shared core:

- `brain/common/`: `config`, `dist`, `optim`, `metrics`, `checkpoint`, `paths`, `windowing`. These replace the per-trainer copies.
- `brain/train/`: `engine.py` (shared loop), `defaults.py` (per-method defaults and smoke config), `__main__.py` (`python -m brain.train <method>`), and one `Trainer` subclass per method: `lejepa2d`, `dino2d`, `lejepa3d`.
- `brain/objectives/`: `sigreg` (SIGReg plus the 2D LeJEPA loss), `dino` (DINO/iBOT/KoLeo, centering), `lejepa3d`, `dense`.
- `brain/models/`: 2D ViT, 3D encoder, `braindino` (DINO head plus model), shared `projector`.
- `brain/data/`: augmentation, volume dataset, 3D crops, batching, slab/slice loading.
- `brain/prep/`: preprocessing pipeline, now with registration and brain mask modules.
- `brain/downstream/`, `brain/datasets/`: paths now come from `brain.common.paths`.

Naming: BrainDINO is the `dino2d` method and is the MRI-foundation-model comparison. The project model is `lejepa3d`. Docs and configs were renamed to match (`docs/brainmodel.md` -> `docs/lejepa3d.md`, `docs/brainmodel_downstream.md` -> `docs/downstream_3d.md`; configs under `configs/{lejepa2d,dino2d,lejepa3d}/`). `out_dir` and `init_from` values in configs were left untouched because they point at existing runs.

## Consolidations

- **Training loop.** One engine owns warmup+cosine LR, atomic `last/best/step_*` checkpoints, resume, SIGTERM preemption with the Slurm requeue flag, NaN guards, the log table and `--set a.b=c` overrides. Methods implement `step`, `state`, `evaluate`, and optional hooks.
- **Config.** Defaults, then YAML, then CLI overrides, via one loader. Override values parse int, then float, then YAML.
- **SIGReg.** A single implementation used by both LeJEPA variants.
- **Projector, parameter groups, effective rank, checkpoint helpers.** One copy each.
- **Jobs.** `jobs/_env.sh` holds environment setup. Nine train scripts became `train.sbatch`, `train_pi.sbatch`, `train_pi4.sbatch`. Other job scripts source `_env.sh`. Cluster paths are overridable via `BRAIN_SCRATCH`, `BRAIN_DATASETS`, `BRAIN_REPO`, `BRAIN_ENV`, `BRAIN_TEMPLATE`, `BRAIN_SYNTHSTRIP`, `BRAIN_STRIPPED`.

## Removed

The VAE model, trainer, reconstruction library and tests; their requirements file and configs; the legacy zarr preprocessing and manifest-resplit scripts; the RSNA regression script; `lib/data`, `lib/zarr_loader`, `lib/progress`; the two frozen-classifier test scripts.

## Preprocessing (format version 5)

Order: read to RAS, N4, centering (registration start pose), rigid then affine Mattes-MI registration to a skull-on T1 template (MNI152), single resample onto a 256x256x128 1 mm grid.

- Registration has sanity limits and falls back to translation on failure. The failure is recorded as `reg_ok=False` and the volume is kept.
- Intensities stay in native units: CT in HU, MRI as robust z-score with floor -6.
- A brain mask is computed (SynthStrip if available, classical fallback otherwise) and stored as `<case>.brain.zip`, indexed as `mask_brain`.
- Native geometry is recorded: `thick_mm`, `thick_axis`, `aniso`.
- Run resume now uses `store.is_current` and retries error rows (previously any existing zip counted as done).
- `process()` itself defaults to no registration so legacy callers and tests work; the CLI and configs default to the new behavior. Changing prep settings needs `--force` or a new root.

## Augmentation (default on)

Appearance order: bias field (MRI only, about the floor), through-plane blur (native units, quadrature to a target FWHM using `thick_mm`), asinh window, quantile-knot monotone contrast curve with optional flip about the median (background kept darkest), then blur, drift, noise. Skull dropout happens at volume level in the dataset using `mask_brain`, and is skipped for fixed validation crops. Crops carry `thick_mm`/`thick_axis` through the batch pipeline. Bias and through-plane assume 1 mm voxels. A guard falls back to the identity curve when too few foreground voxels are sampled.

## Behavior changes to be aware of

- `lejepa2d` now writes `best.pt`.
- Weight-decay exclusions are unified, so cls and register tokens are no longer decayed in 3D.
- Blur kernel radius is adaptive.
- Downstream leaderboard column `vae` is now `ckpt`.
- Latent fixes: wrong repo-root constant in prep after the move, hard-coded scratch paths, ignored transforms in `_roi` and `resample`.

## Not executed (no torch or SimpleITK available where this was built)

Verified by ruff, compile, an import resolver, a config-key checker, and numpy-only tests (`test_config`, `test_brainmask`). Not executed: the engine and all three trainers, the augmentation module, registration and the SimpleITK pipeline paths, SynthStrip, and the ported torch tests. Run on the Mac (CPU):

```
python -m pytest tests -q
python -m brain.train lejepa3d --synthetic --max-steps 8
```

Registration needs `--template` or `BRAIN_TEMPLATE` pointing at an MNI152 T1 (skull on).
