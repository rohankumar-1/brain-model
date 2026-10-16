# 3D LeJEPA (the project model) — current recipe

Code: `brain/train/lejepa3d.py` (trainer), `brain/models/{vit3d,viewnet3d}.py`, `brain/objectives/{lejepa3d,dense,sigreg}.py`, `brain/data/{volumes,crops3d,batch3d,augment}.py`. Run: `python -m brain.train lejepa3d --config configs/lejepa3d/tiny.yaml` (pooled-only baseline: `tiny_pooled.yaml`).
Older design notes (CLS token, masking, v4 canvas) are in `docs/archive/` and no longer match the code.

## Input and tokens
- Volume `(1, 256, 256, 128)` on the shared RAS 1 mm grid, asinh-windowed on the GPU.
- Patch 16×16×8 mm; the full volume is a 16×16×16 = 4096-token grid. RoPE uses millimetres. No CLS token; pooling is a mean over kept tokens.
- Air tokens are dropped by a threshold at the patch centre (`views.gate`). Nothing is reconstructed or masked.

## Views (`brain/data/crops3d.py`)
- Crop origins snap to the coarse token grid, so overlapping crops share tokens exactly. Each crop carries `pos` (its coarse-grid indices) and `parent` (token → position).
- `global` crops are placed freely; later globals are centred inside the first one when `nested`. `local` and `fine` crops lie inside a global crop. `fine` crops use 8×8×4 patches; eight fine tokens average into one coarse position. One extra crop (`win`) is seen under two windows.
- Appearance augmentation, per crop row on the GPU, is described in the next section.

## Appearance augmentation (`brain/data/augment.py`, config `aug`)
The aim is that agreeing across views can only be done through anatomy, never through contrast, resolution or "this scan has a skull". In order, per crop row, each drawn independently:
- `bias`: smooth multiplicative gain on MRI rows (coil shading), applied in native units about the background floor.
- `through_plane`: Gaussian blur along the volume's acquisition thick axis, in native units, until its effective slice thickness reaches a random target in `fwhm_vox` (mm). Uses `thick_mm` / `thick_axis` recorded by prep, so a 1 mm 3D T1 is made to look like a 5 mm axial stack while a scan already thicker than the target is left alone; without the columns the axis is random (60% z).
- `window`: the asinh window with random centre/width (`aug.window`).
- `curve`: a random monotone piecewise-linear contrast curve whose knots sit at quantiles of the crop's own foreground, so each tissue mode can be stretched or squeezed independently; with probability `flip` it is negated about the median (tissue order reversed, as T1 to T2 does). Background stays the darkest value.
- `blur`, `drift`, `noise`: isotropic blur, slow additive drift and noise in windowed units.
- `skull` (volume level, in the loader): with probability `p` a whole volume is cut to its stored brain mask before cropping, so skull-on (RSNA, CQ500, OASIS) and skull-stripped (BraTS, MSD) scans share an appearance. It needs a store built with brain masks (`mask_brain` in `volumes.csv`); with none it does nothing, and validation never uses it.
All are on by default. A config that wants the pre-restructure behaviour sets `aug.curve.p`, `aug.bias.p`, `aug.through_plane.p` and `aug.skull.p` to 0.

## Loss (`brain/objectives/`)
`loss = (1-λ)·inv + λ·sigreg + win_w·win + dense_w·((1-λ)·dense_inv + λ·dense_sig) + predict_w·(pred + pixel_w·pix)`
- `inv`/`sigreg`: the 2D LeJEPA objective on pooled embeddings of the global crops (locals only with `loss.local_pooled`).
- `dense_inv`: every coarse position seen by ≥2 views is pulled to the mean over views of its token embeddings (mean not detached). `dense_sig`: SigReg over `tokens_per_view` random tokens per view. Token head is a LayerNorm MLP (`model.dense_dim`).
- Set `loss.dense.weight: 0` and `local_pooled: true` for the pooled-only baseline (`configs/lejepa3d/tiny_pooled.yaml`).
- Optional `loss.predict` (needs dense; off by default). Consensus between crops rewards embeddings that ignore context, so this adds a term that needs it: blocks of tokens (`blocks`, `block_tokens`) are hidden in each global crop; the encoder sees only the rest (token dropping, RoPE is in mm); a small predictor (`depth`, learned mask token, shares RoPE) outputs for every hidden token (a) `pred`: its token-head embedding from the *unmasked* crop's own pass (stop-grad target, weight 1), and (b) `pix`: the 2×2×2 cell means of the voxels in every kept patch (`pixel_cells`, weight `pixel_weight`), which keeps local detail from being squeezed out. Weight ramps over `ramp_steps`. The predictor is training-only; downstream loaders drop `predictor.*`.
- Continue-from: `output.init_from` loads weights only (fresh optimizer/schedule; predictor starts random). `configs/lejepa3d/tiny_predict.yaml` (8k steps from the 30k dense run) vs `configs/lejepa3d/tiny_continue.yaml` (same schedule, predict weight 0) isolates the effect.

## Training (`brain/train/lejepa3d.py` on `brain/train/engine.py`)
One `ViewNet` forward for all views (DDP/SyncBN friendly), length-bucketed rows, per-GPU batch 192 on 2×L40S, no host syncs in the step. Resumes from `<out_dir>/last.pt`; `--no-resume` starts over; `--synthetic` runs on phantoms without a store; `--set a.b=c` overrides any config key. Logs `erank` (pooled) and `tok_erank` (tokens); both collapsing toward 1 is the failure mode to watch.

## Evaluation
`python -m brain.downstream.probe3d --ckpt A --ckpt B --out runs/probes/x.json` (or `sbatch jobs/probe3d.sbatch`): frozen-token lesion probes on held-out sets, scanner/dataset fingerprint probe, token-position probe, with pixel and random-init controls. 2D reference: lesion lift 6.2, participation ratio 58, position R² 0.84. `--reps N` adds 95% bootstrap intervals over test volumes (test-set sampling only, not run-to-run variance) and `--ref NAME` prints paired differences against one encoder. Task suite and segmentation (`brain/downstream/seg3d.py`) for 3D checkpoints: `docs/downstream_3d.md`.

## Tests
`python -m pytest tests -q` (CPU); the 3D model and objective tests are in `tests/test_lejepa3d.py`, augmentation in `tests/test_augment.py`.
