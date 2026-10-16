# BrainModel (3D) — current recipe

Code: `brainmodel/`, entry point `train_brainmodel.py`, configs `configs/brainmodel_tiny_{lejepa,pooled}.yaml`.
Older design notes (CLS token, masking, v4 canvas) are in `docs/archive/` and no longer match the code.

## Input and tokens
- Volume `(1, 256, 256, 128)` on the shared RAS 1 mm grid, asinh-windowed on the GPU.
- Patch 16×16×8 mm; the full volume is a 16×16×16 = 4096-token grid. RoPE uses millimetres. No CLS token; pooling is a mean over kept tokens.
- Air tokens are dropped by a threshold at the patch centre (`views.gate`). Nothing is reconstructed or masked.

## Views (`brainmodel/data/crops.py`)
- Crop origins snap to the coarse token grid, so overlapping crops share tokens exactly. Each crop carries `pos` (its coarse-grid indices) and `parent` (token → position).
- `global` crops are placed freely; later globals are centred inside the first one when `nested`. `local` and `fine` crops lie inside a global crop. `fine` crops use 8×8×4 patches; eight fine tokens average into one coarse position. One extra crop (`win`) is seen under two windows.
- Appearance augmentations after windowing, per row on the GPU: noise, blur, smooth additive drift (`aug`).

## Loss (`brainmodel/losses/`)
`loss = (1-λ)·inv + λ·sigreg + win_w·win + dense_w·((1-λ)·dense_inv + λ·dense_sig) + predict_w·(pred + pixel_w·pix)`
- `inv`/`sigreg`: the 2D LeJEPA objective on pooled embeddings of the global crops (locals only with `loss.local_pooled`).
- `dense_inv`: every coarse position seen by ≥2 views is pulled to the mean over views of its token embeddings (mean not detached). `dense_sig`: SigReg over `tokens_per_view` random tokens per view. Token head is a LayerNorm MLP (`model.dense_dim`).
- Set `loss.dense.weight: 0` and `local_pooled: true` for the pooled-only baseline (`brainmodel_tiny_pooled.yaml`).
- Optional `loss.predict` (needs dense; off by default). Consensus between crops rewards embeddings that ignore context, so this adds a term that needs it: blocks of tokens (`blocks`, `block_tokens`) are hidden in each global crop; the encoder sees only the rest (token dropping, RoPE is in mm); a small predictor (`depth`, learned mask token, shares RoPE) outputs for every hidden token (a) `pred`: its token-head embedding from the *unmasked* crop's own pass (stop-grad target, weight 1), and (b) `pix`: the 2×2×2 cell means of the voxels in every kept patch (`pixel_cells`, weight `pixel_weight`), which keeps local detail from being squeezed out. Weight ramps over `ramp_steps`. The predictor is training-only; downstream loaders drop `predictor.*`.
- Continue-from: `output.init_from` loads weights only (fresh optimizer/schedule; predictor starts random). `brainmodel_tiny_predict.yaml` (8k steps from the 30k dense run) vs `brainmodel_tiny_continue.yaml` (same schedule, predict weight 0) isolates the effect.

## Training (`brainmodel/train/train_ssl.py`)
One `ViewNet` forward for all views (DDP/SyncBN friendly), length-bucketed rows, per-GPU batch 192 on 2×L40S, no host syncs in the step. Resumes from `<out_dir>/last.pt`; `--no-resume` starts over. Logs `erank` (pooled) and `tok_erank` (tokens); both collapsing toward 1 is the failure mode to watch.

## Evaluation
`python -m downstream.probe3d --ckpt A --ckpt B --out runs/probes/x.json` (or `sbatch jobs/probe3d.sbatch`): frozen-token lesion probes on held-out sets, scanner/dataset fingerprint probe, token-position probe, with pixel and random-init controls. 2D reference: lesion lift 6.2, participation ratio 58, position R² 0.84. `--reps N` adds 95% bootstrap intervals over test volumes (test-set sampling only, not run-to-run variance) and `--ref NAME` prints paired differences against one encoder. Task suite and segmentation (`downstream/seg3d.py`) for 3D checkpoints: `docs/brainmodel_downstream.md`.

## Tests
`python -m tests.test_brainmodel`, `python -m tests.test_probe3d`, `python -m tests.test_seg3d` (CPU).
