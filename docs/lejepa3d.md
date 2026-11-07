# 3D LeJEPA (BrainQuery's self-supervised objective)

Code:
- trainer: `brain/train/lejepa3d.py`;
- models: `brain/models/{vit3d,viewnet3d}.py`;
- objectives: `brain/objectives/{lejepa3d,dense,sigreg}.py`;
- data: `brain/data/{volumes,crops3d,batch3d,augment}.py`.

The current recipe is `configs/lejepa3d/v4_scratch.yaml` (BQ-v4, the encoder R7–R14 start from). The same objective is
the SSL branch of the joint trainer (`configs/joint/r15.yaml`, `brain/train/joint.py`).

## Input and tokens
- Volumes live on the shared RAS 1 mm 256 x 256 x 160 grid in native units (CT HU, MRI robust z) and are windowed on
  the GPU.
- With `window.mode: roles`, a scan becomes three fixed windows: CT brain / subdural / bone, and MRI tissue / wide /
  bright tail. `channels: 1` (BQ-v4) keeps one role window per row; 3 makes them the input channels.
- Patches are 16 x 16 x 8 mm, so the full volume is a 20 x 16 x 16 = 5120-token grid.
- RoPE works in millimetres.
- An optional conv stem (`model.conv_stem`) runs two stride-2 convolution stages before the patchify.
- Air tokens are dropped by the token gate (`views.gate`).

## Views (`brain/data/crops3d.py`)
- Crop origins snap to the coarse token grid, so overlapping crops share tokens exactly. Each crop carries `pos` (its
  coarse-grid indices) and `parent` (token → position).
- `global` crops are placed freely; later globals are centred inside the first one when `nested`.
- `local` and `fine` crops lie inside a global crop. `fine` crops use 8 x 8 x 4 patches, and eight fine tokens average
  into one coarse position.
- `aug.geometric` gives each view its own rotation, scale and left-right mirror. Views are warped from the whole volume
  on the GPU (`batch3d.sample_views`); token bookkeeping follows each token centre into the scan, so dense agreement
  and masked prediction compare the same anatomy.

## Appearance augmentation (`brain/data/augment.py`, config `aug`)
In order, per crop row:
- `histmatch` (MRI only);
- `bias` (coil shading);
- `through_plane` (blur along the acquisition axis, up to a target slice thickness, using the `thick_mm` /
  `thick_axis` prep recorded);
- noise (Rician on MRI);
- `curve`: a monotone contrast curve with knots at the crop's foreground quantiles; with probability `flip` it is
  negated (tissue order reversed);
- `blur`;
- the role windows (jittered by `window.jitter`).

`skull` (with `per_view`) cuts a view to the stored brain mask, so skull-on and skull-stripped scans share an
appearance.

`aug.tie_views` (R15): every view of a volume draws the same contrast (histogram match, curve, window jitter); noise,
blur and bias stay per view. The agreement terms then ask for invariance to crop and degradation, not to contrast, so
sequence-specific information (enhancement, FLAIR hyperintensity) is not trained away. Use it together with
`loss.win_weight: 0`.

## Loss (`brain/objectives/`)
`loss = (1-λ)·inv + λ·sigreg + win_w·win + dense_w·((1-λ)·dense_inv + λ·dense_sig) + predict_w·(pred + pixel_w·pix)`
- `inv` / `sigreg`: LeJEPA on the pooled embeddings of the global crops.
- `win`: one crop seen under two appearances must agree. This is pure contrast invariance; it is off in R15.
- `dense_inv`: every coarse position seen by two or more views is pulled to the mean of its token embeddings.
  `dense_sig`: SIGReg over `tokens_per_view` random tokens per view.
- `pred` / `pix`: blocks of tokens are hidden in each global crop. A small predictor regresses the hidden tokens'
  embeddings from the unmasked crop's own pass (stop-grad target) and the 2 x 2 x 2 voxel cell means. Its weight ramps
  over `ramp_steps`. The predictor is training-only.

## Training
- One `ViewNet` forward covers all views, with length-bucketed rows and no host syncs in the step.
- Resumes from `<out_dir>/last.pt`; `--synthetic` runs on phantoms; `--set a.b=c` overrides any key.
- Watch `erank` (pooled) and `tok_erank` (tokens): both collapsing toward 1 is the failure mode.

## Evaluation
The frozen benchmark (`docs/benchmark.md`) and `python -m brain.downstream.probe3d` (lesion, fingerprint and position
token probes with pixel and random-init controls).

## Tests
`tests/test_lejepa3d.py`, `tests/test_augment.py`, `tests/test_geometric.py`.
