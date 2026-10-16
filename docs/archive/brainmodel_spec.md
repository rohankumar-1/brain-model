# BrainModel: Project Proposal and Implementation Spec

Audience: a coding agent implementing this repo. Sections marked **[Stage 1]** are in scope for the first implementation. Sections marked **[Later]** are design context that should influence interfaces now but should not be built yet.

Values in `<ANGLE_BRACKETS>` or listed under "Open parameters" are unknown and must be read from config or the data, never hard-coded.

See also [brainmodel_stage1_plan.md](brainmodel_stage1_plan.md) for the implementation map against the existing 2D ViT / v4 store.

---

## 1. Goal

Build a foundation vision model for 3D brain CT and MRI that:

1. Learns spatially resolved, physically anchored token representations of full brain volumes via self-supervised pretraining (LeJEPA-style: view invariance + SigReg, no teacher / EMA / stop-gradient).
2. Retains fine-grained pathology (small bleeds, small lesions) and does not collapse into scanner or site signatures.
3. Provides a token interface that a later **reasoning module** can consume: iterative latent computation with variable compute, grounded intermediate heads, and latent trajectory / counterfactual prediction.

Non-goals for Stage 1: language alignment, trajectory modeling, the reasoning module, mesh or surface models.

### Guiding design principles

- **Physical coordinates, not voxel indices.** A token means a physical region in mm. Positional information and patch sizes are defined in physical space.
- **Tokens must be individually meaningful.** Pooled-only losses produce a crop descriptor, not usable tokens. Losses must act at the token level.
- **Invariance teaches only what augmentation varies.** Augmentations are chosen deliberately (see section 6). Left/right flips are off by default because laterality is clinically meaningful.
- **Background is not signal.** Foreground masks gate crop sampling and loss terms.
- **Auditability.** Later-stage intermediate states should be decodable (segmentation, measurements, attention maps). Do not design components that make this impossible.

---

## 2. Data

- All data is 3D volumes, nominally **256 x 256 x 128** voxels, CT and MRI.
- **Do not interpolate z up to 256.** It adds no information, doubles token count, and creates an interpolation-smoothness shortcut along z. Keep native 256x256x128.
- Rectangular (non-cubic) crops are expected and desired.

### Open parameters (read from metadata or config)

| Parameter | Notes |
|---|---|
| Voxel spacing (mm) per volume | Needed for physical patch sizes and coordinates. Read from NIfTI/DICOM header. Check whether it is consistent across the dataset. |
| Modality per volume | CT vs MRI (and MRI sequence if available). |
| Skull-stripped / registered / native space | Affects background fraction and scanner-signature shortcut. Make the foreground-mask step configurable. |
| Site / scanner ID | Used for the nuisance probe only, never as a training input. |
| Patient ID, timepoint | Not used in Stage 1 loss, but **must be loaded and preserved in the dataset index** for later cross-scan positives and trajectory modeling. |

### Preprocessing (offline, cached)

1. Load volume and header. Store spacing and affine.
2. Intensity handling:
   - CT: clip to a configurable HU window (default `[-200, 300]` for parenchyma plus a second channel or augmentation path for a wider window if desired), then scale. Treat HU as physically meaningful. Do not per-volume z-score CT.
   - MRI: normalize using statistics computed **inside the foreground mask only** (background is Rician noise, and whole-volume normalization amplifies it).
3. Compute and cache a **foreground mask** per volume:
   - CT: threshold (default `> -300 HU`, configurable) followed by largest connected component and morphological closing.
   - MRI: Otsu threshold followed by largest connected component and closing.
   - If volumes are zero-padded, mark padding as background and record the padding extent.
4. Cache the per-volume tuple: `(volume, mask, spacing, affine, ids)`.

---

## 3. Architecture **[Stage 1]**

### 3.1 Encoder

A single 3D ViT, shared weights across all view types.

- **Tokenization:** 3D patch embedding (strided Conv3d or linear projection of flattened patches). Patch size is configured in voxels per view type, chosen so the **physical patch size** is meaningful. Support anisotropic patch shapes (e.g. 16x16x8 voxels when spacing is 1x1x2 mm gives 16 mm cubes).
- **Position encoding:** 3D RoPE (or equivalent relative encoding) driven by the **physical center coordinate (mm) of each token**, expressed in a shared per-volume frame (the volume's own origin is fine for Stage 1). It must work for any crop size, shape, and offset. No learned absolute position embeddings tied to a crop size.
- **Scale conditioning:** each token (or view) receives the physical patch size as input, via a scale embedding or FlexiViT-style resizable patch embedding, so the network does not have to infer scale from content.
- **Attention:** full or windowed attention within a view. Views never attend to each other. Make window size configurable so the same code scales to full-volume inference.
- **Outputs per view:** one embedding per token, plus an optional CLS token.
- **Projector:** small MLP mapping tokens (and pooled CLS or mean) into the loss space, as in LeJEPA.
- **Size:** start small (e.g. ViT-S-like) with activation checkpointing. Everything configurable.

### 3.2 Interface contract for later stages

The encoder must expose:

```python
class BrainEncoder(nn.Module):
    def forward(self, view: torch.Tensor,        # (B, C, X, Y, Z)
                coords_mm: torch.Tensor,         # (B, N, 3) token-center physical coords
                patch_size_mm: torch.Tensor,     # (B, 3) or (B, N, 3)
                token_mask: torch.Tensor | None  # (B, N) bool, True = keep
               ) -> dict:
        """returns {'tokens': (B,N,D), 'cls': (B,D), 'proj_tokens': (B,N,P),
                    'proj_cls': (B,P), 'coords_mm': (B,N,3)}"""
```

Tokens always travel with their physical coordinates. Token pruning (dropping background tokens) must be supported via `token_mask` without breaking position encoding.

---

## 4. View generation **[Stage 1]**

Per volume per step, generate views, each with **independent augmentation** and each carrying its **physical bounding box** `(x0,y0,z0,x1,y1,z1)` in mm.

| View type | Count (default) | Extent | Resolution | Patch (voxels) | Tokens |
|---|---|---|---|---|---|
| Global | 2 | rectangular crop covering ~50-100% of each axis | resized to 128x128x64 | 16x16x8 | 512 |
| Local | 4 (up to 8) | rectangular prism, ~64x64x32 to 96x96x48 native voxels, randomized size and aspect ratio | native | 8x8x4 | ~512 |

Rules:

- Local crops: minimum 3 patches along every axis (no extremely thin slabs).
- Local crops must satisfy a **minimum foreground fraction** (default 0.4, configurable) using the cached mask. Resample (bounded retries) if violated. Keep a small configurable fraction (~10%) of boundary-straddling crops so the model learns head edges.
- Global views rarely need the filter but should still check foreground fraction.
- Batch views by shape (as in DINO multi-crop): one forward pass for all globals, one per local shape group.
- All counts, sizes, patch sizes, and thresholds are config values.

---

## 5. Losses **[Stage 1]**

One encoder, shared weights, gradients through all views. No EMA, no teacher, no stop-gradient.

Total: `L = (1 - lambda) * (L_inv + w_same * L_dense_same + w_cross * L_dense_cross) + lambda * (L_sigreg_view + w_tok * L_sigreg_tok)`

Defaults to tune: `lambda ~ 0.05`, token-level weights start modest. All configurable.

### 5.1 View-level invariance

Let `mu` be the mean of the projected global-view CLS (or mean-pooled) embeddings. Pull every view (global and local) toward it:

`L_inv = mean_v || z_v - mu ||^2`

### 5.2 Dense invariance, same scale

Where two views of the same scale overlap physically, tokens at the same physical location should match. Use token center coords and the bounding boxes to find overlaps. If token grids are misaligned, interpolate one view's token map (trilinear on the projected tokens, using physical coordinates) onto the other's token centers. Only overlapped, foreground tokens contribute.

### 5.3 Dense invariance, cross scale (footprint pooling)

Each global token covers many local-token footprints. **Do not** pull every local token to the covering global token (this forces local tokens to be identical over a ~32 mm region and erases fine detail). Instead:

1. For each global token, compute its physical footprint.
2. Collect local tokens (from a local view) whose centers fall inside that footprint, **only if the local crop fully covers the footprint** (configurable minimum coverage).
3. Average those local tokens (foreground tokens only).
4. Match that average to the global token with the same distance as above.

This is the fiddliest component. Implement it as a standalone, unit-tested function `footprint_pool(local_tokens, local_coords, global_coords, global_patch_mm, coverage_thresh)`.

### 5.4 SigReg (LeJEPA)

Sketched Isotropic Gaussian Regularization: project a batch of embeddings onto many random unit 1D directions (resampled each step) and penalize deviation of each 1D marginal from standard normal using the Epps-Pulley characteristic-function test statistic. Apply to:

- the batch of view-level embeddings, and
- the **pool of foreground token-level embeddings** across the batch (thousands of samples per step).

Implementation notes: keep SigReg in a clean module with unit tests (e.g. input drawn from N(0, I) gives near-zero loss; collapsed input gives high loss). Number of projection directions is configurable. Verify it works under DDP (gather embeddings across ranks or document the per-rank behavior).

### 5.5 Token masking in all losses

Each token gets a foreground fraction from the cached mask (computed from the voxels it covers). Tokens below `fg_token_thresh` (default 0.3) are excluded from SigReg, dense invariance, and footprint averaging. They stay in the sequence (so attention sees head boundaries), or are optionally dropped via `token_mask`. Do **not** use a variance filter. Low-variance tissue such as CSF or homogeneous parenchyma is meaningful.

---

## 6. Augmentations **[Stage 1]**

Applied in physical space, independently per view:

- Random crop and resize (as in section 4), small rotations and translations (small angles, configurable).
- Intensity: bias field, gamma, contrast, additive noise, Gaussian blur. For CT, **limited** window/level jitter only, because HU is physically meaningful.
- Resolution and slice-thickness simulation (downsample then resample along z, random blur).
- Random padding or random border treatment to avoid a fixed padding tell.
- **Left/right flip: OFF by default** (configurable flag), since laterality is clinically meaningful.
- Goal: make views from the same scan hard to match by scanner signature. Never share noise or bias-field realizations across views.

---

## 7. Evaluation and diagnostics **[Stage 1]**

Build these before long training runs, so training can be monitored from the start.

1. **Collapse and health metrics (logged every N steps):** per-dimension embedding std, effective rank / covariance eigenvalue spectrum, SigReg value, each loss term separately, fraction of tokens masked.
2. **Scanner / site nuisance probe:** a linear probe on frozen pooled and token embeddings predicting site / scanner / modality (if labels exist). High accuracy means the model encodes nuisance. Track over training.
3. **Token-map visualization:** PCA of token embeddings over a full volume, rendered as RGB overlays on slices. Check spatial smoothness and anatomical coherence (dense-loss weight too high gives overly smooth maps, too low gives noisy ones).
4. **Dense downstream probes (frozen encoder):** linear or small-head segmentation (brain structures and any available lesion labels) and small-lesion detection. Baseline against a plain supervised or nnU-Net-style model where labels exist.
5. **Full-volume inference check:** run the trained encoder on a full 256x256x128 volume (largest rectangular prism) and confirm tokens are consistent with those from sub-crops covering the same region (train-small / infer-big generalization).
6. **Foreground ablation:** report metrics with and without foreground-aware sampling / loss masking.

---

## 8. Repository layout (suggested)

```
brainmodel/
  configs/                 # YAML (Hydra or plain): data, model, views, loss, aug, train
  data/
    index.py               # builds dataset index with patient/timepoint/site/modality
    preprocess.py          # normalization, foreground mask caching
    dataset.py             # returns cached volume, mask, spacing, ids
    views.py               # multi-crop sampler: boxes in mm, fg-fraction filtering
    augment.py             # physical-space augmentations
  models/
    patch_embed.py         # anisotropic 3D patch embed, scale conditioning
    rope3d.py              # physical-coordinate 3D RoPE
    vit3d.py               # BrainEncoder
    projector.py
  losses/
    sigreg.py              # Epps-Pulley, random projections
    invariance.py          # view-level
    dense.py               # same-scale overlap + cross-scale footprint_pool
  train/
    train_ssl.py           # step loop, DDP, AMP, checkpointing, logging
  eval/
    health.py              # collapse metrics
    probes.py              # scanner probe, linear seg probe
    viz.py                 # token PCA overlays
  tests/                   # unit tests, see below
```

### Required unit tests

- Physical-coordinate bookkeeping: crop to bounding box to token centers round-trips correctly for anisotropic spacing and arbitrary crop offsets.
- `footprint_pool` on synthetic inputs with known answers (including partial coverage and empty footprints).
- SigReg behavior on Gaussian vs. collapsed vs. anisotropic inputs.
- RoPE: translating a crop's coordinates produces consistent relative behavior.
- Foreground-fraction filter returns valid crops or fails loudly after the retry limit.
- Smoke test: tiny model, tiny synthetic volumes, loss decreases and no NaNs.

### Engineering requirements

- Mixed precision (bf16), activation checkpointing, DDP. Gradient accumulation does **not** substitute for batch size in SigReg, so document the effective sample count per step.
- Config-driven everything. Log full resolved config with every run.
- Deterministic seeds for view sampling in eval mode.
- Cache the foreground masks and dataset index to disk.

---

## 9. Staged plan

**Stage 1 (implement now):** sections 2-8. Deliverable: pretrained encoder, diagnostics dashboard, frozen-token baselines on dense tasks.

**Gate before Stage 2:** tokens pass the dense probes, the scanner probe is not near-perfect, token maps look anatomically coherent, and full-volume inference is consistent.

---

## 10. Later stages **[Later]** (design context only, do not build yet)

### Stage 2: Reasoning module over full-volume tokens

- Perceiver-style set of latent queries cross-attending to full-volume tokens, optionally anchored to anatomical regions via atlas-aligned coordinates (used as an extra positional signal that the model can ignore when registration is poor).
- **Iterative computation with variable compute:** repeated cross-attention / refinement steps with a halting mechanism, so easy cases stop early.
- Background-token pruning using the foreground mask before this stage to cut token count and allow finer patches inside the head.
- "Look closer here": the module can request high-resolution local crops of regions of interest, using the same encoder with the same physical-coordinate interface.
- **Grounded intermediate heads:** decodable segmentation, measurements (midline shift, hematoma volume, ventricular size), and region attention maps attached to intermediate states for auditability.

### Stage 3: Latent dynamics and counterfactuals

- Predict future latent state from current latent state, delta-t, and optional treatment variables (JEPA-style, in latent space, not pixel space).
- Cross-modal prediction (CT latent to MRI latent) as an auxiliary world-model signal.
- Hypothesis testing by analysis-by-synthesis: condition on a candidate diagnosis or time-since-onset, predict the expected latent, score against observed.
- Deformation field as an auxiliary decoded target.
- Known risks to design around: scanner shortcut (prefer same-site, same-protocol pairs, harmonize), selection bias in rescanned patients, sparse irregular timepoints, confounding by treatment.

### Stage 4: Outcome-grounded training and language interface

- Outcome-grounded objectives: follow-up imaging, clinical outcomes, verifiable measurements as rewards or supervision.
- Report alignment (CLIP/SigLIP-style or a light decoder on frozen or lightly tuned latents), after the visual encoder is solid, so language does not become the bottleneck.
- Case-based retrieval over latent space with outcomes.
- An LLM as orchestrator and report writer calling BrainModel as a tool for structured, verifiable outputs.

---

## 11. Risks and open questions

1. **Dense loss weight:** too heavy smooths away small pathology, too light leaves a crop-level descriptor. Tune against small-lesion probes.
2. **Token-level SigReg vs. spatial correlation:** neighboring tokens are naturally correlated, and an isotropic Gaussian target pushes against that. Watch token-map smoothness.
3. **Spatial knowledge vs. crop invariance:** location is clinically meaningful. Coordinates in the position encoding preserve it. Consider adding atlas-aligned coordinates later.
4. **Memory:** 6-10 views of ~512 tokens each in a 3D ViT is heavy. Begin with fewer local views and a small model.
5. **Scanner shortcut in 3D:** crops from one volume share a scanner signature, which is stronger than in 2D. Mitigate with augmentation, foreground-only loss terms (scalp / table / background carry the signature), and cross-scan positives (same patient, different time or modality) when available. Add these as an optional positive source once patient/timepoint indexing is in place.
6. **Unknowns to resolve from the data:** voxel spacing consistency, modality mix, skull-stripping / registration status, padding conventions, and label availability for probes.
