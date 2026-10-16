> Historical design spec for store format 4. Module paths below predate the `brain/` package (`data/prep` is now `brain/prep`, `lib/*` is split across `brain/common` and `brain/data`). Current behaviour: `docs/prep.md`, `docs/lejepa3d.md`, `docs/restructure.md`.

# v4 spec: native-unit preprocessing and window-randomized pretraining

Status: DRAFT for review. Nothing in this document has been built or run.

## 0. Goals and non-goals

Goals
- Stop committing to one intensity window or one MRI normalization at preprocessing time. Store physical (CT) or robustly standardized (MRI) values with no clipping of clinically relevant range.
- Move all intensity choices to the GPU, at view-generation time, as a randomized, information-preserving, monotone transform centered on a random value.
- Make CT evaluation statistically meaningful (RSNA labels, about 3.8k test studies) and keep every existing downstream task runnable.

Non-goals (kept fixed so the experiment stays interpretable)
- Geometry stays as is: RAS, 1 mm isotropic, 256 x 256 x 128 center crop/pad.
- Subject-level splits stay byte-identical to v3.
- Model (ViT-tiny, 1 input channel), loss (LeJEPA), view counts and crop sizes stay as in v3.
- No masked-patch objective (not approved).
- The 2-GPU batch script is not submitted as part of this spec.

---

# Part 1. General draft

## 1.1 Principle

```
raw volume --(offline, once)--> native-unit volume, float16, no clipping of range
                                  CT : Hounsfield units, [-1024, 3071]
                                  MRI: robust z-score within foreground, background = sentinel
native-unit slice --(GPU, every view)--> warp -> physical noise/blur -> random window -> ViT
```

Offline stage answers "what is the physical signal". Online stage answers "which intensity window do we look through this time".

## 1.2 The window transform

For a view with center `c` and width `w` (both in native units):

```
y = asinh((x - c) / w)
```

- Linear within about +/- w of the center (full local contrast), logarithmic outside (no saturation, no clipping).
- Strictly monotone, so no information is lost between views; bone versus metal, air versus fat stay distinguishable even in a brain-centered view.
- Same function for CT (x in HU) and MRI (x in robust z-units). Only the sampling distributions of (c, w) differ by modality.

## 1.3 Key decisions

| Topic | Decision |
|---|---|
| Storage dtype | float16 native units for both modalities (fp16 resolution is under 0.5 HU up to 1024 HU, about 0.03 HU in the brain range) |
| CT range | HU clipped to [-1024, 3071]; pad and sub-floor values set to -1024 |
| MRI normalization | median/MAD inside an Otsu foreground mask, unclipped tissue range [-5, 20], background sentinel -6 |
| Background handling | no mask array; background is the modality floor value; foreground = derived per modality |
| Window sampling | independent (c, w) per view; presets drawn from radiology windows plus a uniform tail |
| Photometric jitter | gain and gamma removed; native-unit noise and blur kept |
| Downstream window | one fixed canonical window per modality (optional multi-window feature concatenation) |
| Comparability | same splits, same slice-selection logic, same model; new output root (old stores untouched) |

## 1.4 Experimental arms

| Arm | Storage | Views |
|---|---|---|
| v4-fixed | native | canonical window only, noise + blur (isolates the storage change and jitter removal) |
| v4-rand | native | independent random window per view (the proposal) |
| v4-slice | native | one random window per slice, shared by its views (gentler ablation) |

---

# Part 2. Per-dataset considerations

Counts are from the existing summary. "Stripped" means skull-stripped.

| Dataset | Mod. | Volumes | Stripped | Main risk for this redo |
|---|---|---|---|---|
| RSNA ICH | CT (DICOM zip) | 25,262 | n/a | HU calibration, multiple series per study, gantry tilt, thick slices |
| CQ500 + Seg-CQ500 | CT (NIfTI) | 183 | n/a | possible +1024 offset from conversion; scanner diversity; tiny test set |
| BraTS 2023 GLI | MRI x4 seq. | 5,004 vols | yes | bright lesions previously clipped at 3 sigma |
| BraTS 2023 MEN | MRI x4 seq. | 4,000 vols | yes | same |
| BraTS 2024 GLI | MRI x4 seq. | 6,484 vols | yes | same; overlap with 2023 handled by dedup key |
| OASIS-1 | MRI T1 | 436 | no (t88_gfc) | Analyze orientation; already bias-corrected |
| OASIS-2 | MRI T1 | 373 | no | raw, uncorrected bias field, noisy background |
| MSD Task01 | MRI | not in `slices_all` | yes | subset of BraTS; keep excluded (leakage with BraTS test) |

## RSNA ICH
- Read through `materialize_rsna_study` (SimpleITK series reader). SimpleITK applies rescale slope and intercept, so values are HU; verify per study that the air peak sits near -1000.
- The reader keeps the series with the most files. RSNA mixes soft-tissue and bone kernels; the largest-series rule is fine for v4 and is logged, not changed.
- Padding values outside the circular field of view (-2000, -3024) must be set to -1024.
- Gantry-tilted studies and 2.5-5 mm slices are linearly resampled to 1 mm; adjacent z-slices are then near duplicates. Known limitation, out of scope.
- Labels: `stage_2_train.csv` inside the zip has per-image labels (any, epidural, intraparenchymal, intraventricular, subarachnoid, subdural). Study-level label = max over the study's images. This gives a large CT benchmark (about 3.8k test studies) next to CQ500's 27. This is a new downstream task (see 3.9).

## CQ500 + Seg-CQ500
- NIfTI conversions sometimes carry a +1024 offset. Detect (min HU at or above about 0 and median above 500) and subtract 1024; log every case.
- Masks (ICH, brain) go through the unchanged nearest-neighbor path.
- Scanner and slice-thickness diversity is useful for window robustness checks.
- Evaluate with 5-fold CV over all 183 volumes in addition to the fixed 27-volume test split.

## BraTS 2023 GLI / MEN, 2024 GLI
- Already skull-stripped and co-registered; background is exactly 0, so the foreground is easy and consistent across the four sequences of a case.
- Each (case, sequence) is normalized independently. Tissue range is left unclipped (enhancing tumour on T1c and edema on T2/FLAIR exceed 3 sigma).
- Label paths and region definitions are unchanged.
- 2024 MEN-RT cohorts, if present, are ingested through the existing prefix map; log which sequences exist per case.

## OASIS-1
- Use `*_t88_gfc.img` (atlas-registered, gain-field corrected, skull on). Do not use the `_masked_` variant.
- Analyze format: check orientation with a montage before the full run (axial versus coronal confusion would put the wrong plane into `z`).
- No N4 needed (already corrected).

## OASIS-2
- Raw MPRAGE, uncorrected bias, noisy background, anisotropic (1.25 mm sagittal): apply N4 bias-field correction before normalization. About 373 volumes, cheap.
- Longitudinal sessions share a subject; the existing subject-level dedup key keeps splits clean.

## MSD
- Stays excluded: it is a BraTS subset, so including it would leak test subjects.

## Adding new datasets later (goal of the heterogeneity plan)
Each dataset is an adapter that provides: a reader, modality tag, optional labels, and these QC hooks: calibration check (CT) or foreground sanity (MRI), orientation montage, and a split key. Candidates to evaluate later (availability not verified): IXI (T1/T2/PD, skull on), ISLES (stroke MRI/CT perfusion), MSSEG (MS, FLAIR), TCIA glioma collections, fastMRI brain, CT-ICH (PhysioNet).

---

# Part 3. Detailed breakdown

## 3.1 Geometry (unchanged)
- `sitk.DICOMOrient(img, "RAS")`, 1.0 mm isotropic linear resampling, center crop/pad to (Z=128, Y=256, X=256).
- CT pad value becomes -1024 (was the old window minimum); MRI pad value becomes the raw background (0) before normalization, then the floor sentinel after.
- Masks: nearest-neighbor onto the image reference, as now.

## 3.2 Storage format
- Output root: `$BRAIN_SCRATCH/data/processed/slices_zarr_v4/<dataset>/<case_id>.zip`; same ZipStore/Blosc layout; array `image`, shape (128, 1, 256, 256), dtype float16; masks unchanged.
- Zip array attributes: `format_version=4`, `kind` (`ct_hu` or `mri_z`), `floor` (-1024 or -6), plus per-volume QC stats (below).
- Footprint equals the current store (same shape and dtype).
- Index parquet: same columns plus `kind`, `floor`, `qc_flag`; `tissue_frac` is recomputed from the derived foreground. Built by the same index builder as `slices_all`, pointed at the v4 root.
- Old stores stay untouched so v2/v3 results remain reproducible.

## 3.3 CT native units
1. Read, reorient, resample, crop/pad with pad -1024.
2. Offset repair (see CQ500 note) and sub-floor clamp: `x = clip(x, -1024, 3071)`.
3. Store as float16. Nothing else is applied.
4. QC per volume: fraction of voxels in [-1100, -900] (air peak), median HU of voxels in [0, 100] within the foreground (expected about 30-45), min/max, and flags.
5. Derived foreground: `HU > -500`, largest 3D connected component, hole filled. Used for empty-slice filtering, noise masking and foreground-centered local crops. It never alters stored pixels (the table and headrest stay in the data, as in real scans).

## 3.4 MRI normalization
1. Read, reorient, resample, crop/pad (pad 0).
2. Optional N4 (`--n4 none|oasis2|all`, default `oasis2`).
3. Foreground: Gaussian smoothing (sigma 1 voxel), Otsu threshold, largest 3D connected component, fill holes. Flag the volume if the foreground is under 2 percent or over 80 percent of the grid.
4. Statistics inside the foreground: `m = median(x)`, `s = 1.4826 * MAD(x)`.
5. `z = (x - m) / s` inside the foreground, clipped to [-5, 20]; outside the foreground set to the sentinel `-6`.
6. Store as float16.
7. Derived foreground at load time: `z > -5.5`.
- The clip range [-5, 20] is a starting point; the pilot (3.10) sets it from the 99.99th percentiles per dataset.
- Robust statistics are used because mean and standard deviation move with scalp, CSF and lesions; skull-on and stripped volumes then land on comparable scales.

## 3.5 Background convention
- CT background is physical air (-1024); MRI background is the sentinel (-6). The floor value `f` is per modality and known at load.
- Warping trick: warp `(x - f)` with zero padding, then add `f` back, so padding equals the floor for both modalities (plain zero padding would be 0 HU, i.e. water, for CT).

## 3.6 Splits and slice selection
- Dedup key, cohort priority and the 75/10/15 subject split are unchanged; acceptance test: the volume-to-split map is identical to v3 for every shared volume.
- Empty-slice rule keeps `EMPTY_MIN_FRAC=0.05`, evaluated on the derived foreground. Acceptance: kept-slice counts per dataset within 3 percent of v3.
- Gaussian slice preference logic is unchanged.

## 3.7 Training-time view pipeline

Order of operations per view (all on GPU, fp32, cast to bf16 only at the patch embedding):

```
x = slice.float()                               # native units, (B,1,H,W); modality m in {CT, MRI}
x = warp(x - f_m, theta) + f_m                  # existing affine crops, floor padding
x = noise_blur(x, m)                            # native-unit noise on foreground, blur p=0.5
(c, w) = sample_window(m)                       # independent per view
y = asinh((x - c) / w)                          # output roughly in [-5, 5]; scale s=1.0
```

Sampling (per view, per sample). Presets use radiology convention; `w` is the linear half-width of the knee.

CT (HU):

| Preset | Center | w | Probability |
|---|---|---|---|
| brain | 40 | 40 | 0.35 |
| subdural | 75 | 100 | 0.15 |
| stroke | 32 | 20 | 0.15 |
| soft tissue | 40 | 200 | 0.15 |
| bone | 500 | 400 | 0.10 |
| uniform tail | c ~ U(-200, 600), w ~ logU(15, 400) | | 0.10 |

Around each preset: `c += N(0, 10)` (HU), `w *= exp(N(0, 0.25))`.

MRI (robust z-units):

| Preset | Center | w | Probability |
|---|---|---|---|
| tissue | N(0.3, 0.5) | logN(ln 1.5, 0.3) | 0.70 |
| wide | U(-2, 6) | logU(0.5, 4) | 0.30 |

- Noise: CT sigma ~ U(0, 10) HU on foreground; MRI sigma ~ U(0, 0.15) z on foreground. Blur: as now (p=0.5, 5x5 average mix).
- Canonical downstream windows: CT brain (c=40, w=40); MRI (c=0.5, w=1.5).
- Mode flag `views.window.mode`: `fixed` (canonical only), `random` (per view), `slice` (per slice, shared across that slice's views).
- Foreground-centered local crops use the derived foreground (`x > f + margin` for MRI, `HU > -500` for CT) instead of `x > 0`.
- Diagnostics logged with the training metrics: `win_inv` (mean cosine similarity between embeddings of the same crop under two random windows), RankMe (existing), per-modality loss. A montage of 16 random views (images of the same slice under different windows) is written to the run directory at step 0 for visual sign-off.

## 3.8 Loader changes
- `SlabDataset.__getitem__` returns `(slices float16 native, chunk_id, modality_id)`; modality from the index table.
- Conversion to fp32 happens on GPU. Family-balanced sampling, the priority sampler and `take()` are unchanged.

## 3.9 Downstream changes
- `FrozenEncoder.embed` applies the canonical window for the slice's modality before the encoder. Embedding cache version bumps to `v3`; old caches remain valid for old checkpoints.
- `seg_data.PixelCache` stores native fp16 slices; the `Segmenter` applies the canonical window on the fly (both scale 1 and scale 2 paths).
- New flag `--window canonical|multi` in `run_suite.py`. `multi` concatenates features from several windows (CT: brain + subdural + bone).
- New task `rsna_ich`: study-level classification (any, plus five subtypes macro) on the existing 18.9k / 2.5k / 3.8k split, 32 uniform slices per study, same heads.
- CQ500: add 5-fold CV reporting over all 183 labeled volumes as the headline CT metric next to the fixed test split.
- Window-robustness evaluation: CQ500 and RSNA at canonical plus two shifted windows (e.g. (c=60, w=25) and (c=40, w=150)); report the spread.
- Step 0 baseline (independent of the rebuild): evaluate v3 (and epoch-1) on `rsna_ich` with the current store so v4 is judged against a real CT baseline.

## 3.10 QC and tests

Pilot (before the full run): 20 volumes per dataset through the new pipeline, then
- montage per dataset (orientation, fov, bias, foreground outline);
- intensity histograms (HU: air, fat, CSF, brain, bone landmarks; MRI z: per dataset and sequence);
- calibration table for all CT volumes (offset repair counts, flag list);
- percentile report to set the MRI clip range.

Unit tests (CPU):
- window transform: monotone, finite, gradient exists, inverse recovers x, floor maps to a constant;
- warp with floor padding reproduces the floor outside the image;
- MRI normalization: invariant to positive scaling and to additive offsets of the raw image;
- skull-on versus stripped toy volumes give foregrounds of the expected class;
- split map equals v3.

## 3.11 Runs and acceptance

| Stage | Content | Resources |
|---|---|---|
| S0 | RSNA label mapping plus v3 baseline on `rsna_ich` (current store) | 1 GPU, short |
| S1 | QC pilot (3.10) | CPU, short |
| S2 | full preprocess to `slices_zarr_v4` and index rebuild | CPU array; OASIS-2 N4 included |
| S3 | implementation and unit tests; 3,000-step smoke pretrain with montage | 1 GPU |
| S4 | three arms (v4-fixed, v4-rand, v4-slice), 45k steps each, same config as v3 | 2 jobs at 1 GPU (concurrent cap), or the 2-GPU script when approved |
| S5 | downstream suite at canonical window, multi-window, shifted windows | 1 GPU each |

Acceptance for v4-rand (to be confirmed before S4):
1. MRI: OASIS age, CDR and BraTS four-sequence segmentation within the v3 confidence intervals or better.
2. CT: `rsna_ich` any-AUC and CQ500 CV AUC at least match v3 on the same protocol; target is an improvement outside the bootstrap interval.
3. Robustness: spread of CT metrics across canonical and shifted windows smaller than for v3 (v3 evaluated through a linear window adapter for CT).
4. `win_inv` stable and high (threshold set from the smoke run).

## 3.12 File-by-file changes

| File | Change |
|---|---|
| `data/preprocess.py` | native-unit CT/MRI paths, Otsu foreground, offset repair, optional N4, QC stats, `format_version`, new output root, `--n4` flag; remove window constants from the default path |
| index builder used for `slices_all` | point at v4 root, add `kind`, `floor`, `qc_flag`, recompute `tissue_frac` |
| `lib/windowing.py` (new) | `WindowSampler`, `apply_window`, presets and config parsing |
| `lib/views.py` | floor-padded warp, `noise_blur` replaces `photometric`, per-view window, derived foreground for local centers |
| `lib/slabs.py` | return modality id; dtype handling |
| `train_vit.py` | pass modality, log `win_inv`, write view montage |
| `configs/vit_tiny_v4_{fixed,rand,slice}.yaml` (new) | arms; `views.window` block |
| `downstream/embeddings.py`, `seg_data.py`, `seg_train.py` | canonical window application, cache version, native fp16 pixel cache |
| `downstream/tasks.py`, `run_suite.py` | `rsna_ich` task, `--window` flag, CQ500 CV reporting |
| `tests/` (new) | items in 3.10 |

## 3.13 Risks

| Risk | Mitigation |
|---|---|
| Model must infer calibration from landmarks; may become anatomy-only and lose subtle intensity cues (hemorrhage) | half the CT views stay near the brain window; `v4-fixed` and `v4-slice` arms; CQ500 CV and RSNA as direct checks |
| Harder augmentation underfits a small model | smoke run first; optional curriculum narrowing then widening `c` and `w` |
| MRI robust statistics differ from v3 z-score, so MRI results shift for unrelated reasons | `v4-fixed` separates storage effects from window randomization |
| Otsu foreground fails on odd volumes | QC flags, pilot review, volumes excluded by flag list rather than silently processed |
| CT calibration anomalies beyond +1024 offset | calibration table in pilot; unresolved volumes excluded |
| Pipeline change confounded with geometry change | geometry frozen in v4 |
| Cost of full rebuild | resumable per-volume jobs (existing skip logic); validated on the pilot before the array run |

## 3.14 Defaults assumed (change any of these)
1. float16 native storage for both modalities (alternative: int16 HU for CT).
2. MRI reference is median/MAD (alternative: mean/std as in v3).
3. N4 only on OASIS-2 (alternatives: none, or all MRI).
4. MSD excluded.
5. Geometry frozen.
6. RSNA labels become a first-class downstream task.
7. Three arms as in 1.4.
8. Window presets and probabilities as in 3.7 (tunable after the smoke run).

---

# Part 4. Implementation log and amendments (updated as we build)

Code lives in `data/prep/` (`sources`, `geometry`, `centering`, `intensity`, `store`, `pipeline`, `qc`, `pilot`), `lib/windowing.py`, and `tests/test_prep.py` (10 phantom tests, all passing). `data/preprocess.py` is untouched.

Changes versus the spec above
1. **In-plane head centering** (approved). One (x, y) shift per volume, estimated on a 4 mm copy. CT: intracranial blob (HU 0..100, 8 mm opening, largest component) on the cranial-vault slices, median over slices, cross-checked against the head distance-transform core; disagreement over 15 mm falls back to the head estimate and raises `qc_flag`. MRI: distance-transform core of the Otsu foreground.
2. **Single resample** onto the target grid (replaces resample-then-crop), with a 0.5 mm Gaussian pre-smooth on axes finer than 1 mm, applied only to the source region covering the target box. The target grid has an axis-aligned direction, so oblique or tilted acquisitions are also rotated onto the scanner axes.
3. **Source-level CT clamp** to [-1024, 3071] before resampling (FOV padding at -2000 or -3024 cannot blend into tissue), and a +1024 offset repair keyed on "no air and a large mass near 1024 + tissue".
4. **MRI foreground** threshold is relative to the measured background level, not assumed zero.
5. **QC flags** (`qc_flag`): CT grid foreground under 8% or over 80%, air fraction under 5%, MRI foreground under 2% or over 80%, or centering disagreement. Flagged volumes are kept in the store and excluded at index-build time, not silently dropped.
6. **Opt-in z anchoring** (`z_anchor="brain"`), default remains the volume center. `brain_cover` reports the fraction of the brain's z-extent inside the slab.

Pilot findings (CQ500 183 volumes, OASIS-1, OASIS-2, BraTS)
- Centering error against the Seg-CQ500 ground-truth brain mask (51 volumes): mean 0.76 mm, max 1.77 mm.
- 182 of 183 CQ500 volumes use the brain estimator; 5 flagged (one corrupt file with data in 33 of 325 slices and a 6.7 degree gantry tilt, one torso or spine scan, three partial or non-brain scans).
- Median in-plane shift versus the old center crop: CQ500 9 mm (p90 20 mm), OASIS-2 15.5 mm, BraTS 10.5 mm, OASIS-1 1.9 mm.
- Axial orientation verified by head extents (AP longer than LR in every dataset); superior/inferior consistent across datasets. Left/right and anterior/posterior flips need visual sign-off.
- z anchoring changes CQ500 brain coverage from p5 0.91 to 0.97 (5 volumes under 0.9 become 2).
- Throughput per volume on one CPU core: CT 7-8 s, OASIS-1 3.7 s, OASIS-2 7 s, BraTS 2.7 s.

## Amendments after the pilot (full run)
1. **Flagged and failed volumes are not stored** (supersedes item 5 above). Any old file for them is deleted and they are listed in `excluded.csv` with `qc_reason`.
2. **z anchor default is `auto`**: `brain` for CT, `volume` for MRI (CQ500 brain coverage mean 0.97 -> 0.99; Seg-CQ500 worst case 0.78 -> 0.91).
3. **N4 bias correction** on OASIS-2 only (OASIS-1 is gain-field corrected, BraTS is preprocessed).
4. **RSNA is read straight from the zip** by stored byte offsets (`member_offsets.json`) with 16 parallel `pread`s per study. Members of one study are scattered over ~150 GB, so latency, not bandwidth, was the cost: 22 s -> 1.7 s per study read, 2.5 s per study in total (18 core-hours for 25,262 studies). About 40% of RSNA studies have gantry tilt (up to 24 degrees); the target grid removes it.
5. **In-place replacement**: output goes to the same folders and file names as the old store (`slices_zarr`), sharded 4 ways, resumable by manifest. Old index files are backed up in `v3_index_backup/` and rebuilt afterwards with the v3 subject splits. Projected size 245 GB (RSNA 200 GB) versus 123 GB before.

6. **Co-registered sequences share one grid** (found after the first full run). Centering each BraTS contrast on its own left the four sequences of a subject offset from each other: median 1.6 mm, 90th percentile 2.7-3.0 mm, up to 35 mm, and the segmentation (which rides with t1n) misaligned with the other three. `Case.ref` now names the reference sequence (t1n for BraTS, DWI for ISLES 2022); the grid is estimated once from it and reused. BraTS was reprocessed in place; the cross-contrast spread is now exactly 0 for all 3,872 subjects (`tests.test_prep.test_coregistered_sequences_share_one_grid`). Remember this for any new multi-sequence set.
7. **Resume key** is `(dataset, case_id)`: BraTS case ids repeat across the 2023 and 2024 cohorts (188 volumes, about 47 subjects present in both).
8. **Full-run cost** was 55 core-hours (RSNA read 5.2 s per study with four jobs sharing the zip), versus 18 from the single-job benchmark.

## External datasets (added to the plan)
| Role | Dataset | Task |
|---|---|---|
| Pretrain + eval (in-distribution) | ABIDE I (T1, 17 sites) | autism vs control |
| Pretrain + eval (in-distribution) | IXI (T1, T2) | brain age |
| Held out (eval only) | BraTS 2023 Mets | metastasis segmentation (in BrainDINO) |
| Held out | ATLAS v2.0 | stroke lesion segmentation (in BrainDINO) |
| Held out | UCSF-PDGM | IDH mutation (in BrainDINO); check overlap with BraTS GLI first |
| Held out | ISLES 2022 | stroke lesion segmentation on DWI/ADC (not in BrainDINO) |
| Held out | CQ500 (moved out of pretraining) | ICH classification, 5-fold CV |

UPENN-GBM is skipped: BraTS GLI draws on UPenn cases, so it would leak subjects into pretraining. Subject-level leakage is never allowed; dataset-level overlap is allowed only for the two in-distribution sets. Downloads: `data/fetch.py` (IXI, ABIDE I, ISLES 2022 into the pool); Mets (Synapse), ATLAS (ICPSR) and UCSF-PDGM (TCIA Aspera) need a manual sign-in or an Aspera client.
