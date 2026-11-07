# Preprocessing (store format 6)

`python -m brain.prep.run --root STORE --datasets cq500,oasis,brats,rsna --template $BRAIN_TEMPLATE --shard 0/8 --workers 16`, usually through `jobs/prep_v6.sbatch` (`jobs/prep_v6_local.sbatch` for the FOMO / BraTS archives unpacked on node-local disk, `jobs/prep_v6_staged.sbatch` for datasets packed as `<name>.raw.zip` on pool). Then `python -m brain.prep.index --root STORE --manifests "manifest_v6_*.jsonl"` builds `volumes.csv` and `subjects.csv` (subject splits inherited from the existing index; see "Rebuild order" below).

## Per volume (`brain/prep/pipeline.py`)

1. **Read** (`dicom`, `geometry.read`): any NIfTI/Analyze/DICOM series to float32 RAS. The acquisition geometry is recorded as it was (`native_info`): source spacing and size, `thick_mm` / `thick_axis` (the coarsest sampled axis) and `aniso`.
2. **CT repairs, N4**: +1024 offset repair and clamp to [-1024, 3071] for CT; N4 bias correction for raw MRI datasets (`pipeline.needs_n4`).
3. **Center** (`centering`): intracranial blob (CT) or head distance-transform core (MRI) on a 4 mm copy. This is the start pose for registration, or the grid center when registration is off.
4. **Register** (`registration`, default `affine`): Mattes-MI rigid then affine fit of the scan to a skull-on T1 template (MNI152 1 mm; pass `--template` or set `BRAIN_TEMPLATE`, optionally `--template-brain`). Modality-agnostic, so CT and every MRI sequence land in one space. The fit is accepted only if singular values stay in [0.65, 1.5], rotation is at most 35 degrees and the metric improves; otherwise the rigid fit, then the centering translation, is used and `reg_ok` is false. Nothing is dropped. `--registration rigid` keeps head size; `none` reproduces format 4 placement.
5. **Resample once** onto a 256 x 256 x 128 mm, 1 mm grid (anti-aliased where the source is finer), through the fitted transform. Dataset masks use nearest neighbour through the same transform.
6. **Native units**: CT in HU; MRI as a robust z-score with background -6 (`intensity`). The median/MAD come from the brain mask when it passed QC, or failed only the template-Dice check (fraction within `brainmask.PLAUSIBLE_FRAC`: partial slabs, infants); otherwise from the whole head (`pipeline.mri_stats_on_brain`). The 2026-11-03 store got this rule by a post-prep pass (`stats_from = brain_restat`); prep now applies it directly.
7. **Brain mask** (`brainmask`, default `auto`): SynthStrip when `mri_synthstrip` is installed (or `BRAIN_SYNTHSTRIP` names it), else a classical mask (CT: soft-tissue opening and fill; skull-on MRI: head eroded 14 mm; known skull-stripped releases, `Case.stripped`: the foreground). A mask shipped with the dataset wins. The classical masks are approximate, good for statistics and skull dropout, not for anatomy.
8. **QC**: foreground fraction, centering agreement, registration pose, brain mask fraction; `qc_flag` volumes are excluded and listed in `excluded.csv`.

## Store layout

`<folder>/<case>.zip` holds `image`, a float16 Zarr array `(160, 1, 256, 256)` (1 mm, MNI-registered RAS grid), with all prep measurements as attributes, and `brain`, the computed brain mask. Label masks are sibling zips `<case>.<name>.zip` (SynthSeg anatomy: `<case>.synthseg.zip`). `store.FORMAT_VERSION` is 6. Runs skip a case only if its zip is current-format; changing registration or mask settings needs `--force` or a new root. `index` copies the geometry, registration and brain-mask columns of each manifest into `volumes.csv` (`NATIVE_COLS`).

## Resume and failures

Manifests are per shard. Cases with status `ok` or `flagged` are not redone; `error` cases are retried. A failed case never leaves a partial zip.

## Registration backends

`--reg-backend sitk` (default) is SimpleITK's Mattes-MI rigid -> affine fit on the CPU, about 12 s per case (83% of
prep time). `--reg-backend torch` (`brain/prep/registration_torch.py`) runs the same recipe on the GPU in about 1 s.
It uses the same template (brain-only for stripped scans), 2 mm views, transform family, sanity bounds and fallbacks.
The metric is a soft-histogram MI optimised by regular-step gradient descent in mm-scaled parameters, coarse to fine.
When the rigid fit from the identity is rejected, it restarts from pitched poses. On a GPU out-of-memory it retries,
then runs on the CPU; it never drops to the bare translation. Each volume records `reg_backend`.

Validation on real scans (October 2026). Quality is the template Dice of the SynthStrip mask after registration:

| Set | Cases | Median s (sitk / torch) | Mean Dice (sitk / torch) | torch worse by > 0.02 |
|---|---|---|---|---|
| one per dataset folder | 34 | 11.7 / 0.8 | 0.912 / 0.932 | 1 (a tilted RSNA CT) |
| hard cases (ABIDE, CoRR, NKI, OASIS-1, RSNA, CQ500, AISD, ISLES-24, BrainMetShare) | 29 | 12.1 / 1.2 | +0.023 for torch | 3 (all RSNA CT with large tilt) |

torch matches or beats SimpleITK on MRI, both skull-on and stripped, and on most CT. Heavily tilted RSNA CTs are
better with SimpleITK. In the v6 store, phase C (BraTS incl. GoAT, FOMO-HBN) used torch; everything else used sitk.

## Back to the original scan (`brain/prep/native.py`)

Each format-6 volume stores the fitted registration (`reg_affine`, a 4 x 4 map from stored-grid to source physical
mm), the stored grid geometry (`grid_*`), the source geometry (`native_*`) and the source file (`native_source`).
These are enough to put any stored array back on the patient's scan:

```bash
python -m brain.prep.native --image STORE/<folder>/<case>.zip --array brain --out brain_native.nii.gz
python -m brain.prep.native --image STORE/<folder>/<case>.zip --npy prediction.npy --like original.nii.gz --out pred.nii.gz
```

`--like <original file>` writes onto that file's exact grid and header; use it for anything handed to a physician.
`native.volume_ml(mask, attrs)` gives the true source-space volume: stored voxel count x |det(A)|. The affine is
size-normalizing, so raw stored voxel counts overstate volumes by 13-64% (median 35%).

### Finishing a build: transform reruns, index, cleanup, verification

The transform record was added on 2026-10-17, after part of the v6 build had run. Volumes written before it (phase A
tasks 0-3, early BraTS 2023 GLI/MEN) were redone with `--require-attr reg_affine` (or `reg_backend=torch`), which
reprocesses only volumes that lack it; phase A was rerun as `MANIFEST_TAG=A_rerun` on GPU registration. To rerun
some shards of an n-way split, set `SHARD_N=n` and pass only those array indices:

```bash
PART=brats SHARD_N=4 REG_BACKEND=torch MASK_BATCH=4 EXTRA_ARGS="--require-attr reg_affine" \
  sbatch --export=ALL --array=0-3 jobs/prep_v6_local.sbatch
```

Manifest rows carry `written_at`; when a case appears in several manifests the index keeps the newest row (rows from
before timestamps prefer the one with a stored transform). See "Rebuild order" below for what follows prep.

### Rebuild order

1. **Prep**: `jobs/prep_v6.sbatch` (phases A / B), `jobs/prep_v6_local.sbatch` (BraTS incl. GoAT, HBN, infants) and
   `jobs/prep_v6_staged.sbatch` (raw-packed datasets, e.g. `seg_cq500`); the one-off cohorts (ADNI / PPMI, fastMRI+,
   SynthRAD, UCSF-PDGM, UPENN-GBM, RSNA-ICA, Pedi-Cranial-CT) through their jobs in `jobs/archive/`. Each prep job
   ends with the content dedup (`jobs/dedup_content.sbatch`, `brain.prep.dedup --content`), which lists copies in
   `labels/pool_drop/duplicates.csv`.
2. **Subject tables** the index reads for its `dedup_key`: `python -m brain.prep.subjects rsna-patients`
   (`labels/rsna_patients.csv`) and `python -m brain.prep.subjects glioma-twins` (the UCSF-PDGM / UPENN-GBM BraTS twin
   keys in their pool `meta/` folders); `brain.prep.fastmri` writes `labels/fastmri_patients.csv` while staging.
3. **Index**: `sbatch --export=ALL,FULL=1 jobs/index_append.sbatch` (CPU, minutes): backs up `volumes.csv` /
   `subjects.csv`, rebuilds in a node-local work dir inheriting every existing split, and installs by atomic rename.
   Without `FULL=1` it appends only volumes the index lacks.
4. **Check**: `python -m brain.prep.verify_store --root STORE` (`jobs/archive/verify_v6.sbatch`) checks every indexed
   volume (format 6, stored transform, brain array, label masks) and writes `excluded.csv`.
5. **Post-prep labels** (each reads the index and the stored volumes):
   - `brain.prep.synthseg_run` (`jobs/synthseg.sbatch`) and `brain.prep.wmh_run` (`jobs/wmh.sbatch`): anatomy and WMH maps;
   - `brain.prep.synthseg_qc`: SynthSeg QC-net scores (`labels/synthseg/qc_*.csv`);
   - `brain.prep.rsna_slices`: RSNA per-slice labels and positions (`labels/rsna_slices/rsna_slices.parquet`);
   - `brain.prep.rsna_ica_vessels`: RSNA-ICA vessel and aneurysm masks;
   - `brain.prep.rsna_bleed_masks` (`ts`, `model`, `validate`, `fuse`): fused RSNA bleed pseudo-labels (needs `rsna_slices`);
   - `brain.prep.fastmri_labels`: fastMRI+ box annotations as masks;
   - `brain.prep.live_targets`: per-volume live targets (`labels/live/`);
   then `jobs/multitask_labels.sbatch` (`brain.multitask.labels`).
6. **Data QC and decontamination**: `brain.prep.data_qc scans` / `masks` (shards in `labels/data_qc/`), `report`
   and `verify` (CSVs in `runs/diagnostics/data_qc/`), then `python -m brain.prep.decontam bench-exclude`
   (`configs/bench/exclude_v3.csv` plus the git-ignored `exclude_v3.private.csv` for ADNI / PPMI rows) and
   `python -m brain.prep.decontam train-exclude` (`configs/multitask/train_exclude_r15.csv`).

Raw archives are packed last (`jobs/archive/rawpack_pool.sbatch`), because prep reads raw data from pool.

### Known issue: RSNA DICOM assembly

`brain.prep.dicom.read_series_dir` uses SimpleITK's series reader, which assumes evenly spaced, untilted slices.

- About 2% of RSNA studies mix slice thicknesses or repeat slice positions inside one series (for example 5.2 mm
  slices plus a block of 1.3 mm re-reconstructions). They come out compressed in z and striped.
- About 54% of RSNA studies have gantry tilt (median about 16 degrees). They come out sheared. The affine
  registration usually removes the shear; about 12% of tilted studies fall back to rigid and stay sheared.

For now, the index drops RSNA volumes that fail brain-mask QC from SSL (`ssl_qc_drop`; they stay in downstream
tasks). The stored `native_*` geometry of tilted RSNA studies is SimpleITK's untilted reading, so `brain.prep.native`
exports for them are not exact.

**dcm2niix pilot (2026-10-21): not the fix.** `BRAIN_DICOM_READER=dcm2niix` switches `brain.prep.dicom` to
position-based assembly with dcm2niix. On 60 RSNA studies (40 failing SynthSeg QC, 20 passing) it rescued none and
broke 5, and dcm2niix found no gantry tilt in any study (the released DICOMs appear to lack it). The failing studies
lack the posterior fossa in the data itself (cerebellum ~30 ml under both readers): RSNA seems to ship only the main
axial series. Decision: keep the SimpleITK reader and RSNA as is; QC (below) keeps the unusable scans out of training.

### Known issue: OASIS-1 Analyze orientation (fixed 2026-10-21)

OASIS-1 ships Analyze 7.5 files with `orient = 2` (sagittal). ITK ignores that byte and read the 128-slice left-right
axis as superior-inferior, which no affine registration recovers (template brain Dice ~0.83; every scan failed
SynthSeg). `brain.prep.geometry.read` now reorders sagittal Analyze files into RAS (`_read_sagittal_analyze`), and
`sources.oasis` takes the raw MPRAGE (`*_mpr-1_anon`) like OASIS-2 instead of the distributed Talairach image. After
the fix: template Dice 0.91-0.94, SynthStrip masks, SynthSeg QC 6/6. Left-right of these scans cannot be determined
from the files and is taken as stored.

### Dropped from the training pools

`POOL_DROP_DATASETS` (`brain/prep/index.py`) keeps whole datasets out of SSL and multitask training (they stay in the
store): `fomo_hbn_pd` (HBN proton density; many scans are noise / artefact, SynthSeg QC 61% vs 99% for HBN T1w).

Single volumes are dropped by CSVs in `labels/pool_drop/` (columns `volume_id, reason`; every file is read):

- `prep_qc_fomo_small.csv`: 152 WAND T2 partial slabs (thinnest extent < 100 mm; affine registration "succeeds" on
  them but cannot be right) and 184 infant scans whose registration fell back to rigid / translation (tiny unscaled
  brains in the grid).
- `duplicates.csv` (`python -m brain.prep.dedup --new-families ...`): the same scan redistributed under another
  dataset. Shortlist from the index (identical source grid, nearest image fingerprint: brain fraction, brain / head
  ratio), confirmed by the correlation of the registered volumes (>= 0.98); new duplicates are appended. For the
  2026-10 small FOMO cohorts: 140 candidates, median correlation 0.45, no duplicates. Until 2026-10-25 the shortlist
  also used the registration result and the contrast string, so copies registered by the other backend, or labelled
  with another contrast (MSD "" vs BraTS "t1n"), were never compared, and each run overwrote the file.
- `prep_qc_reader.csv`: scans stored mirrored by the old nibabel fallback reader (see below) that are in the pool.

What the drops mean (2026-10-25): `pool_drop` and `ssl_qc_drop` keep a volume out of SSL. Training labels come from
`brain.multitask.labels.label_pool`: duplicates and `mirrored_reader` scans stay out of every task, but the other
SSL-only exclusions (RSNA DICOM-assembly QC failures, partial slabs, unscaled fallback registrations) keep their
study-level labels and image-space masks and lose only registration-sensitive and live (image-measured) targets.
Before, they were out of every task too: the hemorrhage tasks lost 2,200 RSNA studies and infants lost their age
labels. Registration QC (`registration_qc`) also fails rigid / translation fallbacks for the registration-sensitive
targets now (their template-space volumes are native millilitres).

### Fixed 2026-10-25: nibabel fallback reader orientation

`geometry.read` falls back to nibabel when SimpleITK cannot read a NIfTI (non-orthonormal direction cosines). The
fallback built the image from RAS-ordered data but kept ITK's identity (LPS) direction and the RAS origin, so these
images were stored rotated 180 degrees about the superior axis; registration (rotations up to 35 degrees) could not
undo it and fell back to translation. A scan of all 1,673 translation fallbacks (brain-mask Dice with a reference,
as stored vs flipped; `runs/diagnostics/mirrored_scan.py`) found 45 mirrored: 44 BrainMetShare (held out, used by no
task; re-prep before use) and one HBN T2w in the pool (`prep_qc_reader.csv`). The reader now takes direction, spacing
and origin from the full affine. The 1,058 Yale-metastasis translation fallbacks are not mirrored.

### Hemorrhage segmentation cohorts (BHSD, HemSeg-500)

Both are RSNA ICH studies with voxel masks (`jobs/archive/prep_bhsd.sbatch`, `jobs/archive/prep_hemseg.sbatch`). BHSD (190 studies):
codes 1 EDH, 2 IPH, 3 IVH, 4 SAH, 5 SDH, matching RSNA's labels on 92-97% of studies; kept out of SSL (`bhsd`
column) but its masks train BrainQuery. HemSeg-500 (518 studies): images rebuilt from exactly the DICOM
members the authors listed, so the masks line up; each study is IPH-only or IVH-only, so each binary mask is the
whole hemorrhage. Studies in both keep BHSD's masks. Mask zips must be named `<case>.ICH.zip`: the index only finds
that name (the first build wrote `.ich.zip` and left hemorrhage segmentation without training rows).

### Small FOMO cohorts (added 2026-10-23)

16 FOMO-260K folders for variety (`SMALL_FOLDERS` in `brain/prep/fomo.py`): Infant Development (0-3 y), SLIM, WAND,
Wayne, Yale HighRes, Adolescent Brain Development, CHBMP, HBN-SSI, Beijing Enhanced, ACPI, age-ility, AHEAD (7T),
MICA-MICs, CUNMET, Tao Wu, Cleveland CCF. Downloaded with `python -m brain.datasets.download_fomo --folders small`
(structural sequences only; the session zips, mostly diffusion, are deleted after extraction), prepped with
`PHASE=A DATASETS_OVERRIDE=fomo` (only newly extracted scans are found). 5,722 volumes prepped, 5,386 in the pool.
Registration falls back to rigid / translation more often than in the older cohorts (Beijing 22%, HBN-SSI 14%, CHBMP
9%; AHEAD 100% rigid, sub-mm 7T); the adult fallbacks are centred and upright (`runs/diagnostics/fomo_small_qc.png`)
and only lose alignment-dependent labels. FOMO's OpenNeuro folder (64k structural scans) stays out: it holds the AOMIC
and PIXAR evaluation cohorts and would need per-study filtering.

### RSNA Intracranial Aneurysm (downloaded, not yet in the store)

`pool/datasets/rsna_ica/rsna-intracranial-aneurysm-detection.zip` (215 GB; MIRA dataset 7, non-commercial, no
redistribution): 4,348 labelled series, CTA 1,808 / MRA 1,252 / T2 983 / T1 post-contrast 305; per-series aneurysm
presence and 13 locations, 2,254 aneurysm points (`train_localizers.csv`, pixel x / y on one SOP), 178 vessel
segmentations. `sources.rsna_ica` reads series straight from the zip by byte offsets (index cached in
`work/rsna_ica`); its cases read through dcm2niix (`Case.dicom_reader`; some MR series are one enhanced multi-frame file), with
the SimpleITK reader as fallback when dcm2niix refuses a series (derived / secondary CTAs, malformed overlay tags).
Pilot (`jobs/archive/rsna_ica_pilot.sbatch`, 30 series per modality): MRA 30/30 (median template brain Dice 0.93 despite the
slab), T2 30/30, T1 post 27/30 + 3 reader fallbacks, CTA 12 ok + 4 reader fallbacks + 14 flagged.

Known issue: the flagged CTAs carry wrong rescale tags. Their stored pixels are already in HU (air ~ -1000), but the
header says intercept -1024 (GE) or slope 10 / intercept -10240 (one Siemens), so both readers shift tissue to
~ -1000 (air ~ -2000) and centering, skull stripping and registration fail. Fix when the dataset goes in: after
rescaling, if air sits near -2000 rather than -1000, ignore the rescale tags. Until then they stay out.

### Source quirks of evaluation cohorts

- AISD (CT) is distributed pre-windowed: stored values are clipped at 80 HU, so bone is flat (other CT cohorts reach
  1500-2900 HU).
- INSTANCE (CT) volumes are cropped short of the whole head (SynthSeg full field of view on 12% of a sample).
- CQ500 (`cq500`) is read from the original DICOMs (`pool/datasets/cq500_dicom`, `sources.cq500_dicom`) through
  dcm2niix, which de-shears its gantry tilt; Seg-CQ500 (`seg_cq500`) is the third-party NIfTI subset with ICH and brain
  masks (`sources.seg_cq500`). The DICOM reader is chosen per source (`Case.dicom_reader`: dcm2niix for CQ500,
  RSNA-ICA and ADNI / PPMI; SimpleITK for RSNA ICH), with `BRAIN_DICOM_READER` as the fallback.

## Anatomy labels: SynthSeg-robust 2.0 (`brain/prep/synthseg.py`)

A PyTorch port of FreeSurfer's `mri_synthseg --robust` (three U-Nets in cascade; weights `synthseg_robust_2.0.h5`
from FreeSurfer's model annex, in `tools/synthseg/` on scratch). Validated against the Keras model on six scans of
different cohorts and modalities: >= 99.998% voxel agreement, every structure Dice >= 0.995, identical volumes.
TF32 must stay off (the cascade's argmax between networks propagates its rounding). ~1.5 s per volume on an L40S.

    python -m brain.prep.synthseg_run --select --n 30000      # family-balanced train/val selection, once
    for s in 0 1 2 3; do sbatch -J synthseg-$s --export=ALL,SHARD=$s jobs/synthseg.sbatch; done

Output per volume: `<folder>/<case_id>.synthseg.zip` (array `mask`, uint8 index into the 33 FreeSurfer labels) and a
row of structure volumes (template-space ml, i.e. head-size normalised) in `labels/synthseg/volumes_*.csv`.
`brain.multitask.labels.synthseg_volumes` turns them into group masks, volumes, ratios and asymmetries with two flags:
`qc_ok` (cortex asymmetry <= 0.15, thalamus >= 8 ml, hippocampus >= 4 ml with a full field of view, brain 700-2600 ml:
fails gross hemisphere mislabelling, not pathology) and `fov_ok` (cerebellum >= 110 ml and brainstem >= 18 ml: volume
targets need the whole brain in view).
