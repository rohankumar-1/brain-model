# Preprocessing (store format 5)

`python -m brain.prep.run --root STORE --datasets cq500,oasis,brats,rsna --template $BRAIN_TEMPLATE --shard 0/8 --workers 16`, usually through `jobs/prep_v6.sbatch` (and `jobs/prep_v6_local.sbatch` for archives unpacked on node-local disk). Then `python -m brain.prep.index --root STORE --manifests "manifest_v6_*.jsonl"` builds `volumes.csv`, `subjects.csv` and `slices.parquet`.

## Per volume (`brain/prep/pipeline.py`)

1. **Read** (`dicom`, `geometry.read`): any NIfTI/Analyze/DICOM series to float32 RAS. The acquisition geometry is recorded as it was (`native_info`): source spacing and size, `thick_mm` / `thick_axis` (the coarsest sampled axis) and `aniso`.
2. **CT repairs, N4**: +1024 offset repair and clamp to [-1024, 3071] for CT; N4 bias correction for raw MRI datasets (`pipeline.needs_n4`).
3. **Center** (`centering`): intracranial blob (CT) or head distance-transform core (MRI) on a 4 mm copy. This is the start pose for registration, or the grid center when registration is off.
4. **Register** (`registration`, default `affine`): Mattes-MI rigid then affine fit of the scan to a skull-on T1 template (MNI152 1 mm; pass `--template` or set `BRAIN_TEMPLATE`, optionally `--template-brain`). Modality-agnostic, so CT and every MRI sequence land in one space. The fit is accepted only if singular values stay in [0.65, 1.5], rotation is at most 35 degrees and the metric improves; otherwise the rigid fit, then the centering translation, is used and `reg_ok` is false. Nothing is dropped. `--registration rigid` keeps head size; `none` reproduces format 4 placement.
5. **Resample once** onto a 256 x 256 x 128 mm, 1 mm grid (anti-aliased where the source is finer), through the fitted transform. Dataset masks use nearest neighbour through the same transform.
6. **Native units**: CT in HU; MRI as a robust z-score with background -6 (`intensity`). With a brain mask, the median/MAD come from parenchyma rather than from the whole head.
7. **Brain mask** (`brainmask`, default `auto`): SynthStrip when `mri_synthstrip` is installed (or `BRAIN_SYNTHSTRIP` names it), else a classical mask (CT: soft-tissue opening and fill; skull-on MRI: head eroded 14 mm; known skull-stripped releases, `Case.stripped`: the foreground). A mask shipped with the dataset wins. The classical masks are approximate, good for statistics and skull dropout, not for anatomy.
8. **QC**: foreground fraction, centering agreement, registration pose, brain mask fraction; `qc_flag` volumes are excluded and listed in `excluded.csv`.

## Store layout

`<folder>/<case>.zip` holds `image`, a float16 Zarr array `(128, 1, 256, 256)`, with all prep measurements as attributes. Masks are sibling zips `<case>.<name>.zip`; the computed brain mask is `<case>.brain.zip`. `store.FORMAT_VERSION` is 5. Runs skip a case only if its zip is current-format; changing registration or mask settings needs `--force` or a new root. `index` copies the geometry, registration and brain-mask columns of each manifest into `volumes.csv` (`NATIVE_COLS`).

## Resume and failures

Manifests are per shard. Cases with status `ok` or `flagged` are not redone; `error` cases are retried. A failed case never leaves a partial zip.

## Check on your own data first

`python -m brain.prep.pilot --dataset cq500 --n 20 --template $BRAIN_TEMPLATE` writes a QC figure per case (source slice with the crop box, histograms, windows, mask and lesion contours, registration and mask summaries).

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
before timestamps prefer the one with a stored transform). After all prep: `jobs/index_v6.sbatch` (CPU, minutes:
`--no-slices` skips the 2D slice table, which only the 2D trainers read), `jobs/store_cleanup_v6.sbatch` (drops
non-format-6 zips, old manifests, index backups and retired folders) and `jobs/verify_v6.sbatch`, which checks every
indexed volume (format 6, stored transform, brain array, label masks) and writes `excluded.csv` (cases whose latest
manifest row is flagged or errored). Raw archives are packed last (`jobs/rawpack_pool.sbatch`), because phase B reads
raw data from pool.

### Known issue: RSNA DICOM assembly

`brain.prep.dicom.read_series_dir` uses SimpleITK's series reader, which assumes evenly spaced, untilted slices.

- About 2% of RSNA studies mix slice thicknesses or repeat slice positions inside one series (for example 5.2 mm
  slices plus a block of 1.3 mm re-reconstructions). They come out compressed in z and striped.
- About 54% of RSNA studies have gantry tilt (median about 16 degrees). They come out sheared. The affine
  registration usually removes the shear; about 12% of tilted studies fall back to rigid and stay sheared.

For now, the index drops RSNA volumes that fail brain-mask QC from SSL (`ssl_qc_drop`; they stay in downstream
tasks). The stored `native_*` geometry of tilted RSNA studies is SimpleITK's untilted reading, so `brain.prep.native`
exports for them are not exact.

**Fix before the next re-prep:** assemble slices by their true position along the slice normal. Drop duplicate
positions, shift tilted slices back onto the normal axis, interpolate onto even spacing, and flag large gaps.
This is what `dcm2niix` does.
