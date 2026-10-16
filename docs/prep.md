# Preprocessing (store format 5)

`python -m brain.prep.run --root STORE --datasets cq500,oasis,brats,rsna --template $BRAIN_TEMPLATE --shard 0/8 --workers 16`, usually through `jobs/prep_job.sbatch`. Then `python -m brain.prep.index --root STORE` builds `volumes.csv`, `subjects.csv` and `slices.parquet`.

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
