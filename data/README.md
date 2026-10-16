# Imaging Data

Shared-space preprocessing for **all** CT/MRI sources. Each volume is one compressed
Zarr ZipStore, in a folder named for its dataset. Splits and SSL sampling are index tables.

## Pipeline (`data/preprocess.py`)

Every dataset (CQ500, RSNA, BraTS, OASIS, MSD) goes through the same path:

1. Load volume (NIfTI or RSNA DICOM-from-zip → temp NIfTI)
2. RAS → isotropic **1 mm** → center crop/pad **256×256×128**
3. Intensity → `[0, 1]` (CT HU **[-200, 200]**; MRI Z-score ±3σ)
4. Write the **full** volume to `<dataset>/<case_id>.zip` (float16 Zarr ZipStore, Blosc/zstd, chunk z=8)
5. Drop near-empty slices + mid-skull Bernoulli sample → **index rows only** (`image_path`, `z`)
6. Resample segs onto that same grid (nearest-neighbor) as sibling zips: BraTS `seg` on t1n, Seg-CQ500 `ICH` and `brain`

Subject-level `dedup_key` keeps BraTS contrasts / year overlaps in the same split
and drops lower-priority duplicates. MSD is skipped when BraTS GLI is present.

### Outputs (`$BRAIN_SCRATCH/data/processed/slices_zarr/`)

| File | Role |
|---|---|
| `<dataset>/<case_id>.zip` | one image: `(128, 1, 256, 256)` float16, chunks `(8, 1, 256, 256)` |
| `<dataset>/<case_id>.{seg,ICH,brain}.zip` | matching mask, uint8, same grid |
| **`subjects.csv`** | one row per `dedup_key` (canonical `split`) |
| **`volumes.csv`** | one row per volume (`image_path`, source, contrast, mask paths) |
| `slices.parquet` / `slices.csv` | SSL index: `image_path`, `z`, `tissue_frac` (not pixels) |
| `manifest.csv` | alias of the slice index (older scripts) |
| `labels/` | task CSVs only (masks live next to each volume) |

There are **no** `train`/`val`/`test` image arrays and **no** physical shuffle copy.
`DataLoader(shuffle=True)` permutes the index.

```bash
python data/preprocess.py --overwrite --workers 12 --skip-download \
  --train-frac 0.75 --val-frac 0.10 --test-frac 0.15
```

`data/resplit_manifest.py` rewrites split columns on the tables only — never copies pixels.

## Datasets

- **CQ500 / Seg-CQ500** — head CT + ICH masks on pool (`datasets/cq500`, `datasets/seg_cq500`)
- **RSNA ICH** — zip on pool (`datasets/rsna`); same preprocess workers (not a separate npy path)
- **BraTS** — 4 contrasts; 2024 preferred over 2023 on duplicate subjects
- **OASIS-1/2** — T1 sessions
- **MSD BrainTumour** — skipped when BraTS GLI is present
