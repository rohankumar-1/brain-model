# brain-model

Multimodal CT/MRI representation learning: shared-space prep → self-supervised encoders (2D ViT LeJEPA, 3D BrainModel; ConvVAE is legacy) → frozen-embedding downstream suite.

## Storage layout (ORCD)

| | Path |
|---|---|
| Repo | `/home/roku/brain-model` |
| Scratch (processed Zarr only) | `/home/roku/orcd/scratch/brain-model` |
| Pool (all raw archives) | `/home/roku/orcd/pool/datasets` |

Override with `BRAIN_SCRATCH` / `BRAIN_POOL` / `BRAIN_DATASETS`. `data/raw` points at pool; `data/processed` at scratch.

## Datasets

Counts for **volumes / slices** are after shared-space prep (RAS, 1 mm, 256×256×128, empty-slice drop) when available. BraTS/OASIS archives live on pool; they are in the Zarr store once preprocess finishes. BraTS “cases” each include **4 mpMRI** sequences (t1n, t1c, t2w, t2f/FLAIR) unless noted.

| Dataset | Link | Modality | Volumes | Slices | Downstream tasks |
|---|---|---|---:|---:|---|
| **Seg-CQ500** | [GitHub](https://github.com/Mobarakol/SegQure) · [CQ500 paper](https://arxiv.org/abs/1710.04934) | CT | 51 | 6,528 | ICH / hemorrhage subtypes, mass effect, midline shift (multi-label); optional seg masks |
| **CQ500** | [Qure.ai CQ500](http://headctstudy.qure.ai/dataset) · [arXiv](https://arxiv.org/abs/1710.04934) | CT | 132 | 16,284 | Same CQ500 multi-label radiology findings |
| **RSNA ICH** | [RSNA 2019 ICH Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) | CT | 4,707 | 599,107 | Intracranial hemorrhage presence / subtype classification |
| **MSD BrainTumour** (Task01) | [Medical Decathlon](http://medicaldecathlon.com/) · [Nature Comm.](https://doi.org/10.1038/s41467-022-30695-9) | MRI (T1w used; also FLAIR/T1gd/T2) | 484 train + 266 test | 56,227 + 31,135 | Enhancing-tumor burden (`enhancing_high`); glioma-related labels from seg |
| **BraTS 2023 GLI** (=2021 adult glioma) | [Synapse BraTS 2023](https://www.synapse.org/Synapse:syn51156910) · [BraTS](https://www.braints.org/) | MRI mpMRI (t1n/t1c/t2w/t2f) | 1,251 cases (×4 seq) | TBD | Tumor subregion segmentation (ET/TC/WT); glioma SSL |
| **BraTS 2023 MEN** | [Synapse](https://www.synapse.org/Synapse:syn51156910) | MRI mpMRI | 1,000 cases (×4 seq) | TBD | Meningioma segmentation |
| **BraTS 2024 GLI** (post-treatment) | [Synapse BraTS 2024](https://www.synapse.org/Synapse:syn53708249) | MRI mpMRI | 1,350 + 271 add. cases | TBD | Post-treatment glioma segmentation |
| **BraTS 2024 MEN-RT** | [Synapse](https://www.synapse.org/Synapse:syn53708249) | MRI (RT planning) | ~500 cases | TBD | Meningioma RT target segmentation |
| **OASIS-1** | [OASIS-1](https://sites.wustl.edu/oasisbrains/home/oasis-1/) · [JoCN 2007](https://doi.org/10.1162/jocn.2007.19.9.1498) | MRI T1 (MPRAGE) | 416 subjects / 434 sessions | TBD | Brain age; CDR / dementia staging; AD vs nondemented |
| **OASIS-2** | [OASIS-2](https://sites.wustl.edu/oasisbrains/home/oasis-2/) · [JoCN 2010](https://doi.org/10.1162/jocn.2009.21407) | MRI T1 (longitudinal) | 150 subjects / 373 sessions | TBD | Brain age; CDR; longitudinal dementia conversion |

**Notes**

- Seg-CQ500 and CQ500 are **disjoint** case IDs; same outcome taxonomy.
- MSD Task01 overlaps historically with older BraTS glioma releases — prefer **BraTS 2023 GLI** as the glioma source when both are in the mix; drop or de-dup MSD if needed.
- All raw archives live under **pool** `/home/roku/orcd/pool/datasets` (BraTS/OASIS zips, RSNA zip, CQ500/Seg-CQ500 NIfTI+masks, MSD tar). Processed volumes are one compressed ZipStore per case, in a folder per dataset, on scratch.

## 1. Preprocessing (one volume store)

All datasets (CQ500, RSNA, BraTS, OASIS, …) share one pipeline. Each volume is one compressed ZipStore under a dataset folder. Splits and SSL sampling are index tables:

```bash
python data/preprocess.py --overwrite --workers 6 --skip-download \
  --train-frac 0.75 --val-frac 0.10 --test-frac 0.15
# → …/slices_zarr/<dataset>/<case>.zip  plus subjects.csv, volumes.csv, slices.parquet
```

- **`<dataset>/<case_id>.zip`** — float16 `(128, 1, 256, 256)`, Blosc chunks of 8 slices
- **`<dataset>/<case_id>.{seg,ICH,brain}.zip`** — labels on that same grid, when the dataset has them
- **`subjects.csv`** — one row per `dedup_key` with canonical `split`
- **`volumes.csv`** — one row per volume (`image_path`, source, contrast, mask paths)
- **`slices.parquet`** — SSL index (`image_path`, `z`); not a second pixel copy
- Per volume: RAS → 1 mm iso → 256×256×128 → CT HU [-200,200] or MRI Z-score → full volume stored; empty drop + mid-skull sample writes **indices only**

Prefer Slurm: `sbatch jobs/preprocess_zarr.sbatch` (extracts BraTS/OASIS if needed, then writes the per-volume zips).

## 2. Models

**ConvVAE** — 2D `(B,1,256,256)`, latent μ/logvar. Heterogeneous batches: 64 independently sampled `(image_path, z)` rows (`num_workers=8`). Train: `sbatch jobs/train_vae_256.sbatch`.

Checkpoints (atomic `*.tmp` → replace) under the run dir:

- `last.pt` — every epoch and on SIGTERM/SIGINT (model, optimizer, scheduler, RNG, epoch, history)
- `best.pt` — when val reconstruction improves (`vae.pt` alias)
- Auto-resume from `last.pt` unless `--no-resume`. Slurm job uses `--requeue` and `--signal=B:TERM@120`.

**ViT LeJEPA** (`train_vit.py`, `models/vit.py`) — crop-matching plus SIGReg on 2 global (256) and 6 local (96) views. Views are built on the GPU (`lib/views.py`, area-fraction crops, locals centred on foreground) from 8-slice chunk reads (`lib/slabs.py`, family-balanced with `balance_alpha`). Step-based (warmup + cosine), bf16, optional registers / QK-norm, `torchrun` for multi-GPU. `sbatch jobs/train_vit.sbatch` checkpoints on SIGTERM and requeues itself; configs in `configs/vit_{tiny,small}.yaml`.

**3D BrainModel** (`brainmodel/`, `train_brainmodel.py`) — LeJEPA on native-grid 3D crops with a dense, position-aware token objective; see `docs/brainmodel.md`. Train: `sbatch jobs/train_brainmodel_pi.sbatch`. Judge it with the frozen-token probes: `python -m downstream.probe3d` / `sbatch jobs/probe3d.sbatch`.

**Downstream** (`downstream/`) — frozen encoder; ViT features are `[cls, mean patch]`. Classification / regression: pooled logreg / ridge, seed-ensembled slice-attention MLP (early-stopped on val), XGBoost, with bootstrap 95% CIs. Segmentation: `SegDecoder` (UNETR-style, skips from blocks 3/6/9/12) trained on a pixel cache with GPU augmentation, scored as per-volume Dice.

```bash
python -m downstream.run_suite --task cq500 --ckpt runs/vit_tiny_v2/last.pt
python -m downstream.run_suite --task brats2023_gli --ckpt runs/vit_tiny_v2/last.pt
sbatch --export=ALL,CKPT=runs/vit_tiny_v2/last.pt jobs/downstream_vit.sbatch
```

## 3. Splits

Volume-level **train / val / test = 75 / 10 / 15**, stratified by `(modality, dataset_family)`, assigned by subject `dedup_key` (`seed=42`). SSL/VAE fits train only and early-stops on val. `data/resplit_manifest.py` rewrites table `split` columns only — never copies pixels.

**DDP later:** same index + `DistributedSampler`; `num_workers` is per rank (start 4–8 workers/GPU). No extra pixel copies.
