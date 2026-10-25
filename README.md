# brain-model

Modality-agnostic embeddings for brain CT and MRI. One package, `brain/`, holds the pipeline end to end: prep (raw archives to registered 1 mm Zarr volumes) -> SSL pretraining (several methods on one engine) -> task-conditioned multitask post-training -> frozen-encoder downstream evaluation.

| Method | Command | What it is |
|---|---|---|
| `lejepa3d` | `python -m brain.train lejepa3d` | **Project model.** 3D LeJEPA on native-grid crops with a dense token objective, RoPE in mm, optional masked-block prediction. `docs/lejepa3d.md` |
| `mae3d` | `python -m brain.train mae3d` | 3D MAE, Primus-style ViT on 8 x 8 x 4 mm tokens (split voxel target, lesion-aware masking). |
| `multitask` | `python -m brain.train multitask` | Task-conditioned multitask post-training of an SSL encoder on whole volumes. `docs/multitask.md` |
| `joint` | `python -m brain.train joint` | Joint SSL + multitask training of one encoder from scratch, with an SSL -> task curriculum. `brain/train/joint.py` |

## Layout

```
brain/
  common/      config (defaults + YAML + --set), dist (torchrun, StopFlag), optim (lr, param groups), metrics,
               checkpoint (atomic save, resume), windowing (asinh windows), paths
  prep/        read -> center -> register -> resample -> native units -> brain mask; index; QC; pilot;
               SynthSeg-robust 2.0 (PyTorch port) + runner for anatomy labels
  data/        volume datasets, crops (3D; 2D for the UNETR readout), appearance augmentation, token gate
  models/      vit3d, viewnet3d, mae3d, projector, patch_embed, rope3d, conv3d; vit2d (2D evaluation only)
  objectives/  sigreg (+ batch gather), lejepa3d, dense
  train/       engine (CLI, loop, resume, checkpoints), defaults, one Trainer per method
  multitask/   task registry, label builder (incl. registration / SynthSeg QC), data, model, losses, few-shot
  downstream/  frozen-encoder tasks, heads, probes, segmentation, results views
  datasets/    download / fetch scripts, SinoCT overlap
configs/<method>/*.yaml    only the keys that differ from brain/train/defaults.py need be set
jobs/                      _env.sh (shared), train.sbatch (+ _pi, _pi4), prep_v6*, synthseg, multitask_*, eval_run /
                           submit_eval.sh; jobs/archive/ = finished one-off jobs
tests/                     CPU tests (python tests/run.py tests/test_*.py; pytest also works where installed)
docs/                      prep.md, lejepa3d.md, downstream_3d.md, multitask.md; docs/archive/ = superseded plans
```

## Storage (ORCD)

| | Path |
|---|---|
| Repo | `/home/roku/brain-model` |
| Scratch (processed Zarr only) | `/home/roku/orcd/scratch/brain-model` |
| Pool (raw archives) | `/home/roku/orcd/pool/datasets` |

Override with `BRAIN_SCRATCH` / `BRAIN_DATASETS` (`brain/common/paths.py`), `BRAIN_REPO` / `BRAIN_ENV` for jobs, `BRAIN_TEMPLATE` for registration (with `BRAIN_TEMPLATE_BRAIN`, its brain mask: the fixed image for skull-stripped scans and the prior for brain-mask QC), `BRAIN_SYNTHSTRIP` for the SynthStrip binary (`SYNTHSTRIP_NOCSF=1` selects the model without extra-cerebral CSF).

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

## 1. Preprocessing

```bash
PHASE=B sbatch --export=ALL --array=0-7%4 jobs/prep_v6.sbatch     # see jobs/prep_v6*.sbatch for phases A/B and HBN/GoAT
python -m brain.prep.index --root $STORE
```

Each volume is read, centered, **affinely registered to a skull-on T1 template**, resampled once onto a 256 x 256 x 128 mm 1 mm grid, stored in native units (CT HU, MRI robust z), and given a **brain mask**; native acquisition geometry (including slice thickness) is recorded. Details, flags and the store layout: `docs/prep.md`. `--registration none --brain-mask none` reproduces the old head-centering-only store.

## 2. Pretraining

```bash
python -m brain.train lejepa3d --synthetic --max-steps 8                      # CPU smoke run, no data
python -m brain.train lejepa3d --config configs/lejepa3d/tiny.yaml
sbatch --export=ALL,METHOD=lejepa3d,CONFIG=configs/lejepa3d/small.yaml jobs/train_pi.sbatch   # 2 GPUs
sbatch --export=ALL,METHOD=joint,CONFIG=configs/joint/small.yaml jobs/train_pi4.sbatch
python -m brain.train lejepa3d --config configs/lejepa3d/tiny.yaml --set optim.lr=3e-4 --set aug.curve.p=0
```

All methods share: warmup + cosine schedule, bf16, DDP via `torchrun`, atomic `last.pt` / `best.pt` / `step_*.pt`, resume from `<out_dir>/last.pt`, clean SIGTERM preemption with Slurm requeue, NaN guards, and one log table. Checkpoints store a bare backbone under `model`, so every downstream loader reads all three.

Appearance augmentation (contrast curves with inversion, bias field, through-plane degradation, skull dropout) is **on by default** for `lejepa3d`; see `docs/lejepa3d.md` to turn parts off.

## 3. Downstream

```bash
jobs/submit_eval.sh runs/<run>/last.pt            # the full suite (25 tasks + probes) as a Slurm array, then ingest
python -m brain.downstream.db report --wide        # models x tasks from the experiment database
```

Frozen encoder on the 256 x 256 x 160 grid (20 x 16 x 16 tokens). Classification and regression use PCA plane, grid
and global readouts plus a 3D conv probe, with bootstrap 95% CIs. Segmentation uses a token U-Net decoder (`seg3d`).
`probe3d` runs lesion, fingerprint and location token probes with pixel and random-init controls. Every evaluation
writes an experiment record (model, data provenance, protocol, test and val metrics, environment, artifacts) under
`results/records/`; `brain.downstream.db` folds the records into SQLite. See `docs/downstream_3d.md`.

## 4. Splits

Volume-level train / val / test = 75 / 10 / 15, stratified by `(modality, dataset_family)`, assigned by subject `dedup_key` (`seed=42`). SSL fits train only and early-stops on val. Index tables only; no pixel copies.

## Tests

`python -m pytest tests -q` runs on CPU. `tests/test_registration.py` needs SimpleITK and takes a minute.
