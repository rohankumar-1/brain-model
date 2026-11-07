# BrainQuery (BQ)

**BrainQuery** is a 3D foundation model for brain CT and MRI that answers task queries: a ViT encoder whose sequence
carries one token per task, so one pass answers any of its trained tasks, and a new task is learned from a few labelled
scans. One package, `brain/`, holds the pipeline end to end:

1. prep: raw archives → registered 1 mm volumes;
2. self-supervised pretraining;
3. multitask training;
4. evaluation and a versioned benchmark against published encoders.

Checkpoints are named `BQ-...` in `configs/bench/models.yaml`, e.g. BQ-v4 (the self-supervised encoder), BQ-R14 (the
current multitask run) and BQ-R14F (R14 with study-level sequence fusion).

| Method | Command | What it is |
|---|---|---|
| `lejepa3d` | `python -m brain.train lejepa3d` | Self-supervised pretraining: 3D LeJEPA on native-grid crops (pooled + dense agreement, SIGReg, masked-block prediction), RoPE in mm. `configs/lejepa3d/v4_scratch.yaml`, `docs/lejepa3d.md` |
| `multitask` | `python -m brain.train multitask` | Task training on whole volumes: task tokens in the encoder, typed heads, a segmentation decoder, true negatives, study-level fusion of co-registered sequences. `configs/multitask/r*.yaml`, `docs/multitask.md` |
| `joint` | `python -m brain.train joint` | One encoder trained from scratch on LeJEPA and the tasks together, with the SSL weight scheduled from large to even (the multitask trainer plus an SSL branch). `configs/joint/r15.yaml` |

## Layout

```
brain/
  common/      config (defaults + YAML + --set), dist (torchrun, StopFlag), optim, checkpoint, windowing, paths
  prep/        read -> center -> register -> resample -> native units -> brain mask; index; data QC + decontamination;
               SynthSeg-robust (PyTorch port) and WMH runners for tool labels
  data/        SSL volume datasets, crops, appearance augmentation, token gate
  models/      vit3d (encoder, conv stem, sequence fusion), resnet3d, viewnet3d, projector, patch_embed, rope3d, conv3d
  objectives/  sigreg, lejepa3d, dense
  train/       engine (CLI, loop, resume, checkpoints, EMA), defaults, one Trainer per method
  multitask/   the task side: registry, label builder, data, the BrainQuery model, losses, test_eval, few-shot, explain
  downstream/  frozen-encoder tasks, heads and probes, segmentation transfer, experiment records; models.py (any
               encoder, ours or published, behind one interface) and adapters/ (published encoders)
  bench/       the benchmark: versioned spec, run, pin, per-model card, leaderboard, comparisons
  datasets/    download / fetch scripts
configs/       lejepa3d/, multitask/ (runs + the task registry), joint/, single/ (single-task specialists),
               pilot/ (short design pilots), bench/ (benchmark.yaml, models.yaml, pinned readouts, exclusion lists)
jobs/          _env.sh, train.sbatch, eval_run.sbatch, prep / index / label jobs; jobs/archive/ = finished one-off jobs
analysis/      cited checks and reusable analysis tools; analysis/bench_page/ builds the result pages
tests/         CPU tests (python -m pytest tests -q)
docs/          prep.md, lejepa3d.md, multitask.md, downstream_3d.md, benchmark.md; datasets_and_tasks.xlsx
```

Only the keys that differ from `brain/train/defaults.py` need to be set in a config. `tests/test_config.py` checks
every shipped config against the defaults.

## Storage (ORCD)

| | Path |
|---|---|
| Repo | `/home/roku/brain-model` |
| Scratch (preprocessed store, runs, caches) | `/home/roku/orcd/scratch/brain-model` |
| Pool (raw archives, JPEG-XL lossless) | `/home/roku/orcd/pool/datasets` |

Override with:
- `BRAIN_SCRATCH` / `BRAIN_DATASETS` (`brain/common/paths.py`);
- `BRAIN_REPO` / `BRAIN_ENV` for jobs;
- `BRAIN_TEMPLATE` / `BRAIN_TEMPLATE_BRAIN` for registration;
- `BRAIN_SYNTHSTRIP` for the SynthStrip binary.

Data under a data use agreement (ADNI, PPMI): tables stay on pool, owner-only. Id lists the repo needs live in
git-ignored `*.private.csv` files.

## Datasets

The store holds ~129k volumes from CT and MRI cohorts:
- FOMO-60K and its small cohorts;
- RSNA ICH, CQ500, BHSD, HemSeg, INSTANCE, AISD, ISLES24;
- BraTS 2023/2024, UPENN-GBM, UCSF-PDGM, MSD;
- OASIS-1/2, IXI, ABIDE, ATLAS, MS3SEG, VALDO, WMH;
- RSNA-ICA, SynthRAD, fastMRI+, ADNI, PPMI, Pedi-Cranial-CT-Healthy.

Each is stored on one 256 x 256 x 160 grid of 1 mm voxels, RAS, affinely registered to a skull-on T1 template, in
native units (CT HU, MRI robust z). Cohorts, licences and the tasks built from them are listed in
`docs/datasets_and_tasks.xlsx`; the store index is `volumes.csv` on scratch.

Splits are train / val / test = 75 / 10 / 15 by subject (`dedup_key`), stratified by modality and dataset family
(`seed=42`). Benchmark-only cohorts are held out of every training and SSL pool. Pixel-verified duplicates of
evaluation scans leave the training pools (`configs/multitask/train_exclude_r15.csv`) and the test sets
(`configs/bench/exclude_v3*.csv`).

## 1. Preprocessing

```bash
PHASE=B sbatch --export=ALL --array=0-7%4 jobs/prep_v6.sbatch        # jobs/prep_v6*.sbatch: phases, local staging
sbatch jobs/index_append.sbatch                                     # add new datasets to the index, append-only
sbatch jobs/multitask_labels.sbatch                                 # the multitask label table
```

Steps, flags, the store layout and the rebuild order of the post-prep label producers are in `docs/prep.md`.

## 2. Training

```bash
python -m brain.train multitask --synthetic --max-steps 4           # CPU smoke run, no data
sbatch -p pi_dbertsim -G l40s:4 -c 48 --mem=360G -t 2-00:00:00 -J r14f \
  --export=ALL,METHOD=multitask,EXPECT_GPUS=4,CONFIG=configs/multitask/r14f.yaml jobs/train.sbatch
```

All methods share:
- warmup + cosine or WSD schedules, bf16, DDP via `torchrun`;
- atomic `last.pt` / `best.pt` / EMA checkpoints, resume from `<out_dir>/last.pt`;
- SIGTERM checkpointing with Slurm requeue, and a non-finite-step guard.

A checkpoint stores the encoder under `model`, so every downstream loader reads every method.

## 3. Evaluation and the benchmark

```bash
python -m brain.bench run --model BQ-R14 --submit --jobs 1          # the frozen benchmark for one model
python -m brain.bench pin                                           # fix one readout per task (on validation)
python -m brain.bench board                                         # leaderboard
python -m brain.bench card --model BQ-R14                           # every number recorded for a model
python -m brain.multitask.test_eval --ckpts runs/multitask/r14/last.pt --out runs/diagnostics/test_eval_r14
```

The benchmark is version 3 (`configs/bench/benchmark.yaml`, `docs/benchmark.md`):
- frozen encoders, read out from their patch tokens (linear / MLP on the mean token, an octant probe, a 3D-conv token
  probe);
- one readout per task, pinned on validation across models;
- decontaminated test sets and bootstrap 95% intervals.

`test_eval` scores a model's own trained heads on held-out test splits (`--study N` adds each scan's co-registered
sequences). `analysis/bench_page/` turns the results into the two result pages.

## Tests

`python -m pytest tests -q` runs on CPU in a few minutes. `tests/test_registration.py` needs SimpleITK.
