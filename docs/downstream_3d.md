# Downstream evaluation and the experiment database

Every model is scored on one fixed suite of frozen-encoder evaluations. Every evaluation writes an **experiment
record**, and the records fold into one SQLite database. Results from different models, heads, seeds and store
versions can then be compared without re-reading logs. The benchmark (`docs/benchmark.md`) is a versioned subset of
this suite with a leaderboard across our models and published ones.

A model is one of our 3D checkpoints or a `configs/bench/models.yaml` entry (a published encoder or a baseline behind
an adapter; `brain/downstream/models.py`). Frozen classification and regression tasks take either; segmentation and
`probe3d` read our encoders' token grids and take checkpoints only.

## Running the suite

```bash
jobs/submit_eval.sh runs/<run>/last.pt                     # every entry, one Slurm job in sequence, then ingest
jobs/submit_eval.sh runs/<run>/last.pt cq500 wmh probe3d   # a subset
python -m brain.downstream.suite list                      # the entries
python -m brain.downstream.suite run --task wmh --model runs/<run>/last.pt -- --steps 3000  # one entry, extra args
python -m brain.downstream.suite run --task ixi_age --model regional_stats    # a models.yaml entry
```

| Entries | Evaluator | Heads / probes |
|---|---|---|
| Classification (`tasks.CLASSIFICATION_TASKS`: `cq500`, `rsna_ich`, `oasis1_cdr`, `abide_asd`, `msd`, ...) | `run_suite` | `linear` (logistic regression) and `mlp` on the mean foreground token, `spatial` (linear on the 8 octant means), `tok_conv` (3D conv probe on the token grid; `--no-token-heads` skips it) |
| Regression (`tasks.REGRESSION_TASKS`: `ixi_age`, `atlas_lesion_volume`, `adni_mmse`, ...) | `run_suite` | same, with ridge |
| Segmentation: `seg_cq500`, `brats2023_{gli,men,ped}`, `brats2024_gli`, `atlas`, `isles22`, `wmh`, `isles24`, `aisd`, `instance`, ... | `seg3d` (3D checkpoints) | `tok_unet3d`: a 3D U-Net decoder on the frozen token grid, trained on token-grid crops of each training volume and 3 shifted / flipped encoded copies |
| `probe3d` | `probe3d` | dense lesion probe (linear, MLP) per held-out family; fingerprint (modality, dataset); location; pixel and random-init controls |

A random-init baseline (`python -m brain.downstream.random_init --config configs/lejepa3d/small.yaml --out
runs/baselines/random_small/last.pt`) is the floor every trained model should beat.

## Geometry

The volume is 256 × 256 × 160 at 1 mm (`brain/common/grid.py`). The 16 × 16 × 8 patch gives a 20 × 16 × 16 token grid
(5120 tokens, `TOKEN_GRID`); every evaluator reads our encoders on it, one pass over the whole volume
(`brain.downstream.embed3d`). Segmentation `dice_*` is scored on the 64 bin-centre axial slices the retired 2D
reference decoder saw; `dice_full_*` uses the whole volume.

Caches live under `runs/downstream/_cache/v<store format>/`: embeddings, and seg3d token and mask memmaps. A
re-prepped store therefore never reuses tensors of the old shape.

## Experiment records (`brain/downstream/records.py`)

There is one JSON per (model, task, head) or (model, probe, head), written atomically to `results/records/<YYYY-MM>/`
in the repo. That directory is on the backed-up home filesystem and is git-ignored. Each record holds:

- **model**: checkpoint path, content fingerprint, step and best step, run name, training config, and the provenance
  stored in the checkpoint at training time (git commit, store format, grid, index hash).
- **data**: evaluation store format, grid, `volumes.csv` SHA-1, split sizes, and `pretrain_overlap` (the fraction of
  the task's volumes that were in the SSL set).
- **protocol**: every evaluator argument, plus `argv`.
- **metrics**: `test` (headline, CI, per label / class / cohort) and `val`. Training curves stay in the run directory.
- **env**: git commit and dirty flag, host, Slurm job / array ids, GPU, library versions, start and end time.
- **artifacts**: the run directory on scratch, predictions (`.npz` / per-volume CSV), the decoder, and the probe JSON
  with its bootstrap draws.

## Database (`brain/downstream/db.py`)

```bash
python -m brain.downstream.db ingest                 # idempotent; the suite's last job runs it
python -m brain.downstream.db report --wide          # models x tasks, headline test metric
python -m brain.downstream.db report --task cq500 --all-heads
python -m brain.downstream.db report --probes
python -m brain.downstream.db query "select e.task, e.head, m.grp, m.value from metrics m join evals e using(eval_id)
                                     where m.name = 'auc_per_label' and e.model_id like 'brainmodel_small%'"
python -m brain.downstream.db export --out results/evals.csv
```

The tables are `models`, `evals` and `metrics`. `metrics` has one row per (eval, split, metric, group), with the CI
bounds alongside. The report picks each (model, task) head **on val**. It falls back to test only when a head has
no val score, and marks that case `test!`. Deleting `results/experiments.sqlite` and re-ingesting rebuilds the
database from the records.

## Reading the numbers

- Splits are the SSL subject splits from `volumes.csv`, so downstream test subjects were never trained on. Several
  task families were still in the SSL pool as unlabeled images: OASIS, IXI, ABIDE, PIOP/PIXAR/ID1000, BraTS and RSNA.
  `pretrain_overlap` makes that visible per eval. CQ500, ISLES, ATLAS, WMH, AISD, INSTANCE, BrainMetShare and MSD are
  held out entirely.
- CQ500 keeps a study once when Seg-CQ500 re-releases it.
- Label tables are read from the store copy (`labels/sources/`, synced with `python -m brain.downstream.tasks
  --sync-labels`), because pool keeps only archives.

## Small tasks: repeated subject-grouped k-fold CV

`run_suite --cv auto|on|off` (default auto; `brain/downstream/cv.py`). Small frozen-head tasks score every subject
instead of a handful in a fixed test split: 5 folds x 3 repeats, grouped by `dedup_key` (all scans of a person stay
together), stratified by label (regression by quantile bin); inside each fold a random 15% of the training subjects is
the validation set. Each head's score is computed on the pooled out-of-fold predictions per repeat; the record gives
the mean over repeats (`<metric>_repeats` lists them), a bootstrap CI over the first repeat's pooled predictions, and
`cv: {folds, repeats}`. Auto = fewer than 200 subjects and no volume in the SSL pretraining pool (for pretraining
families only the fixed test split was kept out of SSL, so CV would test on images the encoder saw); `--cv on` forces
it and the record keeps `pretrain_overlap`. Today auto selects `valdo_epvs` (40 subjects) and `ms3seg_age` (100).
