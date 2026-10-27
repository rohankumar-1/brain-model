# The benchmark (`brain/bench/`, `configs/bench/`)

One fixed, versioned protocol that any model is scored on: our checkpoints and published encoders alike. Its purpose
is to say how our models compare with published ones, and with each other, without comparing models that learned from
different things as if they had learned from the same.

```bash
python -m brain.bench list                                     # the tasks, by group and modality
python -m brain.bench list --model <ckpt | name>               # the tasks this model is scored on
python -m brain.bench run --model runs/<run>/best.pt --submit  # one Slurm job over its tasks (jobs/eval_run.sbatch)
python -m brain.bench run --model regional_stats               # a models.yaml entry, run here
python -m brain.bench board                                    # -> runs/bench/board_v1.{md,csv}, _tasks.csv, _left_out.csv
```

## What is fixed (`configs/bench/benchmark.yaml`)

- **Tasks**: 22 frozen transfer tasks in 5 groups (CT, stroke, neuro-psychiatric, age, lesion / tumour), chosen
  because they tell models apart (rule in the file); 4 segmentation tasks optional (`--with-seg`, our encoders only).
- **Splits**: the store index's, pinned by its sha1 (`index_sha1`). Results computed on another index (older splits)
  are left out of the leaderboard and counted in its `Stale` column.
- **Readout**: one head per task for every model, the one with the best mean validation score over the evaluated
  models (never chosen on test); a model without that head (e.g. a global-only encoder) uses its best head on
  validation.
- **Statistics**: skill (0 = chance, 1 = perfect; AUC, R², balanced accuracy, Dice on one scale), a headline as the
  mean over groups of the mean over tasks, and a paired subject-level bootstrap against the reference
  (`brain.eval.report`).
- **Reference**: run 6 (`reference`), the model every other one is paired with.

## Who is compared with whom

Every model has a training class (`brain.downstream.models.TRAINING_CLASSES`):

| Class | Learned from | Examples | Role in the leaderboard |
|---|---|---|---|
| `ssl` | images only | our LeJEPA / MAE encoders; BrainIAC, Brain2Vec, CT-FM | the main comparison |
| `supervised` | labels of other tasks | our multitask / joint models; SFCN, BrainFM | compared on tasks whose labels it did not train on |
| `specialist` | the evaluated task itself | our single-task models; nnU-Net | a per-task ceiling |
| `baseline` | nothing (no training) | random init; `regional_stats` | the floor |

Three rules keep the comparisons fair:

1. **Seen labels**: a task whose labels a model trained on is in-domain for it, not transfer, and is left out of its
   comparisons (ours: `benchmark.yaml: seen`, from the tasks a checkpoint trained; registry models: their
   `seen_tasks`).
2. **Modality**: a model is scored only on tasks of the modalities it takes (an MRI-only encoder never gets a CT task);
   the leaderboard has one table per modality.
3. **Pretraining overlap**: the share of a task's scans a model may have seen in pretraining is shown next to its
   score (ours: the SSL pool flags of the index; registry models: the dataset families they declare).

The leaderboard pairs each model with the reference on the tasks both have, so a model's Δ and the reference's skill
are on the same tasks and scans.

## Adding a published model

1. Write an adapter in `brain/downstream/adapters/`: subclass `brain.downstream.models.ExternalEncoder`, set `dim`
   (and `needs_brain = True` to receive the stored brain mask, e.g. for skull stripping), and implement
   `encode(vol_xyz, modality, brain)`. It receives the stored volume (1 mm MNI grid, 256 x 256 x 160 as (X, Y, Z); CT
   in HU, MRI as a robust z-score) and does the model's own resampling, cropping and intensity normalisation. It
   returns `pooled` (a global embedding) and, if the model has spatial tokens, `tokens` / `xyz` / `keep`.
2. Add its entry to `configs/bench/models.yaml`: adapter, class, modalities, weights path, the dataset families it
   pretrained on and any benchmark tasks whose labels it trained on. A survey of candidates (2026-10-27) is listed
   there as planned entries.
3. `python -m brain.bench run --model <name> --submit`, then `python -m brain.bench board`.

`regional_stats` (brain-mask intensity statistics, whole brain and 32 mm blocks; no training) is the worked example
and the hand-crafted floor.

## Not covered (yet)

- Fine-tuning protocols (all transfer here is frozen); the FOMO challenge's fine-tuning evaluation is the natural
  external benchmark for that.
- Segmentation and the dense token probes (`probe3d`) for registry models: they read our 20 x 16 x 16 token grid.
- In-domain and few-shot comparisons are for our multitask models only (`brain.eval.report`).
