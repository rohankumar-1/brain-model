# Task-conditioned multitask post-training (`brain/multitask/`)

An SSL encoder is post-trained on whole 256 x 256 x 160 volumes to answer any subset of a scan's labelled tasks at
once. Each requested task is a query slot; slots cross-attend to the encoder's tokens and to each other, and a head
shared by every task of the slot's output type reads the answer. A new task is a new row of the task-embedding table,
so held-out tasks are evaluated few-shot by fitting that row alone.

```
tokens = encoder(volume)                                (B, N, D) + CLS; v3 LeJEPA ViT-S, 16 x 16 x 8 mm tokens
slot_k = task_emb[task_k] + type_emb[type_k]            one query per requested task
slots  = L x [cross-attend to tokens -> self-attend across slots -> MLP]
answer = head[type_k](slot_k)                           binary / regression / multiclass / survival / segmentation
```

Segmentation slots drive a decoder (token grid + earlier-block taps + an image pyramid for detail below 16 mm),
FiLM-modulated by the slot, predicting at 2 mm.

## Tasks (`configs/multitask/registry.yaml`, `brain/multitask/registry.py`)

A task is semantic, not a dataset (one "whole tumour" task over glioma, meningioma and paediatric cohorts; one "age"
over every cohort with ages). Each task has a type, descriptions (documentation; embedded only for the text ablation),
class names (multiclass), a role (`train` or `held_out`), a group and label sources. Groups balance training: sampling
and loss go over groups first, then over tasks within a group, so the 30 SynthSeg tasks weigh as two groups.

| Group | Tasks |
|---|---|
| demographics | age, sex (FOMO participants tables) |
| diagnosis | BrainLat diagnosis, tumour type |
| hemorrhage | 6 RSNA subtypes |
| tumor | whole tumour, core, enhancing, resection cavity (BraTS masks), tumour volume |
| derived | brain volume, brain mask, left-right mask / intensity asymmetry, CT CSF fraction, CT density |
| anatomy_seg | 12 SynthSeg structure masks (left + right merged) |
| anatomy_volume | 18 SynthSeg volumes, ratios and asymmetries (template space, so head-size normalised) |
| held_out | CQ500 findings, dementia, stroke / WMH / MS lesion / hemorrhage segmentation (few-shot only) |

## Labels (`brain/multitask/labels.py`, `jobs/multitask_labels.sbatch`)

`python -m brain.multitask.labels` writes `labels/multitask/{labels.parquet, stats.json, summary.txt,
regqc_summary.csv}` in the store: one row per (volume, task) with a value, a reference (mask zip and label ids,
`@brain`, or `@live:<name>` for targets the loader computes from the image and brain mask) and the split.

Leakage rules: training sources read only pretraining-pool volumes, and training uses only their `split == train`
rows. Suite tasks that share training labels (`rsna_ich`, BraTS segmentation) or targets (age, hemorrhage, tumour)
are marked `seen` in the results views.

A volume trains nothing when it fails registration QC: brain-mask left-right asymmetry about the grid midline > 0.09
(tilted / shifted heads), template brain Dice < 0.85 (misregistration), or a failed SynthSeg QC. SynthSeg masks need
`qc_ok`; SynthSeg volumes and ratios also need `fov_ok` (whole brain in view). See `docs/prep.md` for the SynthSeg
flags and the cohorts they catch (RSNA without its posterior fossa, M4Raw coverage, HBN PD).

## Training (`brain/train/multitask.py`, `configs/multitask/v3_init.yaml`)

    sbatch --export=ALL,METHOD=multitask,CONFIG=configs/multitask/v3_init.yaml jobs/train_pi.sbatch
    python -m brain.train multitask --synthetic --max-steps 4       # CPU smoke run

Each sample asks k of its available tasks (k uniform in 1..min(n, 8), or all of them 20% of the time; group first,
then task with weight n_task^-0.5; at most 2 segmentation slots). The encoder starts from `model.init_from`, is frozen
for `freeze_steps`, then trains at 0.25 x the LR with layer-wise decay. Checkpoints keep the encoder under `model`, so
the frozen suite loads them like any 3D encoder.

## Evaluation

- Frozen suite: as for any encoder, plus two global views for multitask checkpoints (`brain/multitask/slots.py`):
  `slots` (every trained task asked on its own, slot vectors concatenated) and `taskout` (the model's own answers).
- Few-shot and transfer: `jobs/multitask_fewshot.sbatch CKPT=...` (`brain/multitask/fewshot.py`). Held-out tasks fit
  their embedding row on k = 16 / 64 / 256 training scans, against a linear probe on the frozen pooled embedding;
  trained tasks are asked directly on unseen cohorts (age on OASIS / IXI / MS3SEG, whole tumour on MSD).

## Ablations (config switches)

`model.slot.embedding: learned | text`, `model.slot.width` (bottleneck), `model.slot.mode: slots | shared`,
`model.slot.cross_task: true | false`, `loss.balance: uniform | ema | uncertainty`, `tasks.group_weights`.
