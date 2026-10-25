# Task-conditioned multitask post-training (`brain/multitask/`)

An SSL encoder is post-trained on whole 256 x 256 x 160 volumes to answer any subset of a scan's labelled tasks at
once. Each requested task is a query slot; slots cross-attend to the encoder's tokens and to each other, and a head
shared by every task of the slot's output type reads the answer. A new task is a new row of the task-embedding table,
so held-out tasks are evaluated few-shot by fitting that row alone.

```
tokens  = encoder(volume)                               (B, N, D) + CLS; v3 LeJEPA ViT-S, 16 x 16 x 8 mm tokens
slot_k  = task_emb[task_k] + type_emb[type_k]           one query per requested task
slots   = L x [cross-attend to tokens -> self-attend across slots -> MLP]
answer  = head[type_k](slot_k)                          binary / regression / multiclass / survival / segmentation
general = L x [cross-attend to tokens -> MLP](general_q)   optional task-agnostic slot, trained with LeJEPA (below)
```

Segmentation slots drive a decoder (token grid + earlier-block taps + an image pyramid for detail below 16 mm),
FiLM-modulated by the slot, predicting at 2 mm.

## Tasks (`configs/multitask/registry.yaml`, `brain/multitask/registry.py`)

A task is semantic, not a dataset (one "whole tumour" task over glioma, meningioma and paediatric cohorts; one "age"
over every cohort with ages). Each task has a type, descriptions (documentation; embedded only for the text ablation),
class names (multiclass), a role (`train` or `held_out`), a group and label sources. Groups balance training: sampling
and loss go over groups first, then over tasks within a group, so the 32 SynthSeg-derived tasks weigh as two groups.
75 tasks, 65 trained:

| Group | Tasks | Sources |
|---|---|---|
| demographics | age, sex | FOMO participants tables (incl. infants 0-3 y and the 2026-10 small cohorts) |
| diagnosis | BrainLat diagnosis, tumour type, infant abnormality, ADHD, Parkinson's | FOMO participants (BrainLat, Infant Development, ACPI + CUNMET, Tao Wu) |
| hemorrhage | 6 RSNA subtypes | RSNA ICH |
| lesion | hemorrhage segmentation | BHSD (5-class masks, all subtypes merged) + HemSeg-500 (IPH- or IVH-only scans) |
| tumor | whole tumour, core, enhancing, resection cavity, tumour volume, glioma 12-month survival, lower grade vs GBM, IDH | BraTS masks; TCGA clinical data for the 163 BraTS GLI cases from TCGA (`brain/datasets/tcga_glioma.py`) |
| derived | brain volume, brain mask, mask / intensity asymmetry, CT CSF fraction, CT density, calcification, skull density, grey-white ratio | computed from the image and brain mask (`@live`) or SynthSeg (`measures.py`) |
| anatomy_seg | 12 SynthSeg structure masks (left + right merged) | SynthSeg-robust 2.0 |
| anatomy_volume | 18 SynthSeg volumes / ratios / asymmetries, Evans index, WMH burden | SynthSeg; WMH-SynthSeg on non-tumour FLAIR / T2 (`brain/prep/wmh_run.py`) |
| clinical_scores | MMSE, ADOS calibrated severity | OASIS-1 / OASIS-2, ABIDE (via the downstream task loaders) |
| held_out | CQ500 findings, dementia, stroke / WMH / MS lesion segmentation | few-shot only |

Survival targets are binary for now (alive at 12 months; patients censored earlier are dropped): the survival head
exists but no survival loss is implemented.

## Labels (`brain/multitask/labels.py`, `jobs/multitask_labels.sbatch`)

`python -m brain.multitask.labels` writes `labels/multitask/{labels.parquet, stats.json, summary.txt,
regqc_summary.csv}` in the store: one row per (volume, task) with a value, a reference (mask zip and label ids,
`@brain`, or `@live:<name>` for targets the loader computes from the image and brain mask) and the split.

Leakage rules: training sources read only pretraining-pool volumes (plus BHSD, kept out of SSL only), and training
uses only their `split == train` rows. Suite tasks that share training labels (`rsna_ich`, BraTS segmentation, MMSE,
ADOS, the glioma tasks) or targets (age, hemorrhage, tumour) are marked `seen` in the results views
(`brain/downstream/results_view.py`).

A volume failing registration QC loses only its alignment-dependent targets (template-space volumes, left-right
asymmetries, every `vol_*`); its study-level labels and image-space masks still train. The QC fails a volume on
brain-mask left-right asymmetry about the grid midline > 0.09 (tilted / shifted heads), template brain Dice < 0.85
(misregistration), or a failed SynthSeg QC. SynthSeg masks need `qc_ok`; SynthSeg volumes and ratios also need
`fov_ok` (whole brain in view). See `docs/prep.md` for the SynthSeg flags and the cohorts they catch.

Free-text participant groups map to classes with `map` (exact), `prefix` (first matching prefix), `default`
(unmatched values) and `drop` (exact values removed), e.g. infant findings: `normal -> 0`, `premature` and motion
artefacts dropped, everything else `-> 1`.

### Order of the label pipeline after new data

    python -m brain.prep.synthseg_run --shard i --n-shards 4      # jobs/synthseg.sbatch; new rows appended to select.csv
    python -m brain.prep.wmh_run --shard i --n-shards 2           # jobs/wmh.sbatch (optional; skips done volumes)
    python -m brain.multitask.measures --workers 31               # Evans index, CT measures, midline (all of select.csv)
    sbatch jobs/multitask_labels.sbatch                           # label table + registration QC

## Training (`brain/train/multitask.py`)

    sbatch --export=ALL,METHOD=multitask,CONFIG=configs/multitask/v4_general.yaml jobs/train_pi.sbatch
    python -m brain.train multitask --synthetic --max-steps 4       # CPU smoke run

Each sample asks k of its available tasks (k uniform in 1..min(n, 8), or all of them 20% of the time; group first,
then task with weight n_task^-0.5; at most 2 segmentation slots). The encoder starts from `model.init_from`, is frozen
for `freeze_steps`, then trains at `enc_lr_mult` x the LR (0.1 since run 2) with layer-wise decay. Augmentation:
flips, small rotations / shifts, appearance (bias, through-plane blur, Rician / Gaussian noise, blur, drift) with the
canonical window pinned, and field-of-view cuts on rows asking no whole-brain target. The loss balancer keeps the
headline loss near 1, so raw per-group losses are logged (`demo`, `dx`, `ich`, ...) to follow progress. Checkpoints
keep the encoder under `model`, so the frozen suite loads them like any 3D encoder.

| Run | Config | What changed |
|---|---|---|
| 1 | `v3_init.yaml` | first run: 33 tasks, encoder 0.25 x LR |
| 2 | `v3_synthseg.yaml` | SynthSeg anatomy, task groups, registration QC, appearance + FOV augmentation, encoder 0.1 x LR, batch 8 |
| control | `v3_synthseg_nolabels.yaml` | run 2 restricted to label-free groups (derived, anatomy_seg, anatomy_volume) |
| 3 | `v4_general.yaml` | general-embedding slot; BHSD / HemSeg / small-FOMO data; WMH, hemorrhage segmentation, clinical-score, glioma, infant / ADHD / Parkinson's tasks |

### General-embedding slot (`model.general`, `loss.general`)

A task-agnostic query (`general_q`) runs through the slot layers on its own, so its output never depends on which
tasks a row asks, and a projector feeds LeJEPA (`brain/objectives/sigreg.py`: prediction loss + SIGReg, `lamb` 0.02,
gathered across GPUs). The two global views: the training view of every row, and a second view of the first `frac`
of the batch (`brain.multitask.data.ssl_view`: rotation / shift, a box crop keeping >= `crop_min` of each axis, no
appearance change). Both views go through the encoder in one pass; the second view's rows ask no task. The loss is
added at `weight` (the task loss is ~1 by construction) and logged as `gen_inv` / `gen_sig`. Off by default: with
`model.general.enabled: false` the model has no extra parameters, so older checkpoints load unchanged.

## Evaluation

### The run report (`brain/eval/`, `configs/eval/core.yaml`)

One report per run against a reference run: `jobs/submit_report.sh RUN_CKPT REF_CKPT` runs the three evaluations and
then `python -m brain.eval.report RUN --ref REF`, which writes `runs/eval/<run>/report.md` (three headline numbers,
then group scores) and `report_tasks.csv` (per task).

| Headline | Question | Source |
|---|---|---|
| In-domain | Does it do the trained tasks? | Its heads on the test splits (`test_eval`), paired with the reference |
| Transfer | Is the encoder a good foundation? | Frozen suite on the core tasks (`core.yaml`: 21 tasks in 5 groups) |
| Adaptation | Can it learn a new task from few labels? | Few-shot on the held-out tasks at k = 64 |

Every metric is on one skill scale (`brain/eval/skill.py`): 0 = chance, 1 = perfect (AUROC `(auc - 0.5) / 0.5`,
balanced accuracy rescaled from chance, R2 clipped at 0, Dice as is). A headline is the mean over groups of the mean
over tasks, on tasks both models have, so a group with many tasks counts once; its interval combines the per-task
intervals assuming independence (approximate). The core transfer set keeps tasks that can tell runs apart (rule in
`core.yaml`); the rest of the suite still runs on request and lands in the per-task CSV.

Validation during training uses the same skill scale: `run_eval` logs `val_score` (and `val_group_<g>`), and
`train.best_metric: score` (the multitask and joint default) keeps the checkpoint with the highest score rather than
the lowest loss, which is not comparable across steps under the curriculum or uncertainty weighting. Validation and
test both ask each task on its own (`alone`, as at deployment; the scan is encoded once): before 2026-10-25 validation
asked all of a scan's tasks in one query, so slots attended to each other and validation and test numbers differed.

### Individual evaluations

- Frozen suite: as for any encoder, plus global views for multitask checkpoints (`brain/multitask/slots.py`):
  `slots` (every trained task asked on its own, slot vectors concatenated), `taskout` (the model's own answers) and,
  with a general slot, `general` (its pre-projector embedding).
- Few-shot and transfer: `jobs/multitask_fewshot.sbatch CKPT=...` (`brain/multitask/fewshot.py`). Held-out tasks fit
  their embedding row on k = 16 / 64 / 256 training scans, against a linear probe on the frozen pooled embedding;
  trained tasks are asked directly on unseen cohorts (age on OASIS / IXI / MS3SEG, whole tumour on MSD).
- Suite tasks added 2026-10-23 (`brain/downstream/tasks.py`): ISLES'24 3-month / discharge mRS and NIHSS, occlusion
  side (CTA), Circle of Willis segmentation (CTA); CQ500 bleed side, chronic bleed, EDH, other fracture; AISD remote
  infarct; OASIS-1 / -2 MMSE; ABIDE ADOS; ATLAS lesion volume / side / days post stroke; MS3SEG lesion load; glioma
  12-month survival, grade, IDH. Rare or small tasks run with `--cv on` (the CQ500 findings, the glioma tasks).
  Per-scan tables from masks: `python -m brain.downstream.mask_labels` (lesion volume and side, AISD infarct codes,
  occlusion side and vessel).

## Clinical measures (`brain/multitask/measures.py`)

Per-scan measures from SynthSeg labels, written to `labels/synthseg/measures.csv`: Evans index (validated: age rho
0.51, AD 0.293 vs CN 0.265), CT grey-white ratio, calcification, skull density, and midline shift.

Midline shift (`midline_shift_plane`): the reference is the plane about which the skull (CT > 300 HU) or the head
outline (MRI) is most mirror-symmetric (offset, yaw, roll; coarse-to-fine search), since bone does not move under mass
effect. Displacement from it is measured on the septum pellucidum (midpoint between the lateral ventricles' medial
walls) in the frontal horns (rows anterior to the third ventricle), as the largest per-slice median, and on the
third ventricle's centre; `max_mm` is the larger. On CQ500 (474 scans, 3-reader MidlineShift labels; not trained on):

| Variant | AUROC | Sens / spec at 3 mm | at 5 mm |
|---|---|---|---|
| brain-pole line (first version) | 0.732 | 0.57 / 0.78 | 0.41 / 0.89 |
| skull plane, all ventricle rows | 0.841 | 0.52 / 0.96 | 0.30 / 0.99 |
| skull plane, frontal horns, max over slices | 0.946 | 0.86 / 0.90 | 0.62 / 0.99 |
| `max_mm` (frontal max or third ventricle) | 0.951 | 0.88 / 0.88 | 0.67 / 0.98 |

`runs/diagnostics/midline_cq500.py` (segments CQ500 once and saves the labels; `--from-saved` re-scores) and
`midline_healthy_viz.py` (healthy range on RSNA-negative CT and healthy MRI, figure). ~10% of scans get no value (no
third ventricle or lateral ventricles found). Not yet a training target.

## Ablations (config switches)

`model.slot.embedding: learned | text`, `model.slot.width` (bottleneck), `model.slot.mode: slots | shared`,
`model.slot.cross_task: true | false`, `loss.balance: uniform | ema | uncertainty`, `tasks.group_weights`,
`tasks.groups` (train only some groups), `model.general.enabled` + `loss.general.weight`.
