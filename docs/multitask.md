# Task-conditioned multitask post-training (`brain/multitask/`)

An SSL encoder is post-trained on whole 256 x 256 x 160 volumes to answer any subset of a scan's labelled tasks. Every
trained task has a task token inside the encoder's own sequence (OMNI-style, arXiv 2610.12574): task tokens read the
patch tokens and themselves, never each other, and the patch tokens read them. A head shared by every task of an output
type reads the task token's output. All trained tasks' tokens are always present, so an answer never depends on which
other tasks were asked. A new task is a new row of the task-embedding table with a read-only token (it reads the scan,
nothing reads it), so held-out tasks are evaluated few-shot by fitting that row alone.

```
token_t = task_emb[t] + type_emb[type_t]                one token per trained task (encoder width)
tokens  = encoder([CLS | task tokens | patches])        v4 LeJEPA ViT-S, 16 x 16 x 8 mm patches
answer  = head[type_t](proj(token_t))                   binary / regression / multiclass / survival / segmentation
```

Segmentation tasks drive a decoder (token grid + earlier-block taps + an image pyramid for detail below 16 mm) that
predicts at 2 mm:
- **Per-slot decoder** (R7-R14): FiLM-modulated by the task's output.
- **Shared decoder** (`model.seg.shared`, R15): runs once per scan, with a small per-task MLP head per voxel.

Warm start: the patch tokens' attention to the new, random task tokens starts at -10 logits and opens linearly
(`task_tokens.write_bias` / `write_ramp_steps`); undamped, the tokens shift a pretrained encoder's features by ~40%
before any training.

The 2026-10-26 comparison (12k steps): task tokens beat the earlier slot readout (task queries cross-attending to the
finished encoding) on 10 of 25 tasks and lost none. The slot readout is kept for the ResNet single-task specialists
(`task_tokens.enabled: false`).

## Tasks (`configs/multitask/registry.yaml`, `brain/multitask/registry.py`)

A task is semantic, not a dataset (one "whole tumour" task over glioma, meningioma and paediatric cohorts; one "age"
over every cohort with ages). Each task has a type, descriptions (documentation),
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
ADOS, the glioma tasks) or targets (age, hemorrhage, tumour) are marked `seen` in the benchmark
(`configs/bench/benchmark.yaml` `seen`, `models.yaml` `seen_tasks`).

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

    sbatch -p pi_dbertsim -G l40s:4 -c 48 --mem=360G -t 2-00:00:00 -J r14f \
      --export=ALL,METHOD=multitask,EXPECT_GPUS=4,CONFIG=configs/multitask/r14f.yaml jobs/train.sbatch
    python -m brain.train multitask --synthetic --max-steps 4       # CPU smoke run

Sampling:
- Each sample asks k of its available tasks: k uniform in 1..min(n, 8), or all of them 20% of the time.
- Tasks are drawn group first, then task with weight n_task^-0.5.
- `tasks.all_seg` asks every segmentation mask a scan has (with the shared decoder); otherwise at most `max_seg`.
- `data.subject_beta` 1 counts a subject once, however many sequences or sessions it has.

The encoder:
- starts from `model.init_from` (or from scratch) and trains at `enc_lr_mult` x the LR with layer-wise decay;
- holds the task tokens.

Augmentation:
- flips, small rotations / shifts / shear;
- appearance in native units;
- skull-strip matching for tumour tasks;
- synthetic contrast for anatomy tasks;
- field-of-view cuts on rows asking no whole-brain target.

The uncertainty balancer keeps the headline loss near 1, so raw per-group losses are logged (`demo`, `dx`, `ich`, ...).
Checkpoints keep the encoder under `model`, so the frozen suite loads them like any 3D encoder.

| Run | Config | What changed |
|---|---|---|
| R7 / R7-SSL | `r7_targets.yaml`, `r7_ssl.yaml` | task tokens on the v4 encoder, 12k steps; R7-SSL adds masked reconstruction and contrast synthesis |
| R9 | `r9.yaml` | R8 (synthesis on its own budget) + masked reconstruction + aneurysm tasks, 30k steps |
| R10 | `r10.yaml` | 4 GPUs (batch 32), task-aware augmentation, weight averaging (EMA) |
| R12 | `r12.yaml` | R10 to 60k on QC-filtered tool labels (v9), HL-Gauss regression, re-warmed WSD |
| R13 | `r13.yaml` | token maps (mask occupancy, RSNA slice and fastMRI+ box bags), fastMRI+ findings, 25% token gate |
| R14 | `r14.yaml` | true negatives (empty masks on healthy scans, RSNA bleed pseudo-masks), skull-strip matching, synthetic contrast |
| R14-F | `r14f.yaml` | R14 + study-level input: a study's co-registered sequences fused inside the encoder (below) |
| R15 | `configs/joint/r15.yaml` | fresh ViT-B with conv stem, joint SSL + tasks, labels v12 (written, not run) |

Earlier runs (1-6, A1, the slot / token / MoE arms) used options retired since. Their configs are in git history, and
each checkpoint stores its full config.

### Study-level input (`model.fusion`, `tasks.study`)

A scan's co-registered partners are the same subject's scans with the identical stored registration (BraTS / UPENN
sequences, SynthRAD MR + CT; `data.study_partners`).
- With probability `p_multi` a scan comes with 1..`max_scans`-1 of them.
- Each scan runs encoder blocks 0..`after_block`-1 as its own row.
- `vit3d.SequenceFusion` then fuses the rows per token position: the mean plus a zero-initialised attention residual
  with a sequence-type embedding.
- The remaining blocks and every head run on the fused study.

A scan without partners never enters the fusion, so its path is the single-scan model's exactly. Labels and the
decoder's image pyramid are the anchor scan's. `test_eval --study N` scores the heads with partners.

### Input windows

`model.in_chans` 3 = the role windows (CT brain / subdural / bone); `model.window_bank` = the shared learned bank
(`brain.models.window_bank`): native HU in, the encoder's bank maps them to K channels (`window_bank.k`, the
encoder's in_chans).

### Native geometry (removed 2026-11-11)

From 2026-11-07 to 2026-11-11 CT could be read on its own slices (`<case>.ngeo.zip`: native slice spacing, 0.5 mm
in-plane) through a second, in-plane stem (`NgeoStemEmbed`) and decoder (`seg_ngeo`), alone or mixed with 1 mm MRI
in one batch (`data.ngeo`, `model.ngeo`; full_ngeo_bank, R17). Audits showed it broke CT / MRI transfer: with
identical recipes CT was worse on nearly every task. Every scan, CT included, is now read from the 1 mm template
store through the one shared stem. What remains:
- the data product: `brain/prep/ngeo.py` / `ngeo_run.py` / `ngeo_qc.py` build and check the copies, `brain/data/ngeo.py`
  reads them (docs/prep.md "Native-geometry CT copy");
- old checkpoints: their native-geometry weights (`backbone.ngeo_embed.*`, `seg_ngeo.*`) are dropped on load
  (`vit3d.drop_legacy_ngeo`: frozen encoders, `fewshot.load_model`, and resume / warm start, which then resets the
  optimizer state), so R17 and full_ngeo_bank evaluate on 1 mm input;
- old configs: `data.ngeo` / `model.ngeo` are still accepted keys, but a new run with either enabled stops
  (`defaults.refuse_ngeo`).

## Evaluation

### Scoring a run

- In-domain: `python -m brain.multitask.test_eval --ckpts REF RUN` scores the trained heads on the test splits, paired
  by scan with the reference (`fp` rows: false-positive rate of lesion heads on lesion-free scans).
- Transfer: the frozen benchmark, `python -m brain.bench run --model NAME` (`docs/benchmark.md`).
- Adaptation: few-shot on the held-out tasks (`brain/multitask/fewshot.py`).

Every metric is on one skill scale (`brain/eval/skill.py`): 0 = chance, 1 = perfect (AUROC `(auc - 0.5) / 0.5`,
balanced accuracy rescaled from chance, R2 clipped at 0, Dice as is).

During training, `run_eval` logs `val_score` (and `val_group_<g>`) on that scale, and `train.best_metric: score` keeps
the checkpoint with the highest score. Validation and test ask each task on its own, as at deployment.

### Individual evaluations

- Frozen suite: as for any encoder (its CLS and patch tokens; the model's own task outputs are in-domain, not
  transfer, and are scored by `test_eval`).
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

## Config switches

- `model.task_tokens.enabled` (false: the slot readout, used by the ResNet specialists).
- `model.seg.shared` / `head_hidden` (shared segmentation decoder with per-task MLP heads).
- `model.fusion`, `tasks.study` (study-level input).
- `model.window_bank` (the shared learned window bank).
- `tasks.group_weights`, `tasks.groups` / `tasks.ids` (train only some groups or tasks), `tasks.exclude`.
- `loss.negative_tasks` (lesion tasks with true negatives).
- `data.exclude_datasets` / `exclude_volumes` (benchmark cohorts and duplicates out of training).
