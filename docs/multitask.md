# Task-conditioned multitask post-training (`brain/multitask/`)

An SSL encoder is post-trained on whole 256 x 256 x 160 volumes to answer any subset of a scan's labelled tasks. Every
trained task has a task token inside the encoder's own sequence (OMNI-style, arXiv 2610.12574): task tokens read the
patch tokens and themselves, never each other, and the patch tokens read them. A head shared by every task of an output
type reads the task token's output. All trained tasks' tokens are always present, so an answer never depends on which
other tasks were asked. A new task is a new row of the task-embedding table with a read-only token (it reads the scan,
nothing reads it), so held-out tasks are evaluated few-shot by fitting that row alone.

```
token_t = task_emb[t] + type_emb[type_t]                one token per trained task (encoder width)
tokens  = encoder([CLS | task tokens | patches])        v3 LeJEPA ViT-S, 16 x 16 x 8 mm patches
answer  = head[type_t](proj(token_t))                   binary / regression / multiclass / survival / segmentation
```

Segmentation tasks drive a decoder (token grid + earlier-block taps + an image pyramid for detail below 16 mm),
FiLM-modulated by the task's output, predicting at 2 mm. Warm start: the patch tokens' attention to the (new, random)
task tokens starts at -10 logits and opens linearly over 2k steps (`task_tokens.write_bias` / `write_ramp_steps`);
undamped, the tokens shift v3's features by ~40% before any training. `moe.blocks` optionally turns the top blocks'
MLPs into mixtures of experts (experts copied from the MLP; no clear gain over tokens alone at 12k steps).

The 2026-10-26 comparison (12k steps, batch 8, test set, 25 tasks with >= 20 test scans): task tokens beat the earlier
slot readout (task queries cross-attending to the finished encoding through 3 slot layers) on 10 tasks and lost none,
with 3M fewer parameters; the gains sit on whole-brain scalar targets (CT density, brain volume, calcification,
tumour type). Slot-readout checkpoints (runs 1-6, A1) still load (`task_tokens.enabled: false`).

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

    sbatch --export=ALL,METHOD=multitask,CONFIG=configs/multitask/tokens.yaml jobs/train_pi4.sbatch
    sbatch --export=ALL,METHOD=joint,CONFIG=configs/joint/tokens_warm_v3.yaml jobs/train_pi4.sbatch   # with SSL
    python -m brain.train multitask --synthetic --max-steps 4       # CPU smoke run

Each sample asks k of its available tasks (k uniform in 1..min(n, 8), or all of them 20% of the time; group first,
then task with weight n_task^-0.5; at most 2 segmentation tasks). The encoder starts from `model.init_from` and trains
at `enc_lr_mult` x the LR (0.25) with layer-wise decay from step 0 (task tokens live in it). Augmentation:
flips, small rotations / shifts, appearance (bias, through-plane blur, Rician / Gaussian noise, blur, drift) with the
canonical window pinned, and field-of-view cuts on rows asking no whole-brain target. The loss balancer keeps the
headline loss near 1, so raw per-group losses are logged (`demo`, `dx`, `ich`, ...) to follow progress. Checkpoints
keep the encoder under `model`, so the frozen suite loads them like any 3D encoder.

| Run | Config | What changed |
|---|---|---|
| 1 | `v3_init.yaml` | first run: 33 tasks, encoder 0.25 x LR |
| 2 | `v3_synthseg.yaml` | SynthSeg anatomy, task groups, registration QC, appearance + FOV augmentation, encoder 0.1 x LR, batch 8 |
| control | `v3_synthseg_nolabels.yaml` | run 2 restricted to label-free groups (derived, anatomy_seg, anatomy_volume) |
| 3 | `v4_general.yaml` | general-embedding slot (retired); BHSD / HemSeg / small-FOMO data; WMH, hemorrhage segmentation, clinical-score, glioma, infant / ADHD / Parkinson's tasks |
| 4 | `v5_enclr.yaml` | encoder 0.5 x LR, layer decay 0.9 |
| 5 | `v6_nogen_unc.yaml` | no general slot, uncertainty loss balancing |
| 6 | `v7_nosmall.yaml` | run 5 without the 7 small clinical tasks (now the default exclusion) |
| A1 | `joint/a1_warm_v3.yaml` | joint SSL + tasks warm from v3, slot readout |
| arms A / B / C | `omni_*.yaml` | slots vs task tokens vs task tokens + MoE, 12k steps each (B became the default) |
| next | `multitask/tokens.yaml`, `joint/tokens_warm_v3.yaml` | task tokens at full length; and A1 with task tokens |

All of these configs are in `configs/archive/` (they use options retired on 2026-10-27).

## Evaluation

### The run report (`brain/eval/`, `configs/bench/benchmark.yaml`)

One report per run against a reference run: `jobs/submit_report.sh RUN_CKPT REF_CKPT` runs the three evaluations and
then `python -m brain.eval.report RUN --ref REF`, which writes `runs/eval/<run>/report.md` (three headline numbers,
then group scores) and `report_tasks.csv` (per task).

| Headline | Question | Source |
|---|---|---|
| In-domain | Does it do the trained tasks? | Its heads on the test splits (`test_eval`), paired with the reference |
| Transfer | Is the encoder a good foundation? | Frozen suite on the benchmark tasks (`benchmark.yaml`: 22 tasks in 5 groups) |
| Adaptation | Can it learn a new task from few labels? | Few-shot on the held-out tasks at k = 64 |

Every metric is on one skill scale (`brain/eval/skill.py`): 0 = chance, 1 = perfect (AUROC `(auc - 0.5) / 0.5`,
balanced accuracy rescaled from chance, R2 clipped at 0, Dice as is). A headline is the mean over groups of the mean
over tasks, on tasks both models have, so a group with many tasks counts once; its interval combines the per-task
intervals assuming independence (approximate). The core transfer set keeps tasks that can tell runs apart (rule in
`benchmark.yaml`); the rest of the suite still runs on request and lands in the per-task CSV. In-domain skips tasks
with fewer than 20 test scans (the SynthSeg tasks have ~4). Comparisons with published models: `docs/benchmark.md`.

Validation during training uses the same skill scale: `run_eval` logs `val_score` (and `val_group_<g>`), and
`train.best_metric: score` (the multitask and joint default) keeps the checkpoint with the highest score rather than
the lowest loss, which is not comparable across steps under the curriculum or uncertainty weighting. Validation and
test both ask each task on its own (`alone`, as at deployment; the scan is encoded once): before 2026-10-25 validation
asked all of a scan's tasks in one query, so slots attended to each other and validation and test numbers differed.

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

`model.task_tokens.enabled` (false: the slot readout), `model.moe.blocks`, `loss.balance: uniform | ema | uncertainty`,
`tasks.group_weights`, `tasks.groups` / `tasks.ids` (train only some groups or tasks), `tasks.exclude`. Retired on
2026-10-27 after their ablations: the text task embedding, the slot bottleneck width, the shared (slot-free) readout
and the general-embedding slot.
