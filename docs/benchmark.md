# Evaluation and the benchmark (`brain/bench/`, `configs/bench/`)

Our model is **BrainQuery (BQ)**: a 3D encoder that answers task queries. Every checkpoint we compare has a name in
`configs/bench/models.yaml`: BQ-v3 (the self-supervised encoder), BQ-R2 ... BQ-R6 (multitask runs), BQ-A1 (joint SSL +
tasks), BQ-12k-slots / -tokens / -moe (the readout comparison), BQ-S-hemorrhage (a single-task specialist), BQ-T1 /
BQ-A2 (the task-token runs), BQ-random (the architecture, untrained: the floor). One fixed, versioned protocol scores BQ models and published models alike, without
comparing models that learned from different things as if they had learned from the same.

```bash
python -m brain.bench list [--model BQ-A1]                # the tasks (those a model is scored on)
python -m brain.bench run --model BQ-T1 --full --submit   # every evaluation that applies, as Slurm jobs
python -m brain.bench run --model regional_stats          # a published model / baseline (frozen transfer)
python -m brain.bench card --model BQ-A1                  # every number recorded for one model -> runs/bench/cards/
python -m brain.bench board                               # leaderboard -> runs/bench/board_v2.{md,csv} (+ _tasks, _left_out)
                                                          #   and every task x every model -> runs/bench/board_v2_all_tasks.md
python -m brain.bench pin --models BQ-R6 BQ-v3 BQ-A1      # (once per benchmark version) fix each task's readout
```

## What is fixed (`configs/bench/benchmark.yaml`, version 2)

Version 2 (2026-10-27) changed how cross-validated tasks score validation, the number readouts are chosen on: each
fold's validation subjects are split in two halves (stratified by class, each class in both), and a head is tuned /
early-stopped on one half and scored on the other (`brain.downstream.cv.val_halves`). In v1 the token probe
early-stopped on the scans it was then scored on, so its CV validation AUROC ran 0.17 above test (the linear heads'
0.03), and the v1 pins leaned towards it. Test predictions of the linear heads are unchanged; the token probe's seeds
now each stop on one half. Fixed-split tasks are as in v1.

- **Tasks**: 22 frozen transfer tasks in 5 groups (CT, stroke, neuro-psychiatric, age, lesion / tumour), each tagged
  with its modality, chosen because they tell models apart (rule in the file); 4 segmentation transfer tasks.
- **Splits**: the store index's, pinned by its sha1. Results computed on another index (older splits) are left out of
  comparisons and counted.
- **Labels for in-domain and few-shot**: one label table (`labels_dir`, labels v2) for every model, whatever table a
  checkpoint trained on.
- **Readout**: per task, one head for every model, pinned in `configs/bench/readouts_v2.yaml` (`python -m brain.bench
  pin`): the head with the best mean validation score over the models with complete results when the version was
  pinned (BQ-R6, BQ-v3, BQ-A1), never chosen on test. Pinned, so a model's score does not move when other models are
  evaluated. A model without that head (a global-only encoder) uses its own best head on validation.
- **Statistics**: skill (below), group-first averaging, paired subject-level bootstrap against the reference (BQ-R6).

## Who is compared with whom

| Class | Learned from | Examples | Role |
|---|---|---|---|
| `ssl` | images only | BQ-v3; BrainIAC, Brain2Vec, CT-FM | the main comparison |
| `supervised` | labels of other tasks | BQ-R6, BQ-A1, BQ-T1; SFCN, BrainFM | compared on tasks whose labels it did not train on |
| `specialist` | the evaluated task itself | BQ-S-hemorrhage; nnU-Net | a per-task ceiling |
| `baseline` | nothing | random init; `regional_stats` | the floor |

Three rules: a task whose labels a model trained on is left out of its comparisons (**seen labels**; BQ: the
`seen` map in the spec against the tasks the checkpoint trained; published: their `seen_tasks`); a model is scored only
on tasks of the modalities it takes (**modality**; one leaderboard table per modality); and the share of a task's scans
a model may have seen in pretraining is shown next to its score (**overlap**). Each model is paired with the reference
on the tasks and scans both have.

## Every number recorded for one model

`python -m brain.bench card --model <name>` prints them, section by section.

**Identity**: the name, the records id `<run>/<file>@<fingerprint>` (the fingerprint is a hash of the weights file:
retrained weights are a new model), the training class, modalities, how many registry tasks a BQ checkpoint trained on
and which benchmark tasks that makes in-domain.

**1. Transfer (frozen encoder), any model.** The encoder is frozen; a small readout is trained on each task's training
split. Per task:

| Column | Meaning |
|---|---|
| Metric | AUROC (`auc`; multilabel tasks the macro one-vs-rest AUROC, `auc_ovr_macro`) for classification, R² (`r2`) for regression |
| Readout | the head: `cls_logreg` / `cls_ridge` (linear on the CLS embedding), `plane_*` / `grid_*` / `pca_global_*` (linear on PCA-reduced token summaries: per axial slab, 4 x 4 x 4 bins, whole volume), `tok_conv` (a small 3D conv probe on the full token grid, 5 seeds); `(CV)`: the task is too small for a fixed split, so it is scored by subject-grouped 5-fold cross-validation repeated 3 times, each fold's validation set holding every class twice, its score cross-fitted (version 2; v1 results on CV tasks, some with folds that lacked a class and fitted untuned, are not paired with v2 ones) |
| Test [95% CI] | the metric on the test split, from the saved per-scan predictions (CV: on each repeat's pooled out-of-fold predictions, mean over repeats); the interval is a subject-level bootstrap, the same resampling the leaderboard pairs models with. `(CV, 1st rep.)`: a result from before 2026-10-27, which saved predictions for the first of its 3 repeats only, so that repeat is what is scored and compared |
| Val | the metric on the validation split (CV: pooled cross-fitted inner-validation predictions, each from a head not tuned on that scan); what the readout was chosen on |
| Skill | the metric on a common scale, 0 = chance, 1 = perfect: AUROC as (AUC − 0.5) / 0.5, R² clipped at 0, balanced accuracy rescaled from chance, Dice as is |
| n test | test scans |
| Overlap | the fraction of the task's scans the model may have seen in pretraining (BQ: in the SSL pool; published: in the dataset families it declares) |

Then the skill per group (mean over its tasks) and the **headline** per modality: the mean over groups of the group
means, so a group with many tasks counts once. Behind each task the records also keep accuracy at 0.5, class counts,
Pearson r and MAE (regression), per-label AUROC (multilabel), per-cohort metrics, the readout's chosen regularisation,
and every per-scan prediction (`runs/downstream/<task>/<model>/<run>/predictions/`).

**2. Segmentation transfer, BQ checkpoints.** A 3D decoder trained on the frozen tokens; Dice per task (macro over
the task's regions) on test, on 64 axial slices (`dice_macro`) and on the whole volume (`dice_full_macro`), with
per-region Dice and per-volume Dice in the records.

**3. Token probes (`probe3d`), BQ checkpoints.** What single tokens carry, against pixel-statistics and random-init
controls: lesion detection per held-out lesion family (`ap`, average precision, and `lift`, AP over the lesion
prevalence; `auroc`), the scanner / dataset fingerprint (`dataset_bal_acc`: how well tokens identify the source
dataset; lower means less site shortcut), location (`r2_mean`: how well a token's position is decodable).

**4. In-domain, BQ models with task heads.** The model's own heads on the test splits of the tasks it trained (tasks
with at least 20 test scans): the score with its interval and the paired difference to the reference on the same scans
(Δ with a subject-bootstrap interval). This is what multitask training optimised; it is never mixed into transfer.

**5. Adaptation, BQ models with task heads.** Few-shot on the held-out tasks: only the new task's embedding row is
fitted, on k = 64 labelled scans, five seeds (mean ± sd), next to a linear probe on the frozen pooled embedding with
the same scans (a probe beating the embedding fit means the information is in the encoder but the task readout does
not reach it).

**On the leaderboard** (`board_v2.md`; every task x every model in `board_v2_all_tasks.md`) each model gets, per modality: the tasks it shares with the reference, its skill on them, Δ vs
the reference with the paired 95% interval (`*` when it excludes 0), the mean overlap, how many tasks were left out
(and why, in `_left_out.csv`), and how many results were on another index.

## Published models and our preprocessing

Every stored volume is the scan affinely registered to MNI152 (1 mm), cropped to 256 x 256 x 160 mm, CT in HU and MRI
(N4-corrected) as a robust z-score, plus a SynthStrip brain mask. Each volume also keeps `raw_median` / `raw_mad` (the
z-score is undone exactly by `ExternalEncoder.raw_intensity`, apart from the background, set to -6), its native
spacing and slice thickness, and `reg_affine` plus the source path (`brain/prep/native.py` maps any stored array back
onto the original scan). So an eval-time adapter covers most published encoders:

- **MNI-registered, skull-stripped, intensity-normalised inputs** (BrainIAC, SFCN, Brain2Vec-style T1 models): crop /
  resample the stored volume, apply the brain mask, normalise as the model did. Our registration is close to theirs
  (affine to MNI), so this is the native input up to the template and interpolation.
- **Contrast-agnostic models** (BrainFM, SynthSeg-style): any stored contrast at 1 mm, as is.
- **CT models** (CT-FM, head-CT foundation models): HU are preserved; window and resample as the model does. The
  geometry differs (registered and size-normalised, thick slices interpolated to 1 mm) from the native scans these
  models were trained on: a mild domain shift to report, not a blocker.
- **2D models** (DINOv2): axial slices of the stored volume.

What the store cannot give an adapter: the native resolution and field of view (anything outside the 256 x 256 x
160 mm box, the original thick-slice sampling), raw MRI background, and multi-sequence inputs stacked the way a
model's own pipeline aligned them (sequences are registered independently). A model that needs those runs its own
preprocessing on the source scans (`native_source`, the raw archives on pool); that is a separate path, used only if
a model's result looks limited by our preprocessing.

## Adding a published model

1. Write an adapter in `brain/downstream/adapters/`: subclass `brain.downstream.models.ExternalEncoder`, set `dim`
   (and `needs_brain = True` to receive the brain mask), implement `encode(vol_xyz, modality, brain, attrs)` returning
   `pooled` (the global feature the authors publish) and, if it has spatial features, `tokens` / `xyz` / `keep`.
   The base class does the geometry: `resample` puts the stored volume on the model's own grid (any orientation,
   spacing and field of view, given as a voxel-to-RAS affine), and `to_token_grid` samples the model's feature map
   at our 20 x 16 x 16 token centres, so its tokens take the same readouts as ours (`tok_conv`, plane / grid PCA).
2. Put the weights in `weights/<name>/` (git-ignored), as safetensors: never unpickle a third-party checkpoint.
3. Add its entry to `configs/bench/models.yaml` (adapter, class, modalities, weights, pretraining dataset families,
   tasks whose labels it trained on, reference and licence). Candidates surveyed on 2026-10-27 are listed there.
4. `python -m brain.bench run --model <name> --submit`, then `python -m brain.bench board` and `card`.

Worked examples (`brain/downstream/adapters/`):

| Name | Model | Class | Input it gets | Global feature | Tokens |
|---|---|---|---|---|---|
| `ct_fm` | CT-FM (SegResNet encoder, 77M; SimCLR on 148k IDC CTs; Apache 2.0) | ssl, CT | SPL, 3 x 1 x 1 mm (its pretraining spacing), HU -1024..2048 -> [0, 1], the whole 192 x 256 x 256 mm box | last stage average-pooled, 512-d | last stage (48 x 16 x 16 mm) |
| `brainiac` | BrainIAC (3D ViT-B, SimCLR on ~32k MRIs; non-commercial) | ssl, MRI | its 1 mm head template grid, our brain mask, resized to 96^3, z-scored over brain voxels | first output token (theirs; no CLS), 768-d | 6^3 patches (~28 x 34 x 27 mm) |
| `regional_stats` | brain-mask intensity statistics, whole brain and 32 mm blocks | baseline | the stored volume | the statistics | none |
| `BQ-random` | the BQ-v3 architecture, untrained | baseline | as BQ-v3 | CLS | ours |

## Not covered (yet)

- Fine-tuning protocols (all transfer here is frozen); the FOMO challenge's evaluation is the natural external one.
- Segmentation transfer and token probes for published models: they still read our encoder directly, not the
  adapter's tokens. A random-init model needs neither: frozen transfer is its comparison (the token probes already
  carry a random-init control).
