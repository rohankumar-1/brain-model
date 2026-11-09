# BrainQuery: meeting of 9 November 2026, presentation materials

Everything needed to build the slide deck, self-contained (no cluster access needed). Numbers come from the project's
evaluation outputs on 9 November 2026; the generators live in `analysis/presentation/` and `analysis/bench_page/`.

**BrainQuery** is one 3D model for brain MRI and CT: a transformer encoder that answers ~70 clinical and anatomical tasks
at once (one learned token per task), trained on ~74k scans from 17 cohorts and tested on held-out splits plus 19
external cohorts it never trained on.

## The story (suggested deck order)

1. **Thesis.** One model, trained once on many tasks, transfers to new sites and new tasks without retraining, and is
   competitive with models built for each task.
2. **Data.** `figures/datasets_map_slide.png`: training scans (left) vs test scans (right) per cohort, coloured by task
   type; hatched = external cohorts. 73,821 training scans in 17 cohorts; 14,158 held-out test scans; 3,834 scans in
   19 external cohorts. The store holds 129,120 scans (99,744 MRI, 29,376 CT), 57 dataset families.
3. **Model and training.** `figures/training_diagram.png` (CT / MRI → learned window bank → patch embedding →
   transformer with one token per task → heads shared per output type) and `figures/training_tasks_slide.png` (all
   68 trained tasks in 13 groups, coloured by output type: binary 20, regression 18, multiclass 4, segmentation 20,
   dense / self-supervised 6).
4. **Evaluation, three kinds** (`eval_tables.md`, or `figures/eval_t1.png`, `eval_t2.png`, `eval_t3.png`):
   - *In-training*: trained task, held-out test split (e.g. RSNA ICH).
   - *External*: trained task, unseen cohort, no labels from it (CQ500 read by the RSNA heads).
   - *Transfer*: task never trained; frozen encoder + small readout (ADNI, PPMI, UCSF IDH, CQ500 fracture).
5. **Headline evidence** (strongest first):
   - *Generalisation to a new site without its labels*: RSNA's hemorrhage heads read CQ500 unchanged at any bleed .917,
     intraventricular .981 (above a supervised model trained on ~290k labelled CTs: .931), subarachnoid .935; the
     epidural head loses nothing across sites (RSNA test .878, CQ500 .882). Zero-shot beat every readout fitted on
     CQ500's own labels.
   - *Transfer to unseen tasks*: the frozen encoder matches the best published frozen foundation models (ADNI CN vs AD
     .918 vs .922, CN vs MCI .768 vs .757, UCSF IDH .915 vs .901) and clearly beats a random-init encoder (e.g.
     CN vs MCI .768 vs .641, MMSE .375 vs .198, fracture .854 vs .627). With an attention readout picked on
     validation, CQ500's 8-finding macro AUC rose from .886 to .914 (fracture .759 → .854, SAH .869 → .940).
   - *Multitask beats single-task where data is scarce*: enhancing tumour Dice .656 vs .331, whole tumour .645 vs
     .542, age R² .922 vs .749, sex AUC .935 vs .590, 12-month survival AUC .743 vs .603.
6. **Gap and plan.** Behind supervised models on fine-grained CT (epidural .855 vs .984; calvarial fracture .854 vs
   .962) and on segmentation (whole tumour .645 vs .908); single-task specialists still win RSNA subdural / epidural.
   Cause: everything is resampled to 1 mm (CT loses half its in-plane resolution) and rare findings get little exposure
   in the multitask mix. Already done this weekend: store re-encoded (int16, ~340 GB freed), native-resolution CT copy of
   29,374 scans, learned window bank in the model; the first window-bank run beats R16 on validation (.791 vs .770,
   `data/validation_new_run_vs_r16.csv`); the native-CT model is training now.
7. **Closing line.** "BrainQuery reads unseen hospitals' scans at near-supervised level for tasks it trained on, without
   their labels, transfers to new tasks as well as the best published foundation models, and the remaining gap is
   resolution, which we are now training against."

## Extra slides: what the model does and how it fits (all on the new run, held-out test scans)

Suggested placement: after the headline evidence (5), before gap and plan (6). Strongest first.

- `figures/extra/cq500_fewshot.png`: **"Reading a new hospital's CTs needs no labels from it."** The RSNA heads read
  CQ500 zero-shot as well as readouts fitted with 8 labelled CQ500 positives (any bleed .908 vs .890, subarachnoid
  .931 vs .877), and better with fewer.
- `figures/extra/cq500_roc.png`: ROC curves of that zero-shot read, per finding, beside the supervised CQ500 reference.
- `figures/audit/seg_examples.png`: reference vs predicted masks on typical test scans (the scans nearest each task's
  median Dice, not the best). Median Dice: hemorrhage .76, whole tumour .73, enhancing tumour .85. These are medians
  over 30-40 scans; the eval tables report means over all test scans, which a few complete misses pull down.
- `figures/audit/anatomy.png`: every anatomy head at once on one MRI and one CT test scan (no reference shown: these
  targets are SynthSeg pseudo-labels).
- `figures/extra/mt_vs_single.png`: **one model matches the specialists where they do well and rescues them where
  they fail.** Per trained task, the new run's own head vs the best from-scratch single-task model (57 tasks): within
  ±0.03 on 26 (anatomy masks, hemorrhage types), ahead by > 0.03 on 21 (tumour sub-regions +0.28-0.32, sex, diagnoses,
  survival, tumour type, age), behind by > 0.03 on 10 (CT density / CSF fraction, MRI edema, artifact and mass,
  MGMT, epidural). Gain does not depend much on labelled-data size (median +0.01 below 5,000 scans, 0.00 above);
  it is largest where single-task training struggles. Caveat: specialists were scored in earlier runs; test scans come
  from the same test splits but label tables may differ slightly.
- `figures/extra/regression.png`: predicted vs true age (R² .91, mean error 4.9 years, newborns to 90) and normalised
  brain volume (R² .96) on 1,000 pooled CT + MRI test scans.
- `figures/audit/attention_maps.png` and `attention_lesion.png`: **where task tokens look.** Hemorrhage and tumour
  tokens put 6-8x their fair share of attention on CT bleeds and 2-6x on gliomas; age, brain-volume and ventricle
  tokens about 1x (spread over the brain). Honest reading: lesion tokens find *any* lesion (the tumour token also looks
  at bleeds); they were never trained to ignore the other kind.
- `figures/audit/task_overlap_groups_last.png` (and the full `task_overlap_last.png`): which task groups attend to the
  same places. Lesion-type groups (hemorrhage, lesions, tumour, radiology, diagnosis, clinical scores) share attention
  (.45-.69); anatomy, volumes, asymmetry and CT measures overlap more with each other than with lesion groups;
  demographics overlap little with anything. This matches the earlier gradient-conflict finding (local lesion tasks vs
  global anatomy tasks). Caveat: last-layer attention is concentrated on few image tokens, so some unrelated tasks
  (e.g. age and CT skull density) have near-identical maps.
- `figures/audit/fit_by_data.png`: **data-rich tasks are not memorised; small cohorts are.** Same heads on training vs
  test scans: median train - test gap +0.02 for the 39 tasks with >= 5,000 labelled scans, +0.10 for the 21 with
  < 2,000 (MGMT, survival, neuro diagnosis, MMSE grade near 1.0 on training scans). Message: for the main tasks more
  compute / resolution should still pay; for rare tasks the limit is data. (It does *not* show uniform underfitting.)

Native-CT model (finished 14:10, not in the headline): CQ500 zero-shot below the 1 mm model on every finding (any bleed
.886 vs .908, subdural .841 vs .915); its new stem needs a warm-up before a fair comparison.

## Caveats to keep on the slides or in speaker notes

- Literature values use their own splits and often more labels or sequences; starred (\*) values are supervised or
  fine-tuned upper references, unstarred ones are like-for-like frozen comparisons.
- Several test sets are small (±0.05-0.10 AUC intervals); one training seed per model.
- PPMI (Parkinson's), MGMT and stroke outcome are near chance for every model (random init even wins PPMI); they say
  little about representation quality.
- "Ours" is the best of our two models (the new run and R14-F) per row; readouts are always picked on validation,
  never on test; the zero-shot CQ500 numbers involve no choice at all.

## Files

| Path | What |
|---|---|
| `eval_tables.md` | The three evaluation tables (single-task, literature, ours), with sources |
| `figures/datasets_map_slide.png` | Dataset map, 16:9 |
| `figures/training_diagram.png` | How the model is trained, 16:9 |
| `figures/training_tasks_slide.png` | All 68 training tasks by group and output type, 16:9 |
| `figures/eval_t1.png`, `eval_t2.png`, `eval_t3.png` | The three tables as slides, 16:9 |
| `pages/results.html` | Full results page (datasets, evaluation types, frozen benchmark vs published models, own heads vs single-task, CQ500, literature) |
| `pages/literature.html` | Per-task comparison with published results, with references |
| `pages/dataset_map.html` | Interactive dataset map (log / linear scale) |
| `data/frozen_benchmark_scores.json` | Frozen benchmark v3: per task and model, score, 95% interval, n |
| `data/frozen_readouts_all_heads.csv` | Every readout (linear, MLP, octants, token conv, attention) per task: validation and test score; `picked_on_val` marks the reported one |
| `data/own_heads_test_new_run_vs_r14f.csv` | Own-head test scores, new run vs R14-F on the same scans |
| `data/cq500_from_rsna_heads_*.csv` | CQ500 read by the RSNA heads: zero-shot and few-shot arms |
| `data/validation_new_run_vs_r16.csv` | Validation score per task group, new run vs R16 |
| `data/datasets.json` | Per-cohort train / test / external scan counts and totals |
| `figures/audit/*.png`, `figures/extra/*.png` | Model-audit and extra result slides, 16:9 (see "Extra slides") |
| `data/train_vs_test_fit.csv` | Per task: own-head score on training vs test scans, labelled training scans |
| `data/task_attention_overlap.csv` | Task-token x task-token attention-map similarity (header = token's first task) |

Model names: **new run** = `full_iso_bank` (R14-F + 30k steps with the learned window bank on the re-encoded store,
1 mm grid); **R14-F** = the previous best run; **R16** = the stopped predecessor recipe. Published models run through
our benchmark: CT-FM, BrainIAC, BrainFM, HLIP.
