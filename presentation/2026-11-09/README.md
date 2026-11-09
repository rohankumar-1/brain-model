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
     .918 vs .922, CN vs MCI .757 vs .757, UCSF IDH .915 vs .901) and clearly beats a random-init encoder (e.g.
     CN vs MCI .757 vs .641, MMSE .375 vs .198, fracture .854 vs .627). With an attention readout picked on
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

Model names: **new run** = `full_iso_bank` (R14-F + 30k steps with the learned window bank on the re-encoded store,
1 mm grid); **R14-F** = the previous best run; **R16** = the stopped predecessor recipe. Published models run through
our benchmark: CT-FM, BrainIAC, BrainFM, HLIP.
