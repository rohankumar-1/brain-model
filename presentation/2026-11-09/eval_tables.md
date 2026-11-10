# BrainQuery evaluation, 9 November 2026

**Ours** = best of our two models (the new full-suite run and R14-F); ᵃ marks R14-F, otherwise the new run. **Literature**: \* = supervised or fine-tuned published model, usually on its own split; unstarred = frozen or linear-probe foundation model. – = no comparable value.

## 1. In-training evaluation: trained task, held-out test split of a training cohort

Own trained heads.

| Task | Metric | Single-task | Literature | Ours |
|---|---|---|---|---|
| RSNA ICH · any bleed | AUC | .963 | .967\* | **.964**ᵃ |
| RSNA ICH · subdural | AUC | .950 | .983\* | **.931**ᵃ |
| RSNA ICH · epidural | AUC | .885 | .984\* | **.855** |
| BraTS · whole tumour (one sequence) | Dice | .542 | .908\* | **.645** |
| BraTS · enhancing tumour (T1c) | Dice | .331 | .737\* | **.656** |
| UPenn-GBM · 12-month survival | AUC | .603 | .72\* | **.743** |
| Age (pooled cohorts) | R² | .749 | – | **.922**ᵃ |
| Sex | AUC | .590 | – | **.935** |

Single-task: best of three from-scratch architectures per task family. \* supervised or fine-tuned published model on its own split (Pillar-0; RSNA 2019 winner; nnU-Net and missing-modality models on T1c alone; BrainIAC).

## 2. External evaluation: trained task, unseen cohort, no labels from it

CQ500 (471 scans) read by the RSNA heads.

| Task | Metric | Single-task | Literature | Ours |
|---|---|---|---|---|
| CQ500 · any bleed | AUC | – | .942\* | **.917**ᵃ |
| CQ500 · intraparenchymal | AUC | – | .954\* | **.903**ᵃ |
| CQ500 · intraventricular | AUC | – | .931\* | **.981** |
| CQ500 · subdural | AUC | – | .952\* | **.915** |
| CQ500 · subarachnoid | AUC | – | .957\* | **.935**ᵃ |
| CQ500 · epidural | AUC | – | .973\* | **.882**ᵃ |

Single-task specialist not yet scored on CQ500. \* Chilamkurthy et al. 2018, supervised on ~290k head CTs, CQ500 as its external test.

## 3. Transfer: task never trained, frozen encoder + small readout fitted on the task

Frozen benchmark.

| Task | Metric | Single-task | Literature | Ours |
|---|---|---|---|---|
| ADNI · CN vs AD | AUC | – | .922 | **.921**ᵃ |
| ADNI · CN vs MCI | AUC | – | .757 | **.768** |
| ADNI · MMSE | R² | – | .43 | **.375**ᵃ |
| PPMI · Parkinson's vs control | AUC | – | .610 | **.554** |
| UCSF · IDH mutation | AUC | – | .901 | **.922** |
| CQ500 · calvarial fracture | AUC | – | .962\* | **.854** |
| CQ500 · midline shift | AUC | – | .970\* | **.944**ᵃ |
| ISLES 2024 · LVO side (CTA) | AUC | – | – | **.955**ᵃ |
| MS3SEG · MS lesion load | R² | – | – | **.757**ᵃ |

Single-task models exist only for trained tasks. Literature: the best published frozen / linear-probe foundation model; \* supervised model (no frozen result published). Ours: the readout picked on validation.
