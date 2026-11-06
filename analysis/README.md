# Analysis scripts

One-off checks and studies whose results the code, configs or docs cite as evidence, plus a few reusable tools. They
import `brain` but are not part of the package: run them from the repo root with the project environment
(`source jobs/_env.sh; $PY analysis/<script>.py ...`). Outputs go to `runs/diagnostics/` (scratch, not versioned).
Moved here from the unversioned `runs/diagnostics/` on 2026-11-06; 39 dead one-offs were retired then.

## Benchmark pages

`bench_page/` builds the two result pages (task scores, headline): `collect_scores.py scores.json` (model x task scores
with 95% intervals from the benchmark results DB, pinned v3 readouts), then `build_page.py scores.json OUT.html` and
`build_focus.py scores.json OUT.html`.

## Evidence cited elsewhere

| script | what it shows |
|---|---|
| `cq500_side_verify.py` | CQ500 orientation: the image side of one-sided bleeds matches the reads (benchmark.yaml) |
| `tumor_label_semantics.py` | tumour label values mean the same thing in every cohort (prep pipeline) |
| `mirrored_scan.py` | translation-fallback volumes were stored mirrored by the old reader (docs/prep.md) |
| `fomo_small_qc.py` | visual QC of the small-FOMO registrations (docs/prep.md) |
| `n4_check.py` | bias-field correction is applied consistently per family |
| `synthseg_check.py` | the SynthSeg port matches the original Keras model (needs the retired TF environment to re-run) |
| `synthseg_ct_reliability.py` | test-retest agreement of SynthSeg measures on CT (multitask CT anatomy targets) |
| `synthrad_ct_synthseg.py` | CT vs MR SynthSeg on the same SynthRAD heads (registry CT anatomy rules) |
| `wmh_pilot.py`, `wmh_validate.py` | WMH-SynthSeg speed / volumes, and its agreement with expert WMH masks (labels) |
| `midline_cq500.py`, `midline_healthy_viz.py` | midline-shift measure validation (docs/multitask.md) |
| `aneurysm_finder.py` | whether the aneurysm location map finds aneurysms (R11-R14 config headers) |

## Tools

| script | use |
|---|---|
| `profile_loader.py` | where a training item's CPU time goes (data loading profile) |
| `plot_a1_tasks.py` | validation curves of a multitask / joint run per group and task |
| `token_probe.py` | what each task token knows about a downstream label |
| `registration_qc.py`, `mask_gallery.py` | registration and stored-mask QC galleries across datasets |
| `grad_conflict.py` | gradient conflict between task groups (slot-readout checkpoints) |

## Checks for open items (delete when closed)

| script | item |
|---|---|
| `labels_v12_check.py` | labels v12 (CT Hounsfield masks, lesion-aware anatomy, T1c-only enhancing / core) |
| `pedi_ct_qc.py` | Pedi-Cranial-CT-Healthy prep QC |
| `r14f_check.py` | R14-F fused study input on real data |
| `aisd_nan.py` | AISD frozen-segmentation NaN (not yet explained) |
