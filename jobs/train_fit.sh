#!/bin/bash
# Training-set fit (2026-11-05): each family's best from-scratch specialist and R14 scored on their own training scans
# (test_eval --split train, heads only, up to 400 per task) and on the test split, labels v10.
#   sbatch -p pi_dbertsim,mit_normal_gpu --gres=gpu:l40s:1 -c 12 --mem=96G -t 04:00:00 -J train-fit -o runs/diagnostics/train_fit_%j.log jobs/train_fit.sh
set -uo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
L=/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr/labels/multitask_v10
for spec in hemorrhage:hemorrhage_scratch_hybrid tumor:tumor_scratch_vit demographics:demographics_scratch_resnet \
            anatomy:anatomy_scratch_resnet ct_measures:ct_measures_scratch_vit radiology:radiology_scratch_resnet; do
  fam=${spec%%:*}; run=${spec#*:}
  T=$("$PY" -c "import yaml; print(' '.join(yaml.safe_load(open('configs/single/$run.yaml'))['tasks']['ids']))" 2>/dev/null | tail -1)
  for split in train test; do
    "$PY" -u -m brain.multitask.test_eval --ckpts runs/single/$run/last.pt runs/multitask/r14/last.pt --tasks $T \
      --split $split --labels-dir $L --workers 10 --out runs/diagnostics/train_fit/${fam}_${split} 2>&1 | grep -E "^\[heads\]|Error"
  done
done
