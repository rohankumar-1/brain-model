#!/bin/bash
# Pilot: RSNA studies listed in runs/diagnostics/rsna_pilot/cases.txt re-prepped with the dcm2niix reader into a
# separate pilot store, then SynthSeg QC compared with the main store (runs/diagnostics/rsna_pilot_compare.py).
#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --job-name=rsna-pilot
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/rsna_pilot_%j.out
#SBATCH -e logs/rsna_pilot_%j.out
set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
D="$BRAIN_REPO/runs/diagnostics/rsna_pilot"
export BRAIN_DICOM_READER=dcm2niix OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u -m brain.prep.run --root "$D/store" --datasets rsna --cases "$D/cases.txt" --workers 11 \
  --manifest manifest_pilot.jsonl --brain-mask auto --gpu-mask --mask-batch 4 --reg-backend "${REG_BACKEND:-sitk}" \
  --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN"
"$PY" -u runs/diagnostics/rsna_pilot_compare.py
