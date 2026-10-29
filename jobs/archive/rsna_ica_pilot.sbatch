#!/bin/bash
# Pilot: 30 RSNA-ICA series per modality (half with an aneurysm; runs/diagnostics/rsna_ica_pilot/cases.txt) prepped
# into a separate pilot store with the dcm2niix reader (enhanced multi-frame MR), then QC'd (rsna_ica_pilot_qc.py).
#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --exclude=node3006          # broken CUDA install (torch sees no GPU)
#SBATCH --job-name=rsna-ica-pilot
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/rsna_ica_pilot_%j.out
#SBATCH -e logs/rsna_ica_pilot_%j.out
set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
D="$BRAIN_REPO/runs/diagnostics/rsna_ica_pilot"
export BRAIN_DICOM_READER=dcm2niix OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u -m brain.prep.run --root "$D/store" --datasets rsna_ica --cases "$D/cases.txt" --workers 11 \
  --manifest manifest_pilot.jsonl --brain-mask auto --gpu-mask --mask-batch 4 --reg-backend "${REG_BACKEND:-sitk}" \
  --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN" ${EXTRA:-}
