#!/bin/bash
# CQ500 original DICOMs (brain.datasets.download_cq500): one plain soft-tissue series per study onto pool.
#   sbatch jobs/download_cq500.sbatch
#SBATCH -p mit_data_transfer
#SBATCH --job-name=dl-cq500
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/download_cq500_%j.out
#SBATCH -e logs/download_cq500_%j.out
set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
echo "[dl-cq500] $(date -Is) host=$(hostname) free tmp: $(df -h "$TMPDIR" | tail -1)"
"$PY" -u -m brain.datasets.download_cq500 --dest "${BRAIN_DATASETS:-/home/roku/orcd/pool/datasets}/cq500_dicom" --workers 6
echo "[dl-cq500] done $(date -Is)"
