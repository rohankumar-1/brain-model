#!/bin/bash
# The real 5 mm plain series of CQ500 studies whose regular pick is thin (brain.datasets.download_cq500 --thick), for
# the ngeo copy: one stored zip per study in pool cq500_dicom_thick (~16 MB each; 384 studies ~6 GB; each study zip
# ~165 MB is downloaded to node-local disk and deleted). STUDIES=0-490 (default) or a comma list.
#   STUDIES=0,108,113 sbatch --export=ALL jobs/archive/download_cq500_thick.sbatch
#SBATCH -p mit_data_transfer
#SBATCH --job-name=dl-cq500-thick
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -o logs/download_cq500_thick_%j.out
#SBATCH -e logs/download_cq500_thick_%j.out
set -euo pipefail
export BRAIN_REPO="${BRAIN_REPO:-/home/roku/brain-model}"
source "$BRAIN_REPO/jobs/_env.sh"
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
echo "[dl-cq500-thick] $(date -Is) host=$(hostname) free tmp: $(df -h "$TMPDIR" | tail -1)"
"$PY" -u -m brain.datasets.download_cq500 --thick --dest "${BRAIN_DATASETS:-/home/roku/orcd/pool/datasets}/cq500_dicom_thick" \
  --workers "${WORKERS:-4}" --studies "${STUDIES:-0-490}"
echo "[dl-cq500-thick] done $(date -Is)"
