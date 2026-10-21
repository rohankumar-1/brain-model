#!/bin/bash
# Rebuild one archive from its JPEG-XL chunks, verify, replace the original (brain.prep.transcode merge).
#SBATCH -p mit_data_transfer
#SBATCH --job-name=transcode-merge
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -o logs/transcode_merge_%j.out
#SBATCH -e logs/transcode_merge_%j.out
set -euo pipefail
cd "${BRAIN_REPO:-/home/roku/brain-model}"
source jobs/_env.sh
"$PY" -u -m brain.prep.transcode merge "$1"
