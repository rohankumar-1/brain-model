#!/bin/bash
# Pixel-fingerprint RSNA, probe SinoCT middles, then pull keep reconstructed DICOMs.
# Submit: sbatch jobs/sinoct_overlap.sbatch

#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH --job-name=sinoct-fp
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/sinoct_overlap_%j.out
#SBATCH -e logs/sinoct_overlap_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
"$PY" -u -m brain.datasets.download_sinoct --overlap --pull --workers "${SLURM_CPUS_PER_TASK:-16}"
