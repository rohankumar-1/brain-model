#!/bin/bash
# v4 preprocessing pilot on one small dataset (CPU only).
# Submit: sbatch --export=ALL,DATASET=cq500,N=0,ZA=volume jobs/prep_pilot.sbatch     (N=0: all volumes)

#SBATCH -p mit_normal
#SBATCH --job-name=prep-pilot
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o logs/prep_pilot_%j.out
#SBATCH -e logs/prep_pilot_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
"$PY" -u -m brain.prep.pilot \
  --dataset "${DATASET:?}" --n "${N:-0}" --z-anchor "${ZA:-volume}" --filter "${FILTER:-}" --workers "${SLURM_CPUS_PER_TASK:-8}"
