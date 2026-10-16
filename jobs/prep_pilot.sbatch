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
cd /home/roku/brain-model
mkdir -p logs
unset PYTHONPATH
export PYTHONNOUSERSITE=True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/roku/brain-model:/home/roku/.local/lib/python3.12/site-packages
/home/roku/.conda/envs/ml-torch/bin/python -u -m data.prep.pilot \
  --dataset "${DATASET:?}" --n "${N:-0}" --z-anchor "${ZA:-volume}" --filter "${FILTER:-}" --workers "${SLURM_CPUS_PER_TASK:-8}"
