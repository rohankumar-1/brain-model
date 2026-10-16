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
cd /home/roku/brain-model
mkdir -p logs
unset PYTHONPATH
export PYTHONNOUSERSITE=True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/roku/brain-model:/home/roku/.local/lib/python3.12/site-packages
/home/roku/.conda/envs/ml-torch/bin/python -u -m data.download_sinoct --overlap --pull --workers "${SLURM_CPUS_PER_TASK:-16}"
