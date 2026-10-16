#!/bin/bash
# Generic CPU job for the v4 preprocessing package.
# Submit: sbatch --export=ALL,CMD="-m data.prep.run --root STORE --shard 0/8" [-c 16 --mem 64G] jobs/prep_job.sbatch

#SBATCH -p mit_normal
#SBATCH --job-name=prep
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/prep_%j.out
#SBATCH -e logs/prep_%j.out

set -euo pipefail
cd /home/roku/brain-model
mkdir -p logs
unset PYTHONPATH
export PYTHONNOUSERSITE=True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/roku/brain-model:/home/roku/.local/lib/python3.12/site-packages
/home/roku/.conda/envs/ml-torch/bin/python -u ${CMD:?}
