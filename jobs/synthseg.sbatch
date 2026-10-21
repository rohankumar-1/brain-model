#!/bin/bash
# SynthSeg-robust 2.0 labels (brain.prep.synthseg_run) for labels/synthseg/select.csv, one GPU per array element.
#
#   python -m brain.prep.synthseg_run --select --n 30000     # once
#   sbatch --array=0-3 jobs/synthseg.sbatch                   # N_SHARDS = array size (default 4)

#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --job-name=synthseg
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 6
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/synthseg_%A_%a.out
#SBATCH -e logs/synthseg_%A_%a.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
export OMP_NUM_THREADS=4
"$PY" -u -m brain.prep.synthseg_run --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${N_SHARDS:-4}" --batch 1
