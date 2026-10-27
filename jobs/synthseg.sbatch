#!/bin/bash
# SynthSeg-robust 2.0 labels (brain.prep.synthseg_run) for labels/synthseg/select.csv, one GPU per array element.
#
#   python -m brain.prep.synthseg_run --select --n 30000     # once
#   for s in 0 1 2 3; do sbatch -J synthseg-$s --export=ALL,SHARD=$s jobs/synthseg.sbatch; done   # N_SHARDS default 4
#   N_SHARDS=1 EXTRA_ARGS="--redo-families fomo_infant" sbatch --export=ALL jobs/synthseg.sbatch      # re-prepped family

#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --exclude=node3006          # broken CUDA install (torch sees no GPU)
#SBATCH --job-name=synthseg
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 6
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/synthseg_%j.out
#SBATCH -e logs/synthseg_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
export OMP_NUM_THREADS=4
"$PY" -u -m brain.prep.synthseg_run --shard "${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}" --n-shards "${N_SHARDS:-4}" --batch 1 ${EXTRA_ARGS:-}
