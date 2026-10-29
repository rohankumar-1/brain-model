#!/bin/bash
# WMH-SynthSeg burden (brain.prep.wmh_run) over non-tumour FLAIR / T2 pool volumes, one GPU per shard.
#
#   for s in 0 1; do sbatch -J wmh-$s --export=ALL,SHARD=$s,N_SHARDS=2 jobs/wmh.sbatch; done

#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --exclude=node3006          # broken CUDA install (torch sees no GPU)
#SBATCH --job-name=wmh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 6
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/wmh_%j.out
#SBATCH -e logs/wmh_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u -m brain.prep.wmh_run --shard "${SHARD:-0}" --n-shards "${N_SHARDS:-1}"
