#!/bin/bash
# Build the segmentation pixel caches (4 MRI sequences per BraTS case) without a GPU.
# Submit: sbatch --export=ALL,TASKS=brats2023_gli:brats2023_men jobs/seg_cache.sbatch
# GPU eval jobs can wait on it with --dependency=afterok:<jobid>.

#SBATCH -p mit_normal
#SBATCH --job-name=seg-cache
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/segcache_%j.out
#SBATCH -e logs/segcache_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"

IFS=':' read -r -a TASK_LIST <<< "${TASKS:-brats2023_gli:brats2023_men}"
for t in "${TASK_LIST[@]}"; do
  echo "[cache-start] $t $(date -Is)"
  "$PY" -u -m brain.downstream.run_suite --task "$t" --cache-only
  echo "[cache-done] $t $(date -Is)"
done
