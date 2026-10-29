#!/bin/bash
# N4 bias-field correction of stored MRI that prep did not correct (BraTS, MSD, VALDO, MS3SEG; brain.prep.n4_store).
# Rewrites those image zips in place (registration, grid and masks unchanged); resumable: corrected images carry an
# ``n4`` attribute and are skipped. CPU work; pi_dbertsim only takes GPU jobs (the GPU idles).
#
#   sbatch --array=0-3 jobs/n4_store.sbatch
#SBATCH -p pi_dbertsim,mit_preemptable
#SBATCH --job-name=n4-store
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=node3006
#SBATCH -o logs/n4_store_%A_%a.out
#SBATCH -e logs/n4_store_%A_%a.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
N="${SLURM_ARRAY_TASK_COUNT:-1}"
I="${SLURM_ARRAY_TASK_ID:-0}"
echo "[n4-store] $(date -Is) host=$(hostname) shard=$I/$N"
"$PY" -u -m brain.prep.n4_store --shard "$I/$N" --workers $(( ${SLURM_CPUS_PER_TASK:-12} - 1 ))
echo "[n4-store] done $(date -Is)"
