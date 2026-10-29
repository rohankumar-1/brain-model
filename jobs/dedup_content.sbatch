#!/bin/bash
# The dedup step of a prep round (brain.prep.dedup --content): fingerprint every stored volume (cached per family),
# find copies across subjects on the GPU, confirm by correlation, list them in labels/pool_drop/duplicates.csv and
# delete their store files. Run after the prep jobs and before the index:
#
#   sbatch --dependency=afterok:<prep jobs> jobs/dedup_content.sbatch              # whole store
#   QUERY=fomo_newfam sbatch --export=ALL --dependency=afterok:<prep> jobs/dedup_content.sbatch
#SBATCH -p pi_dbertsim,mit_preemptable
#SBATCH --job-name=dedup
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=node3006
#SBATCH -o logs/dedup_%j.out
#SBATCH -e logs/dedup_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[dedup] $(date -Is) host=$(hostname) query=${QUERY:-all}"
"$PY" -u -m brain.prep.dedup --content --query "${QUERY:-all}" --workers $(( ${SLURM_CPUS_PER_TASK:-12} - 1 ))
sed -n 1,6p ~/orcd/.quota
echo "[dedup] done $(date -Is)"
