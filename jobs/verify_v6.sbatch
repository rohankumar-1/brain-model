#!/bin/bash
# Last step of the v6 build: check every indexed volume (format 6, stored transform, brain array, label masks)
# and write excluded.csv. Exit 1 lists problems in verify_problems.csv.
#
#   sbatch --dependency=afterok:<store-cleanup> jobs/verify_v6.sbatch
#SBATCH -p mit_normal
#SBATCH --job-name=verify-v6
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o logs/verify_v6_%j.out
#SBATCH -e logs/verify_v6_%j.out
set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
echo "[verify-v6] $(date -Is)"
"$PY" -u -m brain.prep.verify_store --root "$STORE" --workers "${SLURM_CPUS_PER_TASK:-4}"
echo "[verify-v6] done $(date -Is)"
