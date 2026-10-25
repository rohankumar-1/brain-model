#!/bin/bash
# After the v6 index: delete what the format-6 store no longer uses (stale v5 zips, .brain.zip files, old
# manifests, retired folders incl. sinoct/) and the v3 split backup (splits now live in volumes.csv / volumes_v5.csv).
#
#   sbatch --dependency=afterok:<index-v6> jobs/store_cleanup_v6.sbatch

#SBATCH -p mit_normal
#SBATCH --job-name=store-cleanup
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o logs/store_cleanup_%j.out
#SBATCH -e logs/store_cleanup_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
SCR="${BRAIN_SCRATCH:-/home/roku/orcd/scratch/brain-model}"
STORE="${STORE:-$SCR/data/processed/slices_zarr}"
echo "[store-cleanup] $(date -Is)"
"$PY" -u -m brain.prep.store_cleanup --root "$STORE" --workers "${SLURM_CPUS_PER_TASK:-8}" --yes
rm -rf "$SCR/data/processed/v3_index_backup" "$SCR/data/processed/smoke_v6"
sed -n 1,6p ~/orcd/.quota
echo "[store-cleanup] done $(date -Is)"
