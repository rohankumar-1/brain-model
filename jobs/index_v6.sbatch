#!/bin/bash
# Rebuild volumes.csv / slices from the format-6 manifests only. Subject splits are inherited from the v5 index
# (copied once to volumes_v5.csv) so train/val/test stay comparable across the grid change.
#
#   sbatch --dependency=afterok:<A>:<B>:<C> jobs/index_v6.sbatch
# CPU only and quick: --no-slices skips the 2D slice table (only the legacy 2D trainers read it).

#SBATCH -p mit_normal
#SBATCH --job-name=index-v6
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/index_v6_%j.out
#SBATCH -e logs/index_v6_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
export BRAIN_FOMO="${BRAIN_DATASETS:-/home/roku/orcd/pool/datasets}/fomo260k"   # holdout lists (scratch copy is gone)
[[ -f "$STORE/volumes_v5.csv" ]] || cp -p "$STORE/volumes.csv" "$STORE/volumes_v5.csv"
echo "[index-v6] $(date -Is) v6 manifests: $(ls "$STORE"/manifest_v6_*.jsonl | wc -l)"
"$PY" -u -m brain.prep.index --root "$STORE" --workers "${SLURM_CPUS_PER_TASK:-16}" \
  --manifests "manifest_v6_*.jsonl" --inherit-splits "$STORE/volumes_v5.csv" --no-slices
echo "[index-v6] done $(date -Is)"
