#!/bin/bash
# Build the multitask label table (brain.multitask.labels): labels/multitask/{labels.parquet, stats.json, summary.txt}.
# Reads small label tables, ~4k BraTS segmentation zips (tumour volumes), ~1.5k volumes (live-target statistics) and
# every pool volume's brain mask once (registration QC, cached in labels/multitask/regqc.parquet).
#
#   sbatch jobs/multitask_labels.sbatch

#SBATCH -p mit_normal
#SBATCH --job-name=mt-labels
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/multitask_labels_%j.out
#SBATCH -e logs/multitask_labels_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
echo "[mt-labels] $(date -Is)"
"$PY" -u -c "from brain.downstream.tasks import sync_label_files; print(len(sync_label_files()), 'label files synced')"
"$PY" -u -m brain.multitask.labels --workers "${SLURM_CPUS_PER_TASK:-16}" --live-sample "${LIVE_SAMPLE:-1500}"
echo "[mt-labels] done $(date -Is)"
