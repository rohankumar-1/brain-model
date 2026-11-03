#!/bin/bash
# Report-only content audit before the 2026-10-29 index rebuild (runs/diagnostics/content_audit.py): UPENN-GBM / UCSF-PDGM
# BraTS twins and evaluation-CT (CQ500, INSTANCE, AISD, ISLES24) copies anywhere in the store, incl. RSNA-ICA. Lists
# nothing for dropping and deletes nothing.
#
#   sbatch jobs/content_audit.sbatch
#SBATCH -p pi_dbertsim,mit_preemptable
#SBATCH --job-name=content-audit
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --exclude=node3006
#SBATCH -o logs/content_audit_%j.out
#SBATCH -e logs/content_audit_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[audit] $(date -Is) host=$(hostname)"
"$PY" -u "${SCRIPT:-runs/diagnostics/content_audit.py}"
sed -n 1,6p ~/orcd/.quota
echo "[audit] done $(date -Is)"
