#!/bin/bash
# MR-RATE (gated Hugging Face dataset, CC BY-NC-SA 4.0, academic use only): stream the selected head studies
# (brain.prep.mr_rate select -> META/selection.csv) one zip at a time through this node's /tmp, keep only the chosen
# series in the home holding folder in chunks (never scratch or pool), and wait while MAX_READY chunks await prep
# (jobs/prep_mr_rate.sbatch preps and deletes them). Resumable: finished chunks and studies are skipped.
#   sbatch jobs/fetch_mr_rate.sbatch
#SBATCH -p mit_data_transfer
#SBATCH --job-name=fetch-mr-rate
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 3-00:00:00
#SBATCH -o logs/fetch_mr_rate_%j.out
#SBATCH -e logs/fetch_mr_rate_%j.out
set -euo pipefail
umask 077
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
HOLD="${HOLD:-/home/roku/brain_tmp/mr_rate}"
LOCAL="/tmp/${USER}_mr_rate_dl_${SLURM_JOB_ID:-local}"
cleanup() { rm -rf "$LOCAL"; }
trap cleanup EXIT
mkdir -p "$HOLD" "$LOCAL"; chmod 700 "$HOLD" "$(dirname "$HOLD")"
echo "[fetch-mr-rate] $(date -Is) host=$(hostname) hold=$HOLD"
"$PY" -u -m brain.prep.mr_rate fetch --hold "$HOLD" --chunk-size "${CHUNK:-300}" --max-ready "${MAX_READY:-2}" \
  --tmp "$LOCAL" --workers "${WORKERS:-8}" ${LIMIT_CHUNKS:+--limit-chunks $LIMIT_CHUNKS}
echo "[fetch-mr-rate] done $(date -Is)"
