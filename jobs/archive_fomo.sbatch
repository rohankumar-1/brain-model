#!/bin/bash
# JPEG-LS pack finished FOMO extracts onto pool, then delete scratch niftis.
#   sbatch --export=ALL,FAMILIES=fomo_nigerian,fomo_m4raw jobs/archive_fomo.sbatch

#SBATCH -p mit_quicktest
#SBATCH --job-name=arch-fomo
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:14:00
#SBATCH --signal=TERM@90
#SBATCH -o logs/archive_fomo_%j.out
#SBATCH -e logs/archive_fomo_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
FAMILIES="${FAMILIES:?set FAMILIES e.g. fomo_nigerian,fomo_m4raw}"

echo "[host] $(hostname) $(date -Is) families=$FAMILIES"
trap 'kill -TERM $pid 2>/dev/null; wait $pid' TERM
"$PY" -u -m brain.prep.archive --families "$FAMILIES" --delete-extracted \
  --workers 4 &
pid=$!
wait $pid
echo "[done] $(date -Is)"
