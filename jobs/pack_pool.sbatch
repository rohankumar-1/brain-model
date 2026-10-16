#!/bin/bash
# JPEG-LS pack a pool image tree, then delete the uncompressed source.
#   TREE=/home/roku/orcd/pool/datasets/brats/extracted \
#   DEST=/home/roku/orcd/pool/datasets/brats/extracted.jls.zip \
#   sbatch --export=ALL,TREE,DEST jobs/pack_pool.sbatch

#SBATCH -p mit_quicktest
#SBATCH --job-name=pack-pool
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 00:14:00
#SBATCH --signal=TERM@90
#SBATCH -o logs/pack_pool_%j.out
#SBATCH -e logs/pack_pool_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
TREE="${TREE:?}"
DEST="${DEST:?}"
DELETE="${DELETE:-1}"
EXTRA=()
if [ "$DELETE" = 1 ]; then
  EXTRA+=(--delete-extracted)
fi

echo "[host] $(hostname) $(date -Is) $TREE -> $DEST delete=$DELETE"
trap 'kill -TERM $pid 2>/dev/null; wait $pid' TERM
"$PY" -u -m brain.prep.archive --tree "$TREE" --dest "$DEST" "${EXTRA[@]}" \
  --workers "${SLURM_CPUS_PER_TASK:-16}" &
pid=$!
wait $pid
echo "[done] $(date -Is)"
