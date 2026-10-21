#!/bin/bash
# Pool -> archives only: one bit-exact <name>.raw.zip per dataset folder next to its original archives; intermediates
# (aborted JPEG-LS runs, old per-dataset stores, extracted copies of a kept archive) removed. Sources are deleted only
# after the zip is re-read and CRC-checked. Smallest datasets first; resumable (re-submit after a timeout).
# Run only after every prep phase has finished reading pool.
#
#   sbatch --dependency=afterok:<A>:<B>:<C> jobs/rawpack_pool.sbatch
# Re-prep later: python -m brain.prep.rawpack stage --dest $TMPDIR/datasets; export BRAIN_DATASETS=$TMPDIR/datasets

#SBATCH -p mit_normal
#SBATCH --job-name=rawpack-pool
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/rawpack_%j.out
#SBATCH -e logs/rawpack_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
echo "[rawpack] $(date -Is) plan"
"$PY" -u -m brain.prep.rawpack plan ${DATASETS:+--datasets "$DATASETS"}
echo "[rawpack] $(date -Is) pack"
"$PY" -u -m brain.prep.rawpack pack ${DATASETS:+--datasets "$DATASETS"}
sed -n 1,6p ~/orcd/.quota
echo "[rawpack] done $(date -Is)"
