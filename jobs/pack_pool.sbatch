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
cd /home/roku/brain-model
mkdir -p logs
unset PYTHONPATH WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
export PYTHONNOUSERSITE=True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/roku/brain-model:/home/roku/.local/lib/python3.12/site-packages
PY=/home/roku/.conda/envs/ml-torch/bin/python
TREE="${TREE:?}"
DEST="${DEST:?}"
DELETE="${DELETE:-1}"
EXTRA=()
if [ "$DELETE" = 1 ]; then
  EXTRA+=(--delete-extracted)
fi

echo "[host] $(hostname) $(date -Is) $TREE -> $DEST delete=$DELETE"
trap 'kill -TERM $pid 2>/dev/null; wait $pid' TERM
"$PY" -u -m data.prep.archive --tree "$TREE" --dest "$DEST" "${EXTRA[@]}" \
  --workers "${SLURM_CPUS_PER_TASK:-16}" &
pid=$!
wait $pid
echo "[done] $(date -Is)"
