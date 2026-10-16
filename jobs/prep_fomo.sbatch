#!/bin/bash
# One FOMO source family → slices_zarr/fomo_<short>/ (contrasts share that folder).
#   sbatch --export=ALL,MATCH=fomo_hbn,EXPECTED=8491 jobs/prep_fomo.sbatch
# Stay at 16 CPUs / 64G so this can sit next to prep-sinoct (32/96G) under mit_quicktest.

#SBATCH -p mit_quicktest
#SBATCH --job-name=prep-fomo
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 00:14:00
#SBATCH --signal=TERM@90
#SBATCH -o logs/prep_fomo_%j.out
#SBATCH -e logs/prep_fomo_%j.out

set -euo pipefail
cd /home/roku/brain-model
mkdir -p logs
unset PYTHONPATH WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
export PYTHONNOUSERSITE=True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/roku/brain-model:/home/roku/.local/lib/python3.12/site-packages
PY=/home/roku/.conda/envs/ml-torch/bin/python
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
MATCH="${MATCH:?set MATCH (e.g. fomo_hbn)}"
DATASETS="${DATASETS:-fomo}"
MANIFEST="${MANIFEST:-manifest_${MATCH}.jsonl}"
EXPECTED="${EXPECTED:-0}"

echo "[host] $(hostname) $(date -Is) store=$STORE datasets=$DATASETS match=$MATCH manifest=$MANIFEST expected=$EXPECTED"

"$PY" -u -m data.prep.run --root "$STORE" --datasets "$DATASETS" --match "$MATCH" \
  --manifest "$MANIFEST" --workers "${SLURM_CPUS_PER_TASK:-16}" --z-anchor auto

N_DONE=$("$PY" -c "
import pandas as pd
from pathlib import Path
p = Path('$STORE') / '$MANIFEST'
print(0 if not p.exists() else pd.read_json(p, lines=True).drop_duplicates(['dataset','case_id']).shape[0])
")
echo "[prep] $MATCH manifest rows=$N_DONE expected=$EXPECTED $(date -Is)"
if [ "$EXPECTED" -gt 0 ] && [ "$N_DONE" -ge "$EXPECTED" ]; then
  echo "[done] $MATCH $N_DONE/$EXPECTED — JPEG-LS archive to pool, drop extracted"
  "$PY" -u -m data.prep.archive --families "$MATCH" --delete-extracted \
    --workers 8
else
  echo "[partial] $MATCH $N_DONE/$EXPECTED — requeue to continue"
fi
exit 0
