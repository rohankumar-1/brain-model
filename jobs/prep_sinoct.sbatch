#!/bin/bash
# Preprocess keep-only SinoCT reconstructed series into slices_zarr, then append SSL slices
# without re-indexing existing volumes.
#   sbatch jobs/prep_sinoct.sbatch

#SBATCH -p mit_quicktest
#SBATCH --job-name=prep-sinoct
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 00:14:00
#SBATCH -o logs/prep_sinoct_%j.out
#SBATCH -e logs/prep_sinoct_%j.out

set -euo pipefail
cd /home/roku/brain-model
mkdir -p logs
unset PYTHONPATH WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
export PYTHONNOUSERSITE=True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/roku/brain-model:/home/roku/.local/lib/python3.12/site-packages
PY=/home/roku/.conda/envs/ml-torch/bin/python
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"

echo "[host] $(hostname) $(date -Is) store=$STORE"
if [ ! -f "$STORE/volumes.csv.bak_before_sinoct" ]; then
  cp "$STORE/volumes.csv" "$STORE/volumes.csv.bak_before_sinoct"
fi

"$PY" -u -m data.prep.run --root "$STORE" --datasets sinoct \
  --manifest manifest_sinoct.jsonl --workers "${SLURM_CPUS_PER_TASK:-32}" --z-anchor auto

N_DONE=$("$PY" -c "
import pandas as pd
from pathlib import Path
p = Path('$STORE') / 'manifest_sinoct.jsonl'
print(0 if not p.exists() else pd.read_json(p, lines=True).drop_duplicates(['dataset','case_id']).shape[0])
")
echo "[prep] manifest rows=$N_DONE"
if [ "$N_DONE" -ge 692 ]; then
  "$PY" -u -m data.prep.index --root "$STORE" --keep-ssl-index --workers "${SLURM_CPUS_PER_TASK:-32}"
  echo "[done] $(date -Is)"
else
  echo "[partial] $N_DONE/692 — requeue to continue $(date -Is)"
  exit 0
fi
