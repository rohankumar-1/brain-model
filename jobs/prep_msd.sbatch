#!/bin/bash
# Extract Medical Decathlon Task01, pack T1w into slices_zarr as held-out, write frozen labels.
#   sbatch jobs/prep_msd.sbatch

#SBATCH -p mit_normal
#SBATCH --job-name=prep-msd
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o logs/prep_msd_%j.out
#SBATCH -e logs/prep_msd_%j.out

set -euo pipefail
cd /home/roku/brain-model
mkdir -p logs
unset PYTHONPATH WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
export PYTHONNOUSERSITE=True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/roku/brain-model:/home/roku/.local/lib/python3.12/site-packages
PY=/home/roku/.conda/envs/ml-torch/bin/python
export STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
export MSD="${MSD:-/home/roku/orcd/pool/datasets/msd_braintumor}"
EXPECTED="${EXPECTED:-484}"

echo "[host] $(hostname) $(date -Is) store=$STORE"
if [ ! -d "$MSD/Task01_BrainTumour/imagesTr" ]; then
  echo "[extract] $MSD/Task01_BrainTumour.tar"
  "$PY" -u -m data.download_msd_braintumor --skip-download --skip-labels --out-dir "$MSD"
fi

if [ ! -f "$STORE/volumes.csv.bak_before_msd" ]; then
  cp "$STORE/volumes.csv" "$STORE/volumes.csv.bak_before_msd"
fi

"$PY" -u -m data.prep.run --root "$STORE" --datasets msd \
  --manifest manifest_msd.jsonl --workers "${SLURM_CPUS_PER_TASK:-32}" --z-anchor auto

N_DONE=$("$PY" -c "
import pandas as pd
from pathlib import Path
p = Path('$STORE') / 'manifest_msd.jsonl'
print(0 if not p.exists() else pd.read_json(p, lines=True).drop_duplicates(['dataset','case_id']).shape[0])
")
echo "[prep] manifest rows=$N_DONE expected=$EXPECTED"
if [ "$N_DONE" -lt "$EXPECTED" ]; then
  echo "[partial] $N_DONE/$EXPECTED — requeue to continue $(date -Is)"
  exit 1
fi

"$PY" -u -m data.prep.index --root "$STORE" --keep-ssl-index --workers "${SLURM_CPUS_PER_TASK:-32}"
"$PY" -u - <<'PY'
from pathlib import Path
from data.preprocess import export_msd_labels
from data.prep.sources import DATASETS
import os
store = Path(os.environ.get("STORE", "/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr"))
export_msd_labels(DATASETS / "msd_braintumor" / "Task01_BrainTumour",
                  store / "labels" / "msd_volume_labels.csv")
PY
echo "[done] $(date -Is)"
