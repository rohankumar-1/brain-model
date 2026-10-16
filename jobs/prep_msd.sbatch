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
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
export STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
export MSD="${MSD:-/home/roku/orcd/pool/datasets/msd_braintumor}"
EXPECTED="${EXPECTED:-484}"

echo "[host] $(hostname) $(date -Is) store=$STORE"
if [ ! -d "$MSD/Task01_BrainTumour/imagesTr" ]; then
  echo "[extract] $MSD/Task01_BrainTumour.tar"
  "$PY" -u -m brain.datasets.download_msd_braintumor --skip-download --skip-labels --out-dir "$MSD"
fi

if [ ! -f "$STORE/volumes.csv.bak_before_msd" ]; then
  cp "$STORE/volumes.csv" "$STORE/volumes.csv.bak_before_msd"
fi

"$PY" -u -m brain.prep.run --root "$STORE" --datasets msd \
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

"$PY" -u -m brain.prep.index --root "$STORE" --keep-ssl-index --workers "${SLURM_CPUS_PER_TASK:-32}"
"$PY" -u - <<'PY'
from pathlib import Path
import os
from brain.common.paths import DATASETS, PROCESSED_ROOT
from brain.datasets.download_msd_braintumor import export_msd_labels
store = Path(os.environ.get("STORE", str(PROCESSED_ROOT)))
export_msd_labels(DATASETS / "msd_braintumor" / "Task01_BrainTumour",
                  store / "labels" / "msd_volume_labels.csv")
PY
echo "[done] $(date -Is)"
