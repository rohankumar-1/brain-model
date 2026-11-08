#!/bin/bash
# ngeo pilot (2026-11-07): <= 50 CT cases into runs/diagnostics/ngeo/pilot (the real store is only read), plus two
# diagnostics: the three thin CQ500 studies with a real 5 mm series rebuilt from their thin series (synthetic slabs,
# pilot_cq500_thin) and the RSNA cases with SimpleITK's store frame ignored (pilot_noframe). Then overlays, QC table,
# index and the size / time estimate.
#   sbatch --dependency=afterok:<cq500 thick download> jobs/archive/ngeo_pilot.sbatch
#SBATCH -p mit_normal,mit_preemptable
#SBATCH --requeue
#SBATCH --job-name=ngeo-pilot
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/ngeo_pilot_%j.out
#SBATCH -e logs/ngeo_pilot_%j.out
set -euo pipefail
export BRAIN_REPO="${BRAIN_REPO:-/home/roku/bm-ngeo-prep}"
source "$BRAIN_REPO/jobs/_env.sh"
STORE=/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr
P=/home/roku/brain-model/runs/diagnostics/ngeo
LOCAL="/tmp/${USER}_ngeo_pilot_${SLURM_JOB_ID:-local}"
cleanup() { rm -rf "$LOCAL"; }
trap cleanup EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
B=(-u -W ignore -m brain.prep.ngeo_run build --root "$STORE" --workers 7 --local "$LOCAL")
echo "[pilot] $(date -Is) host=$(hostname)"
"$PY" "${B[@]}" --out-root "$P/pilot" --cases "$P/pilot/cases.csv"
grep -E "^(dataset|cq500,CQ500-CT-(0|108|113),)" "$P/pilot/cases.csv" > "$LOCAL/cq_thin.csv"
NGEO_CQ500_THICK=/nonexistent "$PY" "${B[@]}" --out-root "$P/pilot_cq500_thin" --cases "$LOCAL/cq_thin.csv" --datasets cq500
grep -E "^(dataset|rsna_ich,)" "$P/pilot/cases.csv" > "$LOCAL/rsna.csv"
NGEO_IGNORE_STORE_FRAME=1 "$PY" "${B[@]}" --out-root "$P/pilot_noframe" --cases "$LOCAL/rsna.csv" --datasets rsna_ich
"$PY" -u -W ignore -m brain.prep.ngeo_qc --out-root "$P/pilot" --n 60
"$PY" -u -W ignore -m brain.prep.ngeo_run index --root "$STORE" --out-root "$P/pilot"
"$PY" -u -W ignore -m brain.prep.ngeo_run estimate --root "$STORE" --out-root "$P/pilot"
echo "[pilot] done $(date -Is)"
