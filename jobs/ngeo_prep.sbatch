#!/bin/bash
# Native-geometry ("ngeo") CT copy (brain.prep.ngeo_run build): one shard of the indexed CT volumes per array task,
# resumable (cases with an ok manifest row and a zip are skipped), CPU only. Raw NIfTI datasets are staged to this
# node's /tmp (never scratch); DICOM datasets (RSNA ICH, RSNA-ICA, CQ500, HemSeg images) are read from pool by offsets.
# The driver checks the space left before every case (statvfs + quota cache - ngeo bytes since) and exits 3 when it
# drops below NGEO_MIN_FREE_GB (default 20).
#
#   # DICOM datasets, 32 shards, at most 8 at a time
#   DATASETS=rsna_ich,rsna_ica_cta,cq500,hemseg SHARD_N=32 sbatch --export=ALL --array=0-31%8 jobs/ngeo_prep.sbatch
#   # NIfTI datasets (each task stages them all, ~10 GB local): 2 shards
#   DATASETS=bhsd,aisd,instance,seg_cq500,isles24_ncct,isles24_cta,pedi_cranial_ct,synthrad_ct,synthrad_cbct \
#     SHARD_N=2 sbatch --export=ALL --array=0-1 jobs/ngeo_prep.sbatch
#   # then: python -m brain.prep.ngeo_run index   (volumes_ngeo.csv in the store root)
# Pilot into another root: OUT_ROOT=runs/diagnostics/ngeo/pilot CASES=<csv> SHARD_N=1 sbatch --array=0 ...
#SBATCH -p mit_normal
#SBATCH --job-name=ngeo-prep
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH --open-mode=append
#SBATCH -o logs/ngeo_prep_%A_%a.out
#SBATCH -e logs/ngeo_prep_%A_%a.out

set -euo pipefail
export BRAIN_REPO="${BRAIN_REPO:-/home/roku/brain-model}"
source "$BRAIN_REPO/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
OUT_ROOT="${OUT_ROOT:-$STORE}"
DATASETS="${DATASETS:?set DATASETS=<comma list of CT datasets>}"
SHARD_N="${SHARD_N:-${SLURM_ARRAY_TASK_COUNT:-1}}"
I="${SLURM_ARRAY_TASK_ID:-0}"
TAG="${TAG:-v1}"
LOCAL="/tmp/${USER}_ngeo_${SLURM_JOB_ID:-local}_${I}"
W=$(( ${SLURM_CPUS_PER_TASK:-12} - 1 ))
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cleanup() { rm -rf "$LOCAL"; }
trap cleanup EXIT
echo "[ngeo] $(date -Is) host=$(hostname) shard=$I/$SHARD_N datasets=$DATASETS out=$OUT_ROOT tmp: $(df -h /tmp | tail -1)"
EXTRA=()
[[ -n "${CASES:-}" ]] && EXTRA+=(--cases "$CASES")
[[ -n "${FORCE:-}" ]] && EXTRA+=(--force)
"$PY" -u -W ignore -m brain.prep.ngeo_run build --root "$STORE" --out-root "$OUT_ROOT" --datasets "$DATASETS" \
  --shard "$I/$SHARD_N" --workers "$W" --local "$LOCAL" --tag "$TAG" "${EXTRA[@]}"
echo "[ngeo] done $(date -Is)"
