#!/bin/bash
# Re-prep Seg-CQ500 (eval only, 51 CT) with the current reader into a CANDIDATE store, then compare it with the
# DICOM CQ500 copies (analysis/segcq500_orientation.py compare). The stored copies date from 2026-10-18, before the
# nibabel-fallback fix of 2026-10-25: the 32 gantry-tilt studies were misplaced / mirrored. The live store is not
# touched here; swapping the candidate in is a separate, checked step (see the script's docstring).
#   sbatch jobs/prep_seg_cq500.sbatch   (CPU: Seg-CQ500 ships brain masks, SynthStrip does not run)
#SBATCH -p mit_preemptable
#SBATCH --job-name=prep-seg-cq500
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/prep_seg_cq500_%j.out
#SBATCH -e logs/prep_seg_cq500_%j.out
set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
CAND="${CAND:-/home/roku/orcd/scratch/brain-model/data/processed/seg_cq500_candidate}"
LOCAL="/tmp/${USER}_segcq_${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cleanup() { rm -rf "$LOCAL"; }
trap cleanup EXIT
echo "[prep-seg-cq500] $(date -Is) host=$(hostname) cand=$CAND"
mkdir -p "$CAND/labels"
"$PY" -u -m brain.prep.rawpack stage --dest "$LOCAL/datasets" --datasets seg_cq500 --workers 11
export BRAIN_DATASETS="$LOCAL/datasets"
"$PY" -u -m brain.prep.run --root "$CAND" --datasets seg_cq500 --shard 0/1 --workers 11 \
  --manifest "manifest_v6_seg_cq500_r2.jsonl" --brain-mask auto \
  --reg-backend sitk --max-scratch-pct 95 --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN"
"$PY" -u analysis/segcq500_orientation.py compare --root "$CAND"
echo "[prep-seg-cq500] done $(date -Is)"
