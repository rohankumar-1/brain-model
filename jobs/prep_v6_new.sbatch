#!/bin/bash
# Format-6 prep for datasets kept on pool as their original downloads (brain.prep.stage_new: ms3seg, valdo).
# Stages them onto this node's local disk (never scratch), writes their label tables to the store, preps, cleans up.
#
#   DATASETS=ms3seg,valdo sbatch jobs/prep_v6_new.sbatch     # then jobs/index_v6.sbatch and jobs/verify_v6.sbatch
#SBATCH -p pi_dbertsim,sched_mit_sloan_gpu_r8
#SBATCH --job-name=prep-v6-new
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --exclude=node3006,node5003,node5801-c1,node5802-c1
#SBATCH -o logs/prep_v6_new_%j.out
#SBATCH -e logs/prep_v6_new_%j.out
set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
DATASETS="${DATASETS:-ms3seg,valdo}"
LOCAL="/tmp/${USER}_new_${SLURM_JOB_ID:-local}"
cleanup() { rm -rf "$LOCAL"; }
trap cleanup EXIT                                 # shared node disk: never leave the copy behind
W=$(( ${SLURM_CPUS_PER_TASK:-10} - 1 ))
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[prep-v6-new] $(date -Is) host=$(hostname) datasets=$DATASETS local=$LOCAL"
"$PY" -u -m brain.prep.stage_new --datasets "$DATASETS" --dest "$LOCAL" --labels "$STORE/labels/sources"
export BRAIN_DATASETS="$LOCAL"
"$PY" -u -m brain.prep.run \
  --root "$STORE" \
  --datasets "$DATASETS" \
  --workers "$W" \
  --manifest "manifest_v6_new_${DATASETS//,/_}_0of1.jsonl" \
  --brain-mask auto --gpu-mask --mask-batch "${MASK_BATCH:-4}" \
  --reg-backend "${REG_BACKEND:-torch}" \
  --max-scratch-pct "${MAX_SCRATCH_PCT:-95}" \
  --template "$BRAIN_TEMPLATE" \
  --template-brain "$BRAIN_TEMPLATE_BRAIN"
echo "[prep-v6-new] done $(date -Is)"
