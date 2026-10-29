#!/bin/bash
# Format-6 prep of datasets that live on pool only as <name>.raw.zip packs (brain.prep.rawpack): stage them onto this
# node's local disk (never scratch), point BRAIN_DATASETS there, prep, and delete the staged copy.
#
#   STAGE=brainmetshare DATASETS=brainmetshare EXTRA_ARGS="--require-attr n4=native" \
#     sbatch --export=ALL jobs/prep_v6_staged.sbatch
#SBATCH -p pi_dbertsim,mit_preemptable
#SBATCH --job-name=prep-v6-staged
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=node3006
#SBATCH -o logs/prep_v6_staged_%j.out
#SBATCH -e logs/prep_v6_staged_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
STAGE="${STAGE:?set STAGE=<pool dataset folders, comma list>}"
DATASETS="${DATASETS:?set DATASETS=<brain.prep.run dataset names>}"
LOCAL="/tmp/${USER}_staged_${SLURM_JOB_ID:-local}"
W=$(( ${SLURM_CPUS_PER_TASK:-12} - 1 ))
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cleanup() { rm -rf "$LOCAL"; }
trap cleanup EXIT
echo "[prep-staged] $(date -Is) host=$(hostname) stage=$STAGE datasets=$DATASETS"
"$PY" -u -m brain.prep.rawpack stage --dest "$LOCAL/datasets" --datasets "$STAGE" --workers "$W"
export BRAIN_DATASETS="$LOCAL/datasets"
EXTRA=()
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=(${EXTRA_ARGS})
"$PY" -u -m brain.prep.run --root "$STORE" --datasets "$DATASETS" --shard 0/1 --workers "$W" \
  --manifest "manifest_v6_S${STAGE//,/_}.jsonl" --brain-mask auto --gpu-mask --mask-batch 8 \
  --reg-backend sitk --max-scratch-pct "${MAX_SCRATCH_PCT:-95}" \
  --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN" "${EXTRA[@]}"
echo "[prep-staged] done $(date -Is)"
