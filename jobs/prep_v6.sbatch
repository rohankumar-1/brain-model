#!/bin/bash
# Format-6 prep (256 x 256 x 160 grid, SynthStrip masks batched on the job's GPU, brain mask inside the image zip).
# Rewrites the store in place; each array task is one shard. Requeue-safe: the shard's manifest records finished
# cases and current-format zips are skipped. Stops dispatching (exit 3) when scratch passes MAX_SCRATCH_PCT.
#
#   PHASE=A sbatch --array=0-7%8 jobs/prep_v6.sbatch     # FOMO families extracted on scratch + ABIDE
#   PHASE=B sbatch --array=0-7%8 jobs/prep_v6.sbatch     # datasets read straight from pool
#   PHASE=SMOKE LIMIT=2 STORE=<scratch smoke root> sbatch -p pi_dbertsim jobs/prep_v6.sbatch
#   Phase-A rerun (store the transform, GPU registration; needs the scratch FOMO/ABIDE extracts, so before cleanup):
#   PHASE=A MANIFEST_TAG=A_rerun REG_BACKEND=torch MASK_BATCH=4 EXTRA_ARGS="--require-attr reg_backend=torch" \
#     sbatch --export=ALL --array=0-7%4 jobs/prep_v6.sbatch
#   (HBN and BraTS incl. GoAT: jobs/prep_v6_local.sbatch, which unpacks them on node-local disk)
#
# pi_dbertsim only accepts GPU jobs; mit_preemptable is the overflow (preempted tasks requeue and resume).
# 12 CPUs + 1 L40S + 96 GB per task: pi_dbertsim allows at most 12 CPUs per GPU (48 CPUs, 4 GPUs per node).

#SBATCH -p pi_dbertsim,mit_preemptable
#SBATCH --exclude=node3006          # broken CUDA install (torch sees no GPU)
#SBATCH --job-name=prep-v6
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/prep_v6_%A_%a.out
#SBATCH -e logs/prep_v6_%A_%a.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
PHASE="${PHASE:?set PHASE=A, B or SMOKE}"
case "$PHASE" in
  A) DATASETS="abide,fomo" ;;
  B) DATASETS="cq500,oasis,rsna,ixi,isles,atlas,pixar,piop1,piop2,id1000,wmh,isles24,instance,aisd,brainmetshare,msd" ;;
  SMOKE) DATASETS="abide,fomo,cq500,oasis,rsna,ixi,isles,atlas,pixar,piop1,piop2,id1000,wmh,isles24,instance,aisd,brainmetshare,msd,brats" ;;
  *) echo "unknown PHASE=$PHASE" >&2; exit 2 ;;
esac
DATASETS="${DATASETS_OVERRIDE:-$DATASETS}"           # e.g. split a phase by registration backend
EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit-per-dataset "$LIMIT")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=(${EXTRA_ARGS})   # space-separated; sbatch --export splits on commas
TAG="${MANIFEST_TAG:-$PHASE}"
N="${SHARD_N:-${SLURM_ARRAY_TASK_COUNT:-1}}"   # SHARD_N: rerun some shards of an n-way split
I="${SLURM_ARRAY_TASK_ID:-0}"
W=$(( ${SLURM_CPUS_PER_TASK:-12} - 1 ))          # one core for the main process (GPU batches, manifest)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # GPU registration workers share the GPU
echo "[prep-v6] $(date -Is) host=$(hostname) phase=$PHASE shard=$I/$N workers=$W gpu=${CUDA_VISIBLE_DEVICES:-none}"
"$PY" -u -m brain.prep.run \
  --root "$STORE" \
  --datasets "$DATASETS" \
  --shard "$I/$N" \
  --workers "$W" \
  --manifest "manifest_v6_${TAG}_${I}of${N}.jsonl" \
  --brain-mask auto --gpu-mask --mask-batch "${MASK_BATCH:-8}" \
  --reg-backend "${REG_BACKEND:-sitk}" \
  --max-scratch-pct "${MAX_SCRATCH_PCT:-95}" \
  --template "$BRAIN_TEMPLATE" \
  --template-brain "$BRAIN_TEMPLATE_BRAIN" \
  "${EXTRA[@]}"
echo "[prep-v6] done $(date -Is)"
