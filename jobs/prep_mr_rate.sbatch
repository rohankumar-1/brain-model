#!/bin/bash
# Prep the MR-RATE chunks jobs/fetch_mr_rate.sbatch marks READY in the home holding folder, oldest first: point
# BRAIN_DATASETS at the chunk, run brain.prep.run (dataset mr_rate, one manifest per chunk), then delete the chunk's
# images and mark it DONE. Loops until the fetch has finished and nothing is left; before the 6 h limit it resubmits
# itself (pending chunks continue in the next job). Owner-only files (umask 077).
#   sbatch jobs/prep_mr_rate.sbatch
#   SBATCH_EXTRA="-p mit_preemptable --gres=gpu:a100:1" sbatch -p mit_preemptable --gres=gpu:a100:1 --export=ALL jobs/prep_mr_rate.sbatch
#   (B200 / RTX PRO 6000 Blackwell nodes: our torch build has no sm_100 / sm_120 kernels; pin the GPU type)
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=prep-mr-rate
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH --exclude=node3006,node4207
#SBATCH -o logs/prep_mr_rate_%j.out
#SBATCH -e logs/prep_mr_rate_%j.out
set -euo pipefail
umask 077
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
HOLD="${HOLD:-/home/roku/brain_tmp/mr_rate}"
BUDGET_S=${BUDGET_S:-18000}            # stop taking new chunks after 5 h; a chunk of 300 studies takes well under 1 h
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T0=$(date +%s)
echo "[prep-mr-rate] $(date -Is) host=$(hostname)"
# Several of these jobs may run at once: a job claims a chunk with an atomic mkdir (CLAIM/<job id>). Claims of jobs
# no longer running, or of this job before a requeue, are released first.
release_stale() {
  for c in "$HOLD"/chunk_*/CLAIM; do
    [[ -d "$c" && ! -f "$(dirname "$c")/DONE" ]] || continue
    owner=$(ls "$c" 2>/dev/null | head -1)
    if [[ -z "$owner" || "$owner" == "${SLURM_JOB_ID:-local}" ]] || ! squeue -h -j "$owner" 2>/dev/null | grep -q .; then
      rm -rf "$c"
    fi
  done
}
release_stale
while true; do
  next=""
  for d in $(ls -d "$HOLD"/chunk_* 2>/dev/null | sort); do
    if [[ -f "$d/READY" && ! -f "$d/DONE" ]] && mkdir "$d/CLAIM" 2>/dev/null; then
      touch "$d/CLAIM/${SLURM_JOB_ID:-local}"; next=$d; break
    fi
  done
  if [[ -z "$next" ]]; then
    if [[ -f "$HOLD/FETCH_DONE" ]]; then
      left=0
      for d in "$HOLD"/chunk_*; do [[ -f "$d/DONE" ]] || left=1; done
      (( left == 0 )) && { echo "[prep-mr-rate] all chunks done $(date -Is)"; exit 0; }
      echo "[prep-mr-rate] remaining chunks are claimed by other jobs; exiting $(date -Is)"; exit 0
    fi
    (( $(date +%s) - T0 > BUDGET_S )) && break
    sleep 120; continue
  fi
  (( $(date +%s) - T0 > BUDGET_S )) && break
  name=$(basename "$next")
  echo "[prep-mr-rate] $name $(date -Is)"
  BRAIN_DATASETS="$next/datasets" "$PY" -u -m brain.prep.run --root "$STORE" --datasets mr_rate --shard 0/1 \
    --workers 11 --manifest "manifest_v6_mr_rate_${name}.jsonl" --brain-mask auto --gpu-mask --mask-batch 8 \
    --reg-backend sitk --max-scratch-pct "${MAX_SCRATCH_PCT:-94}" \
    --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN"
  rm -rf "$next/datasets"
  touch "$next/DONE"
done
echo "[prep-mr-rate] time budget reached $(date -Is); resubmitting"
sbatch ${SBATCH_EXTRA:-} --export=ALL "${BRAIN_REPO:-/home/roku/brain-model}/jobs/prep_mr_rate.sbatch"
