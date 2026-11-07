#!/bin/bash
# Format-6 prep of the LONI IDA held-out cohorts (brain.prep.sources.ida): ADNI (4,979 baseline T1 / T2 / FLAIR of
# 3,129 subjects) and PPMI, read straight from the pool download zips by byte offsets (no staging), DICOM extracted to
# node-local temp, dcm2niix reader (enhanced multi-frame MR) with the SimpleITK fallback. Data use agreements: pool,
# store and work dirs are owner-only. Four shards; re-submit to resume (done cases are skipped).
#   sbatch --export=ALL,DATASET=adni jobs/archive/prep_ida.sbatch
#   sbatch --array=0 --export=ALL,DATASET=adni,N=1,LIMIT=10,TAG=pilot jobs/archive/prep_ida.sbatch      # pilot
#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --job-name=prep-ida
#SBATCH --array=0-3
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --exclude=node3006
#SBATCH -o logs/prep_ida_%A_%a.out
#SBATCH -e logs/prep_ida_%A_%a.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
DATASET=${DATASET:?DATASET=adni or ppmi}
N=${N:-4}
I=${SLURM_ARRAY_TASK_ID:-0}
TAG=${TAG:-${I}of${N}}
export BRAIN_DICOM_READER=dcm2niix OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
umask 077
echo "[prep-ida] $DATASET shard $I/$N $(date -Is)"
"$PY" -u -m brain.prep.run --root "$STORE" --datasets "$DATASET" --shard "$I/$N" --workers 11 \
  --manifest "manifest_v6_${DATASET}_${TAG}.jsonl" --brain-mask auto --gpu-mask --mask-batch 4 --reg-backend sitk \
  ${LIMIT:+--limit-per-dataset $LIMIT} ${EXTRA:-} --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN"
echo "[prep-ida] done $(date -Is)"
