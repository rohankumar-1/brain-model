#!/bin/bash
# Format-6 prep of Pedi-Cranial-CT-Healthy (brain.prep.sources.pedi_cranial_ct; 100 healthy children's head CTs):
# stage the NIfTI from pool's pedi_cranial_ct.raw.zip (brain.prep.rawpack stage) onto this node's /tmp, or, when
# jobs/fetch_pedi_ct.sbatch skipped the archive, read its temporary home copy and delete that afterwards. CT: no N4,
# SynthStrip brain mask on the GPU. Scratch sits above prep's default 95% stop: ~1 GB of new volumes, so 98%.
#   sbatch --dependency=afterok:<fetch job> jobs/prep_pedi_ct.sbatch
#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --job-name=prep-pedi-ct
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --exclude=node3006
#SBATCH -o logs/prep_pedi_ct_%j.out
#SBATCH -e logs/prep_pedi_ct_%j.out

set -euo pipefail
umask 077
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
POOLROOT=/home/roku/orcd/pool/datasets
HOLD=/home/roku/brain_tmp/pedi_cranial_ct
LOCAL="/tmp/${USER}_pedi_ct_prep_${SLURM_JOB_ID:-local}"
TAG=${TAG:-0of1}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cleanup() { rm -rf "$LOCAL"; }
trap cleanup EXIT
echo "[prep-pedi-ct] $(date -Is) host=$(hostname)"
if [ -f "$POOLROOT/pedi_cranial_ct/pedi_cranial_ct.raw.zip" ]; then
  "$PY" -u -m brain.prep.rawpack stage --root "$POOLROOT" --datasets pedi_cranial_ct --dest "$LOCAL/datasets"
else
  mkdir -p "$LOCAL/datasets"
  ln -s "$HOLD" "$LOCAL/datasets/pedi_cranial_ct"
fi
echo "[prep-pedi-ct] $(ls "$LOCAL/datasets/pedi_cranial_ct/nifti" | wc -l) NIfTI staged"
export BRAIN_DATASETS="$LOCAL/datasets"
"$PY" -u -m brain.prep.run --root "$STORE" --datasets pedi_cranial_ct --shard 0/1 --workers 11 \
  --manifest "manifest_v6_pedi_cranial_ct_${TAG}.jsonl" --brain-mask auto --gpu-mask --mask-batch 4 --reg-backend sitk \
  --max-scratch-pct 98 ${LIMIT:+--limit-per-dataset $LIMIT} ${EXTRA:-} \
  --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN"
if [ -z "${LIMIT:-}" ] && [ -d "$HOLD" ]; then rm -rf "$HOLD"; rmdir --ignore-fail-on-non-empty /home/roku/brain_tmp; fi
echo "[prep-pedi-ct] done $(date -Is)"
