#!/bin/bash
# Format-6 prep of the fastMRI+ annotated brain scans (brain.prep.sources.fastmri): stage the kept coil-combined
# images from pool/datasets/fastmri_brain/fastmri_plus_*.zip onto this node's local disk as images (brain.prep.fastmri,
# geometry by convention; writes labels/fastmri_patients.csv for patient-level splits), prep, delete the copy.
# Data use agreement (NYU fastMRI): owner-only files (umask 077). Run after jobs/archive/fetch_fastmri.sbatch.
#   sbatch jobs/archive/prep_fastmri.sbatch
#   sbatch --export=ALL,LIMIT=4,TAG=pilot jobs/archive/prep_fastmri.sbatch
#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --job-name=prep-fastmri
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --exclude=node3006
#SBATCH -o logs/prep_fastmri_%j.out
#SBATCH -e logs/prep_fastmri_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
umask 077
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
LOCAL="/tmp/${USER}_fastmri_${SLURM_JOB_ID:-local}"
TAG=${TAG:-0of1}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cleanup() { rm -rf "$LOCAL"; }
trap cleanup EXIT
echo "[prep-fastmri] $(date -Is) host=$(hostname)"
"$PY" -u -m brain.prep.fastmri --dest "$LOCAL/datasets/fastmri_brain/nifti"
export BRAIN_DATASETS="$LOCAL/datasets"
"$PY" -u -m brain.prep.run --root "$STORE" --datasets fastmri --shard 0/1 --workers 11 \
  --manifest "manifest_v6_fastmri_${TAG}.jsonl" --brain-mask auto --gpu-mask --mask-batch 4 --reg-backend sitk \
  ${LIMIT:+--limit-per-dataset $LIMIT} ${EXTRA:-} --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN"
echo "[prep-fastmri] done $(date -Is)"
