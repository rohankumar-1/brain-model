#!/bin/bash
# Format-6 prep of BHSD (brain.prep.sources.bhsd): the 192 RSNA ICH studies with voxel hemorrhage masks, staged from
# the pool archive onto node-local disk (never scratch). Then jobs/index_v6.sbatch and jobs/multitask_labels.sbatch.
#   sbatch jobs/prep_bhsd.sbatch

#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --job-name=prep-bhsd
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/prep_bhsd_%j.out
#SBATCH -e logs/prep_bhsd_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
LOCAL="/tmp/${USER}_bhsd_${SLURM_JOB_ID:-local}"
cleanup() { rm -rf "${LOCAL:?}"; }
trap cleanup EXIT
mkdir -p "$LOCAL/bhsd/extracted"
"$PY" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
  /home/roku/orcd/pool/datasets/bhsd/label_192.zip "$LOCAL/bhsd/extracted"
echo "[prep-bhsd] staged $(find "$LOCAL/bhsd/extracted" -name '*.nii.gz' | wc -l) files"
export BRAIN_DATASETS="$LOCAL" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u -m brain.prep.run --root "$STORE" --datasets bhsd --workers 11 --manifest manifest_v6_bhsd_0of1.jsonl \
  --brain-mask auto --gpu-mask --mask-batch 4 --reg-backend sitk \
  --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN"
echo "[prep-bhsd] done $(date -Is)"
