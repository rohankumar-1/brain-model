#!/bin/bash
# Format-6 prep of HemSeg-500 (brain.prep.sources.hemseg): 519 RSNA ICH studies with IPH / IVH masks; masks staged from
# the pool archive onto node-local disk (never scratch). Then jobs/index_v6.sbatch and jobs/multitask_labels.sbatch.
#   sbatch jobs/prep_hemseg.sbatch      (images: the listed DICOM slices, read from the RSNA archive on pool)

#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --job-name=prep-hemseg
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/prep_hemseg_%j.out
#SBATCH -e logs/prep_hemseg_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
LOCAL="/tmp/${USER}_hemseg_${SLURM_JOB_ID:-local}"
cleanup() { rm -rf "${LOCAL:?}"; }
trap cleanup EXIT
mkdir -p "$LOCAL/hemseg/extracted"
"$PY" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
  /home/roku/orcd/pool/datasets/hemseg/HemSeg-500.zip "$LOCAL/hemseg/extracted"
cp /home/roku/orcd/pool/datasets/hemseg/annotion_file_info_500.jsonl "$LOCAL/hemseg/"
# images come from the RSNA archive and BHSD overlaps are skipped: both are read from pool, not staged
ln -s /home/roku/orcd/pool/datasets/rsna "$LOCAL/rsna"
ln -s /home/roku/orcd/pool/datasets/bhsd "$LOCAL/bhsd"
echo "[prep-hemseg] staged $(find "$LOCAL/hemseg/extracted" -name '*.nii.gz' | wc -l) masks"
export BRAIN_DATASETS="$LOCAL" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u -m brain.prep.run --root "$STORE" --datasets hemseg --workers 11 --manifest manifest_v6_hemseg_0of1.jsonl \
  --brain-mask auto --gpu-mask --mask-batch 4 --reg-backend sitk \
  --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN"
echo "[prep-hemseg] done $(date -Is)"
