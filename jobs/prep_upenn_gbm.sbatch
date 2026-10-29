#!/bin/bash
# Format-6 prep of UPENN-GBM (brain.prep.sources.upenn_gbm): 671 scans (630 patients) x T1 / T1GD / T2 / FLAIR with the
# tumour segmentations, staged from pool onto node-local disk (never scratch). Held out of pretraining
# (index.HELD_OUT); patients also in BraTS share their BraTS twin's dedup_key once the twins are found by content
# (index.upenn_gbm_key). T1 cases are written first and the other sequences reuse T1's registration.
# Then the index rebuild (jobs/index_v6.sbatch) and the downstream labels.
#   sbatch --export=ALL,LIMIT=3 jobs/prep_upenn_gbm.sbatch     # pilot: 3 scans per sequence (kept; the full run skips them)
#   sbatch jobs/prep_upenn_gbm.sbatch
# EXTRA adds brain.prep.run arguments (no commas).

#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --job-name=prep-upenn-gbm
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --exclude=node3006
#SBATCH -o logs/prep_upenn_gbm_%j.out
#SBATCH -e logs/prep_upenn_gbm_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
SRC=/home/roku/orcd/pool/datasets/upenn_gbm
LOCAL="/tmp/${USER}_upenn_gbm_${SLURM_JOB_ID:-local}"
cleanup() { rm -rf "${LOCAL:?}"; }
trap cleanup EXIT
mkdir -p "$LOCAL/upenn_gbm/nifti" "$LOCAL/upenn_gbm/meta"
ls "$SRC/nifti" | grep -E '_(T1|T1GD|T2|FLAIR|segm|automated_approx_segm)\.nii\.gz$' \
  | sed "s#^#$SRC/nifti/#" | xargs -r cp -t "$LOCAL/upenn_gbm/nifti"
cp "$SRC"/meta/*.csv "$LOCAL/upenn_gbm/meta/"
echo "[prep-upenn] staged $(ls "$LOCAL/upenn_gbm/nifti" | wc -l) files $(date -Is)"
export BRAIN_DATASETS="$LOCAL" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u -m brain.prep.run --root "$STORE" --datasets upenn_gbm --workers 11 --manifest manifest_v6_upenn_gbm_0of1.jsonl \
  --brain-mask auto --gpu-mask --mask-batch 4 --reg-backend sitk ${LIMIT:+--limit-per-dataset $LIMIT} ${EXTRA:-} \
  --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN"
echo "[prep-upenn] done $(date -Is)"
