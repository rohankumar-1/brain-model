#!/bin/bash
# Format-6 prep of UCSF-PDGM (brain.prep.sources.ucsf_pdgm): 501 scans x T1 / T1c / T2 / FLAIR with the shipped brain
# masks and tumour segmentations, staged from pool onto node-local disk (never scratch). Held out of pretraining
# (index.HELD_OUT); patients also in BraTS 2021 share their BraTS twin's dedup_key (index.ucsf_pdgm_key).
# Then the index rebuild (jobs/index_v6.sbatch) and the downstream labels.
#   sbatch --export=ALL,LIMIT=3 jobs/prep_ucsf_pdgm.sbatch     # pilot: 3 scans per sequence (kept; the full run skips them)
#   sbatch jobs/prep_ucsf_pdgm.sbatch
# EXTRA adds brain.prep.run arguments (no commas). 2026-10-29 re-prep of the T1c / T2 / FLAIR volumes onto their T1's
# registration (pipeline.shared_registration; zips without reg_shared are stale):
#   sbatch --export=ALL,EXTRA="--cases logs/ucsf_pdgm_siblings.txt --force" jobs/prep_ucsf_pdgm.sbatch

#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --job-name=prep-ucsf-pdgm
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --exclude=node3006
#SBATCH -o logs/prep_ucsf_pdgm_%j.out
#SBATCH -e logs/prep_ucsf_pdgm_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
SRC=/home/roku/orcd/pool/datasets/ucsf_pdgm
LOCAL="/tmp/${USER}_ucsf_pdgm_${SLURM_JOB_ID:-local}"
cleanup() { rm -rf "${LOCAL:?}"; }
trap cleanup EXIT
mkdir -p "$LOCAL/ucsf_pdgm/nifti" "$LOCAL/ucsf_pdgm/meta"
ls "$SRC/nifti" | grep -E '_(T1|T1c|T2|FLAIR|brain_segmentation|tumor_segmentation)\.nii\.gz$' \
  | sed "s#^#$SRC/nifti/#" | xargs -r cp -t "$LOCAL/ucsf_pdgm/nifti"
cp "$SRC"/meta/*.csv "$LOCAL/ucsf_pdgm/meta/"
echo "[prep-ucsf] staged $(ls "$LOCAL/ucsf_pdgm/nifti" | wc -l) files $(date -Is)"
export BRAIN_DATASETS="$LOCAL" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u -m brain.prep.run --root "$STORE" --datasets ucsf_pdgm --workers 11 --manifest manifest_v6_ucsf_pdgm_0of1.jsonl \
  --brain-mask auto --gpu-mask --mask-batch 4 --reg-backend sitk ${LIMIT:+--limit-per-dataset $LIMIT} ${EXTRA:-} \
  --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN"
echo "[prep-ucsf] done $(date -Is)"
