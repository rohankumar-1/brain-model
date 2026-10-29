#!/bin/bash
# Format-6 prep of RSNA Intracranial Aneurysm Detection (brain.prep.sources.rsna_ica): 4,348 series (CTA 1,808,
# TOF MRA 1,252, T2 983, T1 post-contrast 305) read straight from the 215 GB pool zip by byte offsets (no staging),
# DICOM extracted to node-local temp. dcm2niix reader (enhanced multi-frame MR) with the SimpleITK fallback; CTAs
# whose stored pixels are already HU under a stale rescale header are undone (dicom.fix_ct_rescale, docs/prep.md).
# Pilot: jobs/archive/rsna_ica_pilot.sbatch. Four shards; re-submit to resume (done cases are skipped).
#   sbatch jobs/prep_rsna_ica.sbatch
#   sbatch --array=0 --export=ALL,N=1,EXTRA="--cases logs/rsna_ica_redo.txt --force" jobs/prep_rsna_ica.sbatch
#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --job-name=prep-rsna-ica
#SBATCH --array=0-3
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --exclude=node3006
#SBATCH -o logs/prep_rsna_ica_%A_%a.out
#SBATCH -e logs/prep_rsna_ica_%A_%a.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
N=${N:-4}                                   # N=1 with --array=0 for a case list (EXTRA="--cases FILE")
I=${SLURM_ARRAY_TASK_ID:-0}
export BRAIN_DICOM_READER=dcm2niix OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[prep-rsna-ica] shard $I/$N $(date -Is)"
"$PY" -u -m brain.prep.run --root "$STORE" --datasets rsna_ica --shard "$I/$N" --workers 11 \
  --manifest "manifest_v6_rsna_ica_${I}of${N}.jsonl" --brain-mask auto --gpu-mask --mask-batch 4 --reg-backend sitk \
  ${LIMIT:+--limit-per-dataset $LIMIT} ${EXTRA:-} --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN"
echo "[prep-rsna-ica] done $(date -Is)"
