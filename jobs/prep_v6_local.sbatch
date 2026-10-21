#!/bin/bash
# Format-6 prep for data that lives only as JPEG-LS archives on pool, unpacked onto this node's local disk (never
# scratch) and deleted at the end. One array task = one shard:
#
#   PART=brats sbatch --export=ALL --array=0-3%2 jobs/prep_v6_local.sbatch   # local BraTS cohorts + GoAT (PT009)
#   PART=hbn   sbatch --export=ALL --array=0-3%2 jobs/prep_v6_local.sbatch   # FOMO HBN (PT018)
#
# brats: every task unpacks all of PT009 (25 GB) so case discovery is identical, then takes --shard i/n of the cases.
# hbn:   every task unpacks only its subjects (crc32 of the subject folder, i/n) of PT018 (170 GB) and preps them all.
# Requeue-safe: unpack skips files already there, prep resumes from its manifest and skips current-format zips.
# REG_BACKEND=torch registers on the GPU in every worker (brain.prep.registration_torch); use MASK_BATCH=4 with it.

#SBATCH -p pi_dbertsim
#SBATCH --job-name=prep-v6-local
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/prep_v6_local_%A_%a.out
#SBATCH -e logs/prep_v6_local_%A_%a.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
POOL_FOMO="${BRAIN_DATASETS:-/home/roku/orcd/pool/datasets}/fomo260k"
PART="${PART:?set PART=brats or PART=hbn}"
N="${SHARD_N:-${SLURM_ARRAY_TASK_COUNT:-1}}"   # SHARD_N: rerun some shards of an n-way split
I="${SLURM_ARRAY_TASK_ID:-0}"
LOCAL="/tmp/${USER}_fomo_v6_${PART}_${I}of${N}"   # node-local; one per shard
W=$(( ${SLURM_CPUS_PER_TASK:-12} - 1 ))
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # many processes share one GPU
case "$PART" in
  brats) ZIP=PT009_BraTS-GEN.jls.zip; SUBSET=();               DATASETS=brats; SHARD=(--shard "$I/$N"); NEED_GB=80 ;;
  hbn)   ZIP=PT018_HBN.jls.zip;       SUBSET=(--subset "$I/$N"); DATASETS=fomo;  SHARD=();                NEED_GB=$(( 400 / N + 40 )) ;;
  *) echo "unknown PART=$PART" >&2; exit 2 ;;
esac
echo "[prep-v6-local] $(date -Is) host=$(hostname) part=$PART shard=$I/$N local=$LOCAL"

avail=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)
if (( avail < NEED_GB )) && [[ ! -d "$LOCAL" ]]; then
  echo "[prep-v6-local] only ${avail}G free on /tmp (need ${NEED_GB}G)" >&2
  exit 4
fi
cleanup() { [[ "${KEEP_LOCAL:-0}" == 1 ]] || rm -rf "$LOCAL"; }
trap cleanup EXIT                                 # shared node disk: never leave the copy behind

mkdir -p "$LOCAL/extracted"
for f in mapping.tsv mri_info.tsv participants.tsv holdout_datasets.txt holdout_report.csv README.md; do
  ln -sfn "$POOL_FOMO/$f" "$LOCAL/$f"
done
echo "[prep-v6-local] unpack $ZIP $(date -Is)"
"$PY" -u -m brain.prep.archive --unpack "$POOL_FOMO/$ZIP" --dest "$LOCAL/extracted" --workers "$W" "${SUBSET[@]}"
df -h /tmp | tail -1

export BRAIN_FOMO="$LOCAL"
EXTRA=()
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=(${EXTRA_ARGS})   # e.g. --require-attr reg_backend=torch
"$PY" -u -m brain.prep.run \
  --root "$STORE" \
  --datasets "$DATASETS" \
  --workers "$W" \
  "${SHARD[@]}" \
  --manifest "manifest_v6_C${PART}_${I}of${N}.jsonl" \
  --brain-mask auto --gpu-mask --mask-batch "${MASK_BATCH:-8}" \
  --reg-backend "${REG_BACKEND:-sitk}" \
  --max-scratch-pct "${MAX_SCRATCH_PCT:-95}" \
  --template "$BRAIN_TEMPLATE" \
  --template-brain "$BRAIN_TEMPLATE_BRAIN" \
  "${EXTRA[@]}"
echo "[prep-v6-local] done $(date -Is)"
