#!/bin/bash
# Format-6 prep of SynthRAD2023 brain (brain.prep.sources.synthrad): extract only Task{1,2}/brain/ (9.5 GB) of the
# pool zips onto this node's local disk (never scratch), point BRAIN_DATASETS there, prep, delete the copy.
# 180 CT + T1 MR (task 1) and 180 CT + CBCT (task 2); the secondary scan reuses the primary's registration.
#   sbatch jobs/prep_synthrad.sbatch
#   sbatch --export=ALL,LIMIT=4,TAG=pilot jobs/prep_synthrad.sbatch
#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --job-name=prep-synthrad
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --exclude=node3006
#SBATCH -o logs/prep_synthrad_%j.out
#SBATCH -e logs/prep_synthrad_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
POOL=/home/roku/orcd/pool/datasets/synthrad2023
LOCAL="/tmp/${USER}_synthrad_${SLURM_JOB_ID:-local}"
TAG=${TAG:-0of1}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cleanup() { rm -rf "$LOCAL"; }
trap cleanup EXIT
echo "[prep-synthrad] $(date -Is) host=$(hostname)"
"$PY" - "$POOL" "$LOCAL/datasets/synthrad2023" <<'EOF'
import sys, zipfile
src, dest = sys.argv[1:]
for t in ("Task1", "Task2"):
    with zipfile.ZipFile(f"{src}/{t}.zip") as zf:
        names = [n for n in zf.namelist() if n.startswith(f"{t}/brain/") and n.endswith(".nii.gz")]
        zf.extractall(dest, names)
        print(f"[stage] {t}: {len(names)} files", flush=True)
EOF
export BRAIN_DATASETS="$LOCAL/datasets"
"$PY" -u -m brain.prep.run --root "$STORE" --datasets synthrad --shard 0/1 --workers 11 \
  --manifest "manifest_v6_synthrad_${TAG}.jsonl" --brain-mask auto --gpu-mask --mask-batch 4 --reg-backend sitk \
  ${LIMIT:+--limit-per-dataset $LIMIT} ${EXTRA:-} --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN"
echo "[prep-synthrad] done $(date -Is)"
