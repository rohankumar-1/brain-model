#!/bin/bash
# Frozen VAE embeddings → MSD enhancing_high classifier (L40S).
#   sbatch jobs/test_frozen_clf_msd.sbatch

#SBATCH -p mit_normal_gpu
#SBATCH --job-name=msd_clf
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -G l40s:1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/msd_clf_%j.out
#SBATCH -e /dev/null

set -euo pipefail

VAE_CKPT="${VAE_CKPT:-runs/vae_256_v2/best.pt}"
OUT_DIR="${OUT_DIR:-runs/frozen_clf_msd}"

cd /home/roku/brain-model
mkdir -p logs

unset PYTHONPATH
export PYTHONNOUSERSITE=True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

PY=/home/roku/.conda/envs/ml-torch/bin/python
export PYTHONPATH=/home/roku/brain-model:/home/roku/.local/lib/python3.12/site-packages

NVIDLIBS=$(find \
  /home/roku/.conda/envs/ml-torch/lib/python3.12/site-packages/nvidia \
  /home/roku/.local/lib/python3.12/site-packages/nvidia \
  -type d -name lib 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH="${NVIDLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[host] $(hostname)"
"$PY" -m downstream.run_suite \
  --task msd \
  --ckpt "${VAE_CKPT}" \
  --slices-root data/processed/slices_zarr \
  --out-root "${OUT_DIR}"
echo "[done] $(date -Is)"
