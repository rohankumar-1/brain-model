#!/bin/bash
# Frozen VAE embedding → CQ500 multi-label classifier (L40S).
#   sbatch jobs/test_frozen_clf.sbatch

#SBATCH -p mit_normal_gpu
#SBATCH --job-name=frozen_clf
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -G l40s:1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/frozen_clf_%j.out
#SBATCH -e /dev/null

set -euo pipefail

VAE_CKPT="${VAE_CKPT:-runs/vae_256_v2/best.pt}"
OUT_DIR="${OUT_DIR:-runs/frozen_clf_cq500_v2}"

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
echo "[date] $(date -Is)"
nvidia-smi -L || true
"$PY" - <<'PY'
import torch
print(f"[torch] {torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

echo "[ckpt] ${VAE_CKPT}"
echo "[out]  ${OUT_DIR}"
"$PY" -m downstream.run_suite \
  --task cq500 \
  --ckpt "${VAE_CKPT}" \
  --slices-root data/processed/slices_zarr \
  --out-root "${OUT_DIR}"

echo "[done] $(date -Is)"
