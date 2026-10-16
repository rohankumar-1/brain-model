#!/bin/bash
# Evaluate VAE checkpoint on held-out test fold (L40S).
#   sbatch jobs/test_vae_256.sbatch

#SBATCH -p mit_normal_gpu
#SBATCH --job-name=vae_test
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -G l40s:1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/vae_test_%j.out
#SBATCH -e /dev/null

set -euo pipefail

RUN_DIR="${RUN_DIR:-runs/vae_256_v2}"

cd /home/roku/brain-model
mkdir -p logs

unset PYTHONPATH
export PYTHONNOUSERSITE=True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

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

echo "[run-dir] ${RUN_DIR}"
"$PY" test_vae.py --run-dir "${RUN_DIR}" --fold test
echo "[done] $(date -Is)"
