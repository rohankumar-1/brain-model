#!/bin/bash
# Train ConvVAE on 256x256 CT/MRI slices — single L40S GPU (ORCD Engaging).
# Docs: https://orcd-docs.mit.edu/running-jobs/requesting-resources/
#       https://orcd-docs.mit.edu/recipes/torch-gpu/
#
# Resumes from <out>/last.pt if present. Preemption/time-limit: SIGTERM 120s
# before kill flushes last.pt; --requeue picks the job back up.
#
# Submit:
#   sbatch jobs/train_vae_256.sbatch
# Fresh run (ignore last.pt):
#   sbatch --export=ALL,CONFIG=configs/vae_256_v2.yaml,NO_RESUME=1 jobs/train_vae_256.sbatch

#SBATCH -p mit_normal_gpu
#SBATCH --job-name=vae256v2
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -G l40s:1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH -o logs/vae256_%j.out
#SBATCH -e /dev/null

set -euo pipefail

# Override at submit time: sbatch --export=ALL,CONFIG=configs/foo.yaml jobs/train_vae_256.sbatch
CONFIG="${CONFIG:-configs/vae_256_v2.yaml}"
NO_RESUME="${NO_RESUME:-}"

cd /home/roku/brain-model
mkdir -p logs

# Login shell PYTHONPATH points at a broken/partial user torch; use a clean path.
unset PYTHONPATH
export PYTHONNOUSERSITE=True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Torch + deps live under the conda env and ~/.local (pip CUDA wheels).
PY=/home/roku/.conda/envs/ml-torch/bin/python
export PYTHONPATH=/home/roku/brain-model:/home/roku/.local/lib/python3.12/site-packages

# Pip-installed torch needs NVIDIA shared libs from the wheel packages.
NVIDLIBS=$(find \
  /home/roku/.conda/envs/ml-torch/lib/python3.12/site-packages/nvidia \
  /home/roku/.local/lib/python3.12/site-packages/nvidia \
  -type d -name lib 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH="${NVIDLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[host] $(hostname)"
echo "[date] $(date -Is)"
echo "[cuda] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
"$PY" - <<'PY'
import torch
print(f"[torch] {torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

echo "[config] ${CONFIG}"
if [ -n "${NO_RESUME}" ]; then
  echo "[resume] disabled"
  "$PY" -u train_vae.py --config "${CONFIG}" --no-resume
else
  "$PY" -u train_vae.py --config "${CONFIG}"
fi

echo "[done] $(date -Is)"
echo "[test] run after job finishes:"
echo "  $PY test_vae.py --run-dir <out_dir from config>"
