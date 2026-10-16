#!/bin/bash
# Frozen 2D ViT downstream: clf/reg (minus RSNA) then a few 2D segs. One L40S.
# TASKS is colon-separated. Full 3D clf/reg suite is jobs/downstream_vit_pi.sbatch.
#
# Submit (CKPT and TASKS env are optional):
#   sbatch jobs/downstream_vit.sbatch

#SBATCH -p mit_normal_gpu
#SBATCH --job-name=vit-down
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -G l40s:1
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o logs/downstream_%j.out
#SBATCH -e logs/downstream_%j.out

set -euo pipefail

cd /home/roku/brain-model
mkdir -p logs

unset PYTHONPATH
export PYTHONNOUSERSITE=True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLOSC_NTHREADS=1

PY=/home/roku/.conda/envs/ml-torch/bin/python
export PYTHONPATH=/home/roku/brain-model:/home/roku/.local/lib/python3.12/site-packages

NVIDLIBS=$(find \
  /home/roku/.conda/envs/ml-torch/lib/python3.12/site-packages/nvidia \
  /home/roku/.local/lib/python3.12/site-packages/nvidia \
  -type d -name lib 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH="${NVIDLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[host] $(hostname) $(date -Is)"
"$PY" - <<'PY'
import torch
print(f"[torch] {torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

CKPT="${CKPT:-runs/vit_tiny/last.pt}"
WINDOW="${WINDOW:-multi}"
# Colon-separated. Commas are eaten by sbatch --export.
if [ -n "${TASKS:-}" ]; then
  IFS=':' read -r -a TASK_LIST <<< "${TASKS}"
else
  # Frozen clf/reg minus RSNA (too large for the 6h mit_normal_gpu cap). Seg stays 2D-decoder.
  TASK_LIST=(
    oasis1_age oasis1_cdr oasis2_cdr oasis2_conversion
    abide_asd
    ixi_age pixar_age piop1_age piop2_age id1000_age
    cq500 msd
    seg_cq500 brats2023_gli brats2023_men brats2024_gli
  )
fi
for t in "${TASK_LIST[@]}"; do
  echo "[task-start] $t $(date -Is)"
  "$PY" -u -m downstream.run_suite --task "$t" --ckpt "$CKPT" --window "$WINDOW" ${RUN_ARGS:-}
  echo "[task-done] $t $(date -Is)"
done
echo "[done] $(date -Is)"
