#!/bin/bash
# Frozen clf/reg suite on pi_dbertsim (1 L40S, 12 CPUs). Multi-window asinh by default.
# Default TASK_LIST is every classification and age task. Segmentation is jobs/seg3d.sbatch.
# Override with colon-separated TASKS (commas are eaten by sbatch --export).
#
# Submit:
#   sbatch --export=ALL,CKPT=runs/lejepa3d/last.pt jobs/downstream_vit_pi.sbatch
#   sbatch --export=ALL,CKPT=runs/lejepa2d/last.pt,TASKS=oasis1_age:cq500 jobs/downstream_vit_pi.sbatch
# Native 8x8x4 tokens (32^3, no 2x2x2 pool): BRAIN_FINE_GRID=1 (do not also set BRAIN_FINE_POOL).

#SBATCH -p pi_dbertsim
#SBATCH --job-name=vit-down
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH -t 18:00:00
#SBATCH -o logs/downstream_%j.out
#SBATCH -e logs/downstream_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"

echo "[host] $(hostname) $(date -Is)"
"$PY" - <<'PY'
import torch
print(f"[torch] {torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

CKPT="${CKPT:-runs/lejepa2d/last.pt}"
WINDOW="${WINDOW:-multi}"
SLICES="${SLICES:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
if [ -n "${TASKS:-}" ]; then
  IFS=':' read -r -a TASK_LIST <<< "${TASKS}"
else
  # CLASSIFICATION_TASKS + REGRESSION_TASKS. RSNA is last because it dominates embed time.
  TASK_LIST=(
    oasis1_age oasis1_cdr oasis2_cdr oasis2_conversion
    abide_asd
    ixi_age pixar_age piop1_age piop2_age id1000_age
    cq500 msd
    rsna rsna_ich
  )
fi

for t in "${TASK_LIST[@]}"; do
  echo "[task-start] $t $(date -Is)"
  "$PY" -u -m brain.downstream.run_suite --task "$t" --ckpt "$CKPT" --window "$WINDOW" \
    --slices-root "$SLICES" ${RUN_ARGS:-}
  echo "[task-done] $t $(date -Is)"
done
echo "[done] $(date -Is)"
