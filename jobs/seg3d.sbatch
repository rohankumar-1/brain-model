#!/bin/bash
# Segmentation decoder on frozen 3D BrainModel tokens, 1 L40S. Each task builds its token cache once per
# checkpoint, so tasks run in one job (the shared mask cache is written by the first checkpoint).
#   sbatch --export=ALL,TASKS="wmh seg_cq500",CKPTS="runs/snap/predict_8k.pt runs/snap/continue_8k.pt" jobs/seg3d.sbatch
# EXTRA_ARGS is appended to every brain.downstream.seg3d call (e.g. "--steps 6000").
# Do not put commas in EXTRA_ARGS: Slurm --export splits on them. Use colons, or FINE=1.
#   sbatch --export=ALL,FINE=1,TASKS="wmh",CKPTS="runs/snap/predict_8k.pt" jobs/seg3d.sbatch

#SBATCH -p pi_dbertsim
#SBATCH --job-name=seg3d
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o logs/seg3d_%j.out
#SBATCH -e logs/seg3d_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"

if [ "${FINE:-}" = "1" ]; then
  EXTRA_ARGS="--fine ${EXTRA_ARGS:-}"
fi
echo "[host] $(hostname) $(date -Is) tasks=${TASKS:?set TASKS} ckpts=${CKPTS:?set CKPTS} extra=${EXTRA_ARGS:-}"
for t in ${TASKS}; do
  for c in ${CKPTS}; do
    echo "[seg3d-start] $t $c $(date -Is)"
    "$PY" -u -m brain.brain.downstream.seg3d --task "$t" --ckpt "$c" ${EXTRA_ARGS:-}
    echo "[seg3d-done] $t $c $(date -Is)"
  done
done
echo "[done] $(date -Is)"
