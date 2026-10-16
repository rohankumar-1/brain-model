#!/bin/bash
# Segmentation decoder on frozen 3D BrainModel tokens, 1 L40S. Each task builds its token cache once per
# checkpoint, so tasks run in one job (the shared mask cache is written by the first checkpoint).
#   sbatch --export=ALL,TASKS="wmh seg_cq500",CKPTS="runs/snap/predict_8k.pt runs/snap/continue_8k.pt" jobs/seg3d.sbatch
# EXTRA_ARGS is appended to every downstream.seg3d call (e.g. "--steps 6000").
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
cd /home/roku/brain-model
mkdir -p logs

unset PYTHONPATH
export PYTHONNOUSERSITE=True OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 BLOSC_NTHREADS=1
PY=/home/roku/.conda/envs/ml-torch/bin/python
export PYTHONPATH=/home/roku/brain-model:/home/roku/.local/lib/python3.12/site-packages
NVIDLIBS=$(find /home/roku/.conda/envs/ml-torch/lib/python3.12/site-packages/nvidia \
  /home/roku/.local/lib/python3.12/site-packages/nvidia -type d -name lib 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH="${NVIDLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ "${FINE:-}" = "1" ]; then
  EXTRA_ARGS="--fine ${EXTRA_ARGS:-}"
fi
echo "[host] $(hostname) $(date -Is) tasks=${TASKS:?set TASKS} ckpts=${CKPTS:?set CKPTS} extra=${EXTRA_ARGS:-}"
for t in ${TASKS}; do
  for c in ${CKPTS}; do
    echo "[seg3d-start] $t $c $(date -Is)"
    "$PY" -u -m downstream.seg3d --task "$t" --ckpt "$c" ${EXTRA_ARGS:-}
    echo "[seg3d-done] $t $c $(date -Is)"
  done
done
echo "[done] $(date -Is)"
