#!/bin/bash
# Frozen-token probes (dense lesion, fingerprint, location) on 1 L40S.
#   sbatch --export=ALL,CKPTS="runs/a.pt runs/b.pt",OUT=runs/probes/x.json jobs/probe3d.sbatch
# CONTROLS (default "--pixel --random-init") adds the pixel-statistics and random-init baselines.

#SBATCH -p pi_dbertsim
#SBATCH --job-name=probe3d
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/probe3d_%j.out
#SBATCH -e logs/probe3d_%j.out

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

echo "[host] $(hostname) $(date -Is) ckpts=${CKPTS}"
ARGS=()
for c in ${CKPTS}; do ARGS+=(--ckpt "$c"); done
"$PY" -u -m downstream.probe3d "${ARGS[@]}" ${CONTROLS:---pixel --random-init} --out "${OUT:?set OUT}" ${EXTRA_ARGS:-}
echo "[exit] $(date -Is)"
