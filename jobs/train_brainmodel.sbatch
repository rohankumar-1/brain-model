#!/bin/bash
# 3D BrainModel SSL. Resumes from <out_dir>/last.pt and requeues on SIGTERM.
#
#   sbatch jobs/train_brainmodel.sbatch
#   sbatch --export=ALL,CONFIG=configs/brainmodel_tiny_lejepa.yaml jobs/train_brainmodel.sbatch

#SBATCH -p mit_normal_gpu
#SBATCH --job-name=brainmodel
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -G l40s:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@180
#SBATCH -o logs/brainmodel_%j.out
#SBATCH -e logs/brainmodel_%j.out

set -uo pipefail
cd /home/roku/brain-model
mkdir -p logs

CONFIG="${CONFIG:-configs/brainmodel_tiny_lejepa.yaml}"
NGPU="${SLURM_GPUS_ON_NODE:-1}"
if [ -n "${EXPECT_GPUS:-}" ] && [ "$NGPU" -ne "$EXPECT_GPUS" ]; then
  echo "[abort] allocated ${NGPU} GPUs, expected ${EXPECT_GPUS}"
  exit 3
fi

unset PYTHONPATH
export PYTHONNOUSERSITE=True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1
PY=/home/roku/.conda/envs/ml-torch/bin/python
export PYTHONPATH=/home/roku/brain-model:/home/roku/.local/lib/python3.12/site-packages
NVIDLIBS=$(find /home/roku/.conda/envs/ml-torch/lib/python3.12/site-packages/nvidia \
  /home/roku/.local/lib/python3.12/site-packages/nvidia -type d -name lib 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH="${NVIDLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export REQUEUE_FLAG="logs/.requeue_${SLURM_JOB_ID}"
rm -f "$REQUEUE_FLAG"

echo "[host] $(hostname) $(date -Is) gpus=${NGPU} cpus=${SLURM_CPUS_PER_TASK:-?} config=${CONFIG}"
nvidia-smi -L || true

ARGS=(train_brainmodel.py --config "$CONFIG" ${NO_RESUME:+--no-resume} ${EXTRA_ARGS:-})
if [ "$NGPU" -gt 1 ]; then
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" "${ARGS[@]}" &
else
  "$PY" -u "${ARGS[@]}" &
fi
PID=$!

trap 'kill -TERM "$PID" 2>/dev/null' TERM
wait "$PID"; rc=$?
if [ "$rc" -gt 128 ]; then wait "$PID"; rc=$?; fi
echo "[exit] rc=${rc} $(date -Is)"

if [ -f "$REQUEUE_FLAG" ]; then
  rm -f "$REQUEUE_FLAG"
  echo "[requeue] resuming from last.pt"
  scontrol requeue "$SLURM_JOB_ID"
fi
exit "$rc"
