#!/bin/bash
# SSL pretraining, any method, one L40S. Resumes from <out_dir>/last.pt and requeues on SIGTERM.
#
#   sbatch --export=ALL,METHOD=lejepa3d,CONFIG=configs/lejepa3d/tiny.yaml jobs/train.sbatch
#   sbatch --export=ALL,METHOD=dino2d jobs/train.sbatch                    # CONFIG defaults to configs/<METHOD>/<tiny|small>.yaml
#   sbatch --export=ALL,METHOD=lejepa2d,EXTRA_ARGS="--set optim.lr=3e-4" jobs/train.sbatch
#   jobs/train_pi.sbatch (2 GPUs) and jobs/train_pi4.sbatch (4 GPUs) take the same variables.
# METHOD: lejepa2d | dino2d | lejepa3d. NO_RESUME=1 starts fresh. EXPECT_GPUS aborts if the allocation differs.

#SBATCH -p mit_normal_gpu
#SBATCH --job-name=train
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -G l40s:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@180
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.out

set -uo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"

METHOD="${METHOD:-lejepa3d}"
case "$METHOD" in dino2d) DEFAULT_CFG=small ;; *) DEFAULT_CFG=tiny ;; esac
CONFIG="${CONFIG:-configs/${METHOD}/${DEFAULT_CFG}.yaml}"
NGPU="${SLURM_GPUS_ON_NODE:-1}"
if [ -n "${EXPECT_GPUS:-}" ] && [ "$NGPU" -ne "$EXPECT_GPUS" ]; then
  echo "[abort] allocated ${NGPU} GPUs, expected ${EXPECT_GPUS}"
  exit 3
fi
export REQUEUE_FLAG="logs/.requeue_${SLURM_JOB_ID}"
rm -f "$REQUEUE_FLAG"

echo "[host] $(hostname) $(date -Is) gpus=${NGPU} cpus=${SLURM_CPUS_PER_TASK:-?} method=${METHOD} config=${CONFIG}"
nvidia-smi -L || true

ARGS=(-m brain.train "$METHOD" --config "$CONFIG" ${NO_RESUME:+--no-resume} ${EXTRA_ARGS:-})
if [ "$NGPU" -gt 1 ]; then
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" "${ARGS[@]}" &
else
  "$PY" -u "${ARGS[@]}" &
fi
PID=$!

# SIGTERM reaches this shell only (B:), so forward it to the trainer, then wait for it.
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
