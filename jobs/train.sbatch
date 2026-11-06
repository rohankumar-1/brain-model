#!/bin/bash
# Training job for every method (python -m brain.train METHOD): resumes from <out_dir>/last.pt and requeues on SIGTERM.
# The header asks for one L40S; pass the partition / GPUs / CPUs / memory on the command line for anything bigger:
#
#   sbatch -p pi_dbertsim -G l40s:4 -c 48 --mem=360G -t 2-00:00:00 -J r14f \
#     --export=ALL,METHOD=multitask,EXPECT_GPUS=4,CONFIG=configs/multitask/r14f.yaml jobs/train.sbatch
#   EXTRA_ARGS="--reset-schedule" (or "--set optim.lr=3e-4") adds trainer arguments; NO_RESUME=1 starts fresh.
# METHOD: lejepa3d | multitask | joint (required with CONFIG). EXPECT_GPUS aborts if the allocation differs.

#SBATCH -p mit_normal_gpu
#SBATCH --exclude=node3006          # broken CUDA install (torch sees no GPU)
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

METHOD="${METHOD:?set METHOD (lejepa3d | multitask | joint)}"
CONFIG="${CONFIG:?set CONFIG (a configs/<method>/<run>.yaml)}"
NGPU="${SLURM_GPUS_ON_NODE:-1}"
if [ -n "${EXPECT_GPUS:-}" ] && [ "$NGPU" -ne "$EXPECT_GPUS" ]; then
  echo "[abort] allocated ${NGPU} GPUs, expected ${EXPECT_GPUS}"
  exit 3
fi
# Variable-shape batches (whole volumes, extra views, segmentation slots) fragment the CUDA cache; without this a
# batch that fits can still OOM with GBs "reserved but unallocated" (multitask run 3, first attempt).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export REQUEUE_FLAG="logs/.requeue_${SLURM_JOB_ID}"
export STOP_FILE="logs/.stop_${SLURM_JOB_ID}"
rm -f "$REQUEUE_FLAG" "$STOP_FILE"

echo "[host] $(hostname) $(date -Is) gpus=${NGPU} cpus=${SLURM_CPUS_PER_TASK:-?} method=${METHOD} config=${CONFIG}"
nvidia-smi -L || true

ARGS=(-m brain.train "$METHOD" --config "$CONFIG" ${NO_RESUME:+--no-resume} ${EXTRA_ARGS:-})
if [ "$NGPU" -gt 1 ]; then
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" "${ARGS[@]}" &
else
  "$PY" -u "${ARGS[@]}" &
fi
PID=$!

# Slurm's warning SIGTERM (--signal=B:TERM@180) reaches this shell only. Ask the trainer to stop through STOP_FILE:
# it notices within ``train.stop_poll`` steps, writes last.pt, sets REQUEUE_FLAG and exits cleanly. (Forwarding the
# signal through torchrun kills the ranks and their DataLoader workers before they can checkpoint.) If it has not
# exited after 150 s, fall back to the signal.
on_term() {
  touch "$STOP_FILE"
  echo "[stop] $(date -Is) asked the trainer to checkpoint and stop"
  ( sleep 150; kill -0 "$PID" 2>/dev/null && kill -TERM "$PID" 2>/dev/null ) &
}
trap on_term TERM
wait "$PID"; rc=$?
if [ "$rc" -gt 128 ]; then wait "$PID"; rc=$?; fi
echo "[exit] rc=${rc} $(date -Is)"

rm -f "$STOP_FILE"
if [ -f "$REQUEUE_FLAG" ]; then
  rm -f "$REQUEUE_FLAG"
  echo "[requeue] resuming from last.pt"
  scontrol requeue "$SLURM_JOB_ID"
fi
exit "$rc"
