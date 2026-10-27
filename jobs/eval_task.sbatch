#!/bin/bash
# One downstream-suite entry (task or probe3d) for one checkpoint; every evaluator writes experiment records.
# Normally submitted as an array by jobs/submit_eval.sh, which maps SLURM_ARRAY_TASK_ID to an entry of SUITE_TASKS.
#
#   sbatch --export=ALL,CKPT=runs/x/last.pt,SUITE_TASKS="cq500 ixi_age probe3d" --array=0-2 jobs/eval_task.sbatch
#   EXTRA="--steps 3000" adds evaluator arguments (spaces, no commas: sbatch --export splits on commas).

#SBATCH -p pi_dbertsim
#SBATCH --exclude=node3006          # broken CUDA install (torch sees no GPU)
#SBATCH --job-name=eval
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o logs/eval_%A_%a.out
#SBATCH -e logs/eval_%A_%a.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
CKPT="${CKPT:?set CKPT}"
read -r -a TASKS <<< "${SUITE_TASKS:?set SUITE_TASKS}"
TASK="${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
echo "[eval] $(date -Is) host=$(hostname) task=$TASK ckpt=$CKPT"
# shellcheck disable=SC2086
"$PY" -u -m brain.downstream.suite run --task "$TASK" --ckpt "$CKPT" -- ${EXTRA:-}
echo "[eval] done $(date -Is)"
