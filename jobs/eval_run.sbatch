#!/bin/bash
# Downstream evaluation of one model as ONE job: the suite entries in SUITE_TASKS run one after another on a single GPU
# (a failing entry is logged and skipped), then the records are ingested into results/experiments.sqlite. No array and
# no separate ingest job, so it queues as a single small job. Submitted by jobs/submit_eval.sh and brain.bench run.
# CKPT is a checkpoint or a configs/bench/models.yaml name.
#
#   sbatch --export=ALL,CKPT=runs/x/last.pt,SUITE_TASKS="cq500 ixi_age" jobs/eval_run.sbatch
#   EXTRA="--steps 3000" adds evaluator arguments (spaces, no commas: sbatch --export splits on commas).

#SBATCH -p pi_dbertsim
#SBATCH --exclude=node3006          # broken CUDA install (torch sees no GPU)
#SBATCH --job-name=eval
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o logs/eval_run_%j.out
#SBATCH -e logs/eval_run_%j.out

set -uo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
CKPT="${CKPT:?set CKPT}"
read -r -a TASKS <<< "${SUITE_TASKS:?set SUITE_TASKS}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
echo "[eval-run] $(date -Is) host=$(hostname) ckpt=$CKPT tasks=${#TASKS[@]}: ${TASKS[*]}"
ok=() failed=()
for task in "${TASKS[@]}"; do
  echo "[eval-run] ===== $task  $(date -Is)"
  # shellcheck disable=SC2086
  if "$PY" -u -m brain.downstream.suite run --task "$task" --ckpt "$CKPT" -- ${EXTRA:-}; then ok+=("$task")
  else failed+=("$task"); echo "[eval-run] FAILED $task (continuing)"; fi
done
"$PY" -u -m brain.downstream.db ingest
echo "[eval-run] done $(date -Is): ok=${#ok[@]} [${ok[*]}] failed=${#failed[@]} [${failed[*]}]"
(( ${#failed[@]} == 0 ))
