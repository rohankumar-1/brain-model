#!/bin/bash
# Diagnostic: train/eval context mismatch. Token stability whole-volume vs training-size windows, then frozen tasks
# with windowed encoding (BRAIN_EVAL_TILE=1) for the pretrained model and the random-init floor.
#SBATCH -p pi_dbertsim
#SBATCH --job-name=context-diag
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o logs/context_diag_%j.out
#SBATCH -e logs/context_diag_%j.out
set -uo pipefail
cd /home/roku/brain-model
source jobs/_env.sh
"$PY" -u -m brain.downstream.context_check --ckpt runs/v3/snapshots/v3_cls_s42/step_0018000.pt \
  --ckpt runs/baselines/random_small/last.pt --n 24
export BRAIN_EVAL_TILE=1
TASKS="cq500 oasis1_cdr oasis2_cdr msd oasis1_age ixi_age pixar_age id1000_age ms3seg_age valdo_epvs"
for ck in runs/v3/snapshots/v3_cls_s42_tiled/step_0018000.pt runs/baselines/random_small_tiled/last.pt; do
  for t in $TASKS; do
    echo "===== $ck $t"; "$PY" -u -m brain.downstream.suite run --task "$t" --ckpt "$ck" || echo "FAILED $t"
  done
done
"$PY" -u -m brain.downstream.db ingest
