#!/bin/bash
# Diagnostic: do earlier blocks keep the tissue-contrast information the final layer discards? Frozen tasks read out
# from block 4 alone and from block 4 + the final layer (BRAIN_EVAL_TAPS), and WMH / ATLAS segmentation with block 4
# added to the decoder input; final-layer-only results already exist for comparison.
#SBATCH -p pi_dbertsim
#SBATCH --job-name=layer-diag
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o logs/layer_diag_%j.out
#SBATCH -e logs/layer_diag_%j.out
set -uo pipefail
cd /home/roku/brain-model
source jobs/_env.sh
TASKS="pixar_age ms3seg_age valdo_epvs oasis2_cdr ixi_age oasis1_age msd cq500"
export BRAIN_EVAL_TAPS=3
for mode in only final; do
  if [[ $mode == only ]]; then export BRAIN_EVAL_TAPS_ONLY=1; else unset BRAIN_EVAL_TAPS_ONLY; fi
  ck=runs/v3/snapshots/v3_cls_s42_b4${mode}/step_0018000.pt
  for t in $TASKS; do echo "===== $mode $t"; "$PY" -u -m brain.downstream.suite run --task "$t" --ckpt "$ck" || echo "FAILED $t"; done
done
unset BRAIN_EVAL_TAPS BRAIN_EVAL_TAPS_ONLY
for t in wmh atlas; do
  echo "===== seg $t taps=3"
  "$PY" -u -m brain.downstream.suite run --task "$t" --ckpt runs/v3/snapshots/v3_cls_s42_b4final/step_0018000.pt \
    -- --aug 1 --steps 2000 --patience 3 --taps 3 || echo "FAILED $t"
done
"$PY" -u -m brain.downstream.db ingest
