#!/bin/bash
# What the GPU step spends its time on (2026-11-04): conv-stem ViT-B multitask from scratch at batch 4, 40 steps each,
# with one component switched off at a time; vol/s and peak memory from the last [step] row.
#   sbatch -p pi_dbertsim --gres=gpu:l40s:1 -c 12 --mem=96G -t 01:30:00 -J prof-gpu -o logs/profile_gpu_%j.out jobs/profile_gpu.sh
set -uo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
T=${TMPDIR:-/tmp}/profile_gpu_${SLURM_JOB_ID:-local}
BASE=(--no-resume --max-steps 60 --log-every 20 --set model.init_from=null --set output.resume_from=null
      --set model.size=base --set model.conv_stem=[32,64] --set optim.batch_size=4 --set tasks.max_seg=3
      --set train.eval_every=1000000 --set train.ckpt_every=1000000 --set train.ema_decay=0)
run() {
  local name=$1; shift
  echo "=== $name"
  "$PY" -u -m brain.train multitask --config configs/multitask/r14.yaml --out-dir "$T/$name" "${BASE[@]}" "$@" 2>&1 \
    | grep -E "^\[step +60|OutOfMemory|out of memory|Traceback|Error" | awk '{print $(NF-1), $NF}'
  rm -rf "$T/$name"
}
run baseline
run shared_decoder_3 --set model.seg.shared=true
run shared_decoder_all --set model.seg.shared=true --set tasks.all_seg=true --set tasks.max_seg=16
run shared_decoder_all_no_ckpt --set model.seg.shared=true --set tasks.all_seg=true --set tasks.max_seg=16 --set model.checkpoint=false
echo "=== done"
