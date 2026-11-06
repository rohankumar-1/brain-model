#!/bin/bash
# Memory / throughput profile for the fresh model (2026-11-04): conv-stem ViT-S vs ViT-B, LeJEPA pre-training and
# multitask post-training from scratch, 40 steps each on one GPU; peak memory and vol/s from the [step] rows.
#   sbatch -p pi_dbertsim --gres=gpu:l40s:1 -c 12 --mem=96G -t 02:00:00 -J profile -o logs/profile_%j.out jobs/profile_vitb.sh
set -uo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
T=${TMPDIR:-/tmp}/profile_${SLURM_JOB_ID:-local}
COMMON=(--no-resume --max-steps 40 --log-every 10)
run() {   # name, method, config, extra --set ...
  local name=$1 method=$2 cfg=$3; shift 3
  echo "=== $name"
  "$PY" -u -m brain.train "$method" --config "$cfg" --out-dir "$T/$name" "${COMMON[@]}" "$@" 2>&1 \
    | grep -E "^\[model\]|^\[step +40|OutOfMemory|out of memory|Traceback|Error" | cut -c1-400
  rm -rf "$T/$name"
}
STEM=(--set model.conv_stem=[32,64])
for size in small base; do
  run "ssl_${size}" lejepa3d configs/lejepa3d/v4_scratch.yaml --set model.size=$size "${STEM[@]}"
done
MT=(--set model.init_from=null --set output.resume_from=null --set tasks.max_seg=3 --set train.eval_every=1000000
    --set train.ckpt_every=1000000 --set train.ema_decay=0 "${STEM[@]}")
run mt_small_b8 multitask configs/multitask/r14.yaml --set model.size=small --set optim.batch_size=8 "${MT[@]}"
run mt_base_b4 multitask configs/multitask/r14.yaml --set model.size=base --set optim.batch_size=4 "${MT[@]}"
run mt_base_b8 multitask configs/multitask/r14.yaml --set model.size=base --set optim.batch_size=8 "${MT[@]}"
echo "=== done"
