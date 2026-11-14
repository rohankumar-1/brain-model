#!/bin/bash
# R19 joint (ViT-B, learned window bank in both branches) real-data smoke + memory profile on ONE L40S:
# joint.ssl_batch_size 4, then 2; task batch 4 (optim.batch_size); labels v12 until v13 exists. Both branches are
# active from step 20 (warm-up / ramp given in steps). Checkpoints deleted after each arm.
#   BRAIN_REPO=<repo or worktree> sbatch jobs/r19_joint_smoke.sbatch
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=r19-joint-smoke
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH --exclude=node3006,node4207
#SBATCH -o /home/roku/brain-model/logs/r19_joint_smoke_%j.out
#SBATCH -e /home/roku/brain-model/logs/r19_joint_smoke_%j.out
set -uo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
D=/home/roku/brain-model/runs/diagnostics/r19_joint_smoke
LAB=/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr/labels/multitask_v12
mkdir -p "$D"
echo "[smoke] $(date -Is) host=$(hostname) repo=$PWD $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
for SB in ${SSL_BATCHES:-4 2}; do
  OUT=$D/sslb$SB
  rm -rf "$OUT"
  "$PY" -u -m brain.train joint --config configs/joint/r19.yaml --max-steps "${STEPS:-80}" --log-every 10 \
    --out-dir "$OUT" --set data.labels_dir=$LAB --set joint.ssl_batch_size=$SB --set joint.warmup_steps=10 \
    --set joint.task_ramp_steps=10 --set joint.taper_end_steps=60 --set train.eval_every=100000 \
    --set train.ckpt_every=100000 --set train.keep_every=100000 --set optim.warmup_steps=10
  echo "[smoke] ssl_batch_size=$SB exit=$? $(date -Is)"
  find "$OUT" -maxdepth 1 -name '*.pt' -delete
done
