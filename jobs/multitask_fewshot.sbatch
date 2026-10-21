#!/bin/bash
# Few-shot (held-out tasks) and transfer (trained tasks on new cohorts) evaluation of a multitask checkpoint.
#
#   sbatch --export=ALL,CKPT=runs/multitask/v3init_s42/best.pt jobs/multitask_fewshot.sbatch
#   sbatch --export=ALL,CKPT=...,EXTRA_ARGS="--shots 16 64 --tasks cq500_hemorrhage dementia" jobs/multitask_fewshot.sbatch

#SBATCH -p pi_dbertsim
#SBATCH --job-name=mt-fewshot
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=80G
#SBATCH -t 06:00:00
#SBATCH -o logs/multitask_fewshot_%j.out
#SBATCH -e logs/multitask_fewshot_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
: "${CKPT:?set CKPT=<checkpoint>}"
echo "[mt-fewshot] $(hostname) $(date -Is) ckpt=${CKPT}"
"$PY" -u -m brain.multitask.fewshot --ckpt "$CKPT" --workers 10 ${EXTRA_ARGS:-}
echo "[mt-fewshot] done $(date -Is)"
