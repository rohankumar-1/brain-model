#!/bin/bash
# E1 (configs/multitask/e1_enclr1.yaml) vs BQ-12k-tokens: own heads on the trained tasks' test splits, and own heads
# transferred to CQ500 / external cohorts. The frozen benchmark is submitted separately (python -m brain.bench run).
#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --exclude=node3006
#SBATCH --job-name=e1-eval
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o logs/e1_eval_%j.out
#SBATCH -e logs/e1_eval_%j.out
set -uo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
export OMP_NUM_THREADS=8
E1=runs/multitask/e1_enclr1/best.pt
"$PY" -u -m brain.multitask.test_eval --ckpts runs/multitask/omni_b_tokens/best.pt "$E1" --out runs/diagnostics/test_eval_e1
: "$PY" -u -m brain.multitask.transfer_heads --ckpts BQ-12k-tokens "$E1" --out runs/bench/transfer_e1
