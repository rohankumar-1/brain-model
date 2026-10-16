#!/bin/bash
# ViT-Small win-inv on 2x L40S. Longer wall than Tiny; SIGTERM requeues from last.pt.
#
# Submit:
#   sbatch jobs/train_vit_small_pi.sbatch

#SBATCH -p pi_dbertsim
#SBATCH --job-name=vit-small
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 24
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@180
#SBATCH -o logs/vit_%j.out
#SBATCH -e logs/vit_%j.out

export EXPECT_GPUS=2
export CONFIG="${CONFIG:-configs/vit_small_v4_wininv.yaml}"
source /home/roku/brain-model/jobs/train_vit.sbatch
