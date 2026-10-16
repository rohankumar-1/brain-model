#!/bin/bash
# BrainDINO ViT-S on 2x L40S (leaves the 4-GPU node free for 3D training).
# SIGTERM checkpoints last.pt; --requeue resumes.
#
#   sbatch jobs/train_braindino_pi.sbatch

#SBATCH -p pi_dbertsim
#SBATCH --job-name=braindino
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 24
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@180
#SBATCH -o logs/braindino_%j.out
#SBATCH -e logs/braindino_%j.out

export EXPECT_GPUS=2
export CONFIG="${CONFIG:-configs/braindino_small.yaml}"
source /home/roku/brain-model/jobs/train_braindino.sbatch
