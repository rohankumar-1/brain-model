#!/bin/bash
# SSL pretraining on pi_dbertsim, 2x L40S (DDP). Same variables as jobs/train.sbatch.
# The partition requires a GPU GRES and allows 12 CPUs per GPU; batch_size in the config is per GPU.
# SIGTERM checkpoints last.pt and --requeue resumes. Override wall time with: sbatch -t 2-00:00:00 ...
#
#   sbatch --export=ALL,METHOD=lejepa3d,CONFIG=configs/lejepa3d/small.yaml jobs/train_pi.sbatch

#SBATCH -p pi_dbertsim
#SBATCH --exclude=node3006          # broken CUDA install (torch sees no GPU)
#SBATCH --job-name=train-2gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 24
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@180
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.out

export EXPECT_GPUS=2
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/train.sbatch"
