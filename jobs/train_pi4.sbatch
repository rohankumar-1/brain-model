#!/bin/bash
# SSL pretraining on pi_dbertsim, 4x L40S (one full node) (DDP). Same variables as jobs/train.sbatch.
# The partition requires a GPU GRES and allows 12 CPUs per GPU; batch_size in the config is per GPU.
# SIGTERM checkpoints last.pt and --requeue resumes. Override wall time with: sbatch -t 2-00:00:00 ...
#
#   sbatch --export=ALL,METHOD=lejepa3d,CONFIG=configs/lejepa3d/small.yaml jobs/train_pi4.sbatch

#SBATCH -p pi_dbertsim
#SBATCH --job-name=train-4gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 48
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=192G
#SBATCH -t 2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@180
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.out

export EXPECT_GPUS=4
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/train.sbatch"
