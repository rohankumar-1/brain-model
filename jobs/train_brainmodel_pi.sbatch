#!/bin/bash
# Tiny 3D BrainModel SSL on pi_dbertsim, 2x L40S (DDP).
# Partition requires a GPU GRES and allows 12 CPUs per GPU.
#
#   sbatch jobs/train_brainmodel_pi.sbatch
#   sbatch --export=ALL,CONFIG=configs/brainmodel_tiny_lejepa.yaml jobs/train_brainmodel_pi.sbatch

#SBATCH -p pi_dbertsim
#SBATCH --job-name=bm-tiny
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 24
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@180
#SBATCH -o logs/brainmodel_%j.out
#SBATCH -e logs/brainmodel_%j.out

export EXPECT_GPUS=2
export CONFIG="${CONFIG:-configs/brainmodel_tiny_lejepa.yaml}"
source /home/roku/brain-model/jobs/train_brainmodel.sbatch
