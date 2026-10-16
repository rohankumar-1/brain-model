#!/bin/bash
# Small 3D BrainModel SSL on pi_dbertsim, 4x L40S (one full node, DDP).
# Wall is the partition cap (2 days) so a 15k-step run can finish in one job.
# SIGTERM still checkpoints last.pt; --requeue is a fallback if the cap hits first.
# Partition allows 12 CPUs per GPU. batch_size in the config is per GPU.
#
#   sbatch --export=ALL,CONFIG=configs/brainmodel_small_multiscale.yaml jobs/train_brainmodel_pi4.sbatch

#SBATCH -p pi_dbertsim
#SBATCH --job-name=bm-ms
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 48
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=192G
#SBATCH -t 2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@180
#SBATCH -o logs/brainmodel_%j.out
#SBATCH -e logs/brainmodel_%j.out

export EXPECT_GPUS=4
export CONFIG="${CONFIG:-configs/brainmodel_small_multiscale.yaml}"
source /home/roku/brain-model/jobs/train_brainmodel.sbatch
