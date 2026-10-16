#!/bin/bash
# LeJEPA pretraining on 2x L40S (DDP) on pi_dbertsim, one node.
# The partition requires a GPU GRES and allows 12 CPUs per GPU, so 24 CPUs for 2 GPUs.
# Resumes from <out_dir>/last.pt, checkpoints on SIGTERM, requeues near the time limit.
#
# Submit:
#   sbatch jobs/train_vit_2gpu_pi.sbatch
#   sbatch --export=ALL,CONFIG=configs/vit_tiny_v4.yaml jobs/train_vit_2gpu_pi.sbatch

#SBATCH -p pi_dbertsim
#SBATCH --job-name=vit-v4-2gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 24
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@180
#SBATCH -o logs/vit_%j.out
#SBATCH -e logs/vit_%j.out

export EXPECT_GPUS=2
export CONFIG="${CONFIG:-configs/vit_tiny_v4.yaml}"
source /home/roku/brain-model/jobs/train_vit.sbatch
