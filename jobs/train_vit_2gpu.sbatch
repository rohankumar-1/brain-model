#!/bin/bash
# LeJEPA pretraining on 2x L40S (DDP). Same behavior as train_vit.sbatch: resumes from
# <out_dir>/last.pt, checkpoints on SIGTERM, requeues itself near the time limit.
#
# The GPU count lives in this header (not a command-line override), and the job
# aborts if the allocation is not exactly 2 GPUs, so it never trains at half speed.
# L40S nodes have 16 CPUs per GPU (ORCD docs), hence 32 CPUs for 2 GPUs.
#
# Submit:
#   sbatch --export=ALL,CONFIG=configs/<name>.yaml jobs/train_vit_2gpu.sbatch
# batch_size in the config is per GPU, so the global batch is 2x that value.

#SBATCH -p mit_normal_gpu
#SBATCH --job-name=vit-2gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --gpus-per-node=l40s:2
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@180
#SBATCH -o logs/vit_%j.out
#SBATCH -e logs/vit_%j.out

export EXPECT_GPUS=2
source /home/roku/brain-model/jobs/train_vit.sbatch
