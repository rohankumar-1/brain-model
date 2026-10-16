#!/bin/bash
# Generic CPU job for the v4 preprocessing package.
# Submit: sbatch --export=ALL,CMD="-m brain.prep.run --root STORE --shard 0/8" [-c 16 --mem 64G] jobs/prep_job.sbatch

#SBATCH -p mit_normal
#SBATCH --job-name=prep
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/prep_%j.out
#SBATCH -e logs/prep_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
"$PY" -u ${CMD:?}
