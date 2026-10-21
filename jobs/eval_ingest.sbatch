#!/bin/bash
# Fold new experiment records into results/experiments.sqlite and print the headline report.
#   sbatch --dependency=afterany:<eval array> jobs/eval_ingest.sbatch

#SBATCH -p mit_normal
#SBATCH --job-name=eval-ingest
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o logs/eval_ingest_%j.out
#SBATCH -e logs/eval_ingest_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
"$PY" -u -m brain.downstream.db ingest
"$PY" -u -m brain.downstream.db report --wide
"$PY" -u -m brain.downstream.db report --probes
