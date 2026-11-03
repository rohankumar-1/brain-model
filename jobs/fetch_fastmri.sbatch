#!/bin/bash
# Streaming fetch of the fastMRI+ annotated brain scans (brain.prep.fastmri_fetch): the brain multicoil train / val
# batches (~100 GB of tar.xz each, ~75 min at ~25 MB/s, streamed through xz and never stored); only the 1,001
# annotated scans are kept, as coil-combined images, in pool/datasets/fastmri_brain/fastmri_plus_<batch>.zip.
# Three chain jobs (the per-user submit limit rules out a 13-task array), chain K taking archives K, K+3, ...; a
# finished batch (its "done" marker in work/fastmri) is skipped on resubmission. Data use agreement: owner-only files.
#   for k in 0 1 2; do sbatch --export=ALL,CHAIN=$k,CHAINS=3 jobs/fetch_fastmri.sbatch; done
#   sbatch --export=ALL,ONLY="13 14 15" jobs/fetch_fastmri.sbatch    # test_full 0-2, only if scans are still missing
#SBATCH -p mit_data_transfer,mit_normal
#SBATCH --job-name=fetch-fastmri
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 6
#SBATCH --mem=16G
#SBATCH -t 10:00:00
#SBATCH -o logs/fetch_fastmri_%j.out
#SBATCH -e logs/fetch_fastmri_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
umask 077
ARCHIVES=(brain_multicoil_train_batch_{0..9}.tar.xz brain_multicoil_val_batch_{0..2}.tar.xz
          brain_multicoil_test_full_batch_{0..2}.tar.xz)
CHAIN=${CHAIN:-0}; CHAINS=${CHAINS:-1}
IDX=${ONLY:-$(seq $CHAIN $CHAINS 12)}
WORK=/home/roku/orcd/scratch/brain-model/work/fastmri
export TMPDIR=$(mktemp -d /tmp/fastmri.XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT
for i in $IDX; do
  A=${ARCHIVES[$i]}
  if [ -f "$WORK/done_${A%.tar.xz}" ]; then echo "[fetch-fastmri] $A done before, skipped"; continue; fi
  echo "[fetch-fastmri] $A $(hostname) $(date -Is)"
  "$PY" -u -m brain.prep.fastmri_fetch --archive "$A" && touch "$WORK/done_${A%.tar.xz}"
done
