#!/bin/bash
# One 15-minute worker of the JPEG-LS -> JPEG-XL archive re-encode (brain.prep.transcode). Claims ~1 GB chunks from
# the archives in jobs/transcode_list.txt (in order), stops before the limit, resubmits itself while work remains,
# and submits the merge (jobs/transcode_merge.sbatch) of every archive whose chunks are all done.
#   sbatch jobs/transcode_quick.sbatch          # start one chain; quicktest allows 48 CPUs/user -> run two chains
#   touch logs/.transcode_stop                  # chains end after their current job
#SBATCH -p mit_quicktest
#SBATCH --job-name=transcode
#SBATCH -c 24
#SBATCH --mem=90G
#SBATCH -t 00:15:00
#SBATCH -o logs/transcode_%j.out
#SBATCH -e logs/transcode_%j.out
set -uo pipefail
cd "${BRAIN_REPO:-/home/roku/brain-model}"
source jobs/_env.sh
mapfile -t ARCH < <(grep -v '^#' jobs/transcode_list.txt | grep -v '^$')
pool_pct=$(awk '/POOL/ {print int($7)}' ~/orcd/.quota 2>/dev/null || echo 0)
if (( pool_pct >= 93 )); then echo "[transcode] pool at ${pool_pct}%: not starting new chunks"; rc=10
else
  "$PY" -u -m brain.prep.transcode work "${ARCH[@]}" --workers "$SLURM_CPUS_PER_TASK" --deadline 760; rc=$?
fi
for z in $("$PY" -m brain.prep.transcode ready "${ARCH[@]}"); do
  echo "[transcode] all chunks done: submit merge of $z"
  sbatch --parsable jobs/transcode_merge.sbatch "$z"
done
if [[ ! -f logs/.transcode_stop ]]; then
  if [[ $rc -eq 10 ]]; then sbatch --parsable jobs/transcode_quick.sbatch
  elif [[ $rc -eq 11 ]]; then sbatch --parsable --begin=now+600 jobs/transcode_quick.sbatch   # merge in progress
  fi
fi
exit 0
