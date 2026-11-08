#!/bin/bash
# Store re-encode, format 6 -> 7 (brain.prep.reencode): every image zip listed in STORE/volumes.csv rewritten in place
# (int16 image + scale; masks / brain arrays / attrs / index untouched), one shard per array task. Resumable and
# requeue-safe: zips already at format 7 and paths the shard manifest lists as done are skipped; SIGTERM (preemption,
# time limit) finishes the current zip and exits 4. Stops (exit 3, no requeue) when free space (statvfs) or the scratch
# quota cache plus the shard's own bytes gets within the margins. Manifests: runs/diagnostics/reencode/manifest_<i>of<n>.jsonl
#
#   sbatch --array=0-31%16 jobs/reencode.sbatch                     # whole store, 32 shards, 16 at a time
#   SHARDS=32 sbatch --array=0-31%8 --begin=20:00 jobs/reencode.sbatch   # gentler / off-peak
#   EXTRA_ARGS="--ct-air-below none" sbatch --array=0-31%16 jobs/reencode.sbatch   # no CT air clamp
#
# CPU only (pi_dbertsim needs a GPU). mit_normal (12 h, 96 CPUs per user) with mit_preemptable as overflow: a
# preempted task requeues and resumes from its manifest. 4 CPUs per task: ~1 zip/s, ~12 MB/s of real NFS traffic
# (old read + new write; the verify re-read hits the client cache). %16 -> ~200 MB/s, ~2.5-3 h for the store.
# MAX_MB_S caps each task (counting read + verify read + write twice, ~25 MB per zip).
# Afterwards: python -m brain.prep.verify_store --root STORE (expects format 7 everywhere) and grep the manifests
# for "error" / "stopped".

#SBATCH -p mit_normal,mit_preemptable
#SBATCH --job-name=reencode
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/reencode_%A_%a.out
#SBATCH -e logs/reencode_%A_%a.out

set -uo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
# zstd-9 compression is the slow step (~3.9 CPU-s per CT zip); zarr compresses the 20 chunks in parallel threads
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
SHARDS="${SHARDS:-32}"
I="${SLURM_ARRAY_TASK_ID:-0}"
echo "[reencode] $(date -Is) host=$(hostname) shard ${I}/${SHARDS} store=${STORE} repo=${BRAIN_REPO:-/home/roku/brain-model}"
"$PY" -u -m brain.prep.reencode --root "$STORE" --shard "${I}/${SHARDS}" \
  --max-mb-s "${MAX_MB_S:-30}" --min-free-gb "${MIN_FREE_GB:-200}" --quota-margin-gb "${QUOTA_MARGIN_GB:-15}" \
  ${EXTRA_ARGS:-}
rc=$?
echo "[reencode] $(date -Is) exit ${rc}"
exit $rc
