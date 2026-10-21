#!/bin/bash
# After phase A of the v6 prep: drop the scratch extracts it read. Each FOMO family / the ABIDE tree is deleted only
# when its pool JPEG-LS zip holds a member for every image; the FOMO metadata copies only when pool has identical ones.
# Leaves scratch with the store plus small run metadata (work/rsna_ingest, runs/).
#
#   sbatch --dependency=afterok:<phase A array> jobs/cleanup_extracts_v6.sbatch

#SBATCH -p mit_normal
#SBATCH --job-name=cleanup-extracts
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o logs/cleanup_extracts_%j.out
#SBATCH -e logs/cleanup_extracts_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
SCR="${BRAIN_SCRATCH:-/home/roku/orcd/scratch/brain-model}"
POOL="${BRAIN_DATASETS:-/home/roku/orcd/pool/datasets}"
FOMO="$SCR/datasets/fomo260k"
echo "[cleanup] $(date -Is)"

ok=1
"$PY" -u -m brain.prep.archive --drop-verified --extract-root "$FOMO/extracted" --dest "$POOL/fomo260k" \
  --families PT002_Nigerian_Clinical,PT010_BrainLat,PT014_CoRR,PT024_M4Raw,PT027_NKI,PT035_Yale_Brain_Mets_Longitudinal \
  || ok=0
if [[ -d "$SCR/datasets/abide1/RawData" ]]; then
  "$PY" -u -m brain.prep.archive --drop-verified --tree "$SCR/datasets/abide1/RawData" \
    --dest "$POOL/abide1/RawData.jls.zip" || ok=0
fi

# FOMO metadata: the top-level files must be byte-identical on pool; per-PT copies are redundant (top mapping wins)
for f in mapping.tsv mri_info.tsv participants.tsv holdout_datasets.txt holdout_report.csv README.md; do
  if [[ -f "$FOMO/$f" ]] && ! cmp -s "$FOMO/$f" "$POOL/fomo260k/$f"; then
    echo "[cleanup] KEEP fomo metadata: $f differs from pool"; ok=0
  fi
done
if [[ $ok == 1 ]]; then
  if [[ -d "$FOMO/extracted" ]] && [[ -n "$(ls -A "$FOMO/extracted")" ]]; then
    echo "[cleanup] KEEP $FOMO: extracted/ still has $(ls "$FOMO/extracted")"
  else
    rm -rf "$SCR/datasets"
    echo "[cleanup] removed $SCR/datasets"
  fi
  rmdir "$SCR/work/cq500_download" 2>/dev/null || true
else
  echo "[cleanup] some extracts kept (see above)"
fi
rm -rf "$SCR/data/processed/smoke_v6"
cat ~/orcd/.quota | sed -n 1,6p
echo "[cleanup] done $(date -Is) ok=$ok"
[[ $ok == 1 ]]
