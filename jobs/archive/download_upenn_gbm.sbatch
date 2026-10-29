#!/bin/bash
# UPENN-GBM v2 (TCIA, CC BY 4.0; Bakas et al. 2022 Sci Data) -> pool: the skull-stripped structural MRI (T1, T1GD, T2,
# FLAIR on one SRI24 grid per scan, 671 scans of 630 patients) and the tumour segmentations (manual for 147 scans,
# automated for 611; labels 1 necrosis / 2 oedema / 4 enhancing). ~6 GB of the 69 GB package: the skull-on copies,
# DSC and DTI are left out.
#
# TCIA serves the NIfTI only over IBM Aspera (Faspex 5 public link): ascli (gem aspera-cli, conda env "aspera") with
# the IBM Transfer SDK ascp in ~/.aspera/sdk. Resumable: re-submit to continue (ascp skips complete files).
#   sbatch jobs/download_upenn_gbm.sbatch
# Clinical metadata (survival, IDH1, MGMT, ...) is a separate direct download, already in $ROOT/meta.
#SBATCH -p mit_data_transfer
#SBATCH --job-name=dl-upenn-gbm
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 12:00:00
#SBATCH -o logs/dl_upenn_gbm_%j.out
#SBATCH -e logs/dl_upenn_gbm_%j.out
set -euo pipefail
ROOT=/home/roku/orcd/pool/datasets/upenn_gbm
DEST=$ROOT/nifti
mkdir -p "$DEST" "$ROOT/lists"
E=/home/roku/.conda/envs/aspera
export PATH=$E/bin:$E/share/rubygems/bin:$PATH GEM_HOME=$E/share/rubygems GEM_PATH=$E/share/rubygems
LINK='https://faspex.cancerimagingarchive.net/aspera/faspex?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjYwNCIsInBhc3Njb2RlIjoiYzJiMjI2Mzg5ZjljYWE0NWNkYjc4MzM4NWE4Yzc2MjBjNGU1NDY1MiIsInBhY2thZ2VfaWQiOiI2MDQiLCJlbWFpbCI6ImhlbHBAY2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0='
P=/UPENN-GBM/NIfTI-files
WANT=$ROOT/lists/wanted.txt
browse() { ascli faspex5 packages browse --url="$LINK" "$1" --format=csv --fields=path; }

if [ ! -s "$WANT" ]; then                                  # the file list: one browse per scan folder
  : > "$WANT.part"
  browse "$P/images_segm" | grep -E '_segm\.nii\.gz$' >> "$WANT.part"
  browse "$P/automated_segm" | grep -E '_automated_approx_segm\.nii\.gz$' >> "$WANT.part"
  browse "$P/images_structural" > "$ROOT/lists/scans.txt"
  echo "[dl] $(wc -l < "$ROOT/lists/scans.txt") scan folders"
  while IFS= read -r scan; do
    browse "$scan" | grep -E '_(T1|T1GD|T2|FLAIR)\.nii\.gz$' >> "$WANT.part" || true
  done < "$ROOT/lists/scans.txt"
  mv "$WANT.part" "$WANT"
fi
echo "[dl] $(wc -l < "$WANT") files wanted $(date -Is)"

split -l 400 -d "$WANT" "$ROOT/lists/batch_"
for b in "$ROOT"/lists/batch_*; do
  for attempt in 1 2 3; do
    if ascli faspex5 packages receive --url="$LINK" --to-folder="$DEST" --sources=@lines:@file:"$b" \
         --ts=@json:'{"target_rate_kbps":400000,"resume_policy":"sparse_csum"}' > /dev/null; then break; fi
    echo "[dl] $(basename "$b") attempt $attempt failed; retrying in 60 s"; sleep 60
  done
done
rm -f "$ROOT"/lists/batch_*

have=$(ls "$DEST" | grep -c '\.nii\.gz$' || true)            # flat folder: a plain listing, no find / du on pool
gb=$(ls -l "$DEST" | awk '{s += $5} END {printf "%.1f", s / 1e9}')
echo "[dl] $have of $(wc -l < "$WANT") files in $DEST (${gb} GB) $(date -Is)"
missing=$(comm -23 <(sed 's#.*/##' "$WANT" | sort) <(ls "$DEST" | sort) | wc -l)
[ "$missing" -eq 0 ] || { echo "[dl] $missing files missing: re-submit"; exit 2; }
echo "[dl] done"
