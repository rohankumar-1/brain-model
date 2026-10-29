#!/bin/bash
# UCSF-PDGM v5 (TCIA, CC BY 4.0) -> pool, the structural subset only: T1, T1c, T2, FLAIR, ADC, DWI, SWI (raw, not
# the *_bias copies: prep runs its own N4) and the tumour / brain / parenchyma segmentations. ~10 GB of the 142 GB
# package (the rest is DTI / HARDI / ASL; one DTI_eddy_noreg file alone is ~210 MB per case).
#
# TCIA serves the NIfTI only over IBM Aspera (Faspex 5 public link): ascli (gem aspera-cli, conda env "aspera") with
# the IBM Transfer SDK ascp in ~/.aspera/sdk. Resumable: re-submit to continue (ascp skips complete files).
#   sbatch jobs/download_ucsf_pdgm.sbatch
# Clinical metadata (IDH, MGMT, 1p/19q, grade, OS, BraTS21 IDs) is a separate direct download, already in $ROOT/meta.
#SBATCH -p mit_data_transfer
#SBATCH --job-name=dl-ucsf-pdgm
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 12:00:00
#SBATCH -o logs/dl_ucsf_pdgm_%j.out
#SBATCH -e logs/dl_ucsf_pdgm_%j.out
set -euo pipefail
ROOT=/home/roku/orcd/pool/datasets/ucsf_pdgm
DEST=$ROOT/nifti
mkdir -p "$DEST" "$ROOT/lists"
E=/home/roku/.conda/envs/aspera
export PATH=$E/bin:$E/share/rubygems/bin:$PATH GEM_HOME=$E/share/rubygems GEM_PATH=$E/share/rubygems
LINK='https://faspex.cancerimagingarchive.net/aspera/faspex?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjEwNjUiLCJwYXNzY29kZSI6IjYxZmRlZDkzYzYzM2JhODIxNWYwZjBjMDU2YWRiNjk3Y2IwOTM4MDIiLCJwYWNrYWdlX2lkIjoiMTA2NSIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ=='
KEEP='_(T1|T1c|T2|FLAIR|ADC|DWI|SWI|tumor_segmentation|brain_segmentation|brain_parenchyma_segmentation)\.nii\.gz$'
WANT=$ROOT/lists/wanted.txt

if [ ! -s "$WANT" ]; then                                  # the file list: one browse per case folder
  ascli faspex5 packages browse --url="$LINK" /UCSF-PDGM-v5 --format=csv --fields=path > "$ROOT/lists/cases.txt"
  echo "[dl] $(wc -l < "$ROOT/lists/cases.txt") case folders"
  : > "$WANT.part"
  while IFS= read -r case; do
    ascli faspex5 packages browse --url="$LINK" "$case" --format=csv --fields=path | grep -E "$KEEP" >> "$WANT.part" || true
  done < "$ROOT/lists/cases.txt"
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
