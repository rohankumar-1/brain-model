#!/bin/bash
# Pedi-Cranial-CT-Healthy (TCIA doi:10.7937/MQ33-AJ18, CC BY 4.0; 100 healthy children, 3.2 GB of MetaImage files)
# over Aspera (Faspex 5 public link; ascli from conda env "aspera", see jobs/archive/download_ucsf_pdgm.sbatch) onto
# this node's /tmp, never scratch. brain.prep.pedi_ct turns the .mha into bit-identical .nii.gz plus a FOMO-form
# participants.tsv; brain.prep.rawpack packs the NIfTI as JPEG-XL lossless into pedi_cranial_ct.raw.zip, which goes
# to pool only when it is at most MAX_ARCHIVE_GB (pool is ~91% full). Otherwise the NIfTI go to a temporary home
# folder that jobs/archive/prep_pedi_ct.sbatch reads and then deletes, and the raw data is not archived.
#   sbatch jobs/archive/fetch_pedi_ct.sbatch
#SBATCH -p mit_data_transfer
#SBATCH --job-name=fetch-pedi-ct
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/fetch_pedi_ct_%j.out
#SBATCH -e logs/fetch_pedi_ct_%j.out
set -euo pipefail
umask 077
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
POOL=/home/roku/orcd/pool/datasets/pedi_cranial_ct
HOLD=/home/roku/brain_tmp/pedi_cranial_ct                 # fallback staging when the archive is skipped
MAX_ARCHIVE_GB=${MAX_ARCHIVE_GB:-2.0}
LOCAL="/tmp/${USER}_pedi_ct_${SLURM_JOB_ID:-local}"
cleanup() { rm -rf "$LOCAL"; }
trap cleanup EXIT
mkdir -p "$LOCAL/dl" "$LOCAL/meta" "$LOCAL/pedi_cranial_ct"
echo "[fetch-pedi-ct] $(date -Is) host=$(hostname)"

WP=https://www.cancerimagingarchive.net/wp-content/uploads
DEMO=Pedi-Cranial-CT-Healthy_demographics_v01_20260714.tsv
REPORT=Pedi-Cranial-CT-Healthy_metadata_report_v7_comprehensive-Final_v01_20260714.tsv
for f in "$DEMO" "$REPORT"; do curl -sSf -m 120 -o "$LOCAL/meta/$f" "$WP/$f"; done

E=/home/roku/.conda/envs/aspera
export PATH=$E/bin:$E/share/rubygems/bin:$PATH GEM_HOME=$E/share/rubygems GEM_PATH=$E/share/rubygems
LINK='https://faspex.cancerimagingarchive.net/aspera/faspex?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjEzMTciLCJwYXNzY29kZSI6ImY1ODQ5NzE4NWNmZmIyMjQxMTA2MDM2NTAyOTRkM2IxNmYzNzlkYTYiLCJwYWNrYWdlX2lkIjoiMTMxNyIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ=='
TOP=/Pedi-Cranial-CT-Healthy
: > "$LOCAL/files.csv"
while IFS=, read -r bin _; do
  ascli faspex5 packages browse --url="$LINK" "$bin" --format=csv --fields=path,size >> "$LOCAL/files.csv"
done < <(ascli faspex5 packages browse --url="$LINK" "$TOP" --format=csv --fields=path,size)
cut -d, -f1 "$LOCAL/files.csv" > "$LOCAL/files.txt"
echo "[fetch-pedi-ct] $(wc -l < "$LOCAL/files.txt") files, $(awk -F, '{s += $2} END {printf "%.2f", s / 1e9}' "$LOCAL/files.csv") GB"
for attempt in 1 2 3; do
  if ascli faspex5 packages receive --url="$LINK" --to-folder="$LOCAL/dl" --sources=@lines:@file:"$LOCAL/files.txt" \
       --ts=@json:'{"target_rate_kbps":400000,"resume_policy":"sparse_csum"}' > /dev/null; then break; fi
  echo "[fetch-pedi-ct] attempt $attempt failed; retrying in 60 s"; sleep 60
done
# ascli receive --sources writes the files flat into --to-folder (as in the UCSF-PDGM job); accept the nested layout
# too. Each listed file must be here at its listed size; it moves to $SRC/<age bin>do/<file>, the layout
# brain.prep.pedi_ct stage reads (the bin folder is checked against the demographics table there).
echo "[fetch-pedi-ct] received layout: $(ls "$LOCAL/dl" | head -3 | tr '\n' ' ')... ($(ls "$LOCAL/dl" | wc -l) top-level entries)"
SRC="$LOCAL/src"
bad=0
while IFS=, read -r path size; do
  rel=${path#$TOP/}; base=${path##*/}; f=""
  for c in "$LOCAL/dl/$rel" "$LOCAL/dl${path}" "$LOCAL/dl/$base"; do [ -f "$c" ] && { f=$c; break; }; done
  if [ -n "$f" ] && [ "$(stat -c %s "$f")" -eq "$size" ]; then
    mkdir -p "$SRC/${rel%/*}"; mv "$f" "$SRC/$rel"
  else
    echo "[fetch-pedi-ct] missing / short: $path (found: ${f:-none})"; bad=1
  fi
done < "$LOCAL/files.csv"
[ "$bad" -eq 0 ] || exit 2
echo "[fetch-pedi-ct] download verified $(date -Is)"

"$PY" -u -m brain.prep.pedi_ct stage --src "$SRC" --demographics "$LOCAL/meta/$DEMO" --out "$LOCAL/pedi_cranial_ct"
rm -rf "$LOCAL/dl"

ZIP="$LOCAL/pedi_cranial_ct.raw.zip"
"$PY" -u -m brain.prep.pedi_ct pack --out "$LOCAL/pedi_cranial_ct" --zip "$ZIP" --workers "${SLURM_CPUS_PER_TASK:-8}"
gb=$(stat -c %s "$ZIP" | awk '{printf "%.2f", $1 / 1e9}')
echo "[fetch-pedi-ct] archive ${gb} GB (NIfTI $(du -sb "$LOCAL/pedi_cranial_ct/nifti" | awk '{printf "%.2f", $1 / 1e9}') GB)"

mkdir -p "$POOL/meta" "$STORE/labels/sources/pedi_cranial_ct"
cp "$LOCAL/meta/"*.tsv "$POOL/meta/"
cp "$LOCAL/pedi_cranial_ct/participants.tsv" "$POOL/participants.tsv"
cp "$LOCAL/pedi_cranial_ct/participants.tsv" "$STORE/labels/sources/pedi_cranial_ct/participants.tsv"
if awk -v g="$gb" -v m="$MAX_ARCHIVE_GB" 'BEGIN {exit !(g <= m)}'; then
  cp "$ZIP" "$POOL/pedi_cranial_ct.raw.zip.partial"
  "$PY" -c "import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); bad = z.testzip(); sys.exit(f'CRC {bad}' if bad else 0)" \
    "$POOL/pedi_cranial_ct.raw.zip.partial"
  mv "$POOL/pedi_cranial_ct.raw.zip.partial" "$POOL/pedi_cranial_ct.raw.zip"
  echo "[fetch-pedi-ct] archived to $POOL/pedi_cranial_ct.raw.zip"
else
  echo "[fetch-pedi-ct] archive ${gb} GB > ${MAX_ARCHIVE_GB} GB: NOT archived to pool; NIfTI held in $HOLD for prep"
  mkdir -p "$HOLD"
  cp -r "$LOCAL/pedi_cranial_ct/nifti" "$HOLD/"
  cp "$LOCAL/pedi_cranial_ct/participants.tsv" "$HOLD/"
fi
echo "[fetch-pedi-ct] done $(date -Is)"
