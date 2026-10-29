#!/bin/bash
# ADNI image collection (LONI IDA "Advanced Download" zips) -> pool. ADNI data use agreement: no redistribution.
#
# IDA has no download API: the zip links come from the collection's Advanced Download page and are tied to the
# browser session that made them, so the job needs that session's cookies too. Both stay out of logs and the command
# line: one link per line in ~/.config/brain/adni_links.txt, the browser's ida.loni.usc.edu cookies (Netscape
# cookies.txt) in ~/.config/brain/adni_cookies.txt, both mode 600.
#
#   sbatch --export=ALL,PILOT=1 jobs/download_adni.sbatch     # first 50 MB of link 1: checks the links work here
#   sbatch jobs/download_adni.sbatch                          # every link; resumable, re-submit to continue
#
# Optional: the collection's metadata CSV (IDA "CSV" button) and ADNIMERGE from Study Data are small; copy them to
# $DEST/../tables by hand.
#SBATCH -p mit_data_transfer
#SBATCH --job-name=dl-adni
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 12:00:00
#SBATCH -o logs/dl_adni_%j.out
#SBATCH -e logs/dl_adni_%j.out
set -euo pipefail
DEST=/home/roku/orcd/pool/datasets/adni/zips
LINKS=~/.config/brain/adni_links.txt
COOKIES=~/.config/brain/adni_cookies.txt
mkdir -p "$DEST"
for f in "$LINKS" "$COOKIES"; do
  [ -f "$f" ] || { echo "[dl] missing $f"; exit 1; }
  [ "$(stat -c %a "$f")" = 600 ] || { echo "[dl] $f must be mode 600 (chmod 600 $f)"; exit 1; }
done
n=$(grep -c . "$LINKS")
echo "[dl] $n links -> $DEST  $(date -Is)"

is_zip() { [ "$(head -c 2 "$1" | od -An -c | tr -d ' ')" = "PK" ]; }   # an expired session returns an HTML page
complete() { [ -s "$1" ] && python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1])" "$1" 2>/dev/null; }

if [ "${PILOT:-0}" = 1 ]; then
  out="$DEST/pilot.part"
  curl -sS --fail -b "$COOKIES" -r 0-52428799 -o "$out" --url "$(grep . "$LINKS" | sed -n 1p)"
  if is_zip "$out"; then echo "[dl] pilot ok: $(stat -c %s "$out") bytes of a zip"; rm -f "$out"; exit 0; fi
  echo "[dl] pilot got no zip (first bytes: $(head -c 60 "$out" | tr -cd '[:print:]')) -- links or cookies not accepted here"
  rm -f "$out"; exit 2
fi

i=0
grep . "$LINKS" | while IFS= read -r url; do
  i=$((i + 1))
  out="$DEST/adni_part_$(printf %02d "$i").zip"
  if complete "$out"; then echo "[dl] part $i already complete"; continue; fi
  for attempt in 1 2 3 4 5; do
    if curl -sS --fail -b "$COOKIES" -C - --retry 5 --retry-delay 30 -o "$out" --url "$url"; then break; fi
    echo "[dl] part $i attempt $attempt failed; resuming in 60 s"; sleep 60
  done
  is_zip "$out" || { echo "[dl] part $i is not a zip (session expired?); remove it and refresh the links"; exit 2; }
  python3 - "$out" <<'PY'
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
print(f"[dl] {sys.argv[1].rsplit('/', 1)[-1]}: {len(names)} members, {sum(n.endswith('.dcm') for n in names)} DICOM, "
      f"{sum(n.endswith(('.nii', '.nii.gz')) for n in names)} NIfTI")
PY
done
echo "[dl] done $(date -Is)"; ls -l "$DEST"
