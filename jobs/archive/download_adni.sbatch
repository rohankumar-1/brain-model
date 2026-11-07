#!/bin/bash
# ADNI image collection (LONI IDA "Advanced Download" zips) -> pool. ADNI data use agreement: no redistribution.
#
# IDA has no download API, and its Advanced Download links only work from the IP address that generated them (any
# other IP gets a ~300-byte zip holding a README that says so). dtn001 egresses as 18.13.54.30 (login nodes and
# dtn002 differ), so generate the links in a browser tunnelled through dtn001, then run this job pinned there:
#   1. sbatch -p mit_data_transfer -w dtn001 -c 1 --mem=1G -t 3:00:00 --wrap 'sleep 3h'   # ssh to a dtn needs a job
#   2. laptop: ssh -N -D 1080 -J roku@<login host> roku@orcd-dtn001.mit.edu; browser SOCKS5 proxy localhost:1080
#   3. in that browser: IDA collection -> Advanced Download -> copy the links, one per line, into
#      ~/.config/brain/adni_links.txt (mode 600). Optional: ida.loni.usc.edu cookies (Netscape cookies.txt) in
#      ~/.config/brain/adni_cookies.txt (mode 600), only if IDA turns out to want them too.
#
#   sbatch --export=ALL,PILOT=1 jobs/archive/download_adni.sbatch     # first 50 MB of link 1: checks the links work here
#   sbatch jobs/archive/download_adni.sbatch                          # every link; resumable, re-submit to continue
#   sbatch --export=ALL,PART=2 jobs/archive/download_adni.sbatch      # link 2 only: one job per link runs them in parallel
#                                                             # (IDA serves ~1.5 MB/s per connection, and answers 503
#                                                             # beyond 2 concurrent downloads: queue more with
#                                                             # --dependency=afterany:<running job>)
#   sbatch --export=ALL,DATASET=ppmi jobs/archive/download_adni.sbatch  # PPMI is on IDA too: ppmi_links.txt -> ppmi/zips
#
# Optional: the collection's metadata CSV (IDA "CSV" button) and ADNIMERGE from Study Data are small; copy them to
# $DEST/../tables by hand.
#SBATCH -p mit_data_transfer
#SBATCH -w dtn001
#SBATCH --job-name=dl-adni
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/dl_adni_%j.out
#SBATCH -e logs/dl_adni_%j.out
set -euo pipefail
DATASET=${DATASET:-adni}
DEST=/home/roku/orcd/pool/datasets/$DATASET/zips
LINKS=~/.config/brain/${DATASET}_links.txt
COOKIES=~/.config/brain/${DATASET}_cookies.txt
mkdir -p "$DEST"
cookie_args=()
for f in "$LINKS" "$COOKIES"; do
  [ -f "$f" ] || { [ "$f" = "$COOKIES" ] && continue; echo "[dl] missing $f"; exit 1; }
  [ "$(stat -c %a "$f")" = 600 ] || { echo "[dl] $f must be mode 600 (chmod 600 $f)"; exit 1; }
  [ "$f" = "$COOKIES" ] && cookie_args=(-b "$COOKIES")
done
n=$(grep -c . "$LINKS")
echo "[dl] $n links -> $DEST  from $(hostname) $(curl -s --max-time 10 https://api.ipify.org || true)  $(date -Is)"

is_zip() { [ "$(head -c 2 "$1" | od -An -c | tr -d ' ')" = "PK" ]; }   # an expired session returns an HTML page
# IDA refuses with a zip holding only README.txt (wrong IP, ...): print it and succeed, so callers can bail out
refused() { python3 - "$1" <<'PY' 2>/dev/null
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
if z.namelist() != ["README.txt"]: sys.exit(1)
print(z.read("README.txt").decode(errors="replace").strip())
PY
}
complete() { [ -s "$1" ] && python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1])" "$1" 2>/dev/null && ! refused "$1" >/dev/null; }

if [ "${PILOT:-0}" = 1 ]; then
  out="$DEST/pilot.part"
  curl -sS --fail "${cookie_args[@]}" -r 0-52428799 -o "$out" --url "$(grep . "$LINKS" | sed -n 1p)"
  if msg=$(refused "$out"); then echo "[dl] pilot refused by IDA: $msg"; rm -f "$out"; exit 2; fi
  if is_zip "$out"; then echo "[dl] pilot ok: $(stat -c %s "$out") bytes of a zip"; rm -f "$out"; exit 0; fi
  echo "[dl] pilot got no zip (first bytes: $(head -c 60 "$out" | tr -cd '[:print:]')) -- links not accepted here"
  rm -f "$out"; exit 2
fi

i=0
grep . "$LINKS" | while IFS= read -r url; do
  i=$((i + 1))
  [ -n "${PART:-}" ] && [ "$i" != "$PART" ] && continue
  out="$DEST/${DATASET}_part_$(printf %02d "$i").zip"
  if complete "$out"; then echo "[dl] part $i already complete"; continue; fi
  if msg=$(refused "$out"); then rm -f "$out"; fi   # a stale refusal from an earlier run
  for attempt in 1 2 3 4 5; do
    if curl -sS --fail "${cookie_args[@]}" -C - --retry 5 --retry-delay 30 -o "$out" --url "$url"; then break; fi
    echo "[dl] part $i attempt $attempt failed; resuming in 60 s"; sleep 60
  done
  if msg=$(refused "$out"); then echo "[dl] part $i refused by IDA: $msg"; rm -f "$out"; exit 2; fi
  is_zip "$out" || { echo "[dl] part $i is not a zip (session expired?); remove it and refresh the links"; exit 2; }
  python3 - "$out" <<'PY'
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
print(f"[dl] {sys.argv[1].rsplit('/', 1)[-1]}: {len(names)} members, {sum(n.endswith('.dcm') for n in names)} DICOM, "
      f"{sum(n.endswith(('.nii', '.nii.gz')) for n in names)} NIfTI")
PY
done
echo "[dl] done $(date -Is)"; ls -l "$DEST"
