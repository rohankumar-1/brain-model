#!/bin/bash
# RSNA Intracranial Aneurysm Detection (RSNA-ICA, MIRA dataset 7; non-commercial, no redistribution) -> pool.
# The presigned S3 link (7-day expiry) is read from ~/.config/brain/rsna_ica_url (mode 600) so it never lands in
# logs or the job's command line. Resumable: re-submit to continue a partial download.
#SBATCH -p mit_data_transfer
#SBATCH --job-name=dl-rsna-ica
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 12:00:00
#SBATCH -o logs/dl_rsna_ica_%j.out
#SBATCH -e logs/dl_rsna_ica_%j.out
set -euo pipefail
DEST=/home/roku/orcd/pool/datasets/rsna_ica
mkdir -p "$DEST"
OUT="$DEST/rsna-intracranial-aneurysm-detection.zip"
URL_FILE=~/.config/brain/rsna_ica_url
for attempt in 1 2 3 4 5 6; do
  if curl -sS --fail -C - --retry 5 --retry-delay 30 -o "$OUT" --url "$(cat "$URL_FILE")"; then break; fi
  echo "[dl] attempt $attempt failed; resuming in 60 s" ; sleep 60
done
ls -l "$OUT"
python3 - "$OUT" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); names = z.namelist()
print(f"[dl] zip ok: {len(names)} members; csv: {[n for n in names if n.endswith('.csv')][:10]}")
PY
echo "[dl] done $(date -Is)"
