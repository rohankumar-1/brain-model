#!/bin/bash
# Download MS3SEG (figshare 30393475, CC BY 4.0): 100 MS patients, raw NIfTI + tri-class masks (ventricles, normal
# WMH, MS lesions). Kept on pool as the original RAR volumes; md5-checked against figshare. Lists the contents.
#SBATCH -p mit_data_transfer
#SBATCH --job-name=fetch-ms3seg
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 04:00:00
#SBATCH -o logs/fetch_ms3seg_%j.out
#SBATCH -e logs/fetch_ms3seg_%j.out
set -euo pipefail
DEST="${BRAIN_DATASETS:-/home/roku/orcd/pool/datasets}/ms3seg"
mkdir -p "$DEST"
cd "$DEST"
curl -sf https://api.figshare.com/v2/articles/30393475 > article.json
python3 - <<'PY' > files.tsv
import json
for f in json.load(open("article.json"))["files"]:
    if f["name"].startswith(("MS_100_patient_nifti", "MS_100_patient_masks")):
        print(f["name"], f.get("computed_md5") or f["supplied_md5"], f["download_url"], sep="\t")
PY
while IFS=$'\t' read -r name md5 url; do
  if [[ -f "$name" ]] && [[ "$(md5sum "$name" | cut -d' ' -f1)" == "$md5" ]]; then echo "[ms3seg] $name ok (cached)"; continue; fi
  echo "[ms3seg] fetch $name $(date -Is)"
  curl -sfL --retry 5 -o "$name" "$url"
  [[ "$(md5sum "$name" | cut -d' ' -f1)" == "$md5" ]] || { echo "[ms3seg] md5 mismatch: $name" >&2; exit 1; }
  echo "[ms3seg] $name ok"
done < files.tsv
for a in MS_100_patient_nifti.part1.rar MS_100_patient_masks.rar; do
  echo "== $a"; 7z l "$a" | awk '/^-----/{f=!f; next} f' | awk '{print $NF}' | head -25
  echo "... $(7z l "$a" | awk '/^-----/{f=!f; next} f' | wc -l) entries"
done
ls -la "$DEST"
