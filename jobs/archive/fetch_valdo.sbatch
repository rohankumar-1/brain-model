#!/bin/bash
# Download the VALDO 2021 training data (Zenodo 4687995, CC BY-NC-SA 4.0) to pool as the original tarballs:
# Task1 perivascular spaces, Task2 cerebral microbleeds (T2*), Task3 lacunes. Checks Zenodo's md5 for each file.
#SBATCH -p mit_data_transfer
#SBATCH --job-name=fetch-valdo
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH -t 04:00:00
#SBATCH -o logs/fetch_valdo_%j.out
#SBATCH -e logs/fetch_valdo_%j.out
set -euo pipefail
DEST="${BRAIN_DATASETS:-/home/roku/orcd/pool/datasets}/valdo"
mkdir -p "$DEST"
cd "$DEST"
curl -sf https://zenodo.org/api/records/4687995 > record.json
python3 - <<'PY' > files.tsv
import json
for f in json.load(open("record.json"))["files"]:
    print(f["key"], f["checksum"].split(":", 1)[1], f["links"]["self"], sep="\t")
PY
fetch() {   # name md5 url; resumes a partial file, then checks md5
  local name=$1 md5=$2 url=$3
  if [[ -f "$name" ]] && [[ "$(md5sum "$name" | cut -d' ' -f1)" == "$md5" ]]; then echo "[valdo] $name ok (cached)"; return; fi
  echo "[valdo] fetch $name $(date -Is)"
  curl -sfL --retry 5 -C - -o "$name" "$url" || curl -sfL --retry 5 -o "$name" "$url"
  [[ "$(md5sum "$name" | cut -d' ' -f1)" == "$md5" ]] || { echo "[valdo] md5 mismatch: $name" >&2; return 1; }
  echo "[valdo] $name ok $(date -Is)"
}
pids=()
while IFS=$'\t' read -r name md5 url; do fetch "$name" "$md5" "$url" & pids+=($!); done < files.tsv   # Zenodo throttles per connection
for p in "${pids[@]}"; do wait "$p"; done
for t in Task*_v2.tar.gz; do echo "== $t"; tar -tzf "$t" | awk -F/ 'NF>2{print $1"/"$2}' | sort -u | wc -l; tar -tzf "$t" | head -12; done
ls -la "$DEST"
