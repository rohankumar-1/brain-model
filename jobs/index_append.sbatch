#!/bin/bash
# Add newly prepped datasets to volumes.csv without moving anything (2026-11-02, fastMRI+): back up the current index,
# rebuild from the format-6 manifests inheriting every existing split from the backup, then require the result to be
# append-only (brain.bench.spec.index_check: every old volume still there with every old value). Otherwise the backup
# is restored and the job fails, so jobs depending on it (afterok) do not run on a changed index.
#   sbatch --dependency=afterany:<benchmark jobs> jobs/index_append.sbatch
#SBATCH -p mit_normal
#SBATCH --job-name=index-append
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH -o logs/index_append_%j.out
#SBATCH -e logs/index_append_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
export BRAIN_FOMO="${BRAIN_DATASETS:-/home/roku/orcd/pool/datasets}/fomo260k"
PRE="$STORE/volumes_$(date +%Y-%m-%d)_pre.csv"
cp -p "$STORE/volumes.csv" "$PRE"
trap 'cp -p "$PRE" "$STORE/volumes.csv"; echo "[index-append] failed: restored $PRE"' ERR
echo "[index-append] $(date -Is) backup $PRE"
"$PY" -u -m brain.prep.index --root "$STORE" --workers "${SLURM_CPUS_PER_TASK:-4}" \
  --manifests "manifest_v6_*.jsonl" --inherit-splits "$PRE" --no-slices
CAND="$STORE/volumes_$(date +%Y-%m-%d)_candidate.csv"
cp -p "$STORE/volumes.csv" "$CAND"
# keep every old row exactly as it was and append only volumes the old index lacks (a full rebuild may also refresh
# old rows, e.g. recomputed stats or masks written since: reported here, not applied)
"$PY" - "$PRE" "$CAND" "$STORE/volumes.csv" <<'PYEOF'
import sys
import pandas as pd
old, cand, out = sys.argv[1:]
a = pd.read_csv(old, low_memory=False)
b = pd.read_csv(cand, low_memory=False)
key = ["volume_id", "dataset"]
ak, bk = a.set_index(key), b.set_index(key)
from brain.bench.spec import index_check
print("[index-append] the full rebuild against the old index (reported, not applied):")
index_check(old, cand)
new = bk.loc[bk.index.difference(ak.index)].reset_index()
extra = [c for c in b.columns if c not in a.columns]
if extra:
    print(f"[index-append] new columns not carried (the old rows stay byte-identical): {extra}")
new = new.reindex(columns=list(a.columns))
if "vol_idx" in new:
    new["vol_idx"] = range(len(a), len(a) + len(new))
# the old file's bytes, then the new rows: re-serialising old rows would reformat floats
with open(old, "rb") as f:
    head = f.read()
with open(out, "wb") as f:
    f.write(head if head.endswith(b"\n") else head + b"\n")
    f.write(new.to_csv(index=False, header=False).encode())
print(f"[index-append] wrote old {len(a)} rows + {len(new)} new = {len(a) + len(new)}")
PYEOF
if ! "$PY" -c "import sys; from brain.bench.spec import index_check; sys.exit(0 if index_check('$PRE', '$STORE/volumes.csv') else 1)"; then
  cp -p "$PRE" "$STORE/volumes.csv"
  echo "[index-append] NOT append-only: restored $PRE"
  exit 1
fi
"$PY" -c "
import pandas as pd
v = pd.read_csv('$STORE/volumes.csv', low_memory=False)
f = v[v.dataset.astype(str).str.startswith('fastmri')].drop_duplicates('volume_id')
print('[index-append] fastmri volumes', len(f), f.split.value_counts().to_dict())
"
echo "[index-append] done $(date -Is)"
