#!/bin/bash
# Add newly prepped datasets to volumes.csv without moving anything (2026-11-02, fastMRI+): back up the current index,
# rebuild from the format-6 manifests inheriting every existing split from the backup, then require the result to be
# append-only (brain.bench.spec.index_check: every old volume still there with every old value).
# Since 2026-11-06 (Pedi-Cranial-CT) the rebuild is a candidate in a node-local work dir (brain.prep.index --out-dir):
# the live volumes.csv / subjects.csv are never rewritten in place, only replaced by one atomic rename once the
# appended file passes index_check and the live file is still the backed-up one (a run starting meanwhile reads either
# the old or the new index, never a partial or full rebuild). On any failure the live index is left as it was.
#   sbatch --dependency=afterany:<benchmark jobs> jobs/index_append.sbatch
#   sbatch --export=ALL,ONLY=pedi_cranial_ct jobs/index_append.sbatch   # append only these datasets (comma prefixes)
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
WORK="/tmp/${USER}_index_append_${SLURM_JOB_ID:-local}"
cleanup() { rm -rf "$WORK"; rm -f "$STORE/.volumes.append.partial.csv" "$STORE/.subjects.append.partial.csv"; }
trap cleanup EXIT
mkdir -p "$WORK"
DAY=$(date +%Y-%m-%d)
PRE="$STORE/volumes_${DAY}_pre.csv"
SPRE="$STORE/subjects_${DAY}_pre.csv"
[ -e "$PRE" ] && PRE="$STORE/volumes_${DAY}_${SLURM_JOB_ID:-local}_pre.csv" && SPRE="$STORE/subjects_${DAY}_${SLURM_JOB_ID:-local}_pre.csv"
cp -p "$STORE/volumes.csv" "$PRE"
cp -p "$STORE/subjects.csv" "$SPRE"
PRE_SHA=$(sha1sum "$PRE" | cut -d' ' -f1)
echo "[index-append] $(date -Is) backup $PRE ($PRE_SHA), $SPRE"
"$PY" -u -m brain.prep.index --root "$STORE" --workers "${SLURM_CPUS_PER_TASK:-4}" \
  --manifests "manifest_v6_*.jsonl" --inherit-splits "$PRE" --no-slices --out-dir "$WORK"
# keep every old row exactly as it was and append only volumes the old index lacks (a full rebuild may also refresh
# old rows, e.g. recomputed stats or masks written since: reported here, not applied)
"$PY" - "$PRE" "$WORK/volumes.csv" "$STORE/.volumes.append.partial.csv" "${ONLY:-}" \
  "$SPRE" "$WORK/subjects.csv" "$STORE/.subjects.append.partial.csv" <<'PYEOF'
import sys
import pandas as pd
old, cand, out, only, sold, scand, sout = sys.argv[1:]
a = pd.read_csv(old, low_memory=False)
b = pd.read_csv(cand, low_memory=False)
key = ["volume_id", "dataset"]
ak, bk = a.set_index(key), b.set_index(key)
from brain.bench.spec import index_check
print("[index-append] the full rebuild against the old index (reported, not applied):")
index_check(old, cand)
new = bk.loc[bk.index.difference(ak.index)].reset_index()
if only:
    keep = new["dataset"].astype(str).str.startswith(tuple(x for x in only.split(",") if x))
    if (~keep).any():
        print(f"[index-append] ONLY={only}: not appending {int((~keep).sum())} other new rows "
              f"{new.loc[~keep, 'dataset'].value_counts().to_dict()}")
    new = new[keep]
extra = [c for c in b.columns if c not in a.columns]
if extra:
    print(f"[index-append] new columns not carried (the old rows stay byte-identical): {extra}")
new = new.reindex(columns=list(a.columns))
if "vol_idx" in new:
    new["vol_idx"] = range(len(a), len(a) + len(new))


def append(old_path, rows, out_path):
    # the old file's bytes, then the new rows: re-serialising old rows would reformat floats
    with open(old_path, "rb") as f:
        head = f.read()
    with open(out_path, "wb") as f:
        f.write(head if head.endswith(b"\n") else head + b"\n")
        f.write(rows.to_csv(index=False, header=False).encode())


append(old, new, out)
print(f"[index-append] volumes: old {len(a)} rows + {len(new)} new = {len(a) + len(new)}")
# subjects.csv: the old rows as they were plus the new volumes' subjects (dedup_keys the old table lacks)
sa, sb = pd.read_csv(sold, low_memory=False), pd.read_csv(scand, low_memory=False)
snew = sb[sb["dedup_key"].isin(set(new["dedup_key"])) & ~sb["dedup_key"].isin(set(sa["dedup_key"]))]
snew = snew.reindex(columns=list(sa.columns))
append(sold, snew, sout)
print(f"[index-append] subjects: old {len(sa)} rows + {len(snew)} new = {len(sa) + len(snew)}")
PYEOF
"$PY" -c "import sys; from brain.bench.spec import index_check; sys.exit(0 if index_check(sys.argv[1], sys.argv[2]) else 1)" \
  "$PRE" "$STORE/.volumes.append.partial.csv" || { echo "[index-append] NOT append-only: live index untouched"; exit 1; }
[ "$(sha1sum "$STORE/volumes.csv" | cut -d' ' -f1)" = "$PRE_SHA" ] \
  || { echo "[index-append] volumes.csv changed while this job ran: live index untouched"; exit 1; }
mv "$STORE/.subjects.append.partial.csv" "$STORE/subjects.csv"
mv "$STORE/.volumes.append.partial.csv" "$STORE/volumes.csv"
echo "[index-append] installed: volumes.csv sha1 $(sha1sum "$STORE/volumes.csv" | cut -d' ' -f1) (was $PRE_SHA)"
"$PY" -c "
import pandas as pd
v = pd.read_csv('$STORE/volumes.csv', low_memory=False)
for p in '${ONLY:-fastmri}'.split(','):
    f = v[v.dataset.astype(str).str.startswith(p)].drop_duplicates('volume_id')
    print('[index-append]', p, 'volumes', len(f), f.split.value_counts().to_dict(), 'pretrain', int(f.pretrain.sum()),
          'held_out', int(f.held_out.sum()), 'families', f.dataset_family.unique().tolist())
"
echo "[index-append] done $(date -Is)"
