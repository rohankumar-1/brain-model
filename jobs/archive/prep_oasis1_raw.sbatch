#!/bin/bash
# Re-prep OASIS-1 from its raw MPRAGE (brain.prep.sources.oasis), staged from the pool archive onto local disk.
#   PILOT=6 sbatch jobs/prep_oasis1_raw.sbatch      # first 6 sessions into runs/diagnostics/oasis1_pilot (not the store)
#   sbatch jobs/prep_oasis1_raw.sbatch              # all sessions into the store (--force), then jobs/index_v6.sbatch

#SBATCH -p pi_dbertsim,mit_normal_gpu
#SBATCH --job-name=prep-oasis1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/prep_oasis1_%j.out
#SBATCH -e logs/prep_oasis1_%j.out

set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
STORE="${STORE:-/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr}"
LOCAL="/tmp/${USER}_oasis1_${SLURM_JOB_ID:-local}"
cleanup() { rm -rf "${LOCAL:?}"; }
trap cleanup EXIT
mkdir -p "$LOCAL/oasis"
PILOT="${PILOT:-0}"
# stage the raw OASIS-1 MPRAGEs (hdr + img) of the first PILOT sessions (0 = all)
"$PY" -u - "$LOCAL/oasis" "$PILOT" <<'PYEOF'
import sys, zipfile, re
dest, pilot = sys.argv[1], int(sys.argv[2])
z = zipfile.ZipFile("/home/roku/orcd/pool/datasets/oasis/oasis.raw.zip")
raw = sorted(n for n in z.namelist() if re.search(r"OAS1_\d+_MR\d+_mpr-1_anon\.(hdr|img)$", n))
sessions = sorted({re.search(r"(OAS1_\d+_MR\d+)", n).group(1) for n in raw})
keep = set(sessions[:pilot]) if pilot else set(sessions)
names = [n for n in raw if re.search(r"(OAS1_\d+_MR\d+)", n).group(1) in keep]
for n in names:
    z.extract(n, dest)
print(f"[stage] {len(keep)} OASIS-1 sessions, {len(names)} files -> {dest}", flush=True)
PYEOF
export BRAIN_DATASETS="$LOCAL"
ROOT="$STORE"; FORCE="--force"; MAN="manifest_v6_oasis1raw_0of1.jsonl"
if [[ "$PILOT" != 0 ]]; then ROOT="$BRAIN_REPO/runs/diagnostics/oasis1_pilot"; MAN="manifest_pilot.jsonl"; fi
mkdir -p "$ROOT"
W=$(( ${SLURM_CPUS_PER_TASK:-12} - 1 ))
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" -u -m brain.prep.run --root "$ROOT" --datasets oasis --match oasis1 --workers "$W" --manifest "$MAN" $FORCE \
  --brain-mask auto --gpu-mask --mask-batch 4 --reg-backend torch \
  --template "$BRAIN_TEMPLATE" --template-brain "$BRAIN_TEMPLATE_BRAIN"
echo "[prep-oasis1] done $(date -Is)"
