#!/bin/bash
# Unpack BraTS/OASIS on pool if needed, then write one compressed volume per case.
# Existing <dataset>/<case>.zip files are kept (do not pass --overwrite).

#SBATCH -p mit_normal
#SBATCH --job-name=prep_zarr
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 6:00:00
#SBATCH -o /home/roku/orcd/scratch/brain-model/logs/preprocess_zarr_%j.out
#SBATCH -e /home/roku/orcd/scratch/brain-model/logs/preprocess_zarr_%j.err

set -euo pipefail
cd /home/roku/brain-model
export PYTHONPATH=/home/roku/brain-model:/home/roku/.local/lib/python3.12/site-packages
export PYTHONNOUSERSITE=True
# One thread per worker. SimpleITK/ITK otherwise uses every allocated core
# inside each of the 6 processes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
export BRAIN_SCRATCH=/home/roku/orcd/scratch/brain-model
export BRAIN_POOL=/home/roku/orcd/pool
export BRAIN_DATASETS=/home/roku/orcd/pool/datasets

PY=/home/roku/.conda/envs/ml-torch/bin/python
echo "[host] $(hostname) cpus=${SLURM_CPUS_PER_TASK} mem=96G $(date -Is)"

BRATS_Z="${BRAIN_DATASETS}/brats"
BRATS_E="${BRAIN_DATASETS}/brats/extracted"
OASIS_Z="${BRAIN_DATASETS}/oasis"
OASIS_E="${BRAIN_DATASETS}/oasis/extracted"
RSNA_ZIP="${BRAIN_DATASETS}/rsna/rsna-ihd-dataset.zip"
mkdir -p "$BRATS_E" "$OASIS_E" "${BRAIN_SCRATCH}/logs"

extract_zip() {
  local z="$1" dest="$2" min_nii="${3:-100}"
  mkdir -p "$dest"
  local n
  n=$(find "$dest" -name '*.nii.gz' 2>/dev/null | wc -l)
  if [ "$n" -ge "$min_nii" ]; then
    echo "[skip] $dest already has $n nii.gz"
    return 0
  fi
  if [ "$n" -gt 0 ]; then
    echo "[wipe] incomplete $dest ($n nii.gz < $min_nii)"
    rm -rf "$dest"
    mkdir -p "$dest"
  fi
  echo "[unzip] $z -> $dest"
  unzip -q -o "$z" -d "$dest"
  echo "[ok] $z ($(find "$dest" -name '*.nii.gz' | wc -l) nii.gz)"
}

# Three archives at a time. More than that just contends on the pool disks.
pids=()
n_running() {
  local n=0 pid
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}
launch() {
  while [ "$(n_running)" -ge 3 ]; do
    sleep 5
  done
  "$@" &
  last_pid=$!
  pids+=("$last_pid")
}

launch extract_zip "$BRATS_Z/2023_GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip" "$BRATS_E/2023_GLI"
launch extract_zip "$BRATS_Z/2023_MEN/ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData.zip" "$BRATS_E/2023_MEN"
men_pid=$last_pid
launch extract_zip "$BRATS_Z/2024_GLI/BraTS2024-BraTS-GLI-TrainingData.zip" "$BRATS_E/2024_GLI"
launch extract_zip "$BRATS_Z/2024_GLI/BraTS2024-BraTS-GLI-AdditionalTrainingData.zip" "$BRATS_E/2024_GLI_add"
launch extract_zip "$BRATS_Z/2024_MEN_RT/BraTS2024-MEN-RT-TrainingData.zip" "$BRATS_E/2024_MEN_RT" 10

n_oasis=$(find "$OASIS_E" \( -name '*.img' -o -name '*.nii*' \) 2>/dev/null | wc -l)
if [ "$n_oasis" -ge 100 ]; then
  echo "[skip] oasis already extracted ($n_oasis)"
else
  for t in "$OASIS_Z"/oasis1/oasis_cross-sectional_disc{1..12}.tar.gz \
           "$OASIS_Z"/oasis2/OAS2_RAW_PART{1,2}.tar.gz; do
    launch tar -xzf "$t" -C "$OASIS_E"
  done
fi

if ! wait "$men_pid"; then
  echo "[error] BraTS 2023 MEN extract failed"
  kill "${pids[@]}" 2>/dev/null || true
  exit 1
fi
if [ -f "$BRATS_Z/2023_MEN/BraTS-MEN-TRAIN-FIX-V4.zip" ]; then
  echo "[unzip] MEN FIX overlay"
  unzip -q -o "$BRATS_Z/2023_MEN/BraTS-MEN-TRAIN-FIX-V4.zip" -d "$BRATS_E/2023_MEN"
fi

extract_fail=0
for pid in "${pids[@]}"; do
  if [ "$pid" = "$men_pid" ]; then
    continue
  fi
  if ! wait "$pid"; then
    extract_fail=1
  fi
done
if [ "$extract_fail" -ne 0 ]; then
  echo "[error] one or more archive extracts failed"
  exit 1
fi

n_brats=$(find "$BRATS_E" -name '*.nii.gz' 2>/dev/null | wc -l)
n_oasis=$(find "$OASIS_E" \( -name '*.img' -o -name '*.nii*' \) 2>/dev/null | wc -l)
echo "[check] brats_nii=$n_brats oasis_img=$n_oasis rsna_zip=$(test -f "$RSNA_ZIP" && echo ok || echo MISSING)"
if [ "$n_brats" -lt 100 ]; then
  echo "[error] BraTS extract incomplete at $BRATS_E"
  exit 1
fi
if [ "$n_oasis" -lt 100 ]; then
  echo "[error] OASIS extract incomplete at $OASIS_E"
  exit 1
fi
if [ ! -f "$RSNA_ZIP" ]; then
  echo "[error] RSNA zip missing at $RSNA_ZIP"
  exit 1
fi

"$PY" -u data/preprocess.py \
  --resume \
  --workers 6 \
  --skip-download \
  --slice-sample bernoulli \
  --slice-sigma 50 \
  --train-frac 0.75 \
  --val-frac 0.10 \
  --test-frac 0.15 \
  --brats-root "$BRATS_E" \
  --oasis-root "$OASIS_E" \
  --rsna-zip "$RSNA_ZIP" \
  --out-dir "${BRAIN_SCRATCH}/data/processed/slices_zarr"

echo "[done] $(date -Is)"
