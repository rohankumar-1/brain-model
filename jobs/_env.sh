# Shared environment for every job. Source it; do not submit it.
#   export BRAIN_REPO=/path/to/brain-model   # default below
#   source "$BRAIN_REPO/jobs/_env.sh"
# Sets PY, PYTHONPATH, library paths and thread limits (override OMP_THREADS for CPU-heavy jobs), and cd's to the repo.

BRAIN_REPO="${BRAIN_REPO:-/home/roku/brain-model}"
BRAIN_ENV="${BRAIN_ENV:-/home/roku/.conda/envs/ml-torch}"
BRAIN_USER_SITE="${BRAIN_USER_SITE:-/home/roku/.local/lib/python3.12/site-packages}"

cd "$BRAIN_REPO"
mkdir -p logs

# Standalone SynthStrip (weights + wrapper under scratch; not a full FreeSurfer install).
export FREESURFER_HOME="${FREESURFER_HOME:-/home/roku/orcd/scratch/brain-model/tools/synthstrip}"
export BRAIN_SYNTHSTRIP="${BRAIN_SYNTHSTRIP:-$FREESURFER_HOME/bin/mri_synthstrip}"
# Registration template and its brain mask (also the brain-mask QC prior used by remask).
export BRAIN_TEMPLATE="${BRAIN_TEMPLATE:-/home/roku/orcd/scratch/brain-model/templates/MNI152_T1_1mm.nii.gz}"
export BRAIN_TEMPLATE_BRAIN="${BRAIN_TEMPLATE_BRAIN:-/home/roku/orcd/scratch/brain-model/templates/MNI152_T1_1mm_brain_mask.nii.gz}"
export PATH="$FREESURFER_HOME/bin:$PATH"

unset PYTHONPATH WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
export PYTHONNOUSERSITE=True
export OMP_NUM_THREADS="${OMP_THREADS:-1}" MKL_NUM_THREADS="${OMP_THREADS:-1}" OPENBLAS_NUM_THREADS="${OMP_THREADS:-1}"
export NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1
PY="$BRAIN_ENV/bin/python"
export PYTHONPATH="$BRAIN_REPO:$BRAIN_USER_SITE"
NVIDLIBS=$(find "$BRAIN_ENV/lib/python3.12/site-packages/nvidia" "$BRAIN_USER_SITE/nvidia" -type d -name lib 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH="${NVIDLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
