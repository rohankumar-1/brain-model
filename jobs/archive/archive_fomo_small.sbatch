#!/bin/bash
# Pack the 2026-10 small FOMO cohorts' extracted NIfTIs onto pool (one JPEG-LS zip per PT, brain.prep.archive; every
# volume round-trip checked), then drop the scratch extracts whose archive verifiably holds every file.
#SBATCH -p pi_dbertsim
#SBATCH --gres=gpu:1
#SBATCH --exclude=node3006
#SBATCH --job-name=archive-fomo-small
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/archive_fomo_small_%j.out
#SBATCH -e logs/archive_fomo_small_%j.out
set -euo pipefail
source "${BRAIN_REPO:-/home/roku/brain-model}/jobs/_env.sh"
PTS="PT023_Infant_Development_Brain,PT031_SLIM,PT033_WAND,PT034_Wayne,PT036_Yale_HighRes,PT008_AdolescentBrainDevelopment,PT012_CHBMP,PT017_HBN-SSI,PT022_Bejing_Enhanced,PT004_ACPI,PT025_age_ility,PT006_AHEAD,PT026_MICA_MICs,PT003_CUNMET,PT032_Tao_Wu,PT001_ClevelandCCF"
echo "[archive] pack $(date -Is)"
"$PY" -u -m brain.prep.archive --families "$PTS" --workers 11
echo "[archive] verify (dry run) $(date -Is)"
"$PY" -u -m brain.prep.archive --families "$PTS" --drop-verified --dry-run
echo "[archive] drop verified extracts $(date -Is)"
"$PY" -u -m brain.prep.archive --families "$PTS" --drop-verified
sed -n 1,6p ~/orcd/.quota
echo "[archive] done $(date -Is)"
