#!/bin/bash
# After the step-15k ablation evals: pick the best prefix-token arm (brain.downstream.select_arm) and start its full
# training with the next-run recipe (configs/lejepa3d/small.yaml: grouped balancing, per-view skull dropout, lambda
# 0.03). The time limit ends before the next maintenance window; the trainer checkpoints and requeues itself.
#SBATCH -p mit_normal
#SBATCH --job-name=select-train
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH -o logs/select_train_%j.out
#SBATCH -e logs/select_train_%j.out
set -euo pipefail
cd "${BRAIN_REPO:-/home/roku/brain-model}"
source jobs/_env.sh
SEL=results/selection_15k.json
"$PY" -u -m brain.downstream.select_arm --step 15000 --json "$SEL"
read -r ARM NCLS NREG < <("$PY" -c "import json; d=json.load(open('$SEL')); print(d['winner'], d['n_cls'], d['n_registers'])")
NAME="${ARM%_s42}"
OUT="runs/v2/${NAME}_s42"
MAINT="${MAINT_START:-2026-10-20T00:00:00}"
left=$(( $(date -d "$MAINT" +%s) - $(date +%s) - 1800 ))     # stop 30 min before maintenance
TIME=(-t 2-00:00:00); BEGIN=()
if (( left > 3600 )); then TIME=(-t "$(( left / 3600 )):$(( left % 3600 / 60 )):00")
elif (( left > -86400 )); then BEGIN=(--begin="${MAINT_END:-2026-10-20T21:30:00}"); fi
echo "[select-train] winner=$ARM n_cls=$NCLS n_registers=$NREG out=$OUT ${TIME[*]} ${BEGIN[*]}"
sbatch --parsable --kill-on-invalid-dep=no "${TIME[@]}" "${BEGIN[@]}" --job-name="v2-${NAME}-s42" \
  --export=ALL,METHOD=lejepa3d,CONFIG=configs/lejepa3d/small.yaml,EXTRA_ARGS="--set seed=42 --set model.n_cls=${NCLS} --set model.n_registers=${NREG} --set output.out_dir=${OUT}" \
  jobs/train_pi.sbatch
