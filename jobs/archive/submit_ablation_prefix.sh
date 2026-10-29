#!/bin/bash
# Prefix-token ablation: A no extra tokens / B one CLS / C CLS + 4 registers, each trained with the `small` recipe
# on pi_dbertsim (2 GPUs) and then scored on the full downstream suite, records -> results/.
#
# pi_dbertsim allows 64 queued jobs per user (array tasks count singly), so each arm's 25-entry eval array is not
# queued up front: a small launcher job on mit_normal runs jobs/submit_eval.sh once that arm's training succeeds.
#
#   jobs/submit_ablation_prefix.sh                          # all arms, seed 42, start now
#   AFTER=<jobid> jobs/submit_ablation_prefix.sh            # start training after that job (e.g. the store cleanup)
#   SEEDS="43" ARMS="none cls" jobs/submit_ablation_prefix.sh
#   TRAIN_ID=<jobid> ARMS=none jobs/submit_ablation_prefix.sh   # an arm whose training is already queued: eval only
set -euo pipefail
cd "${BRAIN_REPO:-/home/roku/brain-model}"
ARMS="${ARMS:-none cls cls_reg4}"
SEEDS="${SEEDS:-42}"
for arm in $ARMS; do
  for seed in $SEEDS; do
    out="runs/ablate_prefix/${arm}_s${seed}"
    if [[ -n "${TRAIN_ID:-}" ]]; then
      train_id="$TRAIN_ID"
    else
      dep=(); [[ -n "${AFTER:-}" ]] && dep=(--dependency="afterok:${AFTER}")
      train_id=$(sbatch --parsable --kill-on-invalid-dep=no "${dep[@]}" -t 2-00:00:00 --job-name="abl-${arm}-s${seed}" \
        --export=ALL,METHOD=lejepa3d,CONFIG=configs/archive/lejepa3d/ablate_prefix/${arm}.yaml,EXTRA_ARGS="--set seed=${seed} --set output.out_dir=${out}" \
        jobs/train_pi.sbatch)
    fi
    launch=$(sbatch --parsable --kill-on-invalid-dep=no -p mit_normal -c 1 --mem=2G -t 00:10:00 --dependency="afterok:${train_id}" \
      --job-name="abl-eval-${arm}-s${seed}" -o "logs/abl_eval_launch_%j.out" \
      --wrap "cd ${PWD} && jobs/submit_eval.sh ${out}/last.pt")
    echo "${arm} seed=${seed}: train ${train_id} -> eval launcher ${launch} (submits the suite + ingest when training succeeds)"
  done
done
