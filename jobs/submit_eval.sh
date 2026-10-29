#!/bin/bash
# Run the whole downstream suite (every task + probe3d) for one checkpoint, then ingest the records. By default one
# job runs the entries in sequence (jobs/eval_run.sbatch). For parallel jobs use python -m brain.bench run --submit.
#
#   jobs/submit_eval.sh runs/brainmodel_small/last.pt                # all entries
#   jobs/submit_eval.sh runs/brainmodel_small/last.pt cq500 probe3d  # a subset
#   EXTRA="..." jobs/submit_eval.sh ...                               # evaluator args
#   DEPENDENCY=afterok:123 jobs/submit_eval.sh ...                    # start after another job
#   SBATCH_ARGS="-p mit_normal_gpu -c 10 --gres=gpu:1 -t 06:00:00" jobs/submit_eval.sh ...   # other partitions
set -euo pipefail
cd "${BRAIN_REPO:-/home/roku/brain-model}"
source jobs/_env.sh >/dev/null
CKPT="${1:?usage: jobs/submit_eval.sh CKPT [entry ...]}"; shift
if [[ $# -gt 0 ]]; then TASKS="$*"; else TASKS="$("$PY" -m brain.downstream.suite list | tr '\n' ' ')"; fi
read -r -a ARR <<< "$TASKS"
N=${#ARR[@]}
DEP=(); [[ -n "${DEPENDENCY:-}" ]] && DEP=(--dependency="$DEPENDENCY")
# Time limit from the entries: frozen clf/reg tasks take minutes (budget 15 min each), segmentation and probe3d about
# an hour (90 min each); a tight limit lets the scheduler backfill the job. SBATCH_ARGS="-t ..." overrides it.
SEG=" $("$PY" -c 'from brain.downstream.tasks import SEGMENTATION_TASKS as S; print(" ".join(S))') probe3d "
mins=0; for t in "${ARR[@]}"; do if [[ "$SEG" == *" $t "* ]]; then mins=$((mins + 90)); else mins=$((mins + 15)); fi; done
mins=$(( mins < 30 ? 30 : (mins > 720 ? 720 : mins) ))
TIME=(-t "$(( mins / 60 )):$(( mins % 60 )):00")
JOB=$(SUITE_TASKS="$TASKS" CKPT="$CKPT" sbatch --parsable --kill-on-invalid-dep=no --export=ALL "${DEP[@]}" "${TIME[@]}" ${SBATCH_ARGS:-} \
  --job-name="eval-$(basename "$(dirname "$CKPT")")" jobs/eval_run.sbatch)
echo "eval job $JOB (${TIME[1]}; $N entries in sequence, ingest at the end: $TASKS)"
