#!/bin/bash
# Run the whole downstream suite (every task + probe3d) for one checkpoint, then ingest the records. By default one
# job runs the entries in sequence (jobs/eval_run.sbatch); ARRAY=1 fans them out as an array.
#
#   jobs/submit_eval.sh runs/brainmodel_small/last.pt                # all entries
#   jobs/submit_eval.sh runs/brainmodel_small/last.pt cq500 probe3d  # a subset
#   EXTRA="..." jobs/submit_eval.sh ...                               # evaluator args
#   ARRAY=1 CONCURRENT=4 jobs/submit_eval.sh ...                      # parallel array instead of one sequential job
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
if [[ "${ARRAY:-0}" == 1 ]]; then mins=$(( mins / N + 30 )); fi       # an array element runs one entry
mins=$(( mins < 30 ? 30 : (mins > 720 ? 720 : mins) ))
TIME=(-t "$(( mins / 60 )):$(( mins % 60 )):00")
if [[ "${ARRAY:-0}" == 1 ]]; then             # one array element per entry (parallel), then a separate ingest job
  JOB=$(SUITE_TASKS="$TASKS" CKPT="$CKPT" sbatch --parsable --kill-on-invalid-dep=no --export=ALL "${DEP[@]}" "${TIME[@]}" ${SBATCH_ARGS:-} \
    --array="0-$((N - 1))%${CONCURRENT:-4}" --job-name="eval-$(basename "$(dirname "$CKPT")")" jobs/eval_task.sbatch)
  INGEST=$(sbatch --parsable --kill-on-invalid-dep=no --dependency="afterany:$JOB" jobs/eval_ingest.sbatch)
  echo "eval array $JOB ($N entries: $TASKS)"
  echo "ingest $INGEST (after the array)"
else                                          # default: one job runs the entries in sequence and ingests itself
  JOB=$(SUITE_TASKS="$TASKS" CKPT="$CKPT" sbatch --parsable --kill-on-invalid-dep=no --export=ALL "${DEP[@]}" "${TIME[@]}" ${SBATCH_ARGS:-} \
    --job-name="eval-$(basename "$(dirname "$CKPT")")" jobs/eval_run.sbatch)
  echo "eval job $JOB (${TIME[1]}; $N entries in sequence, ingest at the end: $TASKS)"
fi
