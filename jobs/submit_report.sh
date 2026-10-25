#!/bin/bash
# Evaluate one checkpoint against a reference and write its report (brain.eval.report): heads on the test splits
# (paired with the reference), the core transfer tasks of the frozen suite, few-shot at the report's k, then the
# report on quicktest once all three have ended. The reference needs its suite results and fewshot JSON already.
#
#   jobs/submit_report.sh RUN_CKPT REF_CKPT
#   WITH_SEG=1 jobs/submit_report.sh ...            # also the segmentation transfer group (milestones; slow)
#   DEPENDENCY=afterok:123 jobs/submit_report.sh ...  # start after the training job
set -euo pipefail
cd "${BRAIN_REPO:-/home/roku/brain-model}"
source jobs/_env.sh >/dev/null
RUN="${1:?usage: jobs/submit_report.sh RUN_CKPT REF_CKPT}"; REF="${2:?usage: jobs/submit_report.sh RUN_CKPT REF_CKPT}"
SEG=$([[ "${WITH_SEG:-0}" == 1 ]] && echo True || echo False)
OUT=$("$PY" -c "from pathlib import Path; from brain.eval.report import default_out; print(default_out(Path('$RUN')))")
TASKS=$("$PY" -c "from brain.eval.report import core_tasks; print(' '.join(core_tasks(with_seg=$SEG)))")
K=$("$PY" -c "import yaml; print(yaml.safe_load(open('configs/eval/core.yaml'))['adaptation']['k'])")
DEP=(); [[ -n "${DEPENDENCY:-}" ]] && DEP=(--dependency="$DEPENDENCY" --kill-on-invalid-dep=yes)
GPU=(--exclude=node3006 -p pi_dbertsim,mit_normal_gpu -c 12 --gres=gpu:l40s:1 --mem=96G)
mkdir -p "$OUT"

HEADS=$(sbatch --parsable "${DEP[@]}" "${GPU[@]}" -t 03:00:00 -J rep-heads -o "logs/report_heads_%j.out" --wrap \
  "source jobs/_env.sh; \"\$PY\" -u -m brain.multitask.test_eval --ckpts $REF $RUN --workers 10 --out $OUT")
TRANSFER=$(CKPT="$RUN" SUITE_TASKS="$TASKS" sbatch --parsable --export=ALL "${DEP[@]}" --exclude=node3006 \
  -t $([[ $SEG == True ]] && echo 12:00:00 || echo 06:00:00) -J rep-transfer jobs/eval_run.sbatch)
FEWSHOT=$(sbatch --parsable "${DEP[@]}" "${GPU[@]}" -t 03:00:00 -J rep-fewshot -o "logs/report_fewshot_%j.out" --wrap \
  "source jobs/_env.sh; \"\$PY\" -u -m brain.multitask.fewshot --ckpt $RUN --shots $K --workers 10")
REPORT=$(sbatch --parsable --dependency="afterany:$HEADS:$TRANSFER:$FEWSHOT" -p mit_quicktest -c 2 --mem=16G \
  -t 00:15:00 -J rep-report -o "$OUT/report_%j.log" --wrap \
  "source jobs/_env.sh; \"\$PY\" -u -m brain.eval.report $RUN --ref $REF $([[ $SEG == True ]] && echo --with-seg)")
echo "report for $RUN vs $REF -> $OUT/report.md"
echo "  heads $HEADS, transfer $TRANSFER (${TASKS// /, }), fewshot $FEWSHOT (k=$K), report $REPORT"
