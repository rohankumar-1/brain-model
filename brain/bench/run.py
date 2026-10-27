"""Run a model's benchmark tasks: here, one after another, or as one Slurm job (``jobs/eval_run.sbatch``)."""

from __future__ import annotations

import os
import subprocess
import sys

from brain.common.paths import REPO_ROOT

# the suite job: an L40S on either GPU partition (node3006 has a broken CUDA install)
SBATCH_GPU = ["-p", "pi_dbertsim,mit_normal_gpu", "--gres=gpu:l40s:1", "--exclude=node3006"]


def run(model: str, *, tasks: list[str] | None = None, with_seg: bool = False, submit: bool = False,
        dependency: str | None = None, extra: list[str] | None = None) -> None:
    from brain.bench.spec import load
    from brain.downstream.models import resolve_model
    from brain.downstream.suite import command

    b = load()
    ref = resolve_model(model)
    applicable = b.tasks_for(ref, with_seg=with_seg)
    skipped = [t for t in b.tasks if t not in applicable]             # other modality, or labels seen in training
    todo = applicable
    if tasks:
        unknown = sorted(set(tasks) - set(b.tasks + b.seg_tasks))
        if unknown:
            raise SystemExit(f"not benchmark tasks: {unknown}")
        todo = [t for t in applicable if t in tasks]
    print(f"[bench] {ref.model_id} ({ref.training_class}): {len(todo)} task(s) to run"
          + (f"; not applicable or seen: {', '.join(skipped)}" if skipped else ""), flush=True)
    if not todo:
        return
    if submit:
        hours = min(12, 1 + len(todo) // 3 + (2 if any(t in b.seg_tasks for t in todo) else 0))
        env = {**os.environ, "CKPT": model, "SUITE_TASKS": " ".join(todo), "EXTRA": " ".join(extra or [])}
        cmd = ["sbatch", "--parsable", "--export=ALL", "-t", f"{hours:02d}:00:00", "-J", f"bench-{ref.name[:20]}",
               *SBATCH_GPU, *([f"--dependency={dependency}", "--kill-on-invalid-dep=no"] if dependency else []),
               str(REPO_ROOT / "jobs" / "eval_run.sbatch")]
        job = subprocess.run(cmd, env=env, cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
        print(f"[bench] submitted job {job}: {' '.join(todo)}")
        return
    failed = []
    for t in todo:
        cmd = command(t, model, list(extra or []))
        print("[bench] " + " ".join(cmd[2:]), flush=True)
        if subprocess.call(cmd, cwd=REPO_ROOT):
            failed.append(t)
    subprocess.call([sys.executable, "-m", "brain.downstream.db", "ingest"], cwd=REPO_ROOT)
    print(f"[bench] done: {len(todo) - len(failed)} ok" + (f", failed: {' '.join(failed)}" if failed else ""))
