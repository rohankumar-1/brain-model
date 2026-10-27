"""Run a model's benchmark evaluations: here, one after another, or as Slurm jobs.

Every model: the frozen transfer tasks of its modalities (``brain.downstream.suite``). With ``full``, a BrainQuery
checkpoint also gets segmentation transfer and the token probes (``probe3d``), and, when it has task heads, its own
heads on the trained tasks' test splits paired with the reference (``brain.multitask.test_eval``, written to
``runs/bench/heads/<model>/``) and few-shot on the held-out tasks (``brain.multitask.fewshot``, next to the checkpoint).
"""

from __future__ import annotations

import os
import subprocess
import sys

from brain.common.paths import REPO_ROOT

# the suite jobs: an L40S on either GPU partition (node3006 has a broken CUDA install); mit_normal_gpu caps jobs at 6 h
SBATCH_GPU = ["-p", "pi_dbertsim,mit_normal_gpu", "--gres=gpu:l40s:1", "--exclude=node3006"]


def sbatch_gpu(hours: int) -> list[str]:
    return SBATCH_GPU if hours <= 6 else ["-p", "pi_dbertsim", *SBATCH_GPU[2:]]
HEADS_DIR = REPO_ROOT / "runs" / "bench" / "heads"


def has_heads(model) -> bool:
    """A BrainQuery checkpoint with task heads (multitask, joint, single-task), not an SSL encoder."""
    return model.is_checkpoint and model.training_class in ("supervised", "specialist")


def plan(model, ref, b, *, tasks: list[str] | None, full: bool) -> tuple[list[str], dict[str, list[str]]]:
    """(suite entries, other commands by name) for ``model``."""
    todo = b.tasks_for(model, with_seg=full)
    if tasks:
        unknown = sorted(set(tasks) - set(b.tasks + b.seg_tasks))
        if unknown:
            raise SystemExit(f"not benchmark tasks: {unknown}")
        todo = [t for t in todo if t in tasks]
    if full and model.is_checkpoint and not tasks:
        todo.append("probe3d")
    py = [sys.executable, "-u", "-m"]
    other: dict[str, list[str]] = {}
    if full and has_heads(model) and not tasks:
        labels = ["--labels-dir", b.labels_dir] if b.labels_dir else []
        pair = [str(ref.ckpt), str(model.ckpt)] if ref.is_checkpoint and ref.model_id != model.model_id \
            else [str(model.ckpt)]
        other["heads"] = py + ["brain.multitask.test_eval", "--ckpts", *pair, *labels, "--workers", "10",
                               "--out", str(HEADS_DIR / model.cache_key)]
        k = str((b.raw.get("adaptation") or {}).get("k", 64))
        other["fewshot"] = py + ["brain.multitask.fewshot", "--ckpt", str(model.ckpt), "--shots", k, *labels,
                                 "--workers", "10"]
    return todo, other


def run(model: str, *, tasks: list[str] | None = None, full: bool = False, submit: bool = False,
        dependency: str | None = None, extra: list[str] | None = None) -> None:
    from brain.bench.spec import load
    from brain.downstream.models import resolve_model
    from brain.downstream.suite import command

    b = load()
    ref_model = resolve_model(model)
    ref = resolve_model(b.reference)
    todo, other = plan(ref_model, ref, b, tasks=tasks, full=full)
    skipped = [t for t in b.tasks if t not in b.tasks_for(ref_model)]   # other modality, or labels seen in training
    print(f"[bench] {ref_model.name} = {ref_model.model_id} ({ref_model.training_class}): {len(todo)} suite entries"
          + (f" + {', '.join(other)}" if other else "")
          + (f"; not applicable or seen: {', '.join(skipped)}" if skipped else ""), flush=True)
    dep = [f"--dependency={dependency}", "--kill-on-invalid-dep=no"] if dependency else []
    if submit:
        jobs = []
        # the jobs run this checkout's code (jobs/_env.sh defaults to the main checkout)
        base_env = {**os.environ, "BRAIN_REPO": str(REPO_ROOT)}
        if todo:
            hours = min(24, 1 + len(todo) // 3 + (3 * sum(t in b.seg_tasks for t in todo)) + 2 * ("probe3d" in todo))
            env = {**base_env, "CKPT": model, "SUITE_TASKS": " ".join(todo), "EXTRA": " ".join(extra or [])}
            cmd = ["sbatch", "--parsable", "--export=ALL", "-t", f"{hours:02d}:00:00",
                   "-J", f"bench-{ref_model.cache_key[:20]}", *sbatch_gpu(hours), *dep,
                   str(REPO_ROOT / "jobs" / "eval_run.sbatch")]
            jobs.append(("suite", subprocess.run(cmd, env=env, cwd=REPO_ROOT, check=True, capture_output=True,
                                                 text=True).stdout.strip()))
        for name, argv in other.items():
            wrap = "source jobs/_env.sh; " + " ".join(f'"{a}"' if " " in a else a for a in argv)
            cmd = ["sbatch", "--parsable", "-c", "12", "--mem=96G", "-t", "03:00:00", "-J", f"bench-{name}",
                   "-o", str(REPO_ROOT / "logs" / f"bench_{name}_%j.out"), *SBATCH_GPU, *dep, "--wrap", wrap]
            jobs.append((name, subprocess.run(cmd, env=base_env, cwd=REPO_ROOT, check=True, capture_output=True,
                                              text=True).stdout.strip()))
        print("[bench] submitted " + ", ".join(f"{n} {j}" for n, j in jobs) + "; then: python -m brain.bench card "
              f"--model {model}")
        return
    failed = []
    for t in todo:
        cmd = command(t, model, list(extra or []))
        print("[bench] " + " ".join(cmd[2:]), flush=True)
        if subprocess.call(cmd, cwd=REPO_ROOT):
            failed.append(t)
    for name, argv in other.items():
        print("[bench] " + " ".join(argv[2:]), flush=True)
        if subprocess.call(argv, cwd=REPO_ROOT):
            failed.append(name)
    subprocess.call([sys.executable, "-m", "brain.downstream.db", "ingest"], cwd=REPO_ROOT)
    n = len(todo) + len(other)
    print(f"[bench] done: {n - len(failed)} ok" + (f", failed: {' '.join(failed)}" if failed else ""))
