"""The full downstream suite for one model: which evaluations exist and how each one is run.

    python -m brain.downstream.suite list                         # every suite entry, one per line
    python -m brain.downstream.suite run --task cq500 --model runs/x/last.pt [-- extra args for the evaluator]
    python -m brain.downstream.suite run --task ixi_age --model sfcn          # a model in configs/bench/models.yaml
    jobs/submit_eval.sh runs/x/last.pt                            # one Slurm job over the entries, then db ingest

Entries: every frozen classification/regression task (``brain.downstream.run_suite``: linear/ridge on pooled tokens,
PCA plane/grid readouts, 3D conv probe), every segmentation task (``brain.downstream.seg3d``) and ``probe3d``
(lesion / fingerprint / location token probes with pixel and random-init controls). Segmentation and probe3d read our
3D encoders' token grids, so they take checkpoints only. Every evaluator writes experiment records
(``brain.downstream.records``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone

from brain.common.paths import REPO_ROOT
from brain.downstream.tasks import FROZEN_TASKS, SEGMENTATION_TASKS

PROBES = ("probe3d",)
SUITE = FROZEN_TASKS + SEGMENTATION_TASKS + PROBES


def command(task: str, model: str, extra: list[str]) -> list[str]:
    """``python -m ...`` argv for one suite entry; ``model`` is a checkpoint path or a registry name."""
    from brain.downstream.models import resolve_model

    py = [sys.executable, "-u", "-m"]
    if task not in SUITE:
        raise SystemExit(f"unknown suite entry {task!r}; choices: {', '.join(SUITE)}")
    if task in PROBES or task in SEGMENTATION_TASKS:
        ref = resolve_model(model)
        if not ref.is_checkpoint:
            raise SystemExit(f"{task} reads our 3D encoders' token grids: {model} is a registry model")
        ckpt = str(ref.ckpt)
        if task in PROBES:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out = REPO_ROOT / "runs" / "downstream" / "probes" / f"{ref.cache_key}_{stamp}.json"
            return py + ["brain.downstream.probe3d", "--ckpt", ckpt, "--pixel", "--random-init", "--out", str(out),
                         *extra]
        return py + ["brain.downstream.seg3d", "--task", task, "--ckpt", ckpt, *extra]
    return py + ["brain.downstream.run_suite", "--task", task, "--model", str(model), *extra]


def main() -> None:
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["list", "run"])
    ap.add_argument("--task", default=None)
    ap.add_argument("--model", "--ckpt", dest="model", default=None, help="a checkpoint or a registry model name")
    a = ap.parse_args(argv)
    if a.action == "list":
        print("\n".join(SUITE))
        return
    if a.task is None or a.model is None:
        raise SystemExit("run needs --task and --model")
    cmd = command(a.task, a.model, extra)
    print("[suite] " + " ".join(cmd[2:]), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
