"""The full downstream suite for one checkpoint: which evaluations exist and how each one is run.

    python -m brain.downstream.suite list                         # every suite entry, one per line
    python -m brain.downstream.suite run --task cq500 --ckpt runs/x/last.pt [-- extra args for the evaluator]
    jobs/submit_eval.sh runs/x/last.pt                            # one Slurm array element per entry, then db ingest

Entries: every frozen classification/regression task (``brain.downstream.run_suite``: linear/ridge on pooled tokens,
PCA plane/grid readouts, 3D conv probe), every segmentation task (``brain.downstream.seg3d`` for 3D checkpoints, the
2D UNETR decoder in ``run_suite`` for 2D ViTs), and ``probe3d`` (lesion / fingerprint / location token probes with
pixel and random-init controls). Every evaluator writes experiment records (``brain.downstream.records``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from brain.common.paths import REPO_ROOT
from brain.downstream.tasks import FROZEN_TASKS, SEGMENTATION_TASKS

PROBES = ("probe3d",)
SUITE = FROZEN_TASKS + SEGMENTATION_TASKS + PROBES


def is_brain3d(ckpt: Path) -> bool:
    from brain.downstream.embeddings import peek_is_brain3d

    return bool(peek_is_brain3d(ckpt))


def command(task: str, ckpt: Path, extra: list[str], *, brain3d: bool) -> list[str]:
    """``python -m ...`` argv for one suite entry."""
    py = [sys.executable, "-u", "-m"]
    if task == "probe3d":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = REPO_ROOT / "runs" / "downstream" / "probes" / f"{ckpt.parent.name}_{ckpt.stem}_{stamp}.json"
        return py + ["brain.downstream.probe3d", "--ckpt", str(ckpt), "--pixel", "--random-init", "--out", str(out),
                     *extra]
    if task in SEGMENTATION_TASKS and brain3d:
        return py + ["brain.downstream.seg3d", "--task", task, "--ckpt", str(ckpt), *extra]
    if task in SUITE:
        return py + ["brain.downstream.run_suite", "--task", task, "--ckpt", str(ckpt), *extra]
    raise SystemExit(f"unknown suite entry {task!r}; choices: {', '.join(SUITE)}")


def main() -> None:
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["list", "run"])
    ap.add_argument("--task", default=None)
    ap.add_argument("--ckpt", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.action == "list":
        print("\n".join(SUITE))
        return
    if a.task is None or a.ckpt is None:
        raise SystemExit("run needs --task and --ckpt")
    ckpt = a.ckpt if a.ckpt.is_absolute() else (REPO_ROOT / a.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    cmd = command(a.task, ckpt.resolve(), extra, brain3d=is_brain3d(ckpt))
    print("[suite] " + " ".join(cmd[2:]), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
