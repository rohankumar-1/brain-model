"""python -m brain.bench {list, run, board}: see ``brain/bench/__init__.py``."""

from __future__ import annotations

import argparse
import sys

from brain.bench import __doc__ as DOC


def main() -> None:
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(prog="python -m brain.bench", description=DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="the benchmark tasks (and, with --model, those a model is scored on)")
    ls.add_argument("--model", default=None)
    r = sub.add_parser("run", help="run a model's benchmark tasks (extra evaluator arguments after --)")
    r.add_argument("--model", required=True, help="a checkpoint, or a configs/bench/models.yaml name")
    r.add_argument("--tasks", nargs="*", default=None, help="a subset of the benchmark tasks")
    r.add_argument("--with-seg", action="store_true", help="also segmentation transfer (our 3D encoders)")
    r.add_argument("--submit", action="store_true", help="one Slurm job (jobs/eval_run.sbatch) instead of here")
    r.add_argument("--dependency", default=None, help="Slurm dependency for --submit, e.g. afterok:123")
    bd = sub.add_parser("board", help="the leaderboard")
    bd.add_argument("--models", nargs="*", default=None, help="default: every model with benchmark results")
    bd.add_argument("--ref", default=None, help="default: the spec's reference")
    bd.add_argument("--with-seg", action="store_true")
    bd.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        from brain.bench.spec import load

        b = load()
        print(f"benchmark v{b.version}  index {b.index_sha1}  reference {b.reference}")
        for g, ts in {**b.groups, **{f"{k} (seg)": v for k, v in b.seg_groups.items()}}.items():
            print(f"  {g:22s} " + ", ".join(f"{t} [{b.modality[t]}]" for t in ts))
        if a.model:
            from brain.downstream.models import resolve_model

            m = resolve_model(a.model)
            print(f"{m.model_id} ({m.training_class}): " + ", ".join(b.tasks_for(m, with_seg=True)))
    elif a.cmd == "run":
        from brain.bench.run import run

        run(a.model, tasks=a.tasks, with_seg=a.with_seg, submit=a.submit, dependency=a.dependency, extra=extra)
    else:
        from brain.bench.board import board

        board(a.models, a.ref, with_seg=a.with_seg, reps=a.reps)


if __name__ == "__main__":
    main()
