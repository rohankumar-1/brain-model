"""One training engine for every SSL method.

A method is a ``Trainer`` subclass that owns its model, optimizer, data and loss and implements
``step`` (one optimizer update, returning detached scalar metrics), ``evaluate`` and ``state``. The engine
owns everything around it: CLI and config, distributed setup, resume, warmup+cosine LR, metric averaging
and the log table, validation cadence, best/last/kept checkpoints, NaN guards, and clean preemption
(SIGTERM -> checkpoint -> ``REQUEUE_FLAG``).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from brain.common.checkpoint import atomic_torch_save, find_resume, load_ckpt, state_is_finite
from brain.common.config import load_config, parse_overrides, save_config
from brain.common.dist import Runtime, StopFlag, init_runtime, shutdown, touch_requeue_flag
from brain.common.optim import finite_params, lr_at, set_lr
from brain.common.paths import resolve
from brain.train.defaults import METHODS, default_config, smoke_config


def lr_of(step: int, s: dict) -> float:
    """The LR at ``step`` of a run's saved schedule (schedules saved before ``kind`` existed are cosine)."""
    return lr_at(step, s["max_steps"], s["lr"], s["warmup_steps"], s["floor"], s.get("kind", "cosine"),
                 s.get("decay_steps"))


HIGHER_IS_BETTER = {"score"}     # the multitask validation skill score (brain.eval.skill); every other metric is a loss


def is_new_best(vals: dict, metric: str, best_val: float) -> float | None:
    """``vals[metric]`` beats ``best_val``: higher for ``score``, lower otherwise (a loss metric falls back to
    ``loss``). NaN/Inf never win."""
    higher = metric in HIGHER_IS_BETTER
    v = vals.get(metric) if higher else vals.get(metric, vals.get("loss"))
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) and (v > best_val if higher else v < best_val) else None


class Trainer:
    """Interface between a method and the engine. Subclasses set ``unit`` and ``batch_units``."""

    unit = "samples"      # what throughput counts
    synthetic_ok = False  # whether --synthetic is supported

    def __init__(self, rt: Runtime, cfg: dict, args: argparse.Namespace, ckpt: dict | None, step: int):
        self.rt, self.cfg, self.args = rt, cfg, args
        self.opt: torch.optim.Optimizer | None = None
        self.module: torch.nn.Module | None = None  # the module whose weights are checkpointed / NaN-checked

    # -- required ------------------------------------------------------------------------------------
    batch_units: int = 1  # units consumed per optimizer step across all ranks

    def banner(self) -> str:
        return ""

    def step(self, step: int, lr: float) -> dict[str, torch.Tensor]:
        """One update. Must not read GPU values on the host; return detached scalar tensors."""
        raise NotImplementedError

    def state(self, model_only: bool) -> dict[str, Any]:
        """Method-specific checkpoint content. Always include ``model``."""
        raise NotImplementedError

    # -- optional ------------------------------------------------------------------------------------
    def evaluate(self) -> dict[str, float]:
        """Rank-0 validation. Keys are logged and appended to history verbatim."""
        return {}

    def log_extras(self) -> dict[str, str]:
        """Extra table cells computed from the last step's tensors (called only when logging)."""
        return {}

    def on_start(self, out_dir: Path, step: int) -> None:
        """Rank-0 hook after the output directory exists (QC images, etc.)."""


def _metric_table(prefix_w: int, log, every: int = 10):
    """Prints a label row every ``every`` lines; values sit in fixed columns under those labels."""
    state = {"labels": None, "widths": None, "n": 0}

    def row(prefix: str, values: dict[str, str]) -> None:
        labels = list(values)
        if state["labels"] != labels:
            state["labels"], state["widths"], state["n"] = labels, [max(len(k), 8) for k in labels], 0
        if state["n"] % every == 0:
            log(" " * prefix_w + "   " + "   ".join(f"{k:>{w}}" for k, w in zip(labels, state["widths"])))
        state["n"] += 1
        log(f"{prefix:<{prefix_w}}   " + "   ".join(f"{values[k]:>{w}}" for k, w in zip(labels, state["widths"])))

    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m brain.train", description=__doc__)
    p.add_argument("method", choices=sorted(METHODS))
    p.add_argument("--config", type=Path, default=None, help="YAML merged onto the method's defaults")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--reset-schedule", action="store_true",
                   help="on resume, use the config's LR schedule (lr, warmup, max_steps) instead of the saved one")
    p.add_argument("--synthetic", action="store_true", help="tiny phantom volumes, no store (lejepa3d only)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted config override, repeatable (e.g. --set optim.lr=1e-4)")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> dict:
    over = parse_overrides(args.set)
    if args.out_dir:
        over.setdefault("output", {})["out_dir"] = str(args.out_dir)
    cfg = load_config(args.config, defaults=default_config(args.method), overrides=over)
    return smoke_config(cfg) if args.synthetic else cfg


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    mod = importlib.import_module(METHODS[args.method])
    if args.synthetic and not mod.Trainer.synthetic_ok:
        raise SystemExit(f"--synthetic is not supported by {args.method}")
    rt = init_runtime(int(cfg["seed"]))
    log = rt.log
    t_cfg, out_cfg = cfg["train"], cfg["output"]
    max_steps = int(args.max_steps or t_cfg["max_steps"])
    log_every = int(args.log_every or t_cfg["log_every"])
    out_dir = resolve(out_cfg["out_dir"])
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    best_metric = str(t_cfg.get("best_metric", "loss"))

    step, history, ckpt, resume_src = 0, [], None, None
    best_val, best_step = float("-inf") if best_metric in HIGHER_IS_BETTER else float("inf"), 0
    synthetic_persist = args.synthetic and os.environ.get("BRAIN_PERSIST_SYNTHETIC") == "1"
    if not (args.no_resume or (args.synthetic and not synthetic_persist)):
        resume_from = out_cfg.get("resume_from")
        resume_src = find_resume(out_dir, resolve(resume_from) if resume_from else None)
    if resume_src is not None:
        ckpt = load_ckpt(resume_src, rt.device)
        if not state_is_finite(ckpt["model"]):
            log(f"[resume] ignored {resume_src}: checkpoint contains NaN/Inf weights")
            ckpt, resume_src = None, None
        else:
            step, history = int(ckpt["step"]), list(ckpt.get("history") or [])
            best_val, best_step = float(ckpt.get("best_val", best_val)), int(ckpt.get("best_step", best_step))
    # The LR schedule is part of the run: a resumed run keeps the one it started with (saved in every checkpoint),
    # so a changed config or --max-steps cannot silently bend the cosine; --reset-schedule adopts the config's.
    o = cfg["optim"]
    schedule = {"lr": float(o["lr"]), "warmup_steps": int(o["warmup_steps"]), "max_steps": max_steps,
                "floor": float(o.get("lr_floor", 0.01)), "kind": str(o.get("schedule") or "cosine"),
                "decay_steps": o.get("decay_steps")}
    saved = (ckpt or {}).get("schedule")
    if saved and not args.reset_schedule:
        if saved != schedule:
            log(f"[resume] keeping the saved LR schedule {saved} (config gives {schedule}; --reset-schedule to adopt)")
        schedule = dict(saved)
        max_steps = int(schedule["max_steps"])
    args.max_steps = max_steps                         # the trainer reads it too (end-of-run prefetch)
    if step > 0:
        # fresh random streams after a resume: the same seeds would replay the batches and augmentation draws
        # that followed the previous start
        torch.manual_seed(int(cfg["seed"]) + 1000 * rt.rank + step)

    trainer: Trainer = mod.Trainer(rt, cfg, args, ckpt, step)
    # what this model is trained on, carried in every checkpoint so evaluation records can name it
    prov = None
    if rt.main and not args.synthetic:
        from brain.downstream.records import provenance

        data_cfg = cfg.get("data") or {}
        root = resolve(data_cfg["store_root"]) if data_cfg.get("store_root") else None
        prov = provenance(root, root / data_cfg.get("index_name", "volumes.csv") if root else None)
        prov["method"] = args.method
        prov["started_from"] = str(resume_src) if resume_src is not None else None
    log(f"[model] {args.method} {trainer.banner()} steps={step}/{max_steps} synthetic={args.synthetic}"
        + (f" resume={resume_src}" if resume_src is not None else "")
        + (f" best={best_val:.4f} @ {best_step}" if math.isfinite(best_val) else ""))
    if rt.main:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")
        trainer.on_start(out_dir, step)

    def save(path: Path, model_only: bool = False) -> None:
        if not finite_params(trainer.module):
            log("[nan] weights are non-finite; not writing a checkpoint")
            raise SystemExit(4)
        atomic_torch_save({
            **trainer.state(model_only), "step": step, "config": cfg, "history": history, "provenance": prov,
            "schedule": schedule, "lr": lr_of(step, schedule),
            "best_val": best_val, "best_step": best_step, "best_metric": best_metric,
        }, path)

    persist = not args.synthetic or os.environ.get("BRAIN_PERSIST_SYNTHETIC") == "1"   # tests: resume on phantoms
    stop = StopFlag()
    step_w, prefix_w = len(str(max_steps)), len(f"[step {max_steps}/{max_steps}]")
    put = _metric_table(prefix_w, log)
    keys: list[str] | None = None
    acc, seen, t0 = None, 0, time.time()
    while step < max_steps:
        lr = lr_of(step, schedule)
        set_lr(trainer.opt, lr)
        metrics = trainer.step(step, lr)
        if keys is None:
            keys = list(metrics)
            acc = torch.zeros(len(keys), device=rt.device)
        acc += torch.stack([metrics[k].detach().float() for k in keys])
        step, seen = step + 1, seen + 1
        if step % log_every == 0 or step <= 3:
            a = (acc / seen).tolist()
            if not all(math.isfinite(v) for v in a):
                log(f"[nan] non-finite loss at step {step}: {dict(zip(keys, a))}; not writing a checkpoint")
                raise SystemExit(2)
            cells = {k: f"{v:.4f}" for k, v in zip(keys, a)}
            cells.update(trainer.log_extras())
            cells["lr"] = f"{lr:.2e}"
            cells[f"{trainer.unit}/s"] = f"{seen * trainer.batch_units / max(time.time() - t0, 1e-6):.1f}"
            if rt.device.type == "cuda":
                cells["peak"] = f"{torch.cuda.max_memory_reserved() / 2**30:.1f}GiB"
            put(f"[step {step:>{step_w}}/{max_steps}]", cells)
            acc.zero_()
            seen, t0 = 0, time.time()
        if rt.main and step % int(t_cfg["eval_every"]) == 0:
            vals = trainer.evaluate()
            if vals:
                history.append({"step": step, **vals})
                log(f"[eval {step:>{step_w}}/{max_steps}] " + " ".join(f"{k}={v:.4f}" for k, v in vals.items()))
                new_best = is_new_best({k.removeprefix("val_"): v for k, v in vals.items()}, best_metric, best_val)
                if new_best is not None:
                    best_val, best_step = new_best, step
                    if persist:
                        save(best_path, model_only=True)
                    log(f"[best] step {step} val_{best_metric}={best_val:.4f} -> {best_path.name}")
                (out_dir / "metrics.json").write_text(json.dumps(
                    {"history": history, "best_val": best_val if math.isfinite(best_val) else None,
                     "best_step": best_step, "best_metric": best_metric}, indent=2) + "\n")
        if rt.main and persist and step % int(t_cfg["ckpt_every"]) == 0:
            save(last_path)
            if step % int(t_cfg["keep_every"]) == 0:      # keep_full: with optimizer state, to branch a cooldown
                save(out_dir / f"step_{step:07d}.pt", model_only=not t_cfg.get("keep_full", False))
        if step % int(t_cfg.get("stop_poll", 20)) == 0 and stop.poll(rt.device):
            break

    if rt.main:
        if persist:
            save(last_path)
        preempted = step < max_steps
        log(f"[done] step {step}/{max_steps}" + (" (preempted)" if preempted else ""))
        if preempted:
            touch_requeue_flag()
    shutdown(rt)
