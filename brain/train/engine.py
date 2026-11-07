"""One training engine for every SSL method.

A method is a ``Trainer`` subclass that owns its model, optimizer, data and loss and implements
``step`` (one optimizer update, returning detached scalar metrics), ``evaluate`` and ``state``. The engine
owns everything around it: CLI and config, distributed setup, resume, warmup+cosine LR, metric averaging
and the log table, validation cadence, best/last/kept checkpoints, NaN guards, and clean preemption
(SIGTERM -> checkpoint -> ``REQUEUE_FLAG``).
"""

from __future__ import annotations

import argparse
import contextlib
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
    """The LR at ``step`` of a run's saved schedule (schedules saved before ``kind`` existed are cosine). ``start``: the
    step the schedule began at (a ``--reset-schedule`` resume restarts it there, warmup included)."""
    t0 = int(s.get("start", 0))
    return lr_at(step - t0, s["max_steps"] - t0, s["lr"], s["warmup_steps"], s["floor"], s.get("kind", "cosine"),
                 s.get("decay_steps"))


def resume_schedule(saved: dict | None, config: dict, step: int, reset: bool) -> dict:
    """The LR schedule a (re)started run follows: the saved one, or with ``reset`` the config's, started afresh at the
    resume step (``start``; warmup included). A reset is idempotent: when the saved schedule already is the config's
    (a requeued chunk passes --reset-schedule again), it keeps its start instead of re-warming."""
    if not saved:
        return dict(config)
    if not reset:
        return dict(saved)
    if {k: v for k, v in saved.items() if k != "start"} == config:
        return dict(saved)
    return {**config, "start": step} if step > 0 else dict(config)


HIGHER_IS_BETTER = {"score"}     # the multitask validation skill score (brain.eval.skill); every other metric is a loss


class WeightEMA:
    """Exponential moving average of a module's weights (``train.ema_decay``; rank 0 only), with the usual warmup
    ``min(decay, (1 + n) / (10 + n))`` so early averages are not dominated by the initial weights. Float tensors are
    averaged in fp32; integer buffers are copied. ``swapped`` puts the average into the module for an evaluation or a
    checkpoint and restores the live weights after."""

    def __init__(self, module: torch.nn.Module, decay: float, state: dict | None = None):
        self.decay, self.n = float(decay), 0
        self.avg = {k: v.detach().float().clone() if v.is_floating_point() else v.detach().clone()
                    for k, v in module.state_dict().items()}
        if state:
            self.n = int(state["n"])
            # keys the saved average lacks (a head added on resume, e.g. HL-Gauss bins) start from the live weights
            self.avg = {k: state["avg"][k].to(v.device, v.dtype) if k in state["avg"] else v
                        for k, v in self.avg.items()}

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        avg, live = [], []
        for k, v in module.state_dict().items():
            a = self.avg[k]
            if a.is_floating_point():
                avg.append(a)
                live.append(v.detach() if v.dtype == a.dtype else v.detach().float())
            else:
                a.copy_(v)
        if avg:                                  # one fused update: a <- d a + (1 - d) v
            torch._foreach_lerp_(avg, live, 1 - d)
        self.n += 1

    def state(self) -> dict:
        return {"n": self.n, "avg": self.avg, "decay": self.decay}

    @contextlib.contextmanager
    def swapped(self, module: torch.nn.Module):
        live = {k: v.detach().clone() for k, v in module.state_dict().items()}
        module.load_state_dict({k: self.avg[k].to(v.dtype) for k, v in live.items()})
        try:
            yield
        finally:
            module.load_state_dict(live)


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

    def eval_lines(self, vals: dict[str, float]) -> list[str]:
        """The validation log: the first line follows the ``[eval step/max]`` prefix, later ones stand alone."""
        return [" ".join(f"{k}={v:.4f}" for k, v in vals.items())]


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
                   help="on resume, start the config's LR schedule (lr, warmup, max_steps) afresh at the resume step "
                        "instead of keeping the saved one")
    p.add_argument("--synthetic", action="store_true", help="tiny phantom volumes, no store")
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
    rt = init_runtime(int(cfg["seed"]))
    log = rt.log
    t_cfg, out_cfg = cfg["train"], cfg["output"]
    max_steps = int(args.max_steps or t_cfg["max_steps"])
    log_every = int(args.log_every or t_cfg["log_every"])
    out_dir = resolve(out_cfg["out_dir"])
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    best_metric = str(t_cfg.get("best_metric", "loss"))

    step, history, ckpt, resume_src, branched = 0, [], None, None, False
    best_val, best_step = float("-inf") if best_metric in HIGHER_IS_BETTER else float("inf"), 0
    persist = not args.synthetic or os.environ.get("BRAIN_PERSIST_SYNTHETIC") == "1"   # tests: resume on phantoms
    if persist and not args.no_resume:
        resume_from = out_cfg.get("resume_from")
        resume_src = find_resume(out_dir, resolve(resume_from) if resume_from else None)
    if resume_src is not None:
        ckpt = load_ckpt(resume_src, rt.device)
        if not state_is_finite(ckpt["model"]):
            log(f"[resume] ignored {resume_src}: checkpoint contains NaN/Inf weights")
            ckpt, resume_src = None, None
        else:
            step, history = int(ckpt["step"]), list(ckpt.get("history") or [])
            # a run branched from another run's checkpoint (resume_from outside out_dir) starts its best afresh: the
            # parent's best was measured on its own labels / validation pick
            branched = Path(resume_src).resolve().parent != Path(out_dir).resolve()
            if branched:
                log(f"[resume] branched from {resume_src}: best checkpoints start afresh")
            if not branched and ckpt.get("best_metric", best_metric) == best_metric:   # a changed best_metric too
                best_val, best_step = float(ckpt.get("best_val", best_val)), int(ckpt.get("best_step", best_step))
    # The LR schedule is part of the run: a resumed run keeps the one it started with (saved in every checkpoint),
    # so a changed config or --max-steps cannot silently bend the cosine; --reset-schedule adopts the config's.
    o = cfg["optim"]
    schedule = {"lr": float(o["lr"]), "warmup_steps": int(o["warmup_steps"]), "max_steps": max_steps,
                "floor": float(o.get("lr_floor", 0.01)), "kind": str(o.get("schedule") or "cosine"),
                "decay_steps": o.get("decay_steps")}
    saved = (ckpt or {}).get("schedule")
    if saved and not args.reset_schedule and saved != schedule:
        log(f"[resume] keeping the saved LR schedule {saved} (config gives {schedule}; --reset-schedule to adopt)")
    schedule = resume_schedule(saved, schedule, step, args.reset_schedule)
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

    # weight average (train.ema_decay): evaluated beside the live weights and written as <name>_ema.pt, an ordinary
    # checkpoint whose weights are the average, so the benchmark can score both
    ema = WeightEMA(trainer.module, float(t_cfg["ema_decay"]), (ckpt or {}).get("ema")) \
        if rt.main and t_cfg.get("ema_decay") else None
    best_ema = float("-inf") if best_metric in HIGHER_IS_BETTER else float("inf")
    if ckpt is not None and not branched and ckpt.get("best_metric", best_metric) == best_metric:
        best_ema = float(ckpt.get("best_ema", best_ema))
    if ema is not None:
        log(f"[ema] decay {ema.decay} (updates so far {ema.n})")

    def save(path: Path, model_only: bool = False) -> None:
        if not finite_params(trainer.module):
            log("[nan] weights are non-finite; not writing a checkpoint")
            raise SystemExit(4)
        atomic_torch_save({
            **trainer.state(model_only), "step": step, "config": cfg, "history": history, "provenance": prov,
            "schedule": schedule, "lr": lr_of(step, schedule),
            "best_val": best_val, "best_step": best_step, "best_metric": best_metric,
            **({"ema": ema.state(), "best_ema": best_ema} if ema is not None and not model_only else {}),
        }, path)

    def save_ema(path: Path) -> None:
        with ema.swapped(trainer.module):
            save(path, model_only=True)

    def log_eval(tag: str, vals: dict) -> None:
        lines = trainer.eval_lines(vals)
        log(f"[{tag} {step:>{step_w}}/{max_steps}] " + (lines[0] if lines else ""))
        for line in lines[1:]:
            log(line)

    stop = StopFlag()
    step_w, prefix_w = len(str(max_steps)), len(f"[step {max_steps}/{max_steps}]")
    put = _metric_table(prefix_w, log)
    keys: list[str] | None = None
    acc, seen, t0 = None, 0, time.time()
    while step < max_steps:
        lr = lr_of(step, schedule)
        set_lr(trainer.opt, lr)
        metrics = trainer.step(step, lr)
        if ema is not None:
            ema.update(trainer.module)
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
                log_eval("eval", vals)
                new_best = is_new_best({k.removeprefix("val_"): v for k, v in vals.items()}, best_metric, best_val)
                if new_best is not None:
                    best_val, best_step = new_best, step
                    if persist:
                        save(best_path, model_only=True)
                    log(f"[best] step {step} val_{best_metric}={best_val:.4f} -> {best_path.name}")
                if ema is not None:
                    with ema.swapped(trainer.module):
                        ev = trainer.evaluate()
                    history[-1]["ema"] = ev
                    log_eval("eval-ema", ev)
                    nb = is_new_best({k.removeprefix("val_"): v for k, v in ev.items()}, best_metric, best_ema)
                    if nb is not None:
                        best_ema = nb
                        if persist:
                            save_ema(out_dir / "best_ema.pt")
                        log(f"[best-ema] step {step} val_{best_metric}={best_ema:.4f} -> best_ema.pt")
                (out_dir / "metrics.json").write_text(json.dumps(
                    {"history": history, "best_val": best_val if math.isfinite(best_val) else None,
                     "best_step": best_step, "best_metric": best_metric}, indent=2) + "\n")
        if rt.main and persist and step % int(t_cfg["ckpt_every"]) == 0:
            save(last_path)
            if step % int(t_cfg["keep_every"]) == 0:      # keep_full: with optimizer state, to branch a cooldown
                save(out_dir / f"step_{step:07d}.pt", model_only=not t_cfg.get("keep_full", False))
                if ema is not None:
                    save_ema(out_dir / "last_ema.pt")
        if step % int(t_cfg.get("stop_poll", 20)) == 0 and stop.poll(rt.device):
            break

    if rt.main:
        if persist:
            save(last_path)
            if ema is not None:
                save_ema(out_dir / "last_ema.pt")
        preempted = step < max_steps
        log(f"[done] step {step}/{max_steps}" + (" (preempted)" if preempted else ""))
        if preempted:
            touch_requeue_flag()
    shutdown(rt)
