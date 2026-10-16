"""Atomic training checkpoints that survive Slurm preemption / Ctrl-C."""

from __future__ import annotations

import random
import signal
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def atomic_torch_save(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_ckpt(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    best_val: float,
    best_epoch: int,
    epochs_no_improve: int,
    history: list[dict],
    latent_dim: int,
    beta: float,
    config: dict,
    extra: dict | None = None,
) -> dict[str, Any]:
    ckpt: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_no_improve": int(epochs_no_improve),
        "history": list(history),
        "latent_dim": int(latent_dim),
        "beta": float(beta),
        "config": config,
        "rng": rng_state(),
    }
    if extra:
        ckpt.update(extra)
    return ckpt


def load_ckpt(path: Path, map_location) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


class InterruptSaver:
    """Flush ``save_fn`` on SIGTERM/SIGINT so Slurm requeue can resume."""

    def __init__(self, save_fn: Callable[[], None]):
        self.save_fn = save_fn
        self._armed = True
        self._prev_term = signal.getsignal(signal.SIGTERM)
        self._prev_int = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):  # noqa: ARG002
        if self._armed:
            self._armed = False
            try:
                print(f"[ckpt] signal {signum}: writing last.pt", flush=True)
                self.save_fn()
            except Exception as e:
                print(f"[ckpt] save on signal failed: {e}", flush=True)
        raise SystemExit(128 + int(signum))

    def disarm(self) -> None:
        self._armed = False
        signal.signal(signal.SIGTERM, self._prev_term)
        signal.signal(signal.SIGINT, self._prev_int)
