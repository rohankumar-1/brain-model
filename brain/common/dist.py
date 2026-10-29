"""Distributed helpers: torchrun env, process-group setup, clean preemption, rank-0 logging."""

from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


def dist_info() -> tuple[int, int, int]:
    """(rank, world_size, local_rank) from the torchrun environment, or (0, 1, 0)."""
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


@dataclass
class Runtime:
    rank: int
    world: int
    local: int
    device: torch.device

    @property
    def main(self) -> bool:
        return self.rank == 0

    def log(self, *args, **kwargs) -> None:
        if self.main:
            print(*args, flush=True, **kwargs)


def init_runtime(seed: int) -> Runtime:
    """Process group (when launched by torchrun), device, per-rank seeds and matmul precision."""
    import numpy as np

    rank, world, local = dist_info()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if world > 1:
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    return Runtime(rank, world, local, device)


def shutdown(rt: Runtime) -> None:
    if rt.world > 1:
        dist.destroy_process_group()


class StopFlag:
    """Set by SIGTERM/SIGINT, or by the file named in ``STOP_FILE`` appearing. ``poll`` agrees across ranks so all
    stop on the same step.

    The trainer checks it between steps, so a checkpoint is never written mid-update. The file is how the sbatch
    wrapper asks for a clean stop before the time limit: a SIGTERM sent through ``torchrun`` makes it tear the ranks
    and their DataLoader workers down before they can checkpoint.
    """

    def __init__(self) -> None:
        self.flag = False
        self.stop_file = os.environ.get("STOP_FILE")
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._set)

    def _set(self, *_) -> None:
        self.flag = True

    def poll(self, device: torch.device) -> bool:
        if self.stop_file and Path(self.stop_file).exists():
            self.flag = True
        if not (dist.is_available() and dist.is_initialized()):
            return self.flag
        t = torch.tensor(float(self.flag), device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())


def touch_requeue_flag() -> None:
    """Tell the sbatch wrapper this run stopped early and should be resubmitted."""
    flag = os.environ.get("REQUEUE_FLAG")
    if flag:
        Path(flag).touch()
