"""Joint SSL + multitask training of one encoder from scratch.

  python -m brain.train joint --config configs/joint/r15.yaml
  python -m brain.train joint --synthetic --max-steps 6          # CPU smoke run

One encoder serves two branches in a single forward pass: the 3D LeJEPA objective on multi-crop views of the SSL pool
(``brain.train.lejepa3d``) and BrainQuery's task side on whole volumes (``brain.multitask``). The loss is
``w_ssl(t) * L_ssl + w_task(t) * L_task`` with a curriculum (``joint``, ``schedule``): SSL alone for ``warmup_frac``
of training (0: tasks from the start), the task weight ramps 0 -> 1 over ``task_ramp_frac``, and the SSL weight goes
from ``ssl_start`` to ``ssl_floor`` by ``taper_end_frac`` and stays there (R15: 4 -> 1, SSL dominant early, then even
with the tasks; the floor keeps the end of training self-supervised as well).

The two losses live on different scales (LeJEPA falls from ~0.8 to ~0.25 over training; the balanced task loss stays
near 1), so with ``joint.normalize`` (default) each is divided by a running mean of itself (EMA, no gradient) before
weighting: the weights then mean shares of the update.

The task side IS brain.train.multitask's Trainer (subclassed: same losses, negatives, token maps, NaN guard, resume
handling); ``optim.batch_size`` is the task batch (whole volumes per GPU) and ``joint.ssl_batch_size`` the SSL one.
Checkpoints are multitask checkpoints (``model`` = the encoder, with its SSL projector; ``multitask``) plus
``joint_norm``, so they load everywhere: frozen suite, few-shot, test_eval, explain.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from brain.common.paths import resolve
from brain.data.batch3d import prepare as ssl_prepare
from brain.models.viewnet3d import batch_meta
from brain.objectives.lejepa3d import lejepa_loss
from brain.objectives.sigreg import SIGReg
from brain.train import multitask
from brain.train.lejepa3d import (build_encoder, forever_loader, loss_kwargs, metric_keys, synthetic_loader,
                                  viewnet_from_config)


def _phases(max_steps: int, j: dict) -> tuple[float, float, float]:
    """(warm-up end, task ramp length, taper end) as fractions of ``max_steps``; ``<name>_steps`` win over
    ``<name>_frac`` when set."""
    def at(name: str, frac: float) -> float:
        return float(j[f"{name}_steps"]) / max(max_steps, 1) if j.get(f"{name}_steps") is not None \
            else float(j.get(f"{name}_frac", frac))

    return at("warmup", 0.25), at("task_ramp", 0.15), at("taper_end", 0.6)


def schedule(step: int, max_steps: int, j: dict) -> tuple[float, float]:
    """(w_ssl, w_task) at ``step``: SSL alone (weight ``ssl_start``), then the task weight ramps in (cosine) while SSL
    tapers to the floor."""
    t = step / max(max_steps, 1)
    warm, ramp, taper_end = _phases(max_steps, j)
    floor, start = float(j.get("ssl_floor", 0.5)), float(j.get("ssl_start", 1.0))

    def smooth(x: float) -> float:
        x = min(max(x, 0.0), 1.0)
        return 0.5 - 0.5 * math.cos(math.pi * x)

    w_task = smooth((t - warm) / max(ramp, 1e-6)) if t >= warm else 0.0
    w_ssl = start - (start - floor) * smooth((t - warm) / max(taper_end - warm, 1e-6)) if t >= warm else start
    return w_ssl, w_task


def task_start(max_steps: int, j: dict) -> int:
    """The step the task branch starts (``schedule``'s warm-up end): task tokens open their write path from here."""
    return int(round(_phases(max_steps, j)[0] * max_steps))


class JointNet(nn.Module):
    """The SSL view network and the BrainQuery model over one shared encoder, as one module (one DDP forward)."""

    def __init__(self, core: nn.Module, mt: nn.Module):
        super().__init__()
        self.core, self.mt = core, mt

    def forward(self, ssl_batch: dict, task_args: tuple | None = None, task_kw: dict | None = None) -> dict:
        out = {"ssl": self.core(ssl_batch), "task": None}
        if task_args is not None:
            out["task"] = self.mt(*task_args, **(task_kw or {}))
        return out


class Trainer(multitask.Trainer):
    unit = "vol"

    def __init__(self, rt, cfg, args, ckpt, step):
        self.j = cfg.get("joint") or {}
        self.max_steps = int(args.max_steps or cfg["train"]["max_steps"])
        self.task_start = task_start(self.max_steps, self.j)
        super().__init__(rt, cfg, args, ckpt, step)        # encoder (build_encoder), BrainQuery, task loaders, optimizer
        m, lc, d = cfg["model"], cfg["loss"], cfg["data"]
        dev = rt.device
        self.ssl_batch = int(self.j.get("ssl_batch_size", 4))
        self.buckets = int(m.get("length_buckets", 4))
        self.lk, self.gather = loss_kwargs(cfg), bool(lc.get("gather_ddp", True))
        self.ssl_keys = metric_keys(cfg)
        self.ramp = int((lc.get("predict") or {}).get("ramp_steps", 0))
        self.sigreg = SIGReg(n_proj=int(lc.get("n_proj", 256))).to(dev)
        seed = int(cfg["seed"]) + rt.rank
        if args.synthetic:
            self.ssl_loader = synthetic_loader(cfg, self.ssl_batch, seed)
        else:
            self.ssl_loader = forever_loader(resolve(d["store_root"]), "train", cfg, batch_size=self.ssl_batch,
                                             shuffle=True, seed=seed + int(step), distributed=rt.world > 1)
        self.normalize, self.norm_decay = bool(self.j.get("normalize", True)), 0.99
        self.register_ema = {"ssl": None, "task": None}
        if ckpt is not None:
            self.register_ema.update({k: torch.tensor(v, device=dev) for k, v in (ckpt.get("joint_norm") or {}).items()
                                      if v is not None})
        self.w = (1.0, 0.0)
        self.d_sum, self.d_cnt = torch.zeros(1, device=dev), 0
        sched = {k: v for k, v in self.j.items() if k.endswith(("_frac", "_steps", "floor", "start")) and v is not None}
        self._banner += f" ssl_batch={self.ssl_batch}x{rt.world} schedule={sched}"

    def build_encoder(self, cfg: dict, device: torch.device, synthetic: bool):
        """The LeJEPA encoder (projector, token head, predictor inside it), from scratch."""
        m = cfg["model"]
        if m.get("init_from"):
            raise SystemExit("joint trains its encoder from scratch: model.init_from is not supported")
        enc = build_encoder(cfg, device)
        enc.checkpoint_blocks = bool(m.get("checkpoint", True)) and not synthetic
        if hasattr(enc.patch_embed, "checkpoint"):
            enc.patch_embed.checkpoint = enc.checkpoint_blocks
        # the multitask branch gates and windows with the encoder's gate (e.g. mode: frac)
        m["gate"] = dict(cfg["views"].get("gate") or {})
        return enc.train().requires_grad_(True)

    def wrap(self, rt):
        core = viewnet_from_config(self.raw.backbone, self.cfg)
        self.joint = JointNet(core, self.raw)
        if rt.world > 1:     # the projector's BatchNorm sees one GPU's views otherwise (parameters are reused)
            self.joint = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.joint)
            return torch.nn.parallel.DistributedDataParallel(self.joint, device_ids=[rt.local],
                                                             find_unused_parameters=True)
        return self.joint

    def step(self, step: int, lr: float) -> dict[str, torch.Tensor]:
        w_ssl, w_task = schedule(step, self.max_steps, self.j)
        self.w = (w_ssl, w_task)
        self.tokens.at_step(step, self.task_start)
        dev = self.rt.device
        ssl_b = ssl_prepare(next(self.ssl_loader), self.cfg, dev, self.buckets)
        b = self.task_batch() if w_task > 0 else None      # every rank alike: the schedule is deterministic
        self.opt.zero_grad(set_to_none=True)
        targs, tkw, pairs = self.task_inputs(b, step) if b is not None else (None, None, None)
        with torch.autocast(**self.amp):
            out = self.net(ssl_b, targs, tkw)
        scale = min(1.0, (step + 1) / self.ramp) if self.ramp else 1.0
        parts = lejepa_loss(out["ssl"], batch_meta(ssl_b), self.sigreg, gather_ddp=self.gather,
                            **{**self.lk, "predict_weight": self.lk["predict_weight"] * scale})
        total = w_ssl * self._norm("ssl", parts["loss"])
        if b is not None:
            task_loss, per_task, has = self.task_loss(out["task"], b, pairs)
            # normalised by the magnitude of the data term: uncertainty weighting's total (exp(-s) L + s) crosses zero
            total = total + w_task * self._norm("task", task_loss, self.balancer.data_term)
            self.log_tasks(per_task, has, b)
            self.d_sum += self.balancer.data_term.detach()
            self.d_cnt += 1
        self.apply(total, step)
        res = {k: parts[k].detach() for k in self.ssl_keys}
        res["loss"] = total.detach()
        return res

    def _norm(self, name: str, loss: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
        """``loss`` divided by a running mean of the magnitude of ``scale`` (default ``loss``; no gradient through it)."""
        if not self.normalize:
            return loss
        v = (loss if scale is None else scale).detach().abs().clamp_min(1e-6)
        e = self.register_ema[name]
        e = v if e is None else self.norm_decay * e + (1 - self.norm_decay) * v
        self.register_ema[name] = e
        return loss / e

    def log_extras(self) -> dict[str, str]:
        task = f"{(self.d_sum / max(self.d_cnt, 1)).item():.4f}" if self.d_cnt else "-"
        self.d_sum.zero_()
        self.d_cnt = 0
        return {"w_ssl": f"{self.w[0]:.2f}", "w_task": f"{self.w[1]:.2f}", "task": task, **super().log_extras()}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        return {} if self.w[1] <= 0 else super().evaluate()

    def state(self, model_only: bool) -> dict:
        s = super().state(model_only)
        s["joint_norm"] = {k: (float(v) if v is not None else None) for k, v in self.register_ema.items()}
        return s
