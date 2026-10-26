"""Joint SSL + multitask training of one encoder from scratch.

  python -m brain.train joint --config configs/joint/small.yaml
  python -m brain.train joint --synthetic --max-steps 6          # CPU smoke run

One encoder serves two branches in a single forward pass: the 3D LeJEPA objective on multi-crop views of the SSL pool
(``brain.train.lejepa3d``) and the task-conditioned multitask model on whole volumes (``brain.multitask``). The loss is
``w_ssl(t) * L_ssl + w_task(t) * L_task`` with a curriculum (``joint``): SSL alone for ``warmup_frac`` of training,
then the task weight ramps 0 -> 1 over ``task_ramp_frac`` while the SSL weight tapers to ``ssl_floor`` by
``taper_end_frac`` and stays there. The floor keeps the end of training self-supervised as well, so the encoder does not
drift toward the labelled tasks alone. While ``w_task`` is 0 the task branch is skipped.

The two losses live on different scales (LeJEPA falls from ~0.8 to ~0.25 over training; the balanced task loss stays
near 1), so with ``joint.normalize`` (default) each is divided by a running mean of itself (EMA, no gradient) before
weighting: the weights then mean shares of the update, and ``ssl_floor: 0.5`` keeps SSL at half the task weight.

The config is the 3D LeJEPA config plus the multitask sections (``tasks``, multitask ``aug`` keys, ``model.slot`` /
``seg`` / ``taps``, ``data.labels_dir`` / ``registry``), so checkpoints load everywhere: the encoder under ``model``
(frozen suite), the multitask model under ``multitask`` (few-shot, test_eval, slot readout, explain).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from brain.common.optim import param_groups
from brain.common.paths import resolve
from brain.data.batch3d import prepare as ssl_prepare
from brain.models.viewnet3d import ViewNet, batch_meta
from brain.multitask.data import protected_tasks, task_loaders
from brain.multitask.data import prepare as task_prepare
from brain.multitask.evaluate import run_eval
from brain.multitask.losses import Balancer, slot_losses
from brain.multitask.model import MultitaskModel
from brain.multitask.registry import load_registry
from brain.objectives.lejepa3d import lejepa_loss
from brain.objectives.sigreg import SIGReg
from brain.train import engine
from brain.train.lejepa3d import build_encoder, forever_loader, loss_kwargs, metric_keys, synthetic_loader


def schedule(step: int, max_steps: int, j: dict) -> tuple[float, float]:
    """(w_ssl, w_task) at ``step``: SSL alone, then the task weight ramps in (cosine) while SSL tapers to the floor.
    Phases are fractions of ``max_steps`` unless given in steps (``warmup_steps`` etc.)."""
    t = step / max(max_steps, 1)

    def at(name: str, frac: float) -> float:
        return float(j[f"{name}_steps"]) / max(max_steps, 1) if j.get(f"{name}_steps") is not None \
            else float(j.get(f"{name}_frac", frac))

    warm, ramp, taper_end = at("warmup", 0.25), at("task_ramp", 0.15), at("taper_end", 0.6)
    floor = float(j.get("ssl_floor", 0.5))

    def smooth(x: float) -> float:
        x = min(max(x, 0.0), 1.0)
        return 0.5 - 0.5 * math.cos(math.pi * x)

    w_task = smooth((t - warm) / max(ramp, 1e-6)) if t >= warm else 0.0
    w_ssl = 1.0 - (1.0 - floor) * smooth((t - warm) / max(taper_end - warm, 1e-6)) if t >= warm else 1.0
    return w_ssl, w_task


# encoder settings that are not in the weights (RoPE frequencies are rebuilt from them) or that change what the weights
# mean; a warm start must match its source on all of them
ARCH_KEYS = ("size", "rope_base_mm", "rope_min_period_mm", "qk_norm", "n_cls", "n_registers", "proj_dim", "proj_hidden",
             "dense_dim", "dense_hidden", "proj_norm", "rope_fp32", "fp32_residual")
LEGACY = {"rope_fp32": False, "fp32_residual": False}        # what a config without the key was trained with


def init_encoder(enc: nn.Module, cfg: dict, src_path) -> None:
    """Load an SSL checkpoint's whole view network (encoder, projector, token head, predictor) into ``enc``, after
    checking that this run builds the same architecture and token gate as the source."""
    src = torch.load(src_path, map_location="cpu", weights_only=False)
    sm, sv = (src.get("config") or {}).get("model") or {}, (src.get("config") or {}).get("views") or {}
    m, v = cfg["model"], cfg["views"]
    bad = [f"model.{k}: run {m.get(k, LEGACY.get(k))!r} vs source {sm.get(k, LEGACY.get(k))!r}" for k in ARCH_KEYS
           if m.get(k, LEGACY.get(k)) != sm.get(k, LEGACY.get(k))]
    if list(v["patch_xyz"]) != list(sv.get("patch_xyz", [])):
        bad.append(f"views.patch_xyz: run {v['patch_xyz']} vs source {sv.get('patch_xyz')}")
    gate, sgate = dict(v.get("gate") or {}), dict(sv.get("gate") or {})
    if gate.get("mode", "center") != sgate.get("mode", "center"):
        bad.append(f"views.gate.mode: run {gate.get('mode', 'center')} vs source {sgate.get('mode', 'center')}")
    if bad:
        raise SystemExit("[init] this run's encoder differs from its init_from source:\n  " + "\n  ".join(bad))
    enc.load_state_dict(src["model"])


class JointNet(nn.Module):
    """The SSL view network and the multitask model over one shared encoder, as one module (one DDP forward)."""

    def __init__(self, core: ViewNet, mt: MultitaskModel):
        super().__init__()
        self.core, self.mt = core, mt

    def forward(self, ssl_batch: dict, task_batch: dict | None, encoder_grad: bool = True) -> dict:
        out = {"ssl": self.core(ssl_batch), "task": None}
        if task_batch is not None:
            b = task_batch
            out["task"] = self.mt(b["x"], b["keep"], b["task"], b["seg_pair"], encoder_grad=encoder_grad)
        return out


class Trainer(engine.Trainer):
    unit = "vol"
    synthetic_ok = True

    def __init__(self, rt, cfg, args, ckpt, step):
        super().__init__(rt, cfg, args, ckpt, step)
        m, o, t, lc, d, tk = cfg["model"], cfg["optim"], cfg["train"], cfg["loss"], cfg["data"], cfg["tasks"]
        dev, synthetic = rt.device, args.synthetic
        self.j = cfg.get("joint") or {}
        self.max_steps = int(args.max_steps or t["max_steps"])
        self.batch = int(o["batch_size"])                                     # SSL volumes per GPU
        self.task_batch = int(self.j.get("task_batch_size", 4))               # whole volumes per GPU
        self.batch_units = self.batch * rt.world
        self.buckets = int(m.get("length_buckets", 4))
        self.lk, self.gather = loss_kwargs(cfg), bool(lc.get("gather_ddp", True))
        self.ssl_keys = metric_keys(cfg)
        self.clip = float(o.get("clip_grad", 1.0))
        self.stride = int(lc.get("seg_stride", 2))
        self.ramp = int((lc.get("predict") or {}).get("ramp_steps", 0))
        self.amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
        # the multitask branch gates and windows with the encoder's gate (e.g. mode: max)
        m["gate"] = dict(cfg["views"].get("gate") or {})

        # encoder (from scratch, or warm from an SSL checkpoint), SSL view network, multitask model on the same encoder
        enc = build_encoder(cfg, dev)
        if m.get("init_from") and not synthetic:
            if ckpt is None:
                init_encoder(enc, cfg, resolve(m["init_from"]))
                rt.log(f"[init] encoder, projector and predictor from {m['init_from']}")
            self.ramp = 0                     # the predictor comes trained: no re-ramp of the prediction terms
        pcfg = lc.get("predict") or {}
        dense_on = self.lk["dense_weight"] > 0
        predict_on = dense_on and self.lk["predict_weight"] > 0
        core = ViewNet(enc, local_pooled=bool(lc.get("local_pooled", False)), dense=dense_on, predict=predict_on,
                       predict_fine=predict_on and float(pcfg.get("fine_weight", 0.0)) > 0,
                       cells_xyz=tuple(pcfg.get("pixel_cells", (2, 2, 2))))
        self.reg = load_registry(d.get("registry"))
        self.train_ids = self.reg.train_ids(tk)
        mt = MultitaskModel(enc, self.reg, m)
        self.raw = JointNet(core, mt).to(dev)
        if rt.world > 1:
            self.raw = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.raw)
        self.module = self.raw
        self.encoder = enc
        groups = self.reg.groups
        gw = tk.get("group_weights") or {}
        self.balancer = Balancer(len(self.reg), lc.get("balance", "ema"), float(lc.get("ema_decay", 0.99)),
                                 group_of=[groups.index(x.group) for x in self.reg.tasks],
                                 group_weight=[float(gw.get(g, 1.0)) for g in groups],
                                 clamp=lc.get("uncertainty_clamp")).to(dev)
        self.fov_unsafe, self.aug_unsafe = (x.to(dev) for x in protected_tasks(self.reg))
        self.sigreg = SIGReg(n_proj=int(lc.get("n_proj", 256))).to(dev)

        seed = int(cfg["seed"]) + rt.rank
        if synthetic:
            self.ssl_loader = synthetic_loader(cfg, self.batch, seed)
        else:
            self.ssl_loader = forever_loader(resolve(d["store_root"]), "train", cfg, batch_size=self.batch,
                                             shuffle=True, seed=seed + int(step), distributed=rt.world > 1)
        self.task_loader, self.val_loader, self.stats, _ = task_loaders(
            cfg, self.reg, self.train_ids, batch=self.task_batch, workers=int(self.j.get("task_workers", 6)),
            seed=seed + int(step), main=rt.main, distributed=rt.world > 1, synthetic=synthetic)
        self.task_iter, self.epoch = iter(self.task_loader), 0

        self.opt = torch.optim.AdamW(
            param_groups(self.raw, float(o["weight_decay"])) + [{"params": list(self.balancer.parameters()),
                                                                 "weight_decay": 0.0, "lr_scale": 1.0}],
            lr=float(o["lr"]), betas=tuple(o.get("betas", (0.9, 0.95))), fused=dev.type == "cuda")
        self.normalize, self.ema_decay = bool(self.j.get("normalize", True)), 0.99
        self.register_ema = {"ssl": None, "task": None}
        if ckpt is not None:
            self.raw.load_state_dict(ckpt["joint"])
            self.balancer.load_state_dict(ckpt["balancer"])
            self.opt.load_state_dict(ckpt["optimizer"])
            self.register_ema.update({k: torch.tensor(v, device=dev) for k, v in (ckpt.get("joint_norm") or {}).items()
                                      if v is not None})
        self.net = torch.nn.parallel.DistributedDataParallel(self.raw, device_ids=[rt.local],
                                                             find_unused_parameters=True) \
            if rt.world > 1 else self.raw
        self.w = (1.0, 0.0)
        self.t_sum, self.t_cnt = torch.zeros(1, device=dev), 0
        self._banner = (f"brain-{enc.size} params={enc.count_parameters():,} ssl_batch={self.batch}x{rt.world} "
                        f"task_batch={self.task_batch}x{rt.world} tasks={len(self.train_ids)}/{len(self.reg)} "
                        f"rope_min_period={m.get('rope_min_period_mm')} gate={m['gate'].get('mode', 'center')} "
                        f"schedule={ {k: v for k, v in self.j.items() if k.endswith(('_frac', '_steps', 'floor')) and v is not None} }")

    def banner(self) -> str:
        return self._banner

    def _next_task(self) -> dict:
        try:
            return next(self.task_iter)
        except StopIteration:
            self.epoch += 1
            sampler = getattr(self.task_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)
            self.task_iter = iter(self.task_loader)
            return next(self.task_iter)

    def step(self, step: int, lr: float) -> dict[str, torch.Tensor]:
        w_ssl, w_task = schedule(step, self.max_steps, self.j)
        self.w = (w_ssl, w_task)
        dev = self.rt.device
        ssl_b = ssl_prepare(next(self.ssl_loader), self.cfg, dev, self.buckets)
        task_b = task_prepare(self._next_task(), self.cfg, dev, self.encoder.patch_xyz, train=True,
                              fov_unsafe=self.fov_unsafe, aug_unsafe=self.aug_unsafe) if w_task > 0 else None
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast(**self.amp):
            out = self.net(ssl_b, task_b)
        scale = min(1.0, (step + 1) / self.ramp) if self.ramp else 1.0
        parts = lejepa_loss(out["ssl"], batch_meta(ssl_b), self.sigreg, gather_ddp=self.gather,
                            **{**self.lk, "predict_weight": self.lk["predict_weight"] * scale})
        total = w_ssl * self._norm("ssl", parts["loss"])
        task_loss = torch.zeros((), device=dev)
        if task_b is not None:
            loss, has = slot_losses(out["task"], task_b["task"], task_b["y"], self.raw.mt.task_type, task_b["seg"],
                                    task_b["seg_pair"], self.stride)
            task_loss, _ = self.balancer(loss, has, task_b["task"])
            # normalised by the magnitude of the data term: uncertainty weighting's total (exp(-s) L + s) crosses zero
            total = total + w_task * self._norm("task", task_loss, self.balancer.data_term)
            self.t_sum += task_loss.detach()
            self.t_cnt += 1
        total.backward()
        self.balancer.sync_grads()
        torch.nn.utils.clip_grad_norm_(self.raw.parameters(), self.clip)
        self.opt.step()
        res = {k: parts[k].detach() for k in self.ssl_keys}
        res["loss"] = total.detach()
        return res

    def _norm(self, name: str, loss: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
        """``loss`` divided by a running mean of the magnitude of ``scale`` (default ``loss``; no gradient through it)."""
        if not self.normalize:
            return loss
        v = (loss if scale is None else scale).detach().abs().clamp_min(1e-6)
        e = self.register_ema[name]
        e = v if e is None else self.ema_decay * e + (1 - self.ema_decay) * v
        self.register_ema[name] = e
        return loss / e

    def log_extras(self) -> dict[str, str]:
        task = f"{(self.t_sum / max(self.t_cnt, 1)).item():.4f}" if self.t_cnt else "-"
        self.t_sum.zero_()
        self.t_cnt = 0
        return {"w_ssl": f"{self.w[0]:.2f}", "w_task": f"{self.w[1]:.2f}", "task": task}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.val_loader is None or self.w[1] <= 0:
            return {}
        res = run_eval(self.raw.mt, self.val_loader, self.cfg, self.reg, self.stats, self.rt.device,
                       self.encoder.patch_xyz, self.amp)
        return {f"val_{k}": v for k, v in res.items()}

    def state(self, model_only: bool) -> dict:
        s = {"model": self.encoder.state_dict(), "multitask": self.raw.mt.state_dict(), "joint": self.raw.state_dict(),
             "balancer": self.balancer.state_dict(), "tasks": [x.id for x in self.reg.tasks],
             "joint_norm": {k: (float(v) if v is not None else None) for k, v in self.register_ema.items()}}
        if not model_only:
            s["optimizer"] = self.opt.state_dict()
        return s
