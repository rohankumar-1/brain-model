"""Task-conditioned multitask post-training of an SSL encoder (brain.multitask) on whole volumes.

    python -m brain.train multitask --config configs/multitask/v3_init.yaml
    python -m brain.train multitask --synthetic --max-steps 4          # CPU smoke run on phantoms

The encoder starts from ``model.init_from`` (its architecture and token grid are copied into this run's config, so
frozen evaluation loads these checkpoints like any 3D encoder: ``model`` is the encoder). For ``optim.freeze_steps``
only the slots and heads train; then the encoder joins at ``enc_lr_mult`` x the LR with layer-wise decay.
Checkpoints also hold ``multitask`` (slots, task embeddings, heads, decoder) and ``tasks`` (registry order).
"""

from __future__ import annotations

import torch

from brain.common.optim import NO_DECAY_SUFFIXES
from brain.common.paths import resolve
from brain.multitask.data import prepare, ssl_view, task_loaders
from brain.multitask.evaluate import run_eval
from brain.multitask.losses import Balancer, slot_losses
from brain.multitask.model import MultitaskModel
from brain.multitask.registry import load_registry
from brain.objectives.sigreg import SIGReg, gather_batch, lejepa_loss
from brain.train import engine

NO_DECAY_TABLES = ("task_emb", "type_emb", "cls_emb", "cls_pos")      # identities, not weights to shrink
# log column per task group: the raw (unbalanced) mean loss of the group's tasks, averaged over the steps of the
# logging window in which the group was asked (the balanced ``loss`` sits near 1 by construction under ``ema``)
GROUP_COLUMNS = {"demographics": "demo", "diagnosis": "dx", "hemorrhage": "ich", "tumor": "tumor",
                 "derived": "derived", "anatomy_seg": "aseg", "anatomy_volume": "avol", "lesion": "lesion"}


def build_encoder(cfg: dict, device: torch.device, synthetic: bool):
    """The SSL encoder, trainable. Copies its model / views config into ``cfg`` (what frozen evaluation reads)."""
    from brain.downstream.embed3d import load_brain3d
    from brain.models.vit3d import BrainEncoder

    m = cfg["model"]
    if synthetic or not m.get("init_from"):
        enc = BrainEncoder(size=m.get("size", "test"), patch_xyz=tuple(cfg["views"]["patch_xyz"]),
                           qk_norm=True, n_cls=int(m.get("n_cls", 1)), proj_dim=16, proj_hidden=32)
    else:
        src = torch.load(resolve(m["init_from"]), map_location="cpu", weights_only=False)
        enc = load_brain3d(src, torch.device("cpu")).model
        scfg = src.get("config") or {}
        for k, v in (scfg.get("model") or {}).items():
            if k not in ("checkpoint",) and k not in m:
                m[k] = v
        cfg["views"] = {**(scfg.get("views") or {}), "gate": cfg["views"].get("gate")}
        cfg["views"]["patch_xyz"] = list(enc.patch_xyz)
        m["size"] = enc.size
    enc.checkpoint_blocks = bool(m.get("checkpoint", True)) and not synthetic
    return enc.to(device).train().requires_grad_(True)


def param_groups(model: MultitaskModel, extra: list, wd: float, enc_mult: float, layer_decay: float) -> list[dict]:
    """Encoder blocks at ``enc_mult * layer_decay ** (depth - 1 - i)`` of the LR (embedding at ``** depth``), the rest
    at 1. Biases, norms and learned tokens get no weight decay."""
    depth = model.backbone.depth
    buckets: dict[tuple[float, float], list] = {}
    named = list(model.named_parameters()) + [(f"extra.{i}", p) for i, p in enumerate(extra)]
    for name, p in named:
        if not p.requires_grad:
            continue
        scale = 1.0
        if name.startswith("backbone."):
            if name.startswith("backbone.blocks."):
                scale = enc_mult * layer_decay ** (depth - 1 - int(name.split(".")[2]))
            else:
                scale = enc_mult * layer_decay ** depth
        skip = p.ndim < 2 or name.endswith(NO_DECAY_SUFFIXES) or any(k in name for k in NO_DECAY_TABLES)
        buckets.setdefault((scale, 0.0 if skip else wd), []).append(p)
    return [{"params": ps, "lr": s, "weight_decay": d, "lr_scale": s} for (s, d), ps in buckets.items()]


class Trainer(engine.Trainer):
    unit = "vol"
    synthetic_ok = True

    def __init__(self, rt, cfg, args, ckpt, step):
        super().__init__(rt, cfg, args, ckpt, step)
        dev = rt.device
        o, d, tk = cfg["optim"], cfg["data"], cfg["tasks"]
        self.batch = int(o["batch_size"])
        self.batch_units = self.batch * rt.world
        self.freeze_steps = int(o.get("freeze_steps", 0))
        self.clip = float(o.get("clip_grad", 1.0))
        self.stride = int(cfg["loss"].get("seg_stride", 2))
        self.amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
        self.reg = load_registry(d.get("registry"))
        # ``tasks.groups`` / ``tasks.ids`` restrict training (the no-label control; single-task models)
        only, ids = tk.get("groups"), tk.get("ids")
        self.train_ids = [t.id for t in self.reg.trained if (not only or t.group in set(only)) and (not ids or t.id in set(ids))]
        if not self.train_ids:
            raise SystemExit(f"tasks.groups {only} selects no trained task (groups: {self.reg.groups})")
        enc = build_encoder(cfg, dev, args.synthetic)
        self.patch = tuple(enc.patch_xyz)
        text = None
        if (cfg["model"].get("slot") or {}).get("embedding") == "text":
            from brain.multitask.text import load_embeddings

            text = load_embeddings([t.descriptions[0] for t in self.reg.tasks], self.reg.text_encoder)
        self.raw = MultitaskModel(enc, self.reg, cfg["model"], text).to(dev)
        groups = self.reg.groups
        gw = tk.get("group_weights") or {}
        self.balancer = Balancer(len(self.reg), cfg["loss"].get("balance", "ema"),
                                 float(cfg["loss"].get("ema_decay", 0.99)),
                                 group_of=[groups.index(t.group) for t in self.reg.tasks],
                                 group_weight=[float(gw.get(g, 1.0)) for g in groups]).to(dev)
        self.module = self.raw
        # tasks whose target needs the whole brain in view: no field-of-view cut on rows that ask them
        from brain.multitask.labels import registration_sensitive

        groups = self.reg.groups
        self.group_of = torch.tensor([groups.index(t.group) for t in self.reg.tasks], device=dev)
        self.log_groups = [(i, GROUP_COLUMNS.get(g, g[:8])) for i, g in enumerate(groups) if g != "held_out"]
        self.g_sum = torch.zeros(len(groups), device=dev)
        self.g_cnt = torch.zeros(len(groups), device=dev)
        unsafe = [registration_sensitive(t.id) or any(s.get("source") == "live" for s in t.sources) for t in self.reg.tasks]
        self.fov_unsafe = torch.tensor(unsafe, device=dev)
        # general embedding (LeJEPA on a task-agnostic slot, two global views): model.general.enabled + loss.general
        self.gen = cfg["loss"].get("general") or {}
        self.gen_w = float(self.gen.get("weight", 0.0)) if (cfg["model"].get("general") or {}).get("enabled") else 0.0
        self.sigreg = SIGReg(n_proj=int(self.gen.get("n_proj", 256))).to(dev)
        self.gen_sum, self.gen_cnt = torch.zeros(2, device=dev), 0
        seed = int(cfg["seed"]) + rt.rank
        self.loader, self.val_loader, self.stats, n_train = task_loaders(
            cfg, self.reg, self.train_ids, batch=self.batch, workers=int(d.get("num_workers", 10)), seed=seed + int(step),
            main=rt.main, distributed=rt.world > 1, synthetic=args.synthetic)
        self._banner_extra = f"train_volumes={n_train}"
        self.iter = iter(self.loader)
        self.epoch = 0
        self.opt = torch.optim.AdamW(
            param_groups(self.raw, list(self.balancer.parameters()), float(o["weight_decay"]),
                         float(o.get("enc_lr_mult", 0.25)), float(o.get("layer_decay", 0.8))),
            lr=float(o["lr"]), betas=(0.9, 0.999), fused=dev.type == "cuda")
        if ckpt is not None:
            self.raw.load_state_dict(ckpt["multitask"])
            self.balancer.load_state_dict(ckpt["balancer"])
            self.opt.load_state_dict(ckpt["optimizer"])
        self.net = torch.nn.parallel.DistributedDataParallel(self.raw, device_ids=[rt.local],
                                                             find_unused_parameters=True) \
            if rt.world > 1 else self.raw
        n_enc = sum(p.numel() for p in enc.parameters())
        n_all = sum(p.numel() for p in self.raw.parameters())
        self._banner = (f"enc={enc.size} patch={self.patch} enc_params={n_enc:,} head_params={n_all - n_enc:,} "
                        f"tasks={len(self.train_ids)}/{len(self.reg)} batch={self.batch}x{rt.world} "
                        f"freeze={self.freeze_steps} " + getattr(self, "_banner_extra", ""))

    def banner(self) -> str:
        return self._banner

    def _next(self) -> dict:
        try:
            return next(self.iter)
        except StopIteration:
            self.epoch += 1
            sampler = getattr(self.loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)
            self.iter = iter(self.loader)
            return next(self.iter)

    def _with_general_views(self, raw: dict, b: dict) -> tuple[dict, torch.Tensor]:
        """Append a second view of the first ``loss.general.frac`` rows (asking no task) to the batch, so both views
        go through the encoder in one pass. Returns the widened batch and the rows that have a second view."""
        n = b["x"].shape[0]
        rows = torch.arange(max(2, round(n * float(self.gen.get("frac", 0.5)))), device=b["x"].device)
        x2, keep2 = ssl_view(raw, rows, self.cfg, self.rt.device, self.patch)
        pad = lambda t, v: torch.cat([t, torch.full((len(rows), *t.shape[1:]), v, dtype=t.dtype, device=t.device)])  # noqa: E731
        return {**b, "x": torch.cat([b["x"], x2]), "keep": torch.cat([b["keep"], keep2]),
                "task": pad(b["task"], -1), "y": pad(b["y"], 0)}, rows

    def step(self, step: int, lr: float) -> dict[str, torch.Tensor]:
        raw = self._next()
        b = prepare(raw, self.cfg, self.rt.device, self.patch, train=True, fov_unsafe=self.fov_unsafe)
        n = b["x"].shape[0]
        if self.gen_w > 0:
            b, rows = self._with_general_views(raw, b)
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast(**self.amp):
            out = self.net(b["x"], b["keep"], b["task"], b["seg_pair"], encoder_grad=step >= self.freeze_steps)
        loss, has = slot_losses(out, b["task"], b["y"], self.raw.task_type, b["seg"], b["seg_pair"], self.stride)
        total, per_task = self.balancer(loss, has, b["task"])
        if self.gen_w > 0:
            g = out["general"].float()
            views = torch.stack([g[rows], g[n:]])                                      # (2, R, D)
            gen, inv, reg = lejepa_loss(gather_batch(views), self.sigreg, float(self.gen.get("lamb", 0.02)))
            total = total + self.gen_w * gen
            self.gen_sum += torch.stack([inv.detach(), reg.detach()])
            self.gen_cnt += 1
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.raw.parameters(), self.clip)
        self.opt.step()
        present = torch.zeros_like(per_task, dtype=torch.bool)
        present[b["task"][has]] = True
        g = self.group_of[present]
        n_g = self.g_sum.numel()
        g_loss = torch.zeros(n_g, device=g.device).index_add(0, g, per_task[present])
        g_n = torch.zeros(n_g, device=g.device).index_add(0, g, torch.ones_like(g, dtype=torch.float32))
        on = g_n > 0
        self.g_sum += torch.where(on, g_loss / g_n.clamp_min(1), torch.zeros_like(g_loss))
        self.g_cnt += on.float()
        return {"loss": total.detach(), "k": (b["task"] >= 0).float().sum(1).mean()}

    def log_extras(self) -> dict[str, str]:
        """Raw per-group losses over the logging window (``-`` for a group not asked in it); resets the window."""
        mean = (self.g_sum / self.g_cnt.clamp_min(1)).tolist()
        cnt = self.g_cnt.tolist()
        self.g_sum.zero_()
        self.g_cnt.zero_()
        out = {name: (f"{mean[i]:.4f}" if cnt[i] else "-") for i, name in self.log_groups}
        if self.gen_w > 0:
            inv, reg = (self.gen_sum / max(self.gen_cnt, 1)).tolist()
            out.update(gen_inv=f"{inv:.4f}", gen_sig=f"{reg:.3f}")
            self.gen_sum.zero_()
            self.gen_cnt = 0
        return out

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        res = run_eval(self.raw, self.val_loader, self.cfg, self.reg, self.stats, self.rt.device, self.patch, self.amp)
        return {f"val_{k}": v for k, v in res.items()}

    def state(self, model_only: bool) -> dict:
        s = {"model": self.raw.backbone.state_dict(), "multitask": self.raw.state_dict(),
             "balancer": self.balancer.state_dict(), "tasks": [t.id for t in self.reg.tasks]}
        if not model_only:
            s["optimizer"] = self.opt.state_dict()
        return s
