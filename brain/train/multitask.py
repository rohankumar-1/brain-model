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
from brain.multitask.data import TYPE_ID, MultitaskDataset, SyntheticMultitask, collate, load_labels, make_loader, prepare
from brain.multitask.evaluate import run_eval
from brain.multitask.losses import Balancer, slot_losses
from brain.multitask.model import MultitaskModel
from brain.multitask.registry import load_registry
from brain.train import engine

NO_DECAY_TABLES = ("task_emb", "type_emb", "cls_emb", "cls_pos")      # identities, not weights to shrink
TYPE_KEYS = (("bin", "binary"), ("reg", "regression"), ("mc", "multiclass"), ("seg", "segmentation"))


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
        seed = int(cfg["seed"]) + rt.rank
        if args.synthetic:
            shape = tuple(reversed(cfg["data"]["volume_size_xyz"]))
            ds = SyntheticMultitask(self.reg, shape, n=32, seed=seed)
            self.stats = {}
            self.loader = torch.utils.data.DataLoader(ds, batch_size=self.batch, shuffle=True, collate_fn=collate,
                                                      drop_last=True)
            self.val_loader = torch.utils.data.DataLoader(SyntheticMultitask(self.reg, shape, n=4, seed=999),
                                                          batch_size=2, collate_fn=collate)
        else:
            import pandas as pd

            store = resolve(d["store_root"])
            labels, self.stats = load_labels(resolve(d["labels_dir"]))
            volumes = pd.read_csv(store / d.get("index_name", "volumes.csv"), low_memory=False)
            kw = dict(k_max=int(tk["k_max"]), p_all=float(tk["p_all"]), rare_beta=float(tk["rare_beta"]),
                      max_seg=int(tk["max_seg"]), group_weights=tk.get("group_weights"))
            ds = MultitaskDataset(store, labels, volumes, self.reg, self.stats, split="train", train=True, **kw)
            workers = int(d.get("num_workers", 10))
            self.loader = make_loader(ds, batch_size=self.batch, shuffle=True, seed=seed + int(step), workers=workers,
                                      balance_alpha=float(d.get("balance_alpha", 1.0)),
                                      balance_groups=d.get("balance_groups"), distributed=rt.world > 1)
            self.val_loader = None
            if rt.main:
                val = labels[(labels["split"] == "val") & (labels["use"] == "train")
                             & labels["task"].isin([t.id for t in self.reg.trained])]
                # a fixed set: up to val_per_task volumes per task (deterministic), each asked all its tasks
                pick = (val.sort_values("volume_id").groupby("task").head(int(d.get("val_per_task", 24)))
                        ["volume_id"].unique().tolist())
                vds = MultitaskDataset(store, labels, volumes, self.reg, self.stats, split="val", train=False,
                                       k_max=64, max_seg=int(d.get("val_max_seg", 4)), volume_ids=pick)
                self.val_loader = make_loader(vds, batch_size=2, shuffle=False, seed=0, workers=min(4, workers),
                                              drop_last=False)
            self._banner_extra = f"train_volumes={len(ds)}"
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
                        f"tasks={len(self.reg.trained)}/{len(self.reg)} batch={self.batch}x{rt.world} "
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

    def step(self, step: int, lr: float) -> dict[str, torch.Tensor]:
        b = prepare(self._next(), self.cfg, self.rt.device, self.patch, train=True)
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast(**self.amp):
            out = self.net(b["x"], b["keep"], b["task"], b["seg_pair"], encoder_grad=step >= self.freeze_steps)
        loss, has = slot_losses(out, b["task"], b["y"], self.raw.task_type, b["seg"], b["seg_pair"], self.stride)
        total, per_task = self.balancer(loss, has, b["task"])
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.raw.parameters(), self.clip)
        self.opt.step()
        tt = self.raw.task_type
        res = {"loss": total.detach()}
        present = torch.zeros_like(per_task, dtype=torch.bool)
        present[b["task"][has]] = True
        for key, kind in TYPE_KEYS:
            m = present & (tt == TYPE_ID[kind])
            res[key] = torch.where(m.any(), per_task[m].sum() / m.sum().clamp_min(1), torch.zeros((), device=tt.device))
        res["k"] = (b["task"] >= 0).float().sum(1).mean()
        return res

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
