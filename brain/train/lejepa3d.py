"""3D LeJEPA pretraining on native-grid crops: pooled crop agreement, dense token agreement, SIGReg.

  python -m brain.train lejepa3d --config configs/lejepa3d/tiny.yaml
  python -m brain.train lejepa3d --synthetic --max-steps 8        # CPU smoke run, no store needed
  torchrun --standalone --nproc_per_node=4 -m brain.train lejepa3d --config configs/lejepa3d/small.yaml
"""

from __future__ import annotations

import torch

from brain.common.checkpoint import load_ckpt, strip_module_prefix
from brain.common.metrics import embedding_health
from brain.common.optim import param_groups
from brain.common.paths import resolve
from brain.data.batch3d import prepare
from brain.data.crops3d import raw_views
from brain.data.volumes import collate_volumes, crop_microbatch, make_loader, make_synthetic_volume
from brain.models.viewnet3d import ViewNet, batch_meta
from brain.data.volumes import n_meta
from brain.models.vit3d import BrainEncoder
from brain.objectives.lejepa3d import lejepa_loss
from brain.objectives.sigreg import SIGReg
from brain.train import engine


def build_encoder(cfg: dict, device: torch.device) -> BrainEncoder:
    m = cfg["model"]
    dense = float(cfg["loss"].get("dense", {}).get("weight", 0.0)) > 0
    pr = cfg["loss"].get("predict") or {}
    predict = dense and float(pr.get("weight", 0.0)) > 0
    cells = pr.get("pixel_cells", (2, 2, 2))
    return BrainEncoder(
        size=m["size"],
        patch_xyz=tuple(cfg["views"]["patch_xyz"]),
        in_chans=int(m.get("in_chans", 1)),
        proj_dim=int(m["proj_dim"]),
        proj_hidden=m.get("proj_hidden"),
        qk_norm=bool(m.get("qk_norm", False)),
        rope_base_mm=float(m.get("rope_base_mm", 100.0)),
        rope_min_period_mm=m.get("rope_min_period_mm"),
        rope_fp32=bool(m.get("rope_fp32", False)), fp32_residual=bool(m.get("fp32_residual", False)),
        n_meta=n_meta(m),
        checkpoint_blocks=bool(m.get("checkpoint", True)),
        proj_norm=str(m.get("proj_norm", "batch")),
        dense_dim=int(m.get("dense_dim", 0)) if dense else 0,
        dense_hidden=int(m.get("dense_hidden", 512)),
        predict_depth=int(pr.get("depth", 4)) if predict else 0,
        predict_pixel_dim=int(cells[0]) * int(cells[1]) * int(cells[2])
        if predict and float(pr.get("pixel_weight", 0.0)) > 0 else 0,
        n_cls=int(m.get("n_cls", 0)),
        n_registers=int(m.get("n_registers", 0)),
    ).to(device)


def loss_kwargs(cfg: dict) -> dict:
    lc = cfg["loss"]
    dense, pr = lc.get("dense") or {}, lc.get("predict") or {}
    return dict(
        lamb=float(lc["lamb"]), win_weight=float(lc.get("win_weight", 1.0)),
        dense_weight=float(dense.get("weight", 0.0)), tokens_per_view=int(dense.get("tokens_per_view", 4)),
        predict_weight=float(pr.get("weight", 0.0)), pixel_weight=float(pr.get("pixel_weight", 0.0)),
        stopgrad_mean=bool(dense.get("stopgrad_mean", False)),
        fine_weight=float(dense.get("fine_weight", 0.0)),
        fine_predict_weight=float(pr.get("fine_weight", 0.0)),
        gather_world_ref=lc.get("gather_world_ref"),
    )


def metric_keys(cfg: dict) -> list[str]:
    """Training-log columns implied by the loss config, in log order."""
    lk = loss_kwargs(cfg)
    dense_on = lk["dense_weight"] > 0
    predict_on = dense_on and lk["predict_weight"] > 0
    pix = ["pix"] if lk["pixel_weight"] > 0 else []
    keys = ["loss", "inv", "win", "sigreg"] + (["dense_inv", "dense_sig", "overlap"] if dense_on else [])
    keys += ["fine_inv", "fine_overlap"] if dense_on and lk["fine_weight"] > 0 else []
    keys += ["pred"] + pix if predict_on else []
    keys += ["fine_pred"] + (["fine_pix"] if pix else []) if predict_on and lk["fine_predict_weight"] > 0 else []
    return keys


def _record(obj, stream) -> None:
    """Tell the allocator that ``stream`` also uses these tensors, which were built on another stream."""
    if torch.is_tensor(obj):
        if obj.is_cuda:
            obj.record_stream(stream)
    elif isinstance(obj, dict):
        for v in obj.values():
            _record(v, stream)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _record(v, stream)


def forever_loader(store, split, cfg, *, batch_size, shuffle, seed, distributed, n_volumes=None):
    """Yields lists of crop chunks that together hold ``batch_size`` volumes."""
    loader = make_loader(store, split, cfg, batch_size=batch_size, shuffle=shuffle, seed=seed,
                         distributed=distributed, n_volumes=n_volumes)
    parts = batch_size // crop_microbatch(batch_size)

    def _gen():
        epoch = seed
        while True:
            if isinstance(loader.sampler, torch.utils.data.DistributedSampler):
                loader.sampler.set_epoch(epoch)
            buf = []
            for chunk in loader:
                buf.append(chunk)
                if len(buf) == parts:
                    yield buf
                    buf = []
            epoch += 1

    return _gen()


def synthetic_loader(cfg: dict, batch_size: int, seed: int):
    size = tuple(cfg["data"]["volume_size_xyz"])
    sp = tuple(cfg["data"]["default_spacing_mm"])
    orig = tuple(cfg["data"]["default_origin_mm"])
    n = seed

    def _gen():
        nonlocal n
        while True:
            batch = collate_volumes([
                make_synthetic_volume(size, sp, orig, modality=n % 2, seed=n + i) for i in range(batch_size)
            ])
            n += batch_size
            batch["volume"] = batch["volume"].permute(0, 1, 4, 3, 2)
            yield [raw_views(batch, cfg)]

    return _gen()


class Trainer(engine.Trainer):
    unit = "vol"
    synthetic_ok = True

    def __init__(self, rt, cfg, args, ckpt, step):
        super().__init__(rt, cfg, args, ckpt, step)
        m, o, t, lc = cfg["model"], cfg["optim"], cfg["train"], cfg["loss"]
        dev, synthetic = rt.device, args.synthetic
        self.max_steps = int(args.max_steps or t["max_steps"])
        self.batch = int(o["batch_size"])
        self.batch_units = self.batch * rt.world
        self.buckets = int(m.get("length_buckets", 4))
        self.lk, self.gather = loss_kwargs(cfg), bool(lc.get("gather_ddp", True))
        self.clip = float(o.get("clip_grad", 1.0))
        self.keys = metric_keys(cfg)
        pcfg = lc.get("predict") or {}
        self.dense_on = self.lk["dense_weight"] > 0
        if float(pcfg.get("weight", 0.0)) > 0 and not self.dense_on:
            raise SystemExit("loss.predict needs loss.dense.weight > 0: it predicts the token head's embeddings")
        predict_on = self.dense_on and self.lk["predict_weight"] > 0
        fine_pred_on = predict_on and float(pcfg.get("fine_weight", 0.0)) > 0
        self.ramp = int(pcfg.get("ramp_steps", 0))
        self.amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

        seed = int(cfg["seed"]) + rt.rank
        val_batch = min(self.batch, 96)
        if synthetic:
            self.loader = synthetic_loader(cfg, self.batch, seed)
            self.val_iter = synthetic_loader(cfg, val_batch, seed + 999)
        else:
            store = resolve(cfg["data"]["store_root"])
            # a resumed run continues with sampler epochs / crop draws it has not used yet (seed + step)
            self.loader = forever_loader(store, "train", cfg, batch_size=self.batch, shuffle=True,
                                         seed=seed + int(step), distributed=rt.world > 1)
            self.val_iter = forever_loader(store, "val", cfg, batch_size=val_batch, shuffle=False, seed=0,
                                           distributed=False) if rt.main else None

        raw = build_encoder(cfg, dev)
        if rt.world > 1:
            raw = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw)
        self.raw, self.module = raw, raw
        self.core = ViewNet(raw, local_pooled=bool(lc.get("local_pooled", False)), dense=self.dense_on,
                            predict=predict_on, predict_fine=fine_pred_on,
                            cells_xyz=tuple(pcfg.get("pixel_cells", (2, 2, 2))))
        self.sigreg = SIGReg(n_proj=int(lc.get("n_proj", 256))).to(dev)
        self.opt = torch.optim.AdamW(param_groups(raw, float(o["weight_decay"])), lr=float(o["lr"]),
                                     betas=tuple(o.get("betas", (0.9, 0.95))), fused=dev.type == "cuda")
        if ckpt is not None:
            raw.load_state_dict(ckpt["model"])
            self.opt.load_state_dict(ckpt["optimizer"])
        elif cfg["output"].get("init_from") and not synthetic:
            self._init_from(resolve(cfg["output"]["init_from"]))
        self.net = torch.nn.parallel.DistributedDataParallel(self.core, device_ids=[rt.local]) \
            if rt.world > 1 else self.core

        # Nothing in the step reads a GPU value on the host, so the CPU queues the next batch while the GPU
        # runs this one. Finiteness is checked when logging and before a checkpoint is written.
        self.side = torch.cuda.Stream() if dev.type == "cuda" else None
        self.nxt, self.last_out = None, None
        self.core.train()
        self._banner = (f"brain-{raw.size} params={raw.count_parameters():,} batch={self.batch}x{rt.world} "
                        f"dense={self.dense_on} predict={predict_on} fine={fine_pred_on} proj_norm={raw.projector.norm}")

    def _init_from(self, path) -> None:
        """Weights only: a fresh optimizer and schedule. Heads the source never had (the predictor) start random."""
        src = load_ckpt(path, self.rt.device)
        res = self.raw.load_state_dict(strip_module_prefix(src["model"]), strict=False)
        if res.unexpected_keys or any(not k.startswith("predictor.") for k in res.missing_keys):
            raise SystemExit(f"[init_from] mismatch: missing={res.missing_keys[:4]} unexpected={res.unexpected_keys[:4]}")
        self.rt.log(f"[init] weights from {path} (step {src.get('step')}); new: {len(res.missing_keys)} tensors")

    def banner(self) -> str:
        return self._banner

    def _fetch(self) -> dict:
        """Builds the next batch on a side stream so the copy and windowing overlap the running step."""
        chunks = next(self.loader)
        if self.side is None:
            return prepare(chunks, self.cfg, self.rt.device, self.buckets)
        with torch.cuda.stream(self.side):
            return prepare(chunks, self.cfg, self.rt.device, self.buckets)

    def _adopt(self, batch: dict) -> dict:
        if self.side is not None:
            torch.cuda.current_stream().wait_stream(self.side)
            _record(batch, torch.cuda.current_stream())
        return batch

    def step(self, step: int, lr: float) -> dict[str, torch.Tensor]:
        batch = self._adopt(self.nxt if self.nxt is not None else self._fetch())
        self.nxt = None
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast(**self.amp):
            out = self.net(batch)
        scale = min(1.0, (step + 1) / self.ramp) if self.ramp else 1.0
        parts = lejepa_loss(out, batch_meta(batch), self.sigreg, gather_ddp=self.gather,
                            **{**self.lk, "predict_weight": self.lk["predict_weight"] * scale})
        parts["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.raw.parameters(), self.clip)
        self.opt.step()
        self.last_out = out
        if step + 1 < self.max_steps:
            self.nxt = self._fetch()
        return {k: parts[k].detach() for k in self.keys}

    def log_extras(self) -> dict[str, str]:
        out = self.last_out
        z = out["proj"].detach().float().reshape(-1, out["proj"].size(-1))
        cells = {"erank": f"{embedding_health(z)['erank']:.1f}/{z.size(0)}"}
        if self.dense_on:
            tok = out["dense"]["emb"][0].detach().float()[out["dense"]["valid"][0]]
            cells["tok_erank"] = f"{embedding_health(tok[:2048])['erank']:.1f}"
        return cells

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """One fixed validation batch with fixed windows. The BatchNorm layers use their running statistics."""
        if self.val_iter is None:
            return {}
        dev = self.rt.device
        self.core.eval()
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            torch.manual_seed(0)
            batch = prepare(next(self.val_iter), self.cfg, dev, self.buckets)
            with torch.autocast(**self.amp):
                out = self.core(batch)
            parts = lejepa_loss(out, batch_meta(batch), self.sigreg, gather_ddp=False, **self.lk)
        self.core.train()
        return {f"val_{k}": float(parts[k]) for k in self.keys if k in parts}

    def state(self, model_only: bool) -> dict:
        s = {"model": self.raw.state_dict()}
        if not model_only:
            s["optimizer"] = self.opt.state_dict()
        return s
