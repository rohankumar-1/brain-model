"""3D MAE pretraining on native-grid crops (Primus-style ViT encoder, voxel reconstruction of hidden tokens).

    python -m brain.train mae3d --config configs/mae3d/base.yaml
    python -m brain.train mae3d --synthetic --max-steps 8        # CPU smoke run, no store needed

Crops, geometric / appearance augmentation and family-balanced sampling are the LeJEPA pipeline's
(``brain.data.crops3d`` / ``batch3d``); masking hides ``loss.predict.mask_ratio`` of each crop's foreground tokens.
Checkpoints keep the encoder under ``model`` (so frozen evaluation loads it like any 3D encoder) and the decoder
under ``decoder``.
"""

from __future__ import annotations

import torch

from brain.common.checkpoint import load_ckpt, strip_module_prefix
from brain.common.optim import param_groups
from brain.common.paths import resolve
from brain.data.batch3d import prepare
from brain.models.mae3d import MAE3D, MAEDecoder
from brain.models.vit3d import BrainEncoder
from brain.train import engine
from brain.train.lejepa3d import _record, forever_loader, synthetic_loader


def build_encoder(cfg: dict, device: torch.device) -> BrainEncoder:
    m = cfg["model"]
    return BrainEncoder(
        size=m["size"], patch_xyz=tuple(cfg["views"]["patch_xyz"]), in_chans=int(m.get("in_chans", 1)),
        proj_dim=int(m.get("proj_dim", 128)), proj_hidden=m.get("proj_hidden"), qk_norm=bool(m.get("qk_norm", True)),
        rope_base_mm=float(m.get("rope_base_mm", 100.0)), checkpoint_blocks=bool(m.get("checkpoint", True)),
        n_cls=int(m.get("n_cls", 0)), n_registers=int(m.get("n_registers", 0)),
        mlp=str(m.get("mlp", "gelu")), layer_scale=m.get("layer_scale"), drop_path=float(m.get("drop_path", 0.0)),
        rope_fp32=bool(m.get("rope_fp32", False)), fp32_residual=bool(m.get("fp32_residual", False)),
    ).to(device)


def build_decoder(cfg: dict, enc: BrainEncoder, device: torch.device) -> MAEDecoder:
    d = cfg["model"].get("decoder") or {}
    px, py, pz = (int(p) for p in cfg["views"]["patch_xyz"])
    extra = 2 if str((cfg["loss"].get("predict") or {}).get("target", "voxels")) == "split" else 0
    return MAEDecoder(enc.embed_dim, int(d.get("dim", 384)), int(d.get("depth", 4)), int(d.get("heads", 6)),
                      px * py * pz + extra, float(cfg["model"].get("rope_base_mm", 100.0)),
                      qk_norm=bool(cfg["model"].get("qk_norm", True))).to(device)


class Trainer(engine.Trainer):
    unit = "vol"
    synthetic_ok = True
    keys = ("loss", "r2", "hidden")

    def __init__(self, rt, cfg, args, ckpt, step):
        super().__init__(rt, cfg, args, ckpt, step)
        o, t = cfg["optim"], cfg["train"]
        dev = rt.device
        self.max_steps = int(args.max_steps or t["max_steps"])
        self.batch = int(o["batch_size"])
        self.batch_units = self.batch * rt.world
        self.buckets = int(cfg["model"].get("length_buckets", 4))
        self.clip = float(o.get("clip_grad", 1.0))
        self.amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
        seed = int(cfg["seed"]) + rt.rank
        val_batch = min(self.batch, 32)
        if args.synthetic:
            self.loader = synthetic_loader(cfg, self.batch, seed)
            self.val_iter = synthetic_loader(cfg, val_batch, seed + 999)
        else:
            store = resolve(cfg["data"]["store_root"])
            self.loader = forever_loader(store, "train", cfg, batch_size=self.batch, shuffle=True,
                                         seed=seed + int(step), distributed=rt.world > 1)
            self.val_iter = forever_loader(store, "val", cfg, batch_size=val_batch, shuffle=False, seed=0,
                                           distributed=False) if rt.main else None
        enc = build_encoder(cfg, dev)
        dec = build_decoder(cfg, enc, dev)
        pr = cfg["loss"].get("predict") or {}
        self.raw = MAE3D(enc, dec, target=str(pr.get("target", "voxels")),
                         lesion_weight=float((pr.get("lesion") or {}).get("weight", 1.0)))
        self.module = enc
        self.opt = torch.optim.AdamW(param_groups(self.raw, float(o["weight_decay"])), lr=float(o["lr"]),
                                     betas=tuple(o.get("betas", (0.9, 0.95))), fused=dev.type == "cuda")
        if ckpt is not None:
            enc.load_state_dict(ckpt["model"])
            dec.load_state_dict(ckpt["decoder"])
            self.opt.load_state_dict(ckpt["optimizer"])
        elif cfg["output"].get("init_from") and not args.synthetic:
            src = load_ckpt(resolve(cfg["output"]["init_from"]), dev)
            enc.load_state_dict(strip_module_prefix(src["model"]))
        # the encoder's projector (built for the joint-embedding methods) gets no gradient here
        self.net = torch.nn.parallel.DistributedDataParallel(self.raw, device_ids=[rt.local],
                                                             find_unused_parameters=True) \
            if rt.world > 1 else self.raw
        self.side = torch.cuda.Stream() if dev.type == "cuda" else None
        self.nxt = None
        self.raw.train()
        self._banner = (f"enc={enc.size} params={enc.count_parameters():,} dec={dec.blocks.__len__()}x"
                        f"{dec.out.in_features} patch={tuple(cfg['views']['patch_xyz'])} "
                        f"mask={cfg['loss']['predict'].get('mask_ratio')} batch={self.batch}x{rt.world}")

    def banner(self) -> str:
        return self._banner

    def _fetch(self) -> dict:
        chunks = next(self.loader)
        if self.side is None:
            return prepare(chunks, self.cfg, self.rt.device, self.buckets)
        with torch.cuda.stream(self.side):
            return prepare(chunks, self.cfg, self.rt.device, self.buckets)

    def step(self, step: int, lr: float) -> dict[str, torch.Tensor]:
        batch = self.nxt if self.nxt is not None else self._fetch()
        if self.side is not None:
            torch.cuda.current_stream().wait_stream(self.side)
            _record(batch, torch.cuda.current_stream())
        self.nxt = None
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast(**self.amp):
            out = self.net(batch)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.raw.parameters(), self.clip)
        self.opt.step()
        if step + 1 < self.max_steps:
            self.nxt = self._fetch()
        return {k: out[k].detach() for k in self.keys}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """One fixed validation batch (fixed crops and masks)."""
        if self.val_iter is None:
            return {}
        dev = self.rt.device
        self.raw.eval()
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            torch.manual_seed(0)
            batch = prepare(next(self.val_iter), self.cfg, dev, self.buckets)
            with torch.autocast(**self.amp):
                out = self.raw(batch)
        self.raw.train()
        return {f"val_{k}": float(out[k]) for k in self.keys}

    def state(self, model_only: bool) -> dict:
        s = {"model": self.raw.enc.state_dict(), "decoder": self.raw.dec.state_dict()}
        if not model_only:
            s["optimizer"] = self.opt.state_dict()
        return s
