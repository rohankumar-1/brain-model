"""2D LeJEPA pretraining of a ViT on CT/MRI slices: crop matching plus SIGReg.

  python -m brain.train lejepa2d --config configs/lejepa2d/tiny.yaml
  torchrun --standalone --nproc_per_node=4 -m brain.train lejepa2d --config configs/lejepa2d/small.yaml

Views are made on the GPU from raw slabs, so every view of a slice shares one read. Two optional
auxiliary terms: ``views.window.inv_weight`` (same crop under two windows) and ``views.window.seq_weight``
(same crop in two co-registered BraTS sequences).
"""

from __future__ import annotations

from pathlib import Path

import torch

from brain.common.metrics import effective_rank
from brain.common.optim import param_groups
from brain.common.paths import resolve
from brain.data.crops2d import multi_crop, same_crop_seq_pair, same_crop_window_pair
from brain.data.slabs import loader_workers, make_slab_loader, take, unpack_batch
from brain.models.vit2d import VisionTransformer
from brain.objectives.sigreg import SIGReg, lejepa_loss, pairwise_cosine, per_sample_invariance, prediction_loss
from brain.train import engine


def write_window_montage(path: Path, x: torch.Tensor, mod: torch.Tensor, n: int = 8, n_win: int = 4,
                         mix: str = "v4") -> None:
    """Grid of the same native slices under independent random windows, for visual sign-off."""
    from torchvision.utils import save_image

    from brain.common.windowing import asinh_window, sample_windows

    n = min(n, x.size(0))
    x, mod = x[:n], mod[:n]
    rows = []
    for _ in range(n_win):
        c, w = sample_windows(mod, "random", mix=mix)
        y = asinh_window(x, c, w)
        lo, hi = y.amin(dim=(-2, -1), keepdim=True), y.amax(dim=(-2, -1), keepdim=True)
        rows.append((y - lo) / (hi - lo).clamp_min(1e-6))
    save_image(torch.cat(rows), path, nrow=n, normalize=False)


def window_pair_loss(net, x, mod, crop_kw):
    """Same-crop two-window prediction loss and cosine, in projector space."""
    pair = same_crop_window_pair(
        x, mod, area=crop_kw["global_area"], max_rot=crop_kw["max_rot"], window_mix=crop_kw["window_mix"])
    proj = net(pair, None)
    return prediction_loss(proj), pairwise_cosine(proj.detach())


def seq_pair_loss(net, x, sib, mod, ok, crop_kw):
    """Same-crop BraTS sequence pair. Loss/cosine are masked to rows with a sibling."""
    pair = same_crop_seq_pair(
        x, sib, mod, area=crop_kw["global_area"], max_rot=crop_kw["max_rot"], window_mix=crop_kw["window_mix"])
    proj = net(pair, None)
    per = per_sample_invariance(proj)
    inv = (per * ok).sum() / ok.sum().clamp_min(1.0)
    mask = ok > 0.5
    cos = pairwise_cosine(proj[:, mask].detach()) if mask.any() else proj.new_zeros(())
    return inv, cos, ok.mean()


class Trainer(engine.Trainer):
    unit = "slices"

    def __init__(self, rt, cfg, args, ckpt, step):
        super().__init__(rt, cfg, args, ckpt, step)
        m, o, d, t = cfg["model"], cfg["optim"], cfg["data"], cfg["train"]
        win = cfg["views"]["window"]
        self.batch, self.lamb, self.n_global = int(o["batch_size"]), float(o["lamb"]), int(o["n_global"])
        self.win_w, self.seq_w = float(win.get("inv_weight", 0.0)), float(win.get("seq_weight", 0.0))
        self.window_mix = str(win.get("mix", "v4"))
        self.crop_kw = dict(
            n_global=self.n_global, n_local=int(o["n_local"]),
            global_area=tuple(o["global_area"]), local_area=tuple(o["local_area"]),
            local_size=int(o["local_size"]), max_rot=float(o.get("max_rot", 10.0)),
            window=str(win.get("mode", "random")), window_mix=self.window_mix,
        )
        self.clip = float(o.get("clip_grad", 1.0))
        self.n_eval = int(t["eval_batches"])
        self.batch_units = self.batch * rt.world

        workers = loader_workers(int(o["num_workers"]), rt.world)
        root = resolve(d["slices_root"])
        kw = dict(batch_slices=self.batch, k=int(d["slices_per_read"]), alpha=float(d["balance_alpha"]),
                  load_siblings=self.seq_w > 0, prefetch=int(o.get("prefetch_factor", 2)))
        self.loader = make_slab_loader(
            root, "train", workers=workers, seed=(int(cfg["seed"]) + rt.rank) * 100_003 + step,
            priority_beta=float(d.get("priority_beta", 0.0)), priority_clip=float(d.get("priority_clip", 3.0)), **kw)
        self.val_loader = make_slab_loader(
            root, "val", workers=min(2, workers), seed=0, persistent=False, **kw) if rt.main else None
        self.it = iter(self.loader)

        self.raw = VisionTransformer(
            size=m["size"], img_size=int(m["img_size"]), patch_size=int(m["patch_size"]),
            proj_dim=int(m["proj_dim"]), proj_hidden=m.get("proj_hidden"),
            registers=int(m.get("registers", 0)), qk_norm=bool(m.get("qk_norm", False)),
        ).to(rt.device)
        self.module = self.raw
        self.sigreg = SIGReg().to(rt.device)
        self.opt = torch.optim.AdamW(param_groups(self.raw, float(o["weight_decay"])), lr=float(o["lr"]),
                                     betas=(0.9, 0.95), fused=rt.device.type == "cuda")
        if ckpt is not None:
            self.raw.load_state_dict(ckpt["model"])
            self.opt.load_state_dict(ckpt["optimizer"])
        self.net = self.raw
        if rt.world > 1:
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.raw, device_ids=[rt.local], broadcast_buffers=False)
        if o.get("compile", True) and rt.device.type == "cuda":
            self.net = torch.compile(self.net)
        self.raw.train()
        self._banner = (f"vit-{self.raw.size} params={self.raw.count_parameters():,} registers={self.raw.n_registers} "
                        f"batch={self.batch}x{rt.world} k={kw['k']} workers={workers} window={self.crop_kw['window']} "
                        f"mix={self.window_mix} win_inv={self.win_w:g} seq_inv={self.seq_w:g}")

    def banner(self) -> str:
        return self._banner

    def on_start(self, out_dir: Path, step: int) -> None:
        if step:
            return
        try:
            x, _, mod, _, _ = unpack_batch(next(iter(self.val_loader)), self.rt.device)
            write_window_montage(out_dir / "window_montage.png", x, mod, mix=self.window_mix)
            self.rt.log(f"[qc] wrote {out_dir / 'window_montage.png'}")
        except Exception as e:  # QC only; never block training
            self.rt.log(f"[qc] window montage skipped: {type(e).__name__}: {e}")

    def _autocast(self):
        return torch.autocast(self.rt.device.type, dtype=torch.bfloat16, enabled=self.rt.device.type == "cuda")

    def step(self, step: int, lr: float) -> dict[str, torch.Tensor]:
        dev = self.rt.device
        x, chunk_ids, mod, sib, ok = unpack_batch(next(self.it), dev)
        gv, lv = multi_crop(x, mod, **self.crop_kw)
        self.opt.zero_grad(set_to_none=True)
        with self._autocast():
            proj = self.net(gv, lv)
            loss, inv, reg = lejepa_loss(proj, self.sigreg, self.lamb, self.n_global)
        sampler = self.loader.sampler
        if sampler.beta > 0:  # hardness of each chunk relative to this batch
            hard = per_sample_invariance(proj.detach().float(), self.n_global).view(len(chunk_ids), -1).mean(1)
            sampler.update(chunk_ids, hard / hard.mean())
        loss.backward()
        out = {"loss": loss.detach(), "inv": inv.detach(), "sigreg": reg.detach()}
        # Auxiliary pairs get their own backward so they do not sit on top of the LeJEPA graph (OOM).
        if self.win_w:
            with self._autocast():
                win_inv, win_cos = window_pair_loss(self.net, x, mod, self.crop_kw)
                term = self.win_w * win_inv
            term.backward()
            out["loss"] = out["loss"] + term.detach()
            out["win_inv"], out["win_cos"] = win_inv.detach(), win_cos.detach()
        if self.seq_w:
            with self._autocast():
                seq_inv, seq_cos, seq_frac = seq_pair_loss(self.net, x, sib, mod, ok, self.crop_kw)
                term = self.seq_w * seq_inv
            term.backward()
            out["loss"] = out["loss"] + term.detach()
            out["seq_inv"], out["seq_cos"], out["seq_frac"] = seq_inv.detach(), seq_cos.detach(), seq_frac.detach()
        torch.nn.utils.clip_grad_norm_(self.raw.parameters(), self.clip)
        self.opt.step()
        return out

    def log_extras(self) -> dict[str, str]:
        s = self.loader.sampler
        return {"seen": f"{s.visited():.0%}"} if s.beta > 0 else {}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """LeJEPA loss on fixed validation slabs plus effective rank of the class token."""
        raw, dev = self.raw, self.rt.device
        raw.eval()
        keys = ["loss", "inv", "sigreg"] + (["win_inv", "win_cos"] if self.win_w else []) \
            + (["seq_inv", "seq_cos", "seq_frac"] if self.seq_w else [])
        tot, feats = torch.zeros(len(keys), device=dev), []
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            torch.manual_seed(0)
            for batch in take(self.val_loader, self.n_eval):
                x, _, mod, sib, ok = unpack_batch(batch, dev)
                gv, lv = multi_crop(x, mod, **self.crop_kw)
                with self._autocast():
                    loss, inv, reg = lejepa_loss(raw(gv, lv), self.sigreg, self.lamb, self.n_global)
                    feats.append(raw.encode(x).float())
                    row = {"inv": inv, "sigreg": reg}
                    if self.win_w:
                        row["win_inv"], row["win_cos"] = window_pair_loss(raw, x, mod, self.crop_kw)
                        loss = loss + self.win_w * row["win_inv"]
                    if self.seq_w:
                        row["seq_inv"], row["seq_cos"], row["seq_frac"] = seq_pair_loss(raw, x, sib, mod, ok, self.crop_kw)
                        loss = loss + self.seq_w * row["seq_inv"]
                    row["loss"] = loss
                    tot += torch.stack([row[k].float() for k in keys])
        raw.train()
        vals = (tot / max(self.n_eval, 1)).tolist()
        return {**{f"val_{k}": v for k, v in zip(keys, vals)}, "erank": effective_rank(torch.cat(feats))}

    def state(self, model_only: bool) -> dict:
        s = {"model": self.raw.state_dict()}
        if not model_only:
            s["optimizer"] = self.opt.state_dict()
        return s
