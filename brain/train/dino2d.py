"""BrainDINO-style DINOv2 + iBOT on the 2D slab store (the MRI-only foundation-model comparison).

  python -m brain.train dino2d --config configs/dino2d/small.yaml

Checkpoints write ``model`` as a bare ``VisionTransformer`` state so
``brain.downstream.embeddings.load_frozen_encoder`` reads them like a LeJEPA run.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from brain.common.metrics import effective_rank
from brain.common.optim import param_groups
from brain.common.paths import resolve
from brain.data.crops2d import apply_window, multi_crop
from brain.data.slabs import loader_workers, make_slab_loader, take, unpack_batch
from brain.models.braindino import BrainDINO
from brain.models.vit2d import VisionTransformer
from brain.objectives.dino import sample_masks, update_center
from brain.train import engine


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def make_backbone(cfg: dict) -> VisionTransformer:
    m = cfg["model"]
    return VisionTransformer(
        size=m["size"], img_size=int(m["img_size"]), patch_size=int(m["patch_size"]),
        proj_dim=int(m.get("proj_dim", 128)), proj_hidden=m.get("proj_hidden"),
        registers=int(m.get("registers", 4)), qk_norm=bool(m.get("qk_norm", True)),
    )


class Trainer(engine.Trainer):
    unit = "slices"

    def __init__(self, rt, cfg, args, ckpt, step):
        super().__init__(rt, cfg, args, ckpt, step)
        m, o, d, t, lc = cfg["model"], cfg["optim"], cfg["data"], cfg["train"], cfg["loss"]
        dev = rt.device
        self.batch = int(o["batch_size"])
        self.batch_units = self.batch * rt.world
        proto = int(m.get("proto_dim", 65536))
        self.n_global, n_local = int(o["n_global"]), int(o["n_local"])
        self.student_temp, self.teacher_temp = float(lc["student_temp"]), float(lc["teacher_temp"])
        self.ema_m, self.ibot_w, self.koleo_w = float(lc["ema"]), float(lc["ibot_weight"]), float(lc["koleo_weight"])
        self.mask_p, self.mask_ratio = float(lc["mask_prob"]), tuple(lc["mask_ratio"])
        self.center_m = float(lc["center_momentum"])
        win = cfg["views"]["window"]
        self.window_mode, self.window_mix = str(win.get("mode", "random")), str(win.get("mix", "v4"))
        self.crop_kw = dict(
            n_global=self.n_global, n_local=n_local,
            global_area=tuple(o["global_area"]), local_area=tuple(o["local_area"]),
            local_size=int(o["local_size"]), max_rot=float(o.get("max_rot", 10.0)),
            window=self.window_mode, window_mix=self.window_mix,
        )
        self.clip = float(o.get("clip_grad", 3.0))
        self.n_eval = int(t["eval_batches"])
        self.n_tok = (int(m["img_size"]) // int(m["patch_size"])) ** 2

        workers = loader_workers(int(o["num_workers"]), rt.world)
        root = resolve(d["slices_root"])
        kw = dict(batch_slices=self.batch, k=int(d["slices_per_read"]), alpha=float(d["balance_alpha"]),
                  prefetch=int(o.get("prefetch_factor", 2)))
        self.loader = make_slab_loader(
            root, "train", workers=workers, seed=(int(cfg["seed"]) + rt.rank) * 100_003 + step,
            priority_beta=float(d.get("priority_beta", 0.0)), priority_clip=float(d.get("priority_clip", 3.0)), **kw)
        self.val_loader = make_slab_loader(
            root, "val", workers=min(2, workers), seed=0, persistent=False, **kw) if rt.main else None
        self.it = iter(self.loader)

        self.student = BrainDINO(make_backbone(cfg).to(dev), proto, hidden=int(m.get("head_hidden", 2048)),
                                 bottleneck=int(m.get("head_bottleneck", 256))).to(dev)
        self.student.backbone.projector.requires_grad_(False)  # LeJEPA head is unused; DDP would hang on it
        self.teacher = deepcopy(self.student).to(dev)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        self.module = self.student
        self.center = torch.zeros(proto, device=dev)
        self.center_patch = torch.zeros(proto, device=dev)
        self.opt = torch.optim.AdamW(
            param_groups(self.student, float(o["weight_decay"]), float(o.get("layer_decay", 0.9)), float(o["lr"])),
            lr=float(o["lr"]), betas=(0.9, 0.999), fused=dev.type == "cuda")
        if ckpt is not None:
            if "student" in ckpt:
                self.student.load_state_dict(ckpt["student"])
                self.teacher.load_state_dict(ckpt["teacher"])
            else:  # a bare-backbone checkpoint
                self.student.backbone.load_state_dict(ckpt["model"], strict=False)
                self.teacher.load_state_dict(self.student.state_dict())
            if "optimizer" in ckpt:
                self.opt.load_state_dict(ckpt["optimizer"])
            if ckpt.get("center") is not None:
                self.center.copy_(ckpt["center"])
                self.center_patch.copy_(ckpt["center_patch"])
        self.net = self.student
        if rt.world > 1:
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.student, device_ids=[rt.local], broadcast_buffers=False, find_unused_parameters=True)
        self.student.train()
        self._banner = (f"vit-{self.student.backbone.size} proto={proto} "
                        f"params={sum(p.numel() for p in self.student.parameters()):,} batch={self.batch}x{rt.world} "
                        f"n_global={self.n_global} n_local={n_local} ema={self.ema_m:g}")

    def banner(self) -> str:
        return self._banner

    def step(self, step: int, lr: float) -> dict[str, torch.Tensor]:
        dev = self.rt.device
        x, _, mod, _, _ = unpack_batch(next(self.it), dev)
        gv, lv = multi_crop(x, mod, **self.crop_kw)
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            masks = torch.stack([sample_masks(gv.size(1), self.n_tok, dev, self.mask_p, self.mask_ratio)
                                 for _ in range(self.n_global)])
            t_dino, t_ibot = [], []
            for v in range(self.n_global):
                c, p = self.teacher.tokens(gv[v])
                t_dino.append(self.teacher.dino_head(c))
                t_ibot.append(self.teacher.ibot_head(p))
            t_dino, t_ibot = torch.stack(t_dino).detach(), torch.stack(t_ibot).detach()
            l_dino, l_ibot, l_koleo, cls0 = self.net(
                gv, lv, masks, t_dino, t_ibot, self.center, self.center_patch, self.student_temp, self.teacher_temp)
            loss = l_dino + self.ibot_w * l_ibot + self.koleo_w * l_koleo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.clip)
        self.opt.step()
        ema_update(self.student, self.teacher, self.ema_m)
        update_center(self.center, t_dino.flatten(0, 1), self.center_m)
        update_center(self.center_patch, t_ibot.flatten(0, 1).flatten(0, 1), self.center_m)
        return {"loss": loss.detach(), "dino": l_dino.detach(), "ibot": l_ibot.detach(),
                "koleo": l_koleo.detach(), "cls_std": cls0.detach().float().std()}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        dev = self.rt.device
        self.teacher.eval()
        feats = []
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            torch.manual_seed(0)
            for batch in take(self.val_loader, self.n_eval):
                x, _, mod, _, _ = unpack_batch(batch, dev)
                y = apply_window(x, mod, "fixed" if self.window_mode == "fixed" else "random", mix=self.window_mix)
                cls, _ = self.teacher.tokens(y)
                feats.append(cls.float())
        z = torch.cat(feats)
        return {"erank": effective_rank(z), "cls_std": float(z.std())}

    def state(self, model_only: bool) -> dict:
        s = {"model": self.student.backbone_state(), "student": self.student.state_dict(),
             "teacher": self.teacher.state_dict(), "center": self.center.detach().cpu(),
             "center_patch": self.center_patch.detach().cpu()}
        if not model_only:
            s["optimizer"] = self.opt.state_dict()
        return s
