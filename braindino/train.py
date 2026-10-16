"""BrainDINO-style DINOv2 + iBOT on the existing 2D slab store.

Does not modify ``train_vit.py``. Checkpoints write ``model`` as a bare
``VisionTransformer`` state so ``downstream.embeddings.load_frozen_encoder`` works.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.checkpoint import atomic_torch_save, load_ckpt
from lib.config import load_config, save_config
from lib.slabs import make_slab_loader, take
from lib.train_utils import StopFlag, dist_info, effective_rank, lr_at
from lib.views import apply_window, multi_crop
from models.vit import VisionTransformer

from braindino.encoder import BrainDINO
from braindino.heads import update_center
from braindino.loss import sample_masks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "braindino_small.yaml")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def unpack_batch(batch, device):
    slabs, chunk_ids, mods = batch[0], batch[1], batch[2]
    k = slabs.shape[1]
    x = slabs.to(device, non_blocking=True).flatten(0, 1).float()
    mod = mods.to(device, non_blocking=True).repeat_interleave(k)
    return x, chunk_ids, mod


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def param_groups(net: BrainDINO, lr: float, wd: float, layer_decay: float) -> list[dict]:
    n_blocks = net.backbone.depth
    buckets: dict[tuple[float, float], list] = {}
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone.blocks."):
            i = int(name.split(".")[2])
            scale = layer_decay ** (n_blocks - 1 - i)
        elif name.startswith("backbone."):
            scale = layer_decay ** n_blocks
        else:
            scale = 1.0
        skip = name.endswith(("bias", "pos_embed", "cls_token", "register_tokens", "mask_token")) or p.ndim < 2
        key = (scale, 0.0 if skip else wd)
        buckets.setdefault(key, []).append(p)
    return [{"params": ps, "lr": lr * s, "weight_decay": d, "lr_scale": s} for (s, d), ps in buckets.items()]


def make_backbone(cfg: dict) -> VisionTransformer:
    m = cfg["model"]
    return VisionTransformer(
        size=m["size"], img_size=int(m["img_size"]), patch_size=int(m["patch_size"]),
        proj_dim=int(m.get("proj_dim", 128)), proj_hidden=m.get("proj_hidden"),
        registers=int(m.get("registers", 4)), qk_norm=bool(m.get("qk_norm", True)),
    )


def n_tokens(img_size: int, patch: int) -> int:
    return (img_size // patch) ** 2


@torch.no_grad()
def evaluate(teacher: BrainDINO, loader, crop_kw, device, n_batches: int, window: str, mix: str) -> dict:
    teacher.eval()
    feats = []
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(0)
        for batch in take(loader, n_batches):
            x, _, mod = unpack_batch(batch, device)
            y = apply_window(x, mod, "fixed" if window == "fixed" else "random", mix=mix)
            cls, _ = teacher.tokens(y)
            feats.append(cls.float())
    teacher.train()
    z = torch.cat(feats)
    return {"erank": effective_rank(z), "cls_std": float(z.std())}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides={"output": {"out_dir": str(args.out_dir)}} if args.out_dir else None)
    rank, world, local = dist_info()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if world > 1:
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
    log = print if rank == 0 else (lambda *a, **k: None)

    seed = int(cfg["seed"]) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")

    m, o, d, t = cfg["model"], cfg["optim"], cfg["data"], cfg["train"]
    loss_cfg = cfg.get("loss") or {}
    max_steps = int(args.max_steps or t["max_steps"])
    batch = int(o["batch_size"])
    proto = int(m.get("proto_dim", 65536))
    n_global, n_local = int(o["n_global"]), int(o["n_local"])
    student_temp = float(loss_cfg.get("student_temp", 0.1))
    teacher_temp = float(loss_cfg.get("teacher_temp", 0.04))
    ema_m = float(loss_cfg.get("ema", 0.994))
    ibot_w = float(loss_cfg.get("ibot_weight", 1.0))
    koleo_w = float(loss_cfg.get("koleo_weight", 0.1))
    mask_p = float(loss_cfg.get("mask_prob", 0.5))
    mask_ratio = tuple(loss_cfg.get("mask_ratio", (0.1, 0.5)))
    window_cfg = (cfg.get("views") or {}).get("window") or {}
    window_mode = str(window_cfg.get("mode", "random"))
    window_mix = str(window_cfg.get("mix", "v4"))
    crop_kw = dict(
        n_global=n_global, n_local=n_local,
        global_area=tuple(o["global_area"]), local_area=tuple(o["local_area"]),
        local_size=int(o["local_size"]), max_rot=float(o.get("max_rot", 10.0)),
        window=window_mode, window_mix=window_mix,
    )
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)
    workers = int(o["num_workers"])
    if cpus > 1:
        workers = max(1, min(workers, cpus // world - 1))

    out_dir = resolve(cfg["output"]["out_dir"])
    last_path = out_dir / "last.pt"
    step, history, ckpt = 0, [], None
    resume_src = None
    if not args.no_resume:
        if last_path.exists():
            resume_src = last_path
        elif cfg.get("output", {}).get("resume_from"):
            resume_src = resolve(cfg["output"]["resume_from"])
    if resume_src is not None:
        ckpt = load_ckpt(resume_src, device)
        step, history = int(ckpt["step"]), list(ckpt.get("history") or [])

    slices_root = resolve(d["slices_root"])
    loader_kw = dict(batch_slices=batch, k=int(d["slices_per_read"]), alpha=float(d["balance_alpha"]))
    loader = make_slab_loader(
        slices_root, "train", workers=workers, seed=seed * 100_003 + step,
        priority_beta=float(d.get("priority_beta", 0.0)), priority_clip=float(d.get("priority_clip", 3.0)),
        **loader_kw,
    )
    val_loader = make_slab_loader(
        slices_root, "val", workers=min(2, workers), seed=0, persistent=False, **loader_kw,
    ) if rank == 0 else None

    student = BrainDINO(
        make_backbone(cfg).to(device), proto,
        hidden=int(m.get("head_hidden", 2048)), bottleneck=int(m.get("head_bottleneck", 256)),
    ).to(device)
    student.backbone.projector.requires_grad_(False)  # LeJEPA head is unused; DDP would hang on it
    teacher = deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    center = torch.zeros(proto, device=device)
    center_patch = torch.zeros(proto, device=device)
    opt = torch.optim.AdamW(
        param_groups(student, float(o["lr"]), float(o["weight_decay"]), float(o.get("layer_decay", 0.9))),
        lr=float(o["lr"]), betas=(0.9, 0.999), fused=device.type == "cuda",
    )
    if ckpt is not None:
        if "student" in ckpt:
            student.load_state_dict(ckpt["student"])
            teacher.load_state_dict(ckpt["teacher"])
        else:
            student.backbone.load_state_dict(ckpt["model"], strict=False)
            teacher.load_state_dict(student.state_dict())
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("center") is not None:
            center.copy_(ckpt["center"])
            center_patch.copy_(ckpt["center_patch"])
    net = student
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[local], broadcast_buffers=False, find_unused_parameters=True,
        )
    raw = student

    log(f"[model] braindino vit-{raw.backbone.size} proto={proto} params={sum(p.numel() for p in raw.parameters()):,} "
        f"batch={batch}x{world} n_global={n_global} n_local={n_local} ema={ema_m:g} "
        f"steps={step}/{max_steps}" + (f" resume={resume_src}" if resume_src is not None else ""))
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")

    def save(path: Path, model_only: bool = False) -> None:
        state = {
            "model": raw.backbone_state(),
            "student": raw.state_dict(),
            "teacher": teacher.state_dict(),
            "center": center.detach().cpu(),
            "center_patch": center_patch.detach().cpu(),
            "step": step, "config": cfg, "history": history,
        }
        if not model_only:
            state["optimizer"] = opt.state_dict()
        atomic_torch_save(state, path)

    stop = StopFlag()
    it, acc, seen, t0 = iter(loader), torch.zeros(5, device=device), 0, time.time()
    n_tok_g = n_tokens(int(m["img_size"]), int(m["patch_size"]))
    student.train()
    while step < max_steps:
        lr = lr_at(step, max_steps, float(o["lr"]), int(o["warmup_steps"]))
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)
        x, _, mod = unpack_batch(next(it), device)
        gv, lv = multi_crop(x, mod, **crop_kw)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            masks = torch.stack([sample_masks(gv.size(1), n_tok_g, device, mask_p, mask_ratio)
                                 for _ in range(n_global)])
            t_dino, t_ibot = [], []
            for v in range(n_global):
                c, p = teacher.tokens(gv[v])
                t_dino.append(teacher.dino_head(c))
                t_ibot.append(teacher.ibot_head(p))
            t_dino = torch.stack(t_dino).detach()
            t_ibot = torch.stack(t_ibot).detach()
            l_dino, l_ibot, l_koleo, cls0 = net(
                gv, lv, masks, t_dino, t_ibot, center, center_patch, student_temp, teacher_temp,
            )
            loss = l_dino + ibot_w * l_ibot + koleo_w * l_koleo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), float(o.get("clip_grad", 3.0)))
        opt.step()
        ema_update(raw, teacher, ema_m)
        cm = float(loss_cfg.get("center_momentum", 0.9))
        update_center(center, t_dino.flatten(0, 1), cm)
        update_center(center_patch, t_ibot.flatten(0, 1).flatten(0, 1), cm)
        acc += torch.stack([loss.detach(), l_dino.detach(), l_ibot.detach(), l_koleo.detach(),
                            cls0.detach().float().std()])
        step, seen = step + 1, seen + 1

        if step % int(t["log_every"]) == 0:
            a = (acc / seen).tolist()
            log(f"[step {step}/{max_steps}] loss={a[0]:.4f} dino={a[1]:.4f} ibot={a[2]:.4f} "
                f"koleo={a[3]:.4f} cls_std={a[4]:.4f} lr={lr:.2e} "
                f"{seen * batch * world / (time.time() - t0):.0f} slices/s")
            acc.zero_()
            seen, t0 = 0, time.time()
        if rank == 0 and step % int(t["eval_every"]) == 0:
            row = {"step": step, **evaluate(
                teacher, val_loader, crop_kw, device, int(t["eval_batches"]), window_mode, window_mix,
            )}
            history.append(row)
            log("[eval] " + " ".join(f"{k}={v:.4f}" if k != "step" else f"{k}={v}" for k, v in row.items()))
            (out_dir / "metrics.json").write_text(json.dumps({"history": history}, indent=2) + "\n")
        if rank == 0 and step % int(t["ckpt_every"]) == 0:
            save(last_path)
            if step % int(t["keep_every"]) == 0:
                save(out_dir / f"step_{step:07d}.pt", model_only=True)
        if step % int(t.get("stop_poll", 20)) == 0 and stop.poll(device):
            break

    if rank == 0:
        save(last_path)
        preempted = step < max_steps
        log(f"[ckpt] wrote {last_path} at step {step}" + (" (preempted)" if preempted else " (done)"))
        if preempted and os.environ.get("REQUEUE_FLAG"):
            Path(os.environ["REQUEUE_FLAG"]).touch()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
