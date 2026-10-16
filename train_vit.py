#!/usr/bin/env python3
"""LeJEPA pretraining of a ViT on CT/MRI slices: crop matching plus SIGReg.

  python train_vit.py --config configs/vit_tiny.yaml
  torchrun --standalone --nproc_per_node=4 train_vit.py --config configs/vit_small.yaml

Training is step-based (warmup + cosine) and resumes from ``last.pt`` mid-epoch.
Views are made on the GPU from raw slabs, so every view of a slice shares one read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.checkpoint import atomic_torch_save, load_ckpt
from lib.config import load_config, save_config
from lib.slabs import make_slab_loader, take
from lib.train_utils import StopFlag, dist_info, effective_rank, lr_at
from lib.views import multi_crop, same_crop_seq_pair, same_crop_window_pair
from models.sigreg import SIGReg, lejepa_loss, pairwise_cosine, per_sample_invariance, prediction_loss
from models.vit import VisionTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "vit_tiny.yaml")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="override train.max_steps")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def param_groups(model: torch.nn.Module, wd: float) -> list[dict]:
    """No weight decay on biases, norms, or the learned tokens / positions."""
    skip = ("pos_embed", "cls_token", "register_tokens")
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and not n.endswith(skip)]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and not n.endswith(skip))]
    return [{"params": decay, "weight_decay": wd}, {"params": other, "weight_decay": 0.0}]


def unpack_batch(batch, device):
    """Loader yields (slabs, chunk_ids, modality[, sibling, ok]). Repeat across k slices."""
    slabs, chunk_ids, mods = batch[0], batch[1], batch[2]
    k = slabs.shape[1]
    x = slabs.to(device, non_blocking=True).flatten(0, 1).float()
    mod = mods.to(device, non_blocking=True).repeat_interleave(k)
    sib = ok = None
    if len(batch) >= 5:
        sib = batch[3].to(device, non_blocking=True).flatten(0, 1).float()
        ok = batch[4].to(device, non_blocking=True).repeat_interleave(k).float()
    return x, chunk_ids, mod, sib, ok


def write_window_montage(path: Path, x: torch.Tensor, mod: torch.Tensor, n: int = 8, n_win: int = 4,
                         mix: str = "v4") -> None:
    """Grid of the same native slices under independent random windows, for visual sign-off."""
    from torchvision.utils import save_image

    from lib.windowing import asinh_window, sample_windows

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
        x, mod,
        area=crop_kw["global_area"],
        max_rot=crop_kw["max_rot"],
        window_mix=crop_kw["window_mix"],
    )
    proj = net(pair, None)
    return prediction_loss(proj), pairwise_cosine(proj.detach())


def seq_pair_loss(net, x, sib, mod, ok, crop_kw):
    """Same-crop BraTS sequence pair. Loss/cosine are masked to rows with a sibling."""
    pair = same_crop_seq_pair(
        x, sib, mod,
        area=crop_kw["global_area"],
        max_rot=crop_kw["max_rot"],
        window_mix=crop_kw["window_mix"],
    )
    proj = net(pair, None)
    per = per_sample_invariance(proj)
    inv = (per * ok).sum() / ok.sum().clamp_min(1.0)
    mask = ok > 0.5
    if mask.any():
        cos = pairwise_cosine(proj[:, mask].detach())
    else:
        cos = proj.new_zeros(())
    return inv, cos, ok.mean()


@torch.no_grad()
def evaluate(raw, sigreg, loader, crop_kw, lamb, n_global, device, n_batches: int,
             win_inv_weight: float = 0.0, seq_inv_weight: float = 0.0) -> dict:
    """LeJEPA loss on fixed validation slabs plus effective rank of the class token."""
    raw.eval()
    n_met = 3 + (2 if win_inv_weight else 0) + (3 if seq_inv_weight else 0)
    tot, feats = torch.zeros(n_met, device=device), []
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(0)
        for batch in take(loader, n_batches):
            x, _, mod, sib, ok = unpack_batch(batch, device)
            gv, lv = multi_crop(x, mod, **crop_kw)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, inv, reg = lejepa_loss(raw(gv, lv), sigreg, lamb, n_global)
                feats.append(raw.encode(x).float())
                parts = [loss, inv, reg]
                if win_inv_weight:
                    win_inv, win_cos = window_pair_loss(raw, x, mod, crop_kw)
                    loss = loss + win_inv_weight * win_inv
                    parts[0] = loss
                    parts += [win_inv, win_cos]
                if seq_inv_weight:
                    seq_inv, seq_cos, seq_frac = seq_pair_loss(raw, x, sib, mod, ok, crop_kw)
                    loss = parts[0] + seq_inv_weight * seq_inv
                    parts[0] = loss
                    parts += [seq_inv, seq_cos, seq_frac]
                tot += torch.stack(parts)
    raw.train()
    n = max(n_batches, 1)
    vals = (tot / n).tolist()
    row = {"val_loss": vals[0], "val_inv": vals[1], "val_sigreg": vals[2],
           "erank": effective_rank(torch.cat(feats))}
    i = 3
    if win_inv_weight:
        row["val_win_inv"], row["val_win_cos"] = vals[i], vals[i + 1]
        i += 2
    if seq_inv_weight:
        row["val_seq_inv"], row["val_seq_cos"], row["val_seq_frac"] = vals[i], vals[i + 1], vals[i + 2]
    return row


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
    max_steps = int(args.max_steps or t["max_steps"])
    batch, lamb, n_global = int(o["batch_size"]), float(o["lamb"]), int(o["n_global"])
    window_cfg = (cfg.get("views") or {}).get("window") or {}
    window_mode = str(window_cfg.get("mode", "random"))
    window_mix = str(window_cfg.get("mix", "v4"))
    win_inv_weight = float(window_cfg.get("inv_weight", 0.0))
    seq_inv_weight = float(window_cfg.get("seq_weight", 0.0))
    crop_kw = dict(
        n_global=n_global, n_local=int(o["n_local"]),
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
    step = 0
    history: list[dict] = []
    ckpt = None
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
    loader_kw = dict(
        batch_slices=batch, k=int(d["slices_per_read"]), alpha=float(d["balance_alpha"]),
        load_siblings=seq_inv_weight > 0,
    )
    loader = make_slab_loader(
        slices_root, "train", workers=workers, seed=seed * 100_003 + step,
        priority_beta=float(d.get("priority_beta", 0.0)), priority_clip=float(d.get("priority_clip", 3.0)),
        **loader_kw,
    )
    val_loader = make_slab_loader(
        slices_root, "val", workers=min(2, workers), seed=0, persistent=False, **loader_kw
    ) if rank == 0 else None

    raw = VisionTransformer(
        size=m["size"], img_size=int(m["img_size"]), patch_size=int(m["patch_size"]),
        proj_dim=int(m["proj_dim"]), proj_hidden=m.get("proj_hidden"),
        registers=int(m.get("registers", 0)), qk_norm=bool(m.get("qk_norm", False)),
    ).to(device)
    sigreg = SIGReg().to(device)
    opt = torch.optim.AdamW(
        param_groups(raw, float(o["weight_decay"])), lr=float(o["lr"]), betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    if ckpt is not None:
        raw.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
    net = raw
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(raw, device_ids=[local], broadcast_buffers=False)
    if o.get("compile", True) and device.type == "cuda":
        net = torch.compile(net)

    log(f"[model] vit-{raw.size} params={raw.count_parameters():,} registers={raw.n_registers} "
        f"batch={batch}x{world} k={loader_kw['k']} workers={workers} window={window_mode} mix={window_mix} "
        f"win_inv={win_inv_weight:g} seq_inv={seq_inv_weight:g} steps={step}/{max_steps}"
        + (f" resume={resume_src}" if resume_src is not None else ""))
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")
        if step == 0:
            try:
                preview, _, preview_mod, _, _ = unpack_batch(next(iter(val_loader)), device)
                write_window_montage(out_dir / "window_montage.png", preview, preview_mod, mix=window_mix)
                log(f"[qc] wrote {out_dir / 'window_montage.png'}")
            except Exception as e:
                log(f"[qc] window montage skipped: {type(e).__name__}: {e}")

    def save(path: Path, model_only: bool = False) -> None:
        state = {"model": raw.state_dict(), "step": step, "config": cfg, "history": history}
        if not model_only:
            state["optimizer"] = opt.state_dict()
        atomic_torch_save(state, path)

    stop = StopFlag()
    n_acc = 3 + (2 if win_inv_weight else 0) + (3 if seq_inv_weight else 0)
    it, acc, seen, t0 = iter(loader), torch.zeros(n_acc, device=device), 0, time.time()
    raw.train()
    while step < max_steps:
        lr = lr_at(step, max_steps, float(o["lr"]), int(o["warmup_steps"]))
        for g in opt.param_groups:
            g["lr"] = lr
        x, chunk_ids, mod, sib, ok = unpack_batch(next(it), device)
        gv, lv = multi_crop(x, mod, **crop_kw)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            proj = net(gv, lv)
            loss, inv, reg = lejepa_loss(proj, sigreg, lamb, n_global)
        if loader.sampler.beta > 0:  # hardness of each chunk relative to this batch
            hard = per_sample_invariance(proj.detach().float(), n_global).view(len(chunk_ids), -1).mean(1)
            loader.sampler.update(chunk_ids, hard / hard.mean())
        loss.backward()
        win_inv = win_cos = seq_inv = seq_cos = seq_frac = None
        if win_inv_weight:
            # Separate backward so the pair does not sit on top of the LeJEPA graph (OOM).
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                win_inv, win_cos = window_pair_loss(net, x, mod, crop_kw)
                win_term = win_inv_weight * win_inv
            win_term.backward()
            loss = loss.detach() + win_term.detach()
        if seq_inv_weight:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                seq_inv, seq_cos, seq_frac = seq_pair_loss(net, x, sib, mod, ok, crop_kw)
                seq_term = seq_inv_weight * seq_inv
            seq_term.backward()
            loss = loss.detach() + seq_term.detach()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), float(o.get("clip_grad", 1.0)))
        opt.step()
        parts = [loss.detach(), inv.detach(), reg.detach()]
        if win_inv_weight:
            parts += [win_inv.detach(), win_cos.detach()]
        if seq_inv_weight:
            parts += [seq_inv.detach(), seq_cos.detach(), seq_frac.detach()]
        acc += torch.stack(parts)
        step, seen = step + 1, seen + 1

        if step % int(t["log_every"]) == 0:
            a = (acc / seen).tolist()
            extra = ""
            i = 3
            if win_inv_weight:
                extra += f" win_inv={a[i]:.4f} win_cos={a[i + 1]:.3f}"
                i += 2
            if seq_inv_weight:
                extra += f" seq_inv={a[i]:.4f} seq_cos={a[i + 1]:.3f} seq_frac={a[i + 2]:.2f}"
            log(f"[step {step}/{max_steps}] loss={a[0]:.4f} inv={a[1]:.4f} sigreg={a[2]:.3f}{extra} "
                f"lr={lr:.2e} {seen * batch * world / (time.time() - t0):.0f} slices/s"
                + (f" seen={loader.sampler.visited():.0%}" if loader.sampler.beta > 0 else ""))
            acc.zero_()
            seen, t0 = 0, time.time()
        if rank == 0 and step % int(t["eval_every"]) == 0:
            row = {"step": step, **evaluate(
                raw, sigreg, val_loader, crop_kw, lamb, n_global, device,
                int(t["eval_batches"]), win_inv_weight=win_inv_weight, seq_inv_weight=seq_inv_weight,
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
