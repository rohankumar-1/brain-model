"""3D LeJEPA pretraining.

  python train_brainmodel.py --config configs/brainmodel_tiny_lejepa.yaml
  python train_brainmodel.py --synthetic --max-steps 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainmodel.config import load_config, save_config, smoke_config
from brainmodel.data.crops import raw_views
from brainmodel.data.dataset import collate_volumes, crop_microbatch, make_loader, make_synthetic_volume
from brainmodel.data.lejepa import prepare
from brainmodel.eval.health import embedding_health
from brainmodel.losses.lejepa import lejepa_loss
from brainmodel.losses.sigreg import SIGReg
from brainmodel.models.viewnet import ViewNet, batch_meta
from brainmodel.models.vit3d import BrainEncoder
from lib.checkpoint import atomic_torch_save, load_ckpt
from lib.train_utils import StopFlag, dist_info, lr_at


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "brainmodel_tiny_lejepa.yaml")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="tiny ellipsoid volumes; no store required")
    p.add_argument("--log-every", type=int, default=None)
    return p.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def param_groups(model: torch.nn.Module, wd: float) -> list[dict]:
    decay = [p for p in model.parameters() if p.ndim >= 2]
    other = [p for p in model.parameters() if p.ndim < 2]
    return [{"params": decay, "weight_decay": wd}, {"params": other, "weight_decay": 0.0}]


def _finite_params(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(p).all() for p in model.parameters() if p.is_floating_point())


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
        checkpoint_blocks=bool(m.get("checkpoint", True)),
        proj_norm=str(m.get("proj_norm", "batch")),
        dense_dim=int(m.get("dense_dim", 0)) if dense else 0,
        dense_hidden=int(m.get("dense_hidden", 512)),
        predict_depth=int(pr.get("depth", 4)) if predict else 0,
        predict_pixel_dim=int(cells[0]) * int(cells[1]) * int(cells[2]) if predict and float(pr.get("pixel_weight", 0.0)) > 0 else 0,
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
    )


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
    loader = make_loader(
        store, split, cfg, batch_size=batch_size, shuffle=shuffle, seed=seed,
        distributed=distributed, n_volumes=n_volumes,
    )
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


@torch.no_grad()
def evaluate(core: ViewNet, val_iter, cfg: dict, device: torch.device, sigreg: SIGReg, buckets: int, amp: dict) -> dict:
    """One fixed validation batch with fixed windows. The BatchNorm layers use their running statistics."""
    core.eval()
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(0)
        batch = prepare(next(val_iter), cfg, device, buckets)
        with torch.autocast(**amp):
            out = core(batch)
        parts = lejepa_loss(out, batch_meta(batch), sigreg, gather_ddp=False, **loss_kwargs(cfg))
    core.train()
    # loss first, matching the train log; remaining keys keep lejepa_loss order
    return {"loss": float(parts["loss"]), **{k: float(v) for k, v in parts.items() if k != "loss"}}


def is_new_best(vals: dict, metric: str, best_val: float) -> float | None:
    """Lower ``vals[metric]`` (fallback ``loss``) beats ``best_val``. NaN/Inf never win."""
    v = vals.get(metric, vals.get("loss"))
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) and v < best_val else None


def _metric_table(keys: list[str], extras: list[str], prefix_w: int, log, every: int = 10):
    """Prints a label row every ``every`` lines; values sit in fixed columns under those labels."""
    extra_min = {"erank": 9, "tok_erank": 9, "lr": 8, "vol/s": 6, "peak": 7}
    labels = keys + extras
    widths = [max(len(k), 8 if k in keys else extra_min.get(k, 8)) for k in labels]
    n = 0

    def header() -> None:
        log(" " * prefix_w + "   " + "   ".join(f"{k:>{w}}" for k, w in zip(labels, widths)))

    def row(prefix: str, values: dict[str, str]) -> None:
        nonlocal n
        if n % every == 0:
            header()
        n += 1
        log(f"{prefix:<{prefix_w}}   " + "   ".join(f"{values.get(k, ''):>{w}}" for k, w in zip(labels, widths)))

    return row


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides={"output": {"out_dir": str(args.out_dir)}} if args.out_dir else None)
    if args.synthetic:
        cfg = smoke_config(cfg)
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
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False

    m, o, t = cfg["model"], cfg["optim"], cfg["train"]
    max_steps = int(args.max_steps or t["max_steps"])
    log_every = int(args.log_every or t["log_every"])
    batch_size = int(o["batch_size"])
    out_dir = resolve(cfg["output"]["out_dir"])
    last_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"
    best_metric = str(t.get("best_metric", "loss"))

    step, history, ckpt, resume_src = 0, [], None, None
    best_val, best_step = float("inf"), 0
    if not args.no_resume and not args.synthetic:
        if last_path.exists():
            resume_src = last_path
        elif cfg.get("output", {}).get("resume_from"):
            resume_src = resolve(cfg["output"]["resume_from"])
    if resume_src is not None:
        ckpt = load_ckpt(resume_src, device)
        if any(torch.is_floating_point(v) and not torch.isfinite(v).all() for v in ckpt["model"].values()):
            log(f"[resume] ignored {resume_src}: checkpoint contains NaN/Inf weights")
            ckpt, resume_src = None, None
        else:
            step, history = int(ckpt["step"]), list(ckpt.get("history") or [])
            best_val = float(ckpt.get("best_val", best_val))
            best_step = int(ckpt.get("best_step", best_step))

    val_batch = min(batch_size, 96)
    if args.synthetic:
        loader = synthetic_loader(cfg, batch_size, seed)
        val_iter = synthetic_loader(cfg, val_batch, seed + 999)
    else:
        store = resolve(cfg["data"]["store_root"])
        loader = forever_loader(store, "train", cfg, batch_size=batch_size, shuffle=True, seed=seed, distributed=world > 1)
        val_iter = forever_loader(
            store, "val", cfg, batch_size=val_batch, shuffle=False, seed=0, distributed=False,
        ) if rank == 0 else None

    raw = build_encoder(cfg, device)
    if world > 1:
        raw = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw)
    dense_on = float(loss_kwargs(cfg)["dense_weight"]) > 0
    pcfg = cfg["loss"].get("predict") or {}
    predict_on = dense_on and float(pcfg.get("weight", 0.0)) > 0
    if float(pcfg.get("weight", 0.0)) > 0 and not dense_on:
        raise SystemExit("loss.predict needs loss.dense.weight > 0: it predicts the token head's embeddings")
    fine_pred_on = predict_on and float(pcfg.get("fine_weight", 0.0)) > 0
    core = ViewNet(raw, local_pooled=bool(cfg["loss"].get("local_pooled", False)), dense=dense_on, predict=predict_on,
                   predict_fine=fine_pred_on, cells_xyz=tuple(pcfg.get("pixel_cells", (2, 2, 2))))
    sigreg = SIGReg(n_proj=int(cfg["loss"].get("n_proj", 256))).to(device)
    opt = torch.optim.AdamW(
        param_groups(raw, float(o["weight_decay"])), lr=float(o["lr"]), betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    if ckpt is not None:
        raw.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
    elif cfg.get("output", {}).get("init_from") and not args.synthetic:
        # Weights only: a fresh optimizer and schedule. Heads the source never had (the predictor) start random.
        src = load_ckpt(resolve(cfg["output"]["init_from"]), device)
        state = {k.removeprefix("module."): v for k, v in src["model"].items()}
        res = raw.load_state_dict(state, strict=False)
        if res.unexpected_keys or any(not k.startswith("predictor.") for k in res.missing_keys):
            raise SystemExit(f"[init_from] mismatch: missing={res.missing_keys[:4]} unexpected={res.unexpected_keys[:4]}")
        log(f"[init] weights from {cfg['output']['init_from']} (step {src.get('step')}); new: {len(res.missing_keys)} tensors")
    net = torch.nn.parallel.DistributedDataParallel(core, device_ids=[local]) if world > 1 else core
    buckets = int(m.get("length_buckets", 4))
    lk, gather = loss_kwargs(cfg), bool(cfg["loss"].get("gather_ddp", True))

    log(f"[model] brain-{raw.size} params={raw.count_parameters():,} batch={batch_size}x{world} "
        f"steps={step}/{max_steps} synthetic={args.synthetic} dense={dense_on} predict={predict_on} "
        f"fine={fine_pred_on} proj_norm={raw.projector.norm}"
        + (f" resume={resume_src}" if resume_src is not None else "")
        + (f" best={best_val:.4f} @ {best_step}" if math.isfinite(best_val) else ""))
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")

    def save(path: Path, model_only: bool = False) -> None:
        if not _finite_params(raw):
            log("[nan] weights are non-finite; not writing a checkpoint")
            raise SystemExit(4)
        state = {"model": raw.state_dict(), "step": step, "config": cfg, "history": history,
                 "best_val": best_val, "best_step": best_step, "best_metric": best_metric}
        if not model_only:
            state["optimizer"] = opt.state_dict()
        atomic_torch_save(state, path)

    stop = StopFlag()
    fine_on = dense_on and float((cfg["loss"].get("dense") or {}).get("fine_weight", 0.0)) > 0
    keys = ["loss", "inv", "win", "sigreg"] + (["dense_inv", "dense_sig", "overlap"] if dense_on else [])
    keys += ["fine_inv", "fine_overlap"] if fine_on else []
    keys += (["pred"] + (["pix"] if float(pcfg.get("pixel_weight", 0.0)) > 0 else [])) if predict_on else []
    keys += (["fine_pred"] + (["fine_pix"] if float(pcfg.get("pixel_weight", 0.0)) > 0 else [])) if fine_pred_on else []
    extras = ["erank"] + (["tok_erank"] if dense_on else []) + ["lr", "vol/s"]
    if device.type == "cuda":
        extras.append("peak")
    step_w = len(str(max_steps))
    prefix_w = len(f"[step {max_steps}/{max_steps}]")
    put = _metric_table(keys, extras, prefix_w, log)
    ramp = int(pcfg.get("ramp_steps", 0))
    acc, seen, t0 = torch.zeros(len(keys), device=device), 0, time.time()
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    clip = float(o.get("clip_grad", 1.0))
    core.train()
    # Nothing in the step reads a GPU value on the CPU, so the CPU queues the next batch while the GPU
    # runs this one. Finiteness is checked when logging and before a checkpoint is written.
    side = torch.cuda.Stream() if device.type == "cuda" else None

    def fetch() -> dict:
        """Builds the next batch on a side stream so the copy and windowing overlap the running step."""
        chunks = next(loader)
        if side is None:
            return prepare(chunks, cfg, device, buckets)
        with torch.cuda.stream(side):
            return prepare(chunks, cfg, device, buckets)

    def adopt(batch: dict) -> dict:
        if side is not None:
            torch.cuda.current_stream().wait_stream(side)
            _record(batch, torch.cuda.current_stream())
        return batch

    nxt = fetch() if step < max_steps else None
    while step < max_steps:
        lr = lr_at(step, max_steps, float(o["lr"]), int(o["warmup_steps"]))
        for g in opt.param_groups:
            g["lr"] = lr
        lbatch, nxt = adopt(nxt), None
        opt.zero_grad(set_to_none=True)
        with torch.autocast(**amp):
            out = net(lbatch)
        scale = min(1.0, (step + 1) / ramp) if ramp else 1.0
        parts = lejepa_loss(out, batch_meta(lbatch), sigreg, gather_ddp=gather,
                            **{**lk, "predict_weight": lk["predict_weight"] * scale})
        parts["loss"].backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), clip)
        opt.step()
        acc += torch.stack([parts[k].detach() for k in keys])
        step, seen = step + 1, seen + 1
        if step < max_steps:
            nxt = fetch()
        if step % log_every == 0 or step <= 3:
            a = (acc / seen).tolist()
            if not all(math.isfinite(v) for v in a):
                log(f"[nan] non-finite loss at step {step}: {dict(zip(keys, a))}; not writing a checkpoint")
                raise SystemExit(2)
            z = out["proj"].detach().float().reshape(-1, out["proj"].size(-1))
            cells = {k: f"{v:.4f}" for k, v in zip(keys, a)}
            cells["erank"] = f"{embedding_health(z)['erank']:.1f}/{z.size(0)}"
            if dense_on:
                tok = out["dense"]["emb"][0].detach().float()[out["dense"]["valid"][0]]
                cells["tok_erank"] = f"{embedding_health(tok[:2048])['erank']:.1f}"
            cells["lr"] = f"{lr:.2e}"
            cells["vol/s"] = f"{seen * batch_size * world / max(time.time() - t0, 1e-6):.1f}"
            if device.type == "cuda":
                cells["peak"] = f"{torch.cuda.max_memory_reserved() / 2**30:.1f}GiB"
            put(f"[step {step:>{step_w}}/{max_steps}]", cells)
            acc.zero_()
            seen, t0 = 0, time.time()
        if rank == 0 and step % int(t["eval_every"]) == 0 and val_iter is not None:
            vals = evaluate(core, val_iter, cfg, device, sigreg, buckets, amp)
            row = {"step": step, **{f"val_{k}": vals[k] for k in keys if k in vals}}
            history.append(row)
            put(f"[eval {step:>{step_w}}/{max_steps}]", {k: f"{vals[k]:.4f}" for k in keys if k in vals})
            new_best = is_new_best(vals, best_metric, best_val)
            if new_best is not None:
                best_val, best_step = new_best, step
                if not args.synthetic:
                    save(best_path, model_only=True)
                log(f"[best] step {step} val_{best_metric}={best_val:.4f} -> {best_path.name}")
            (out_dir / "metrics.json").write_text(
                json.dumps({"history": history, "best_val": best_val, "best_step": best_step,
                            "best_metric": best_metric}, indent=2) + "\n")
        if rank == 0 and not args.synthetic and step % int(t["ckpt_every"]) == 0:
            save(last_path)
            if step % int(t["keep_every"]) == 0:
                save(out_dir / f"step_{step:07d}.pt", model_only=True)
        if step % int(t.get("stop_poll", 20)) == 0 and stop.poll(device):
            break

    if rank == 0:
        if not args.synthetic:
            save(last_path)
        preempted = step < max_steps
        log(f"[done] step {step}/{max_steps}" + (" (preempted)" if preempted else ""))
        if preempted and os.environ.get("REQUEUE_FLAG"):
            Path(os.environ["REQUEUE_FLAG"]).touch()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
