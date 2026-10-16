#!/usr/bin/env python3
"""Train ConvVAE with native train/val splits and interruption-safe checkpoints.

Resume: if ``last.pt`` exists in the run dir, training continues automatically.
  python train_vae.py --config configs/vae_256.yaml
  python train_vae.py --config configs/vae_256.yaml --no-resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.checkpoint import (
    InterruptSaver,
    atomic_torch_save,
    build_ckpt,
    load_ckpt,
    restore_rng_state,
)
from lib.config import load_config, save_config
from lib.recon import evaluate_recon_mse, format_recon_report, save_recon_grid
from lib.progress import EpochBar
from lib.zarr_loader import make_split_loader
from models.vae import ConvVAE, vae_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "vae_256.yaml")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume (default: <out>/last.pt)")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.out_dir is not None:
        overrides["output"] = {"out_dir": str(args.out_dir)}
    cfg = load_config(args.config, overrides=overrides or None)

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[config] {args.config}")

    slices_root = resolve_path(cfg["data"]["slices_root"])
    has_store = (slices_root / "slices.parquet").exists() or (slices_root / "slices.csv").exists()
    if not has_store:
        raise SystemExit(f"Missing processed volumes under {slices_root}; run data/preprocess.py")

    batch_size = int(cfg["optim"]["batch_size"])
    num_workers = int(cfg["optim"].get("num_workers", 8))
    prefetch = int(cfg["optim"].get("prefetch_factor", 2))
    aug_cfg = cfg["data"].get("augment") or {"enabled": True}

    print(f"[data] map-style random slices @ {slices_root} workers={num_workers}")
    train_loader = make_split_loader(
        slices_root,
        "train",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        augment=aug_cfg,
        seed=seed,
        drop_last=True,
        prefetch_factor=prefetch,
    )
    val_loader = make_split_loader(
        slices_root,
        "val",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        augment=False,
        seed=seed,
        drop_last=False,
        prefetch_factor=prefetch,
    )

    latent_dim = int(cfg["model"]["latent_dim"])
    beta = float(cfg["optim"]["beta"])
    lr = float(cfg["optim"]["lr"])
    weight_decay = float(cfg["optim"].get("weight_decay", 0.0))
    max_epochs = int(cfg["train"]["max_epochs"])
    patience = int(cfg["train"]["patience"])
    min_delta = float(cfg["train"]["min_delta"])
    sched_cfg = cfg["optim"].get("scheduler") or {}
    use_scheduler = bool(sched_cfg.get("enabled", False))

    model = ConvVAE(latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 3)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )

    out_dir = resolve_path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    alias_path = out_dir / "vae.pt"
    metrics_path = out_dir / "metrics.json"

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    start_epoch = 1
    global_step = 0
    stopped_early = False

    resume_path = None if args.no_resume else (args.resume or last_path)
    if resume_path is not None and Path(resume_path).exists():
        ckpt = load_ckpt(Path(resume_path), device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            opt.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val = float(ckpt.get("best_val", ckpt.get("val_recon_mse", float("inf"))))
        best_epoch = int(ckpt.get("best_epoch", 0))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        history = list(ckpt.get("history") or [])
        restore_rng_state(ckpt.get("rng"))
        print(
            f"[resume] {resume_path} next_epoch={start_epoch} "
            f"best_val={best_val:.6f} @ {best_epoch}"
        )
    else:
        print("[resume] fresh run")

    def snapshot(epoch: int) -> dict:
        return build_ckpt(
            model=model,
            optimizer=opt,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_val=best_val,
            best_epoch=best_epoch,
            epochs_no_improve=epochs_no_improve,
            history=history,
            latent_dim=latent_dim,
            beta=beta,
            config=cfg,
            extra={"val_recon_mse": best_val},
        )

    last_completed = start_epoch - 1
    saver = InterruptSaver(
        lambda: atomic_torch_save(snapshot(max(last_completed, 0)), last_path)
    )

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            model.train()
            running_loss = 0.0
            running_recon = 0.0
            n_batches = 0
            bar = EpochBar(train_loader, desc=f"epoch {epoch} train", total=len(train_loader))
            for x, _ in bar:
                x = x.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                recon, mu, logvar = model(x)
                loss, recon_mse, _kl = vae_loss(recon, x, mu, logvar, beta=beta)
                loss.backward()
                opt.step()
                running_loss += float(loss.item())
                running_recon += float(recon_mse.item())
                n_batches += 1
                global_step += 1
                bar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

            val_metrics = evaluate_recon_mse(
                model, val_loader, device, progress=f"epoch {epoch} val"
            )
            if scheduler is not None:
                scheduler.step(val_metrics["total"])
            current_lr = float(opt.param_groups[0]["lr"])
            row = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": running_loss / max(n_batches, 1),
                "train_recon_mse": running_recon / max(n_batches, 1),
                "val_recon_mse": val_metrics["total"],
                "val_recon_mse_mri": val_metrics["mri"],
                "val_recon_mse_ct": val_metrics["ct"],
            }
            history.append(row)
            print(
                f"epoch {epoch}: lr={current_lr:.2e} "
                f"train_loss={row['train_loss']:.6f} "
                f"train_recon={row['train_recon_mse']:.6f} "
                f"val_recon={row['val_recon_mse']:.6f} "
                f"(mri={row['val_recon_mse_mri']:.6f}, ct={row['val_recon_mse_ct']:.6f})"
            )

            improved = row["val_recon_mse"] < (best_val - min_delta)
            if improved:
                best_val = row["val_recon_mse"]
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            last_completed = epoch
            ckpt = snapshot(epoch)
            atomic_torch_save(ckpt, last_path)
            metrics_path.write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_val_recon_mse": best_val,
                        "history": history,
                    },
                    indent=2,
                )
                + "\n"
            )
            if improved:
                atomic_torch_save(ckpt, best_path)
                atomic_torch_save(ckpt, alias_path)
                print(f"  -> new best val_recon={best_val:.6f} (saved {best_path.name})")
            else:
                print(
                    f"  -> no improve ({epochs_no_improve}/{patience}), "
                    f"best={best_val:.6f} @ epoch {best_epoch}"
                )

            if epochs_no_improve >= patience:
                stopped_early = True
                print(f"[early-stop] patience={patience} reached at epoch {epoch}")
                break
    finally:
        saver.disarm()

    if best_path.exists():
        ckpt = load_ckpt(best_path, device)
        model.load_state_dict(ckpt["model"])
        print(f"[best] epoch={best_epoch} val_recon_mse={best_val:.6f}")

    val_best = evaluate_recon_mse(model, val_loader, device)
    print(format_recon_report("Val (best checkpoint)", val_best))

    if cfg["output"].get("save_recon_grid", True):
        try:
            save_recon_grid(
                model,
                val_loader,
                device,
                out_dir / "recon_grid_val.png",
                title="VAE val reconstructions (best ckpt)",
            )
            print(f"[ok] wrote {out_dir / 'recon_grid_val.png'}")
        except Exception as e:
            print(f"[warn] recon grid skipped: {e}")

    metrics = {
        "device": str(device),
        "config_path": str(args.config),
        "stopped_early": stopped_early,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_recon_mse": best_val,
        "val_recon_mse": {
            "total": val_best["total"],
            "mri": val_best["mri"],
            "ct": val_best["ct"],
        },
        "val_counts": {
            "total": int(val_best["total_n"]),
            "mri": int(val_best["mri_n"]),
            "ct": int(val_best["ct_n"]),
        },
        "history": history,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[ok] wrote {metrics_path}")
    print("[note] held-out test: python test_vae.py --run-dir", out_dir)


if __name__ == "__main__":
    main()
