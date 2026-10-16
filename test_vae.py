#!/usr/bin/env python3
"""Evaluate a trained ConvVAE on the held-out test fold.

Reports reconstruction MSE (total / MRI / CT) and optionally writes a recon grid.

Examples:
  python test_vae.py --run-dir runs/vae_256
  python test_vae.py --config configs/vae_256.yaml --ckpt runs/vae_256/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.checkpoint import load_ckpt
from lib.config import load_config
from lib.recon import evaluate_recon_mse, format_recon_report, save_recon_grid
from lib.zarr_loader import make_split_loader
from models.vae import ConvVAE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training run directory (reads config.yaml + best.pt by default)",
    )
    p.add_argument("--config", type=Path, default=None, help="Config used for data paths / folds")
    p.add_argument("--ckpt", type=Path, default=None, help="Checkpoint path (default: run-dir/best.pt)")
    p.add_argument(
        "--fold",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Which split to evaluate",
    )
    p.add_argument("--out-json", type=Path, default=None, help="Where to write metrics JSON")
    p.add_argument("--no-grid", action="store_true", help="Skip recon grid image")
    return p.parse_args()


def resolve_path(path_str: str | Path, base: Path = ROOT) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def main() -> None:
    args = parse_args()

    run_dir = resolve_path(args.run_dir) if args.run_dir else None
    if run_dir is not None:
        cfg_path = args.config or (run_dir / "config.yaml")
        ckpt_path = args.ckpt or (run_dir / "best.pt")
        if not ckpt_path.exists() and (run_dir / "vae.pt").exists():
            ckpt_path = run_dir / "vae.pt"
        if not ckpt_path.exists() and (run_dir / "last.pt").exists():
            ckpt_path = run_dir / "last.pt"
    else:
        if args.config is None or args.ckpt is None:
            raise SystemExit("Provide --run-dir, or both --config and --ckpt")
        cfg_path = args.config
        ckpt_path = args.ckpt

    cfg_path = resolve_path(cfg_path)
    ckpt_path = resolve_path(ckpt_path)
    cfg = load_config(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[config] {cfg_path}")
    print(f"[ckpt] {ckpt_path}")

    slices_root = resolve_path(cfg["data"]["slices_root"])
    num_workers = int(cfg["optim"].get("num_workers", 8))
    prefetch = int(cfg["optim"].get("prefetch_factor", 2))
    print(f"[data] split={args.fold} @ {slices_root} workers={num_workers}")

    ckpt = load_ckpt(ckpt_path, device)
    latent_dim = int(ckpt.get("latent_dim", cfg["model"]["latent_dim"]))
    model = ConvVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = make_split_loader(
        slices_root,
        args.fold,
        batch_size=int(cfg["optim"]["batch_size"]),
        num_workers=num_workers,
        shuffle=False,
        augment=False,
        seed=int(cfg["seed"]),
        drop_last=False,
        prefetch_factor=prefetch,
    )
    metrics = evaluate_recon_mse(model, loader, device)
    print(format_recon_report(f"{args.fold.capitalize()}", metrics))

    out = {
        "fold": args.fold,
        "device": str(device),
        "checkpoint": str(ckpt_path),
        "config_path": str(cfg_path),
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_val_recon_mse": ckpt.get("val_recon_mse", ckpt.get("best_val")),
        "recon_mse": {
            "total": metrics["total"],
            "mri": metrics["mri"],
            "ct": metrics["ct"],
        },
        "counts": {
            "total": int(metrics["total_n"]),
            "mri": int(metrics["mri_n"]),
            "ct": int(metrics["ct_n"]),
        },
    }

    out_json = args.out_json
    if out_json is None and run_dir is not None:
        out_json = run_dir / f"test_metrics_{args.fold}.json"
    if out_json is not None:
        out_json = resolve_path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"[ok] wrote {out_json}")

    if not args.no_grid:
        grid_dir = run_dir if run_dir is not None else out_json.parent if out_json else Path(".")
        grid_path = Path(grid_dir) / f"recon_grid_{args.fold}.png"
        try:
            save_recon_grid(
                model,
                loader,
                device,
                grid_path,
                title=f"VAE {args.fold} reconstructions",
            )
            print(f"[ok] wrote {grid_path}")
        except Exception as e:
            print(f"[warn] recon grid skipped: {e}")


if __name__ == "__main__":
    main()
