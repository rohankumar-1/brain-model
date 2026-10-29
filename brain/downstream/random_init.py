"""Write an untrained 3D encoder checkpoint: the floor every pretrained model should beat on the downstream suite.

    python -m brain.downstream.random_init --like runs/v3/cls_s42/best.pt --out runs/baselines/random_v3/last.pt
    python -m brain.bench run --model BQ-random --submit

The checkpoint has the same layout as a training checkpoint (``model``, ``config``, ``step`` = 0, ``provenance``), so
every evaluator loads it unchanged and its experiment records sit next to trained models in the database.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from brain.common.config import load_config
from brain.common.paths import resolve
from brain.downstream.records import provenance
from brain.train.defaults import default_config
from brain.train.lejepa3d import build_encoder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="the architecture to instantiate (a lejepa3d config)")
    src.add_argument("--like", type=Path, help="a checkpoint whose architecture (its saved config) to build")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = torch.load(resolve(a.like), map_location="cpu", weights_only=False, mmap=True)["config"] if a.like \
        else load_config(a.config, defaults=default_config("lejepa3d"))
    torch.manual_seed(a.seed)
    enc = build_encoder(cfg, torch.device("cpu"))
    root = resolve(cfg["data"]["store_root"]) if cfg.get("data", {}).get("store_root") else None
    prov = provenance(root) | {"method": "random_init", "seed": a.seed, "architecture_config": str(a.like or a.config)}
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": enc.state_dict(), "config": cfg, "step": 0, "provenance": prov}, out)
    print(f"[random-init] {sum(p.numel() for p in enc.parameters()) / 1e6:.1f}M params -> {out}")


if __name__ == "__main__":
    main()
