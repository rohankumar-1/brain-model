#!/usr/bin/env python3
"""Rewrite split labels on the index tables — never copies pixels.

Reassigns train/val/test on ``subjects.csv`` / ``volumes.csv`` via
``dedup_key`` and remaps ``slices.csv`` / ``manifest.csv`` / parquet.
Does not rewrite the per-volume zip stores.

Example:
  python data/resplit_manifest.py \\
    --slices-root data/processed/slices_zarr \\
    --drop-dataset msd_brats
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.preprocess import (  # noqa: E402
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
    dataset_family,
    stratified_volume_split,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--slices-root",
        type=Path,
        default=ROOT / "data" / "processed" / "slices_zarr",
    )
    p.add_argument("--drop-dataset", action="append", default=None)
    p.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)
    p.add_argument("--test-frac", type=float, default=TEST_FRAC)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _load_slices(root: Path) -> tuple[pd.DataFrame, Path]:
    for name in ("slices.parquet", "slices.csv", "manifest.csv"):
        p = root / name
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p), p
            return pd.read_csv(p), p
    raise SystemExit(f"Missing slices table under {root}")


def main() -> None:
    args = parse_args()
    drop = args.drop_dataset or []
    root = args.slices_root
    vol_path = root / "volumes.csv"
    subj_path = root / "subjects.csv"
    if not vol_path.exists():
        raise SystemExit(f"Missing {vol_path} (master volume table)")

    vols = pd.read_csv(vol_path)
    slices, slice_src = _load_slices(root)
    print(f"[in] volumes={len(vols)} slices={len(slices)} from {slice_src.name}")

    if drop:
        vols = vols[~vols["dataset"].isin(drop)].copy()
        if "dataset" in slices.columns:
            slices = slices[~slices["dataset"].isin(drop)].copy()
        print(f"[drop] datasets={drop} → volumes={len(vols)} slices={len(slices)}")

    if "dedup_key" not in vols.columns:
        vols["dedup_key"] = vols["volume_id"]
    if "dataset_family" not in vols.columns:
        vols["dataset_family"] = vols["dataset"].map(dataset_family)

    lookup = stratified_volume_split(
        vols.to_dict("records"),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    vols["split"] = vols["volume_id"].map(lookup)
    if "volume_id" in slices.columns:
        slices["split"] = slices["volume_id"].map(lookup)
    elif "vol_idx" in slices.columns:
        idx_map = dict(zip(vols["vol_idx"], vols["split"]))
        slices["split"] = slices["vol_idx"].map(idx_map)

    print(vols.groupby("split").size().to_string())
    if args.dry_run:
        print("[dry-run] not writing")
        return

    shutil.copy2(vol_path, root / "volumes_before_resplit.csv.bak")
    vols.to_csv(vol_path, index=False)

    subj = (
        vols.groupby("dedup_key", as_index=False)
        .agg(
            split=("split", "first"),
            modality=("modality", "first"),
            dataset_family=("dataset_family", "first"),
            n_volumes=("volume_id", "nunique"),
        )
    )
    if subj_path.exists():
        shutil.copy2(subj_path, root / "subjects_before_resplit.csv.bak")
    subj.to_csv(subj_path, index=False)

    for name in ("slices.csv", "manifest.csv"):
        dest = root / name
        if dest.exists():
            shutil.copy2(dest, root / f"{dest.stem}_before_resplit.csv.bak")
        slices.to_csv(dest, index=False)
    pq = root / "slices.parquet"
    try:
        slices.to_parquet(pq, index=False)
    except Exception as e:
        print(f"[warn] parquet skipped: {e}")

    print(f"[ok] wrote split labels on tables under {root}")
    print("[note] zip stores unchanged (index-only resplit)")


if __name__ == "__main__":
    main()
