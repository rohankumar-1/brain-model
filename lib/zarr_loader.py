#!/usr/bin/env python3
"""Map-style loader: random (image zip, z) reads for heterogeneous batches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader, Dataset, DistributedSampler, get_worker_info

from lib.data import SliceAugment


def read_volume_slice(path: Path, z: int, cache: dict | None = None) -> np.ndarray:
    """Read one axial slice from a per-volume ZipStore. ``cache`` keeps a few stores open."""
    from zarr.storage import ZipStore

    path = Path(path)
    key = str(path)
    if cache is not None and key in cache:
        arr = cache[key][1]
    else:
        store = ZipStore(path, mode="r")
        arr = zarr.open_group(store, mode="r")["image"]
        if cache is None:
            try:
                return np.asarray(arr[int(z)], dtype=np.float32)
            finally:
                store.close()
        if len(cache) >= 4:
            _, (old, _) = cache.popitem()
            try:
                old.close()
            except Exception:
                pass
        cache[key] = (store, arr)
    return np.asarray(arr[int(z)], dtype=np.float32)


def _load_slice_table(root: Path) -> pd.DataFrame:
    for name in ("slices.parquet", "slices.csv", "manifest.csv"):
        p = root / name
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p)
    raise FileNotFoundError(f"No slices.parquet / slices.csv / manifest.csv under {root}")


class ZarrSliceIndexDataset(Dataset):
    """One row of the slice index = one 2D example from ``<dataset>/<case>.zip``."""

    def __init__(
        self,
        store_root: str | Path,
        split: str,
        augment: SliceAugment | None = None,
    ):
        self.store_root = Path(store_root)
        self.split = str(split)
        self.augment = augment
        self._cache: dict = {}
        df = _load_slice_table(self.store_root)
        if "image_path" not in df.columns:
            raise ValueError(f"{self.store_root} slice table needs image_path")
        df = df.loc[df["image_path"].astype(str).str.len().gt(0)].copy()
        if "pretrain" in df.columns:
            df = df.loc[df["pretrain"].astype(bool)].copy()
        if "split" not in df.columns:
            raise ValueError(f"{self.store_root} slice table needs split")
        if "z" not in df.columns and "slice_idx" in df.columns:
            df = df.rename(columns={"slice_idx": "z"})
        if "z" not in df.columns:
            raise ValueError(f"{self.store_root} slice table needs z")
        self.df = df.loc[df["split"].astype(str) == self.split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No slices for split={self.split} in {self.store_root}")
        print(f"[dataset] {self.store_root} split={self.split} n={len(self.df)}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        rel = Path(str(row["image_path"]))
        path = rel if rel.is_absolute() else self.store_root / rel
        slc = read_volume_slice(path, int(row["z"]), cache=self._cache)
        if slc.ndim == 2:
            slc = slc[None, ...]
        tensor = torch.from_numpy(np.ascontiguousarray(slc))
        if self.augment is not None:
            tensor = self.augment(tensor)
        modality = str(row["modality"]) if "modality" in self.df.columns else ""
        mid = {"ct": 0, "mri": 1}.get(modality, -1)
        return tensor, mid


def make_split_loader(
    store_root: str | Path,
    split: str,
    *,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = True,
    augment: bool | dict | None = False,
    seed: int = 42,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    distributed: bool = False,
) -> DataLoader:
    """Heterogeneous random-slice loader. ``distributed=True`` uses DistributedSampler."""
    if isinstance(augment, dict):
        aug = SliceAugment.from_config(augment)
    elif augment:
        aug = SliceAugment()
    else:
        aug = None
    dataset = ZarrSliceIndexDataset(store_root, split=split, augment=aug)
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed)
        shuffle = False
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    g = torch.Generator()
    g.manual_seed(int(seed))

    def _worker_init(worker_id: int) -> None:
        wseed = int(seed) + int(worker_id)
        np.random.seed(wseed)
        torch.manual_seed(wseed)

    kwargs: dict = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=drop_last,
        generator=g,
        worker_init_fn=_worker_init if num_workers > 0 else None,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)
