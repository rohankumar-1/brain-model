"""Map-style dataset: one volume, its spacing, and its ids."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from brain.data.index import enrich


ROW_KEYS = ("modality", "thick_mm", "thick_axis")  # per-volume (B,) vectors; view tensors are (V, B, ...)


class VolumeDataset(Dataset):
    def __init__(
        self,
        store_root: str | Path,
        split: str,
        *,
        index_name: str = "volumes.csv",
        default_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
        default_origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
        crop_cfg: dict | None = None,
        fixed_crops: bool = False,
    ):
        self.store_root = Path(store_root)
        self.split = str(split)
        self.crop_cfg = crop_cfg
        self.fixed_crops = bool(fixed_crops)
        self.default_spacing_mm = tuple(float(v) for v in default_spacing_mm)
        self.default_origin_mm = tuple(float(v) for v in default_origin_mm)
        parquet = self.store_root / "brainmodel_index.parquet"
        src = parquet if parquet.exists() else self.store_root / index_name
        df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
        df = enrich(df)
        if "split" not in df.columns:
            raise ValueError(f"{src} needs a split column")
        df = df.loc[df["split"].astype(str) == self.split].copy()
        if "pretrain" in df.columns:
            df = df.loc[df["pretrain"].astype(bool)].copy()
        if "ssl_keep" in df.columns:
            df = df.loc[df["ssl_keep"].astype(bool)].copy()
        self.df = df.reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No volumes for split={self.split} in {src}")

    def __len__(self) -> int:
        return len(self.df)

    def _spacing_origin(self, row) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-volume spacing and origin from optional columns; v4 volumes are 1 mm, origin 0."""
        sp, orig = list(self.default_spacing_mm), list(self.default_origin_mm)
        for i, axis in enumerate("xyz"):
            if f"spacing_{axis}" in self.df.columns and pd.notna(row[f"spacing_{axis}"]):
                sp[i] = float(row[f"spacing_{axis}"])
            if f"origin_{axis}" in self.df.columns and pd.notna(row[f"origin_{axis}"]):
                orig[i] = float(row[f"origin_{axis}"])
        return torch.tensor(sp, dtype=torch.float32), torch.tensor(orig, dtype=torch.float32)

    def _native(self, row) -> tuple[torch.Tensor, torch.Tensor]:
        """Acquisition thickness (mm) and its grid axis from prep; (1.0, -1) when the index has no record."""
        mm = float(row["thick_mm"]) if "thick_mm" in self.df.columns and pd.notna(row["thick_mm"]) else 1.0
        ax = int(row["thick_axis"]) if "thick_axis" in self.df.columns and pd.notna(row["thick_axis"]) else -1
        return torch.tensor([mm], dtype=torch.float32), torch.tensor([ax], dtype=torch.long)

    def _skull_dropout(self, volume: torch.Tensor, row, modality: int, gen) -> torch.Tensor:
        """Cut the volume to its brain mask with probability ``aug.skull.p``; a no-op when no mask is stored."""
        p = float(((self.crop_cfg or {}).get("aug") or {}).get("skull", {}).get("p", 0.0))
        rel = str(row["mask_brain"]) if "mask_brain" in self.df.columns and pd.notna(row["mask_brain"]) else ""
        if p <= 0 or not rel or rel == "nan":
            return volume
        if float(torch.rand(1, generator=gen)) >= p:  # draw first so unstripped volumes never read the mask
            return volume
        from brain.data.augment import skull_dropout
        from brain.prep import store

        brain = torch.from_numpy(np.ascontiguousarray(store.read(self.store_root / rel, "mask")))
        return skull_dropout(volume, brain, modality, 1.0)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[int(idx)]
        rel = Path(str(row["image_path"]))
        img_path = rel if rel.is_absolute() else self.store_root / rel
        from brain.prep import store

        vol = store.read(img_path, "image")
        if vol.ndim == 4:
            vol = vol[:, 0]
        volume = torch.from_numpy(np.ascontiguousarray(vol))  # (Z, Y, X) in the stored dtype
        if volume.ndim != 3:
            raise ValueError(f"expected (Z,Y,X), got {tuple(volume.shape)}")
        spacing, origin = self._spacing_origin(row)
        modality = 0 if str(row["modality"]).lower().startswith("ct") else 1
        from brain.data.crops3d import raw_views

        gen = torch.Generator().manual_seed(1000 + int(idx)) if self.fixed_crops else None
        if not self.fixed_crops:  # validation keeps whole volumes so its loss is comparable between runs
            volume = self._skull_dropout(volume, row, modality, gen)
        thick, axis = self._native(row)
        return raw_views({
            "volume": volume.unsqueeze(0).unsqueeze(0),
            "modality": torch.tensor([modality]),
            "spacing_mm": spacing.unsqueeze(0),
            "origin_mm": origin.unsqueeze(0),
            "thick_mm": thick,
            "thick_axis": axis,
        }, self.crop_cfg, gen)


def make_synthetic_volume(
    size_xyz=(32, 32, 16),
    spacing_mm=(1.0, 1.0, 2.0),
    origin_mm=(0.0, 0.0, 0.0),
    modality: int = 0,
    seed: int = 0,
) -> dict:
    """Ellipsoid phantom in (C, X, Y, Z) for tests and ``--synthetic`` training."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = (int(v) for v in size_xyz)
    x = (np.arange(nx) + 0.5) * spacing_mm[0] + origin_mm[0]
    y = (np.arange(ny) + 0.5) * spacing_mm[1] + origin_mm[1]
    z = (np.arange(nz) + 0.5) * spacing_mm[2] + origin_mm[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    cx, cy, cz = x.mean(), y.mean(), z.mean()
    rx, ry, rz = 0.35 * (x[-1] - x[0]), 0.4 * (y[-1] - y[0]), 0.4 * (z[-1] - z[0])
    r = ((xx - cx) / max(rx, 1e-3)) ** 2 + ((yy - cy) / max(ry, 1e-3)) ** 2 + ((zz - cz) / max(rz, 1e-3)) ** 2
    inside = r < 1.0
    if modality == 0:
        vol = np.full((nx, ny, nz), -1000.0, np.float32)
        vol[inside] = 35.0 + rng.normal(0, 8, size=int(inside.sum())).astype(np.float32)
    else:
        vol = np.full((nx, ny, nz), -6.0, np.float32)
        vol[inside] = rng.normal(0.0, 1.0, size=int(inside.sum())).astype(np.float32)
    return {
        "volume": torch.from_numpy(vol).unsqueeze(0),
        "spacing_mm": torch.tensor(spacing_mm, dtype=torch.float32),
        "origin_mm": torch.tensor(origin_mm, dtype=torch.float32),
        "modality": modality,
        "volume_id": f"synth{seed}",
        "patient_id": f"synth{seed}",
        "timepoint": "0",
        "site": "synth",
        "scanner": "",
        "dataset": "synth",
        "sequence": "",
    }


def collate_volumes(batch: list[dict]) -> dict:
    keys_stack = ("volume", "spacing_mm", "origin_mm")
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys_stack}
    out["modality"] = torch.tensor([b["modality"] for b in batch], dtype=torch.long)
    for name in ("volume_id", "patient_id", "timepoint", "site", "scanner", "dataset", "sequence"):
        out[name] = [b[name] for b in batch]
    return out


def collate_crops(batch: list[dict]) -> dict:
    """Join per-volume crops. View tensors are (V, 1, ...) and join on the batch axis."""
    out = {}
    for key, value in batch[0].items():
        if torch.is_tensor(value):
            out[key] = torch.cat([b[key] for b in batch], dim=0 if key in ROW_KEYS else 1)
        else:
            out[key] = value
    return out


class FamilySampler(DistributedSampler):
    """Draws volumes with replacement so a dataset family gets sampling mass proportional to ``n ** alpha``.

    ``alpha=1`` is the natural mix and ``alpha=0`` is uniform over families. Every rank draws the same list
    for an epoch (seeded by ``seed + epoch``) and takes its own stride of it, like ``DistributedSampler``.
    """

    def __init__(self, dataset: VolumeDataset, alpha: float, *, seed: int = 0, num_replicas: int = 1, rank: int = 0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed)
        fam = dataset.df["dataset_family"].astype(str)
        counts = fam.map(fam.value_counts()).to_numpy(dtype=np.float64)
        self.weights = torch.from_numpy(counts ** (float(alpha) - 1.0))

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        draw = torch.multinomial(self.weights, self.total_size, replacement=True, generator=g)
        return iter(draw[self.rank : self.total_size : self.num_replicas].tolist())


def _worker_threads(_worker_id: int) -> None:
    torch.set_num_threads(1)


def crop_microbatch(batch_size: int) -> int:
    """Volumes a worker crops before handing a chunk over. Few chunks in flight keeps file descriptors low."""
    micro = min(8, int(batch_size))
    return micro if int(batch_size) % micro == 0 else int(batch_size)


def make_loader(
    store_root: str | Path,
    split: str,
    cfg: dict,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    distributed: bool = False,
    n_volumes: int | None = None,
) -> DataLoader:
    """Loader of crop chunks of ``crop_microbatch(batch_size)`` volumes, cut inside the workers.

    ``shuffle=False`` is for validation: a fixed random subset of ``n_volumes`` volumes with fixed crops.
    """
    d = cfg["data"]
    ds = VolumeDataset(
        store_root, split,
        index_name=d.get("index_name", "volumes.csv"),
        default_spacing_mm=tuple(d.get("default_spacing_mm", (1.0, 1.0, 1.0))),
        default_origin_mm=tuple(d.get("default_origin_mm", (0.0, 0.0, 0.0))),
        crop_cfg=cfg,
        fixed_crops=not shuffle,
    )
    workers = int(d.get("num_workers", 8))
    if not shuffle:
        keep = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[: int(n_volumes or batch_size)]
        ds = torch.utils.data.Subset(ds, keep.tolist())
        workers = min(workers, 2)
    alpha = float(d.get("balance_alpha", 1.0))
    if shuffle and alpha != 1.0:
        world, rank = (torch.distributed.get_world_size(), torch.distributed.get_rank()) if distributed else (1, 0)
        sampler = FamilySampler(ds, alpha, seed=seed, num_replicas=world, rank=rank)
    else:
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed) if distributed else None
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    if cpus:
        workers = max(1, min(workers, cpus // world - 1))
    # Keep about two full batches of chunks in flight so the GPU never waits on a batch boundary.
    prefetch = max(2, -(-2 * (batch_size // crop_microbatch(batch_size)) // workers)) if workers else None
    return DataLoader(
        ds, batch_size=crop_microbatch(batch_size), shuffle=shuffle and sampler is None, sampler=sampler,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0, prefetch_factor=prefetch,
        collate_fn=collate_crops, drop_last=True, worker_init_fn=_worker_threads,
        generator=torch.Generator().manual_seed(seed),
    )
