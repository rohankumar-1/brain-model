"""Slab loader for pretraining: one item decodes one 8-slice chunk, returns k slices.

A single-slice read decodes the whole chunk, so taking k slices per decode cuts
CPU cost per slice. Sampling is balanced across dataset families: a family with
``n`` chunks gets probability mass proportional to ``n ** alpha`` (alpha=1 is the
natural mix, alpha=0 is uniform over families).
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from lib.zarr_loader import _load_slice_table

CHUNK = 8
BRATS_CONTRASTS = ("t1n", "t1c", "t2w", "t2f")


def brats_contrast(path: str) -> str | None:
    name = Path(path).name
    for tag in BRATS_CONTRASTS:
        if name.endswith(f"-{tag}.zip"):
            return tag
    return None


def swap_brats_contrast(path: str, contrast: str) -> str:
    tag = brats_contrast(path)
    if tag is None:
        raise ValueError(f"not a BraTS contrast path: {path}")
    p = Path(path)
    return str(p.with_name(p.name[: -len(f"-{tag}.zip")] + f"-{contrast}.zip"))


def _read_picks(path: str, start: int, pick: np.ndarray) -> torch.Tensor:
    import zarr
    from zarr.storage import ZipStore

    store = ZipStore(path, mode="r")
    try:
        block = np.asarray(zarr.open_group(store, mode="r")["image"][start : start + CHUNK])
    finally:
        store.close()
    if block.ndim == 3:
        block = block[:, None]
    return torch.from_numpy(block[pick - start])


class SlabDataset(Dataset):
    def __init__(self, root: str | Path, split: str, k: int = 2, alpha: float = 0.5,
                 load_siblings: bool = False):
        root = Path(root)
        df = _load_slice_table(root)
        if "pretrain" in df.columns:
            df = df[df["pretrain"].astype(bool)]
        df = df[(df["split"].astype(str) == split) & df["image_path"].astype(str).ne("")]
        if "modality" not in df.columns:
            raise ValueError(f"{root} slice table needs a modality column")
        df = df.assign(chunk=df["z"] // CHUNK, family=df["dataset_family"].astype(str))
        slabs = df.groupby(["image_path", "chunk"], sort=False).agg(
            z=("z", list), family=("family", "first"), modality=("modality", "first"),
        ).reset_index()
        self.k = int(k)
        self.load_siblings = bool(load_siblings)
        self.paths = [
            p if Path(p).is_absolute() else str(root / p) for p in slabs["image_path"].astype(str)
        ]
        self.path_set = set(self.paths)
        self.chunks = slabs["chunk"].to_numpy()
        self.zs = [np.asarray(z) for z in slabs["z"]]
        self.mod = slabs["modality"].map({"ct": 0, "mri": 1}).fillna(-1).astype(int).to_numpy()
        counts = slabs["family"].map(slabs["family"].value_counts()).to_numpy(dtype=np.float64)
        self.weights = torch.from_numpy(counts ** (alpha - 1.0))
        n_brats = sum(brats_contrast(p) is not None for p in self.paths)
        print(f"[slabs] {split}: {len(slabs)} chunks from {len(df)} slices, k={k}, alpha={alpha}"
              + (f", siblings on {n_brats} BraTS chunks" if self.load_siblings else ""))
        self._rng = None

    def __len__(self) -> int:
        return len(self.paths)

    def _sibling_path(self, path: str) -> str | None:
        tag = brats_contrast(path)
        if tag is None:
            return None
        cands = [
            swap_brats_contrast(path, other)
            for other in BRATS_CONTRASTS
            if other != tag and swap_brats_contrast(path, other) in self.path_set
        ]
        if not cands:
            return None
        return cands[int(self._rng.integers(len(cands)))]

    def __getitem__(self, i: int):
        if self._rng is None:  # per-worker stream, seeded by the DataLoader
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
        zs = self.zs[i]
        pick = self._rng.choice(zs, self.k, replace=len(zs) < self.k)
        start = int(self.chunks[i]) * CHUNK
        x = _read_picks(self.paths[i], start, pick)
        if not self.load_siblings:
            return x, i, int(self.mod[i])  # (k, 1, H, W), chunk, mod
        sib, ok = torch.zeros_like(x), 0.0
        other = self._sibling_path(self.paths[i])
        if other is not None:
            try:
                sib = _read_picks(other, start, pick)
                ok = 1.0
            except Exception:
                sib = torch.zeros_like(x)
        return x, i, int(self.mod[i]), sib, torch.tensor(ok, dtype=torch.float32)


class PrioritySampler(Sampler):
    """Infinite weighted sampler that can favor chunks the model still finds hard.

    ``update`` receives each drawn chunk's hardness relative to its batch mean, so
    scores stay comparable while the model improves. Each chunk keeps an EMA of
    that ratio; its sampling weight is ``base * clip(score, 1/clip, clip) ** beta``.
    Unseen chunks count as average. ``beta=0`` is plain ``base`` sampling.
    """

    BLOCK = 4096

    def __init__(self, base: torch.Tensor, seed: int, beta: float = 0.0, clip: float = 3.0, ema: float = 0.5):
        self.base, self.seed, self.beta, self.clip, self.ema = base, int(seed), float(beta), float(clip), float(ema)
        self.score = torch.full((len(base),), float("nan"))

    def weights(self) -> torch.Tensor:
        if self.beta == 0:
            return self.base
        s = torch.nan_to_num(self.score, nan=1.0).clamp(1 / self.clip, self.clip)
        return self.base * s.double() ** self.beta

    def update(self, idx: torch.Tensor, hardness: torch.Tensor) -> None:
        idx, h = idx.cpu(), hardness.detach().float().cpu()
        old = self.score[idx]
        self.score[idx] = torch.where(old.isnan(), h, self.ema * old + (1 - self.ema) * h)

    def visited(self) -> float:
        return float(1 - self.score.isnan().float().mean())

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            yield from torch.multinomial(self.weights(), self.BLOCK, True, generator=g).tolist()


def make_slab_loader(
    root: str | Path,
    split: str,
    *,
    batch_slices: int,
    k: int,
    workers: int,
    seed: int,
    alpha: float = 0.5,
    prefetch: int = 2,
    persistent: bool = True,
    priority_beta: float = 0.0,
    priority_clip: float = 3.0,
    load_siblings: bool = False,
) -> DataLoader:
    """Infinite loader of ((B, k, 1, H, W) native float16, chunk ids, modality ids).

    With ``load_siblings``, BraTS rows also yield a co-registered other contrast
    and a 0/1 mask (same z / chunk; skipped when the sibling zip is absent).
    ``loader.sampler`` is the :class:`PrioritySampler`; call its ``update`` with chunk ids.
    """
    ds = SlabDataset(root, split, k=k, alpha=alpha, load_siblings=load_siblings)
    return DataLoader(
        ds,
        batch_size=batch_slices // k,
        sampler=PrioritySampler(ds.weights, seed, priority_beta, priority_clip),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent and workers > 0,
        prefetch_factor=prefetch if workers > 0 else None,
        generator=torch.Generator().manual_seed(seed),
        drop_last=True,
    )


def take(loader: DataLoader, n: int):
    return islice(loader, n)
