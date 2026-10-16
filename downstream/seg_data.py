"""Pixel cache, foreground-weighted sampler and GPU augmentation for segmentation.

The cache holds the uniform slices (image fp16, mask uint8) of every task volume
as ``.npy`` memmaps, built once per task and shared by all checkpoints. BraTS
cases carry four MRI sequences (t1n, t1c, t2w, t2f) that are cached as channels.
Training then never touches the zarr zips again.
"""

from __future__ import annotations

import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from downstream.embeddings import UNIFORM_SLICES, read_uniform, resolve_path
from lib.views import affine_theta, warp, warp_floor
from lib.windowing import FLOOR

SIDE = 256
BRATS_SEQUENCES = ("t1n", "t1c", "t2w", "t2f")


@dataclass
class PixelCache:
    volume_ids: list[str]
    images: np.ndarray  # (N, S, C, H, W) float16 memmap, C = sequences
    masks: np.ndarray  # (N, S, H, W) uint8 memmap
    fg: np.ndarray  # (N, S) bool, slice holds any foreground
    z_index: np.ndarray  # (N, S) source z of each cached slice


def sequence_paths(rel: str, all_sequences: bool) -> list[str]:
    """Image zips of one case: the four BraTS sequences, or just ``rel``."""
    if all_sequences and rel.endswith("-t1n.zip"):
        return [rel.replace("-t1n.zip", f"-{s}.zip") for s in BRATS_SEQUENCES]
    return [rel]


def _read_volume(args: tuple) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    i, image_paths, mask_path, n = args
    reads = [read_uniform(p, "image", n) for p in image_paths]
    zs = reads[0][0]
    img = np.stack([r[1].reshape(len(zs), SIDE, SIDE) for r in reads], axis=1)  # (S, C, H, W)
    return i, zs, img, read_uniform(mask_path, "mask", n)[1]


def load_pixel_cache(
    volume_ids: list[str], image_paths: dict, mask_paths: dict, slices_root: Path,
    cache_dir: Path, task: str, n_slices: int = UNIFORM_SLICES, all_sequences: bool = True,
    threads: int = 8,
) -> PixelCache:
    ids = [str(v) for v in volume_ids]
    seqs = {v: sequence_paths(image_paths[v], all_sequences) for v in ids}
    c = len(seqs[ids[0]])
    root = Path(cache_dir) / "_pixels" / f"{task}_c{c}_s{n_slices}"
    meta_path, meta = root / "meta.json", {"volume_ids": ids, "n_slices": n_slices, "channels": c}
    if meta_path.exists() and json.loads(meta_path.read_text()) == meta:
        return PixelCache(
            ids, np.load(root / "images.npy", mmap_mode="r"), np.load(root / "masks.npy", mmap_mode="r"),
            np.load(root / "fg.npy"), np.load(root / "z_index.npy"),
        )
    root.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(root / "images.npy", "w+", np.float16, (len(ids), n_slices, c, SIDE, SIDE))
    masks = np.lib.format.open_memmap(root / "masks.npy", "w+", np.uint8, (len(ids), n_slices, SIDE, SIDE))
    z_index = np.zeros((len(ids), n_slices), dtype=np.int16)
    jobs = [(i, [resolve_path(slices_root, p) for p in seqs[v]], resolve_path(slices_root, mask_paths[v]), n_slices)
            for i, v in enumerate(ids)]
    with ThreadPoolExecutor(threads) as pool:
        for k, (i, zs, img, msk) in enumerate(pool.map(_read_volume, jobs), 1):
            images[i], masks[i], z_index[i] = img, msk, zs
            if k % 200 == 0 or k == len(ids):
                print(f"[seg-cache] {task} ({c} ch): {k}/{len(ids)} volumes", flush=True)
    images.flush(), masks.flush()
    fg = np.asarray(masks.reshape(len(ids), n_slices, -1).max(-1) > 0)
    np.save(root / "fg.npy", fg)
    np.save(root / "z_index.npy", z_index)
    meta_path.write_text(json.dumps(meta))
    return PixelCache(ids, images, masks, fg, z_index)


def train_batches(cache: PixelCache, vols: np.ndarray, batch: int, steps: int, seed: int,
                  fg_boost: float = 3.0, prefetch: int = 6):
    """Yield ``steps`` (image (B,C,H,W), mask (B,H,W)) host batches. Foreground slices are drawn (1+boost)x as often."""
    n_s = cache.fg.shape[1]
    flat = np.stack(np.meshgrid(vols, np.arange(n_s), indexing="ij"), -1).reshape(-1, 2)
    w = 1.0 + fg_boost * cache.fg[flat[:, 0], flat[:, 1]]
    w /= w.sum()
    q: queue.Queue = queue.Queue(prefetch)

    def produce() -> None:
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            v, s = flat[rng.choice(len(flat), batch, p=w)].T
            q.put((torch.from_numpy(cache.images[v, s]), torch.from_numpy(cache.masks[v, s])))
        q.put(None)

    threading.Thread(target=produce, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def augment(x: torch.Tensor, y: torch.Tensor, modality: str = "mri") -> tuple[torch.Tensor, torch.Tensor]:
    """Shared zoom / rotation / flip in native units; intensity window is applied in Segmenter."""
    b, c, h, w = x.shape
    theta = affine_theta(b, (0.6, 1.0), device=x.device, ratio=(0.85, 1.18), max_rot=15.0)
    floor = x.new_tensor(FLOOR["ct" if str(modality).lower().startswith("ct") else "mri"])
    x = warp_floor(x.reshape(b * c, 1, h, w), theta.repeat_interleave(c, 0), h, floor).reshape(b, c, h, w)
    y = warp(y[:, None].float(), theta, y.size(-1), mode="nearest")[:, 0].long()
    return x, y
