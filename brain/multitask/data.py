"""Whole-volume batches for the multitask model: image, the tasks asked of it, and their targets.

Each training sample asks a random subset of the scan's labelled (trained) tasks: ``k`` uniform in
``1..min(n_available, k_max)``, or every available task (up to ``k_max``) with probability ``p_all``. Tasks are drawn
without replacement, group first: each draw picks one of the scan's task groups (registry ``group``) with weight
``group_weights[group]`` (default 1), then a task inside it with weight ``n_task ** -rare_beta`` (rare tasks asked
more often than their label count alone would). So the 30 SynthSeg tasks are two groups among several, not 30 of the
draws. At most ``max_seg`` segmentation tasks per sample (each runs the decoder once). The draw order is random, so
slots carry no order information.

Workers read the image (and the brain mask / segmentation masks the drawn tasks need), compute live targets, and ship
raw intensities; ``prepare`` (GPU) flips and rotates image and masks together, gates foreground tokens on the raw
values and windows the image. Regression targets are standardised with ``stats.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from brain.common.windowing import asinh_window, canonical_c_w
from brain.data.gate import token_keep
from brain.data.volumes import FamilySampler, _worker_threads
from brain.multitask.labels import live_targets
from brain.multitask.registry import Registry

TYPE_ID = {"binary": 0, "regression": 1, "multiclass": 2, "segmentation": 3, "survival": 4}


class MultitaskDataset(Dataset):
    """One item per volume with at least one label among ``tasks``.

    ``train=True`` draws a task subset per item; ``train=False`` asks every available task in registry order
    (deterministic), capped at ``k_max`` / ``max_seg``.
    """

    def __init__(self, store_root: str | Path, labels: pd.DataFrame, volumes: pd.DataFrame, reg: Registry,
                 stats: dict, *, split: str | list[str], tasks: list[str] | None = None, train: bool = True,
                 k_max: int = 8, p_all: float = 0.2, rare_beta: float = 0.5, max_seg: int = 2,
                 use: str | None = "train", volume_ids: list[str] | None = None,
                 group_weights: dict[str, float] | None = None):
        self.store_root = Path(store_root)
        self.reg, self.train = reg, bool(train)
        self.k_max, self.p_all, self.max_seg = int(k_max), float(p_all), int(max_seg)
        task_ids = tasks or [t.id for t in reg.trained]
        splits = [split] if isinstance(split, str) else list(split)
        lab = labels[labels["task"].isin(task_ids) & labels["split"].isin(splits)]
        if use is not None:
            lab = lab[lab["use"] == use]
        if volume_ids is not None:
            lab = lab[lab["volume_id"].isin(set(volume_ids))]
        lab = lab.assign(t=lab["task"].map(reg.index).astype(int))
        # standardised regression targets (mean / std of the task's training labels)
        mean = lab["task"].map(lambda t: stats.get(t, {}).get("mean", 0.0)).astype(float)
        std = lab["task"].map(lambda t: stats.get(t, {}).get("std", 1.0)).astype(float)
        is_reg = lab["t"].map(lambda i: reg[i].type == "regression")
        lab = lab.assign(y=np.where(is_reg, (lab["value"].astype(float) - mean) / std, lab["value"].astype(float)))
        self.mean = {t: float(s.get("mean", 0.0)) for t, s in stats.items()}
        self.std = {t: float(s.get("std", 1.0)) for t, s in stats.items()}
        counts = lab.groupby("t").size()
        self.task_weight = {int(t): float(n) ** -float(rare_beta) for t, n in counts.items()}
        gw = group_weights or {}
        self.group_of = {i: reg[i].group for i in range(len(reg))}
        self.group_weight = {g: float(gw.get(g, 1.0)) for g in reg.groups}
        self.items: list[tuple[str, list[tuple[int, float, str]]]] = []
        grouped = lab.groupby("volume_id", sort=True)
        for vid, g in grouped:
            self.items.append((str(vid), list(zip(g["t"].astype(int), g["y"].astype(float),
                                                  g["ref"].fillna("").astype(str)))))
        meta = volumes.set_index("volume_id").loc[[v for v, _ in self.items]]
        self.df = meta.reset_index()[["volume_id", "image_path", "modality", "dataset_family"]]
        self.paths = self.df["image_path"].astype(str).tolist()
        self.mods = self.df["modality"].astype(str).str.lower().tolist()

    def __len__(self) -> int:
        return len(self.items)

    def _draw(self, avail: list[tuple[int, float, str]], rng: np.random.Generator) -> list[tuple[int, float, str]]:
        if not self.train:
            seg = 0
            out = []
            for a in sorted(avail):
                if self.reg[a[0]].type == "segmentation":
                    if seg >= self.max_seg:
                        continue
                    seg += 1
                out.append(a)
            return out[: self.k_max]
        cap = min(len(avail), self.k_max)
        k = cap if rng.random() < self.p_all else int(rng.integers(1, cap + 1))
        left: dict[str, list] = {}
        for a in avail:
            left.setdefault(self.group_of[a[0]], []).append(a)
        out, seg = [], 0
        while len(out) < k:
            if seg >= self.max_seg:                       # segmentation budget spent: only non-seg tasks remain
                left = {g: [a for a in v if self.reg[a[0]].type != "segmentation"] for g, v in left.items()}
            left = {g: v for g, v in left.items() if v}
            if not left:
                break
            groups = list(left)
            gw = np.asarray([self.group_weight.get(g, 1.0) for g in groups])
            g = groups[rng.choice(len(groups), p=gw / gw.sum())]
            tw = np.asarray([self.task_weight.get(a[0], 1.0) for a in left[g]])
            a = left[g].pop(int(rng.choice(len(tw), p=tw / tw.sum())))
            seg += self.reg[a[0]].type == "segmentation"
            out.append(a)
        return out

    def __getitem__(self, idx: int) -> dict:
        from brain.prep import store

        vid, avail = self.items[int(idx)]
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))) if self.train else idx)
        drawn = self._draw(avail, rng)
        need_brain = any(r == "@brain" or r.startswith("@live:") for _, _, r in drawn)
        rel = Path(self.paths[idx])
        arrays = store.read_many(rel if rel.is_absolute() else self.store_root / rel,
                                 ("image", "brain") if need_brain else ("image",))
        img = arrays["image"][:, 0] if arrays["image"].ndim == 4 else arrays["image"]
        mod = self.mods[idx]
        live = None
        tasks, ys, seg_masks, seg_slot = [], [], [], []
        mask_cache: dict[str, np.ndarray] = {}
        for t, y, ref in drawn:
            if ref.startswith("@live:"):
                if live is None:
                    live = live_targets(img, arrays["brain"], mod)
                tid = self.reg[t].id
                y = (live[ref.split(":", 1)[1]] - self.mean.get(tid, 0.0)) / self.std.get(tid, 1.0)
                if not math.isfinite(y):
                    continue
            slot = -1
            if self.reg[t].type == "segmentation":
                if ref == "@brain":
                    m = np.asarray(arrays["brain"]) > 0
                else:
                    path, ids = ref.split("|")
                    if path not in mask_cache:
                        raw = np.asarray(store.read(self.store_root / path, "mask"))
                        mask_cache[path] = raw[:, 0] if raw.ndim == 4 else raw
                    m = np.isin(mask_cache[path], [int(i) for i in ids.split(",")])
                if m.shape != img.shape:
                    continue
                slot = len(seg_masks)
                seg_masks.append(torch.from_numpy(np.ascontiguousarray(m, dtype=np.uint8)))
                y = float("nan")
            tasks.append(t)
            ys.append(y)
            seg_slot.append(slot)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)),           # (Z, Y, X) raw, stored dtype
            "modality": 0 if mod.startswith("ct") else 1,
            "task": torch.tensor(tasks, dtype=torch.long),
            "y": torch.tensor(ys, dtype=torch.float32),
            "seg_slot": torch.tensor(seg_slot, dtype=torch.long),
            "seg": torch.stack(seg_masks) if seg_masks else torch.zeros(0, *img.shape, dtype=torch.uint8),
            "volume_id": vid,
        }


def collate(batch: list[dict]) -> dict:
    """Pads slots to the batch's longest task list (``task == -1``); segmentation masks are stacked over the batch
    with ``seg_pair`` (P, 2) = (row, slot) for each."""
    k = max(max(len(b["task"]), 1) for b in batch)
    b_n = len(batch)
    task = torch.full((b_n, k), -1, dtype=torch.long)
    y = torch.full((b_n, k), float("nan"))
    pairs, segs = [], []
    for i, b in enumerate(batch):
        n = len(b["task"])
        task[i, :n], y[i, :n] = b["task"], b["y"]
        for j in range(n):
            s = int(b["seg_slot"][j])
            if s >= 0:
                pairs.append((i, j))
                segs.append(b["seg"][s])
    shape = batch[0]["image"].shape
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "modality": torch.tensor([b["modality"] for b in batch]),
        "task": task, "y": y,
        "seg": torch.stack(segs) if segs else torch.zeros(0, *shape, dtype=torch.uint8),
        "seg_pair": torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
        "volume_id": [b["volume_id"] for b in batch],
    }


def make_loader(ds: MultitaskDataset, *, batch_size: int, shuffle: bool, seed: int, workers: int,
                balance_alpha: float = 1.0, balance_groups: dict | None = None, distributed: bool = False,
                drop_last: bool = True) -> DataLoader:
    if shuffle and balance_alpha != 1.0:
        world, rank = (torch.distributed.get_world_size(), torch.distributed.get_rank()) if distributed else (1, 0)
        sampler = FamilySampler(ds, balance_alpha, seed=seed, num_replicas=world, rank=rank, groups=balance_groups)
    else:
        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed) if distributed else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      prefetch_factor=2 if workers else None, collate_fn=collate, drop_last=drop_last,
                      worker_init_fn=_worker_threads, generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------------------------------------ GPU side
def _rotation(deg: torch.Tensor) -> torch.Tensor:
    """(B, 3) angles in degrees about x, y, z -> (B, 3, 3) rotation matrices (x, y, z order)."""
    a = torch.deg2rad(deg)
    c, s = a.cos(), a.sin()
    one, zero = torch.ones_like(c[:, 0]), torch.zeros_like(c[:, 0])
    rx = torch.stack([one, zero, zero, zero, c[:, 0], -s[:, 0], zero, s[:, 0], c[:, 0]], -1).view(-1, 3, 3)
    ry = torch.stack([c[:, 1], zero, s[:, 1], zero, one, zero, -s[:, 1], zero, c[:, 1]], -1).view(-1, 3, 3)
    rz = torch.stack([c[:, 2], -s[:, 2], zero, s[:, 2], c[:, 2], zero, zero, zero, one], -1).view(-1, 3, 3)
    return rz @ ry @ rx


def augment_theta(b: int, shape_zyx, aug: dict, device) -> torch.Tensor | None:
    """Per-row affine (B, 3, 4) for ``F.affine_grid`` (normalised x, y, z): a rotation in millimetres about the grid
    centre and a shift, applied with probability ``aug.p``. None when nothing is drawn."""
    p = float(aug.get("p", 0.0))
    if p <= 0:
        return None
    on = (torch.rand(b, device=device) < p).float().unsqueeze(-1)
    deg = (torch.rand(b, 3, device=device) * 2 - 1) * torch.tensor(aug.get("rot_deg", [10, 10, 10]), device=device)
    shift = (torch.rand(b, 3, device=device) * 2 - 1) * float(aug.get("shift_mm", 0.0))
    half = torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], device=device, dtype=torch.float32) / 2
    rot = _rotation(deg * on)
    a = rot * half.view(1, 1, 3) / half.view(1, 3, 1)               # diag(1/h) R diag(h)
    return torch.cat([a, (shift * on / half).unsqueeze(-1)], dim=-1)


def prepare(batch: dict, cfg: dict, device, patch_xyz, train: bool) -> dict:
    """Raw batch -> encoder input (B, 1, Z, Y, X) windowed, foreground keep (B, N), and float segmentation targets
    flipped / rotated with their image. Rows whose modality is CT window with the CT canonical window, MRI with the
    MRI one (as frozen evaluation does)."""
    aug = (cfg.get("aug") or {}) if train else {}
    raw = batch["image"].to(device, non_blocking=True).float().unsqueeze(1)
    mod = batch["modality"].to(device)
    b = raw.shape[0]
    seg = batch["seg"].to(device, non_blocking=True).float().unsqueeze(1)
    rows = batch["seg_pair"][:, 0].to(device)
    if train and float(aug.get("flip_lr", 0.0)) > 0:
        flip = torch.rand(b, device=device) < float(aug["flip_lr"])
        raw = torch.where(flip.view(-1, 1, 1, 1, 1), raw.flip(-1), raw)
        if len(seg):
            seg = torch.where(flip[rows].view(-1, 1, 1, 1, 1), seg.flip(-1), seg)
    theta = augment_theta(b, raw.shape[2:], aug.get("affine") or {}, device) if train else None
    if theta is not None:
        bg = raw.amin(dim=(1, 2, 3, 4), keepdim=True)
        grid = F.affine_grid(theta, list(raw.shape), align_corners=False)
        raw = F.grid_sample(raw - bg, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + bg
        if len(seg):
            seg = F.grid_sample(seg, grid[rows], mode="bilinear", padding_mode="zeros", align_corners=False)
        del grid
    xyz = raw.permute(0, 1, 4, 3, 2)                                   # gate works on (B, 1, X, Y, Z)
    gate = (cfg.get("model") or {}).get("gate") or {}
    keep = token_keep(xyz, mod, patch_xyz, ct_hu=tuple(gate.get("ct_hu", (-500.0, 200.0))),
                      mri_lo=float(gate.get("mri_lo", -5.5))).reshape(b, -1)
    cw = torch.tensor([canonical_c_w("ct"), canonical_c_w("mri")], device=device, dtype=torch.float32)[mod]
    x = asinh_window(raw, cw[:, 0].view(-1, 1, 1, 1, 1), cw[:, 1].view(-1, 1, 1, 1, 1))
    return {"x": x, "keep": keep, "modality": mod, "task": batch["task"].to(device), "y": batch["y"].to(device),
            "seg": seg, "seg_pair": batch["seg_pair"].to(device)}


def load_labels(labels_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    d = Path(labels_dir)
    return pd.read_parquet(d / "labels.parquet"), json.loads((d / "stats.json").read_text())


class SyntheticMultitask(Dataset):
    """Phantom volumes with random labels for every trained task type (``--synthetic`` and tests): a bright ellipsoid
    "brain" on a dark background, a cube "lesion" as the segmentation target."""

    def __init__(self, reg: Registry, shape_zyx, n: int = 64, k_max: int = 4, seed: int = 0):
        self.reg, self.shape, self.n, self.k_max, self.seed = reg, tuple(shape_zyx), int(n), int(k_max), int(seed)
        self.tasks = [reg.index[t.id] for t in reg.trained]
        self.df = pd.DataFrame({"dataset_family": ["a", "b"] * (self.n // 2) + ["a"] * (self.n % 2)})

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        g = torch.Generator().manual_seed(self.seed + int(idx))
        z, y, x = self.shape
        zz, yy, xx = torch.meshgrid(*[torch.linspace(-1, 1, s) for s in self.shape], indexing="ij")
        brain = (zz**2 + yy**2 + xx**2) < 0.6
        img = torch.where(brain, 1.0, -2.0) + 0.1 * torch.randn(self.shape, generator=g)
        lesion = torch.zeros(self.shape, dtype=torch.uint8)
        c = [int(torch.randint(s // 4, 3 * s // 4, (1,), generator=g)) for s in self.shape]
        lesion[c[0] - z // 8:c[0] + z // 8, c[1] - y // 8:c[1] + y // 8, c[2] - x // 8:c[2] + x // 8] = 1
        k = int(torch.randint(1, self.k_max + 1, (1,), generator=g))
        pick = torch.randperm(len(self.tasks), generator=g)[:k].tolist()
        tasks, ys, slots, segs = [], [], [], []
        for i in pick:
            t = self.tasks[i]
            kind = self.reg[t].type
            slot = -1
            if kind == "binary":
                v = float(torch.randint(0, 2, (1,), generator=g))
            elif kind == "multiclass":
                v = float(torch.randint(0, len(self.reg[t].classes), (1,), generator=g))
            elif kind == "segmentation":
                if len(segs) >= 2:
                    continue
                v, slot = float("nan"), len(segs)
                segs.append(lesion)
            else:
                v = float(torch.randn(1, generator=g))
            tasks.append(t)
            ys.append(v)
            slots.append(slot)
        return {"image": img.half(), "modality": 1, "task": torch.tensor(tasks, dtype=torch.long),
                "y": torch.tensor(ys, dtype=torch.float32), "seg_slot": torch.tensor(slots, dtype=torch.long),
                "seg": torch.stack(segs) if segs else torch.zeros(0, *self.shape, dtype=torch.uint8),
                "volume_id": f"synthetic:{idx}"}
