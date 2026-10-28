"""Whole-volume batches for BrainQuery: image, the tasks asked of it, and their targets.

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

from brain.common.windowing import asinh_window, canonical_c_w, floors, role_windows
from brain.data.augment import appearance
from brain.data.gate import token_keep
from brain.data.volumes import FamilySampler, _worker_threads, meta_id
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
                 group_weights: dict[str, float] | None = None, meta_drop: float = 0.0, skull_strip: float = 0.0):
        """``meta_drop``: chance (training) that a scan's metadata id is replaced by unknown. ``skull_strip``: chance
        (training) that an MRI scan with a QC-passed SynthStrip mask is skull-stripped, unless a drawn task
        ``needs_skull``: tumour / stroke cohorts arrive stripped, so stripping is otherwise a cohort fingerprint."""
        self.store_root = Path(store_root)
        self.meta_drop, self.skull_strip = float(meta_drop), float(skull_strip)
        self.skull_tasks = {i for i, t in enumerate(reg.tasks) if t.needs_skull}
        self.reg, self.train = reg, bool(train)
        self.k_max, self.p_all, self.max_seg = int(k_max), float(p_all), int(max_seg)
        task_ids = tasks or [t.id for t in reg.trained]
        splits = [split] if isinstance(split, str) else list(split)
        lab = labels[labels["task"].isin(task_ids) & labels["split"].isin(splits)]
        if use is not None:
            lab = lab[lab["use"] == use]
        if volume_ids is not None:
            lab = lab[lab["volume_id"].isin(set(volume_ids))]
        # an older label table (labels/multitask v1, for old checkpoints) still lists scans the prep-time dedup removed;
        # an id the index lists twice (cq500 and seg_cq500 copies of one study since 2026-10-26) names no single image
        ambiguous = set(volumes.loc[volumes["volume_id"].duplicated(), "volume_id"])
        indexed = lab["volume_id"].isin(set(volumes["volume_id"]) - ambiguous)
        if not indexed.all():
            gone = lab.loc[~indexed, "volume_id"].nunique()
            print(f"[data] {gone} labelled volumes are not in the store index once (deduplicated / dropped / "
                  f"ambiguous): skipped", flush=True)
            lab = lab[indexed]
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
        meta = meta.reset_index()
        for col, fill in (("thick_mm", 1.0), ("thick_axis", -1)):
            if col not in meta:
                meta[col] = fill
        self.df = meta[["volume_id", "image_path", "modality", "dataset_family", "thick_mm", "thick_axis"]]
        self.thick = list(zip(self.df["thick_mm"].fillna(1.0).astype(float),
                              self.df["thick_axis"].fillna(-1).astype(int)))
        self.paths = self.df["image_path"].astype(str).tolist()
        self.mods = self.df["modality"].astype(str).str.lower().tolist()
        get = lambda c, fill="": meta[c] if c in meta else pd.Series(fill, index=meta.index)  # noqa: E731
        self.meta_ids = [meta_id(c, m, ds) for c, m, ds in zip(get("contrast", None), self.mods, get("dataset"))]
        mask_ok = get("brain_ok").astype(str).str.lower().isin(["true", "1", "1.0"]) \
            & get("brain_method").astype(str).eq("synthstrip")
        stripped = get("stripped_input").astype(str).str.lower().isin(["true", "1", "1.0"])
        self.strippable = (mask_ok & ~stripped & ~pd.Series(self.mods, index=meta.index).str.startswith("ct")).tolist()

    def __len__(self) -> int:
        return len(self.items)

    def _draw(self, avail: list[tuple[int, float, str]], rng: np.random.Generator) -> list[tuple[int, float, str]]:
        if not self.train:
            # every non-segmentation task, plus ``max_seg`` segmentation tasks taken from a per-volume rotation of the
            # volume's list (``rng`` is seeded by the item index), so a validation set covers all of them
            avail = sorted(avail)
            seg = [a for a in avail if self.reg[a[0]].type == "segmentation"]
            if len(seg) > self.max_seg:
                start = int(rng.integers(len(seg)))
                seg = (seg[start:] + seg[:start])[: self.max_seg]
            return ([a for a in avail if self.reg[a[0]].type != "segmentation"] + seg)[: self.k_max]
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
        strip = (self.train and self.skull_strip > 0 and self.strippable[idx] and rng.random() < self.skull_strip
                 and not any(t in self.skull_tasks for t, _, _ in drawn))
        need_brain = strip or any(r == "@brain" or r.startswith("@live:") for _, _, r in drawn)
        rel = Path(self.paths[idx])
        arrays = store.read_many(rel if rel.is_absolute() else self.store_root / rel,
                                 ("image", "brain") if need_brain else ("image",))
        img = store.squeeze_channel(arrays["image"])
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
                        mask_cache[path] = store.read_image(self.store_root / path, "mask")
                    m = np.isin(mask_cache[path], [int(i) for i in ids.split(",")])
                if m.shape != img.shape:
                    continue
                slot = len(seg_masks)
                seg_masks.append(torch.from_numpy(np.ascontiguousarray(m, dtype=np.uint8)))
                y = float("nan")
            tasks.append(t)
            ys.append(y)
            seg_slot.append(slot)
        if strip:                       # after the live targets (measured inside the brain mask, so unchanged)
            img = np.where(np.asarray(arrays["brain"]) > 0, img, img.min())
        meta_ = 0 if self.train and self.meta_drop > 0 and rng.random() < self.meta_drop else self.meta_ids[idx]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)),           # (Z, Y, X) raw, stored dtype
            "modality": 0 if mod.startswith("ct") else 1,
            "task": torch.tensor(tasks, dtype=torch.long),
            "y": torch.tensor(ys, dtype=torch.float32),
            "seg_slot": torch.tensor(seg_slot, dtype=torch.long),
            "seg": torch.stack(seg_masks) if seg_masks else torch.zeros(0, *img.shape, dtype=torch.uint8),
            "volume_id": vid,
            "thick": torch.tensor(self.thick[idx], dtype=torch.float32),     # (mm, grid axis or -1)
            "meta": meta_,
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
        "thick": torch.stack([b.get("thick", torch.tensor([1.0, -1.0])) for b in batch]),
        "meta": torch.tensor([int(b.get("meta", 0)) for b in batch], dtype=torch.long),
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


def augment_theta(b: int, shape_zyx, aug: dict, device, skip: torch.Tensor | None = None) -> torch.Tensor | None:
    """Per-row affine (B, 3, 4) for ``F.affine_grid`` (normalised x, y, z): a rotation in millimetres about the grid
    centre and a shift, applied with probability ``aug.p`` (never on ``skip`` rows). None when nothing is drawn."""
    p = float(aug.get("p", 0.0))
    if p <= 0:
        return None
    on = torch.rand(b, device=device) < p
    if skip is not None:
        on &= ~skip
    on = on.float().unsqueeze(-1)
    deg = (torch.rand(b, 3, device=device) * 2 - 1) * torch.tensor(aug.get("rot_deg", [10, 10, 10]), device=device)
    shift = (torch.rand(b, 3, device=device) * 2 - 1) * float(aug.get("shift_mm", 0.0))
    half = torch.tensor([shape_zyx[2], shape_zyx[1], shape_zyx[0]], device=device, dtype=torch.float32) / 2
    rot = _rotation(deg * on)
    a = rot * half.view(1, 1, 3) / half.view(1, 3, 1)               # diag(1/h) R diag(h)
    return torch.cat([a, (shift * on / half).unsqueeze(-1)], dim=-1)


def foreground_tokens(raw: torch.Tensor, mod: torch.Tensor, patch_xyz, gate: dict | None = None) -> torch.Tensor:
    """Foreground token mask (B, N) of native-unit volumes ``raw`` (B, 1, Z, Y, X)."""
    gate = gate or {}
    return token_keep(raw.permute(0, 1, 4, 3, 2), mod, patch_xyz, mode=str(gate.get("mode", "center")),
                      ct_hu=tuple(gate.get("ct_hu", (-500.0, 200.0))),
                      mri_lo=float(gate.get("mri_lo", -5.5))).reshape(raw.shape[0], -1)   # gate works on (B,1,X,Y,Z)


def canonical_window(raw: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
    """Each row in its modality's canonical window (CT brain / MRI tissue), as frozen evaluation encodes."""
    cw = torch.tensor([canonical_c_w("ct"), canonical_c_w("mri")], device=raw.device, dtype=torch.float32)[mod]
    return asinh_window(raw, cw[:, 0].view(-1, 1, 1, 1, 1), cw[:, 1].view(-1, 1, 1, 1, 1))


def encoder_window(raw: torch.Tensor, mod: torch.Tensor, in_chans: int = 1) -> torch.Tensor:
    """The unaugmented encoder input: the canonical window, or for a three-channel encoder the three role windows
    (``windowing.ROLES``, the first of them the canonical one)."""
    return role_windows(raw, mod) if int(in_chans) == 3 else canonical_window(raw, mod)


def gate_and_window(raw: torch.Tensor, mod: torch.Tensor, patch_xyz, gate: dict | None = None, in_chans: int = 1):
    """Native-unit volumes ``raw`` (B, 1, Z, Y, X) -> the windowed encoder input (``encoder_window``) and the
    foreground token mask (B, N) gated on the raw values."""
    return encoder_window(raw, mod, in_chans), foreground_tokens(raw, mod, patch_xyz, gate)


def fov_cut(raw: torch.Tensor, seg: torch.Tensor, rows: torch.Tensor, mod: torch.Tensor, allowed: torch.Tensor,
            cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Blank a random inferior slab (``mm`` range, 1 mm slices) of each allowed row with probability ``p``: the image
    goes to its background floor and its masks to 0, as in a scan whose coverage stops above the skull base
    (RSNA, M4Raw), so truncation stops being a cohort cue."""
    b, z = raw.shape[0], raw.shape[2]
    on = (torch.rand(b, device=raw.device) < float(cfg.get("p", 0.0))) & allowed
    if not on.any():
        return raw, seg
    lo, hi = cfg.get("mm", (10, 40))
    cut = torch.randint(int(lo), int(hi) + 1, (b,), device=raw.device) * on
    below = torch.arange(z, device=raw.device).view(1, 1, z, 1, 1) < cut.view(-1, 1, 1, 1, 1)
    raw = torch.where(below, floors(mod).view(-1, 1, 1, 1, 1).to(raw.device), raw)
    if len(seg):
        seg = torch.where(below[rows], torch.zeros((), device=seg.device), seg)
    return raw, seg


def protected_tasks(reg) -> tuple[torch.Tensor, torch.Tensor]:
    """Per registry task, two bool tensors for ``prepare``: ``fov_unsafe`` (the target needs the whole brain in view:
    alignment-dependent and live targets; no field-of-view cut) and ``aug_unsafe`` (a live target, measured by
    ``labels.live_targets`` on the unaugmented, template-aligned image: no rotation / shift, bias field or drift, which
    would move the image away from its label; the mirror flip is fine, the measures are flip-invariant)."""
    from brain.multitask.labels import registration_sensitive

    live = [any(s.get("source") == "live" for s in t.sources) for t in reg.tasks]
    fov = [registration_sensitive(t.id) or lv for t, lv in zip(reg.tasks, live)]
    return torch.tensor(fov), torch.tensor(live)


def prepare(batch: dict, cfg: dict, device, patch_xyz, train: bool, fov_unsafe: torch.Tensor | None = None,
            aug_unsafe: torch.Tensor | None = None) -> dict:
    """Raw batch -> encoder input (B, C, Z, Y, X) windowed, foreground keep (B, N), and float segmentation targets
    moved with their image. Training: flip, rotation / shift (not on rows asking a task in ``aug_unsafe``), a
    field-of-view cut on rows asking no task in ``fov_unsafe`` (n_tasks bool: targets that need the whole brain in
    view; see ``protected_tasks``; None = every row is unsafe, no cut), then appearance augmentation
    (``aug.appearance``: bias / through-plane blur / noise / blur / drift, ``brain.data.augment``) with the window
    pinned to each modality's canonical one (frozen evaluation's; CT density targets live in it). The foreground
    gate is taken before appearance augmentation, so noise cannot flip background tokens. ``aug_unsafe`` rows also
    get no bias field or drift. A three-channel encoder (``model.in_chans`` 3) gets the three role windows, unjittered,
    with every appearance augmentation in native units (``augment``'s roles path: no drift)."""
    in_chans = int((cfg.get("model") or {}).get("in_chans", 1))
    aug = (cfg.get("aug") or {}) if train else {}
    raw = batch["image"].to(device, non_blocking=True).float().unsqueeze(1)
    mod = batch["modality"].to(device)
    b = raw.shape[0]
    task = batch["task"].to(device)
    asked = task >= 0
    live_rows = (aug_unsafe.to(device)[task.clamp_min(0)] & asked).any(1) if aug_unsafe is not None else None
    seg = batch["seg"].to(device, non_blocking=True).float().unsqueeze(1)
    rows = batch["seg_pair"][:, 0].to(device)
    if train and float(aug.get("flip_lr", 0.0)) > 0:
        flip = torch.rand(b, device=device) < float(aug["flip_lr"])
        raw = torch.where(flip.view(-1, 1, 1, 1, 1), raw.flip(-1), raw)
        if len(seg):
            seg = torch.where(flip[rows].view(-1, 1, 1, 1, 1), seg.flip(-1), seg)
    theta = augment_theta(b, raw.shape[2:], aug.get("affine") or {}, device, skip=live_rows) if train else None
    if theta is not None:
        bg = raw.amin(dim=(1, 2, 3, 4), keepdim=True)
        grid = F.affine_grid(theta, list(raw.shape), align_corners=False)
        raw = F.grid_sample(raw - bg, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + bg
        if len(seg):
            seg = F.grid_sample(seg, grid[rows], mode="bilinear", padding_mode="zeros", align_corners=False)
        del grid
    if train and (aug.get("fov_cut") or {}).get("p", 0.0):
        unsafe = fov_unsafe.to(device)[task.clamp_min(0)] & (task >= 0) if fov_unsafe is not None \
            else torch.ones_like(task, dtype=torch.bool)
        raw, seg = fov_cut(raw, seg, rows, mod, ~unsafe.any(1), aug["fov_cut"])
    keep = foreground_tokens(raw, mod, patch_xyz, (cfg.get("model") or {}).get("gate"))
    app = aug.get("appearance")
    if train and app:
        thick = batch["thick"].to(device) if "thick" in batch else None
        window = {"mode": "roles", "jitter": 0.0} if in_chans == 3 else {"mode": "fixed"}
        x = appearance(raw, mod, {**app, "window": window},
                       thick=(thick[:, 0], thick[:, 1].long()) if thick is not None else None, protect=live_rows)
    else:
        x = encoder_window(raw, mod, in_chans)
    return {"x": x, "keep": keep, "modality": mod, "task": task, "y": batch["y"].to(device),
            "seg": seg, "seg_pair": batch["seg_pair"].to(device),
            "meta": batch["meta"].to(device) if "meta" in batch else None}


def pick_val_volumes(labels: pd.DataFrame, volumes: pd.DataFrame, reg: Registry, task_ids: list[str],
                     n: int) -> list[str]:
    """The fixed training-time validation set: up to ``n`` split-``val`` scans per task, each later asked all its tasks.

    Scans are ordered by a stable hash of their id (ids start with the dataset name, so sorting by id took a single
    cohort per task) and taken round-robin across dataset families; binary tasks get up to n/2 positives so both
    classes are present (epidural hemorrhage: 2% positives). Tasks sharing scans pick the same ones first, which keeps
    the union small (64 per task -> ~800 scans for the 65 trained tasks)."""
    import hashlib

    types = {t.id: t.type for t in reg.tasks}
    val = labels[(labels["split"] == "val") & (labels["use"] == "train") & labels["task"].isin(task_ids)]
    val = val.merge(volumes[["volume_id", "dataset_family"]], on="volume_id", how="left")
    val = val.assign(h=val["volume_id"].map(lambda s: int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)))
    picks = []
    for t, g in val.groupby("task"):
        g = g.sort_values("h")
        if types.get(t) == "binary":
            pos, neg = g[g["value"] == 1], g[g["value"] == 0]
            k = min(len(pos), n // 2)
            g = pd.concat([pos.head(k), neg.head(n - k)])
        else:
            g = g.assign(r=g.groupby("dataset_family").cumcount()).sort_values(["r", "h"]).head(n)
        picks.append(g["volume_id"])
    return pd.concat(picks).unique().tolist() if picks else []


def task_loaders(cfg: dict, reg: Registry, train_ids: list[str], *, batch: int, workers: int, seed: int, main: bool,
                 distributed: bool, synthetic: bool = False):
    """Training loader, fixed validation loader (main rank only; ``pick_val_volumes``), label statistics and the number
    of training scans, for the multitask and joint trainers. ``synthetic`` gives phantom volumes (CPU smoke runs)."""
    from torch.utils.data import DataLoader

    from brain.common.paths import resolve

    d, tk = cfg["data"], cfg["tasks"]
    if synthetic:
        shape = tuple(reversed(d["volume_size_xyz"]))
        ds = SyntheticMultitask(reg, shape, n=32, seed=seed)
        val = DataLoader(SyntheticMultitask(reg, shape, n=4, seed=999), batch_size=2, collate_fn=collate)
        return DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate, drop_last=True), val, {}, len(ds)
    store = resolve(d["store_root"])
    labels, stats = load_labels(resolve(d["labels_dir"]))
    volumes = pd.read_csv(store / d.get("index_name", "volumes.csv"), low_memory=False)
    ds = MultitaskDataset(store, labels, volumes, reg, stats, split="train", train=True, tasks=train_ids,
                          k_max=int(tk["k_max"]), p_all=float(tk["p_all"]), rare_beta=float(tk["rare_beta"]),
                          max_seg=int(tk["max_seg"]), group_weights=tk.get("group_weights"),
                          meta_drop=float((cfg["model"].get("meta") or {}).get("drop", 0.0)),
                          skull_strip=float(((cfg.get("aug") or {}).get("skull_strip") or {}).get("p", 0.0)))
    loader = make_loader(ds, batch_size=batch, shuffle=True, seed=seed, workers=workers,
                         balance_alpha=float(d.get("balance_alpha", 1.0)), balance_groups=d.get("balance_groups"),
                         distributed=distributed)
    val = None
    if main:
        pick = pick_val_volumes(labels, volumes, reg, train_ids, int(d.get("val_per_task", 64)))
        vds = MultitaskDataset(store, labels, volumes, reg, stats, split="val", train=False, tasks=train_ids, k_max=64,
                               max_seg=int(d.get("val_max_seg", 4)), volume_ids=pick)
        val = make_loader(vds, batch_size=2, shuffle=False, seed=0, workers=min(4, workers), drop_last=False)
    return loader, val, stats, len(ds)


def load_labels(labels_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    d = Path(labels_dir)
    return pd.read_parquet(d / "labels.parquet"), json.loads((d / "stats.json").read_text())


LEGACY_STATS = "stats_legacy.json"     # the label table's regression stats before the 2026-10-25 rebuild


def checkpoint_stats(saved: dict | None, labels_dir: str | Path, current: dict) -> dict:
    """The regression-target normalisation a checkpoint was trained with: ``saved`` in it (``label_stats``, runs from
    2026-10-25 on), else the frozen pre-rebuild copy (``stats_legacy.json``), else the label table's current stats.
    The loader standardises targets with these: other stats than the training ones bias the comparison."""
    if saved:
        return dict(saved)
    legacy = Path(labels_dir) / LEGACY_STATS
    return json.loads(legacy.read_text()) if legacy.exists() else current


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
                "volume_id": f"synthetic:{idx}", "thick": torch.tensor([1.0, -1.0])}
