"""Whole-volume batches for BrainQuery: image, the tasks asked of it, and their targets.

Each training sample asks a random subset of the scan's labelled (trained) tasks: ``k`` uniform in
``1..min(n_available, k_max)``, or every available task (up to ``k_max``) with probability ``p_all``. Tasks are drawn
without replacement, group first: each draw picks one of the scan's task groups (registry ``group``) with weight
``group_weights[group]`` (default 1), then a task inside it with weight ``n_task ** -rare_beta`` (rare tasks asked
more often than their label count alone would). So the 30 SynthSeg tasks are two groups among several, not 30 of the
draws. At most ``max_seg`` segmentation tasks per sample (each runs the decoder once; dense tasks count against it
too unless ``max_dense`` gives them their own budget). The draw order is random, so
slots carry no order information.

Workers read the image (and the brain mask / segmentation masks the drawn tasks need), compute live targets, and ship
raw intensities; ``prepare`` (GPU) flips and rotates image and masks together, gates foreground tokens on the raw
values and windows the image. Regression targets are standardised with ``stats.json``.

Self-supervised tasks (registry ``ssl``, run option ``tasks.ssl``) are spatial ``dense`` tasks. With probability
``tasks.mask_p`` a sample that has a ``masked_recon`` row becomes a masked sample (always, when that is its only task):
it asks reconstruction plus its other self-supervised tasks only, and ``prepare`` blanks ``tasks.mask_ratio`` of its
foreground tokens (SimMIM-style: the encoder sees the blank tokens; the decoder's image pyramid sees the blanked image),
keeping the unblanked encoder input as the target. A ``synth_*`` row loads the partner contrast (identical grid) as raw
intensities; ``prepare`` moves it with the image and windows it in its modality's canonical window.
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
from brain.multitask.registry import SPATIAL_TYPES, Registry

TYPE_ID = {"binary": 0, "regression": 1, "multiclass": 2, "segmentation": 3, "survival": 4, "dense": 5}
RECON, NO_DENSE = -2, -1        # ``dense_slot`` codes: reconstruction (target from the image) / not a dense task


class MultitaskDataset(Dataset):
    """One item per volume with at least one label among ``tasks``.

    ``train=True`` draws a task subset per item; ``train=False`` asks every available task in registry order
    (deterministic), capped at ``k_max`` / ``max_seg``.
    """

    def __init__(self, store_root: str | Path, labels: pd.DataFrame, volumes: pd.DataFrame, reg: Registry,
                 stats: dict, *, split: str | list[str], tasks: list[str] | None = None, train: bool = True,
                 k_max: int = 8, p_all: float = 0.2, rare_beta: float = 0.5, max_seg: int = 2,
                 use: str | None = "train", volume_ids: list[str] | None = None,
                 group_weights: dict[str, float] | None = None, meta_drop: float = 0.0, skull_strip: float = 0.0,
                 mask_p: float = 0.0, max_dense: int | None = None, task_aware: bool = False,
                 slices: dict | None = None, boxes: dict | None = None):
        """``meta_drop``: chance (training) that a scan's metadata id is replaced by unknown. ``skull_strip``: chance
        (training) that an MRI scan with a QC-passed SynthStrip mask is skull-stripped, unless a drawn task
        ``needs_skull``: tumour / stroke cohorts arrive stripped, so stripping is otherwise a cohort fingerprint.
        ``slices``: RSNA slice labels per volume id (``load_slices``), passed through to ``prepare``."""
        self.store_root = Path(store_root)
        self.slices, self.boxes = slices, boxes
        self.meta_drop, self.skull_strip, self.mask_p = float(meta_drop), float(skull_strip), float(mask_p)
        self.ssl_tasks = {i for i, t in enumerate(reg.tasks) if t.ssl}
        # stripping removes the skull and the vessels outside the brain mask: never on a scan asked a skull task, nor
        # (``task_aware``) a fine one
        self.skull_tasks = {i for i, t in enumerate(reg.tasks) if t.needs_skull or (task_aware and t.fine)}
        self.reg, self.train = reg, bool(train)
        self.k_max, self.p_all, self.max_seg = int(k_max), float(p_all), int(max_seg)
        # dense (self-supervised) tasks' own decoder budget; None: they share ``max_seg`` with segmentation
        self.max_dense = None if max_dense is None else int(max_dense)
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

    def _spatial(self, t: int) -> bool:
        return self.reg[t].type in SPATIAL_TYPES

    def _budget(self, t: int) -> str | None:
        """The decoder budget task ``t`` draws on: ``seg`` (``max_seg``), ``dense`` (``max_dense``) or None."""
        kind = self.reg[t].type
        if kind == "dense" and self.max_dense is not None:
            return "dense"
        return "seg" if kind in SPATIAL_TYPES else None

    def _draw(self, avail: list[tuple[int, float, str]], rng: np.random.Generator,
              taken: dict[str, int] | None = None) -> list[tuple[int, float, str]]:
        """``taken``: decoder slots already asked of the sample per budget, counted against ``k_max`` and the
        budgets."""
        taken = taken or {}
        if not self.train:
            # every non-spatial task, plus ``max_seg`` spatial (segmentation / dense) tasks taken from a per-volume
            # rotation of the volume's list (``rng`` is seeded by the item index), so a validation set covers all of them
            avail = sorted(avail)
            seg = [a for a in avail if self._spatial(a[0])]
            if len(seg) > self.max_seg:
                start = int(rng.integers(len(seg)))
                seg = (seg[start:] + seg[:start])[: self.max_seg]
            return ([a for a in avail if not self._spatial(a[0])] + seg)[: self.k_max]
        cap = min(len(avail), self.k_max - sum(taken.values()))
        if cap <= 0:
            return []
        k = cap if rng.random() < self.p_all else int(rng.integers(1, cap + 1))
        left: dict[str, list] = {}
        for a in avail:
            left.setdefault(self.group_of[a[0]], []).append(a)
        out, used = [], {"seg": taken.get("seg", 0), "dense": taken.get("dense", 0)}
        limit = {"seg": self.max_seg, "dense": self.max_dense if self.max_dense is not None else self.max_seg}
        while len(out) < k:
            spent = {b for b, n in used.items() if n >= limit[b]}
            if spent:                                     # a decoder budget spent: its tasks leave the draw
                left = {g: [a for a in v if self._budget(a[0]) not in spent] for g, v in left.items()}
            left = {g: v for g, v in left.items() if v}
            if not left:
                break
            groups = list(left)
            gw = np.asarray([self.group_weight.get(g, 1.0) for g in groups])
            g = groups[rng.choice(len(groups), p=gw / gw.sum())]
            tw = np.asarray([self.task_weight.get(a[0], 1.0) for a in left[g]])
            a = left[g].pop(int(rng.choice(len(tw), p=tw / tw.sum())))
            b = self._budget(a[0])
            if b:
                used[b] += 1
            out.append(a)
        return out

    def _draw_sample(self, avail: list[tuple[int, float, str]],
                     rng: np.random.Generator) -> tuple[list[tuple[int, float, str]], bool]:
        """The asked tasks and whether the sample is masked (see the module docstring). Validation never masks."""
        recon = [a for a in avail if a[2] == "@recon"]
        rest = [a for a in avail if a[2] != "@recon"]
        if not (self.train and recon and self.mask_p > 0 and (not rest or rng.random() < self.mask_p)):
            return self._draw(rest, rng), False
        return recon[:1] + self._draw([a for a in rest if a[0] in self.ssl_tasks], rng,
                                      taken={self._budget(recon[0][0]): 1}), True

    def __getitem__(self, idx: int) -> dict:
        from brain.prep import store

        vid, avail = self.items[int(idx)]
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))) if self.train else idx)
        drawn, masked = self._draw_sample(avail, rng)
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
        dense_slot, dense_raw, dense_mod = [], [], []
        mask_cache: dict[str, np.ndarray] = {}
        for t, y, ref in drawn:
            dslot = NO_DENSE
            if ref == "@recon":
                dslot = RECON
            elif ref.startswith("@synth:"):
                path, pmod = ref.split(":", 1)[1].rsplit("|", 1)
                other = store.read_image(self.store_root / path)
                if other.shape != img.shape:
                    continue
                dslot = len(dense_raw)
                dense_raw.append(other)
                dense_mod.append(0 if pmod.startswith("ct") else 1)
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
                    path, ids, *ign = ref.split("|")         # ``path|ids[|ignored ids]``
                    if path not in mask_cache:
                        mask_cache[path] = store.read_image(self.store_root / path, "mask")
                    m = np.isin(mask_cache[path], [int(i) for i in ids.split(",")]).astype(np.uint8)
                    if ign:                                  # voxels left out of the loss (2): an annotator's "ignore"
                        m[np.isin(mask_cache[path], [int(i) for i in ign[0].split(",")]) & (m == 0)] = 2
                if m.shape != img.shape:
                    continue
                slot = len(seg_masks)
                seg_masks.append(torch.from_numpy(np.ascontiguousarray(m, dtype=np.uint8)))
                y = float("nan")
            tasks.append(t)
            ys.append(float("nan") if dslot != NO_DENSE else y)
            seg_slot.append(slot)
            dense_slot.append(dslot)
        if strip:                       # after the live targets (measured inside the brain mask, so unchanged)
            brain = np.asarray(arrays["brain"]) > 0
            img = np.where(brain, img, img.min())
            dense_raw = [np.where(brain, o, o.min()) for o in dense_raw]     # a synthesis target keeps the input's FOV
        meta_ = 0 if self.train and self.meta_drop > 0 and rng.random() < self.meta_drop else self.meta_ids[idx]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)),           # (Z, Y, X) raw, stored dtype
            "modality": 0 if mod.startswith("ct") else 1,
            "task": torch.tensor(tasks, dtype=torch.long),
            "y": torch.tensor(ys, dtype=torch.float32),
            "seg_slot": torch.tensor(seg_slot, dtype=torch.long),
            "seg": torch.stack(seg_masks) if seg_masks else torch.zeros(0, *img.shape, dtype=torch.uint8),
            "dense_slot": torch.tensor(dense_slot, dtype=torch.long),
            "dense": torch.stack([torch.from_numpy(np.ascontiguousarray(o, dtype=np.float16)) for o in dense_raw])
            if dense_raw else torch.zeros(0, *img.shape, dtype=torch.float16),
            "dense_mod": torch.tensor(dense_mod, dtype=torch.long),
            "masked": masked,
            "volume_id": vid,
            "thick": torch.tensor(self.thick[idx], dtype=torch.float32),     # (mm, grid axis or -1)
            "meta": meta_,
            "slab": self.slices.get(vid) if self.slices else None,
            "boxes": self.boxes.get(vid) if self.boxes else None,
        }


def collate(batch: list[dict]) -> dict:
    """Pads slots to the batch's longest task list (``task == -1``); segmentation masks are stacked over the batch
    with ``seg_pair`` (P, 2) = (row, slot) for each; dense slots likewise as ``dense_pair`` (D, 2), with ``dense_src``
    (D,) the index of the slot's raw target in ``dense`` (-1: reconstruction) and ``dense_row`` / ``dense_mod`` per
    raw target."""
    k = max(max(len(b["task"]), 1) for b in batch)
    b_n = len(batch)
    task = torch.full((b_n, k), -1, dtype=torch.long)
    y = torch.full((b_n, k), float("nan"))
    pairs, segs = [], []
    dpairs, dsrc, draws, drows, dmods = [], [], [], [], []
    for i, b in enumerate(batch):
        n = len(b["task"])
        task[i, :n], y[i, :n] = b["task"], b["y"]
        for j in range(n):
            s = int(b["seg_slot"][j])
            if s >= 0:
                pairs.append((i, j))
                segs.append(b["seg"][s])
            d = int(b["dense_slot"][j]) if "dense_slot" in b else NO_DENSE
            if d != NO_DENSE:
                dpairs.append((i, j))
                dsrc.append(-1 if d == RECON else len(draws))
                if d >= 0:
                    draws.append(b["dense"][d])
                    drows.append(i)
                    dmods.append(int(b["dense_mod"][d]))
    shape = batch[0]["image"].shape
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "modality": torch.tensor([b["modality"] for b in batch]),
        "task": task, "y": y,
        "seg": torch.stack(segs) if segs else torch.zeros(0, *shape, dtype=torch.uint8),
        "seg_pair": torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
        "dense_pair": torch.tensor(dpairs, dtype=torch.long).reshape(-1, 2),
        "dense_src": torch.tensor(dsrc, dtype=torch.long),
        "dense": torch.stack(draws) if draws else torch.zeros(0, *shape, dtype=torch.float16),
        "dense_row": torch.tensor(drows, dtype=torch.long),
        "dense_mod": torch.tensor(dmods, dtype=torch.long),
        "masked": torch.tensor([bool(b.get("masked", False)) for b in batch]),
        "volume_id": [b["volume_id"] for b in batch],
        "thick": torch.stack([b.get("thick", torch.tensor([1.0, -1.0])) for b in batch]),
        "meta": torch.tensor([int(b.get("meta", 0)) for b in batch], dtype=torch.long),
        **collate_slabs([b.get("slab") for b in batch]),
        **collate_boxes([b.get("boxes") for b in batch]),
    }


# RSNA slice labels (brain.prep.rsna_slices): bit k of a slice's labels is task SLAB_TASKS[k]
SLAB_TASKS = ("hemorrhage_any", "hemorrhage_epidural", "hemorrhage_intraparenchymal", "hemorrhage_intraventricular",
              "hemorrhage_subarachnoid", "hemorrhage_subdural")
SLAB_NO_LABEL = 255


def load_slices(path: str | Path) -> dict:
    """volume id -> (g (3,), c, spacing, slice positions (L,), label bits (L,)) from rsna_slices.parquet. Studies whose
    scan origin is not on an end slice (rsna_slices_qc.csv beside it: 55 of 25k, mixed series the reader assembled
    differently) are left out: their slice-to-voxel map cannot be trusted."""
    d = pd.read_parquet(path)
    qc = Path(path).with_name(Path(path).stem + "_qc.csv")
    if qc.exists():
        q = pd.read_csv(qc)
        d = d[~d["volume_id"].isin(q.loc[q["origin_end_mm"] > 0.5, "volume_id"])]
    return {str(r.volume_id): (np.array([r.g_z, r.g_y, r.g_x], np.float32), float(r.c), float(r.spacing),
                               np.asarray(r.s, np.float32), np.asarray(r.bits, np.int16)) for r in d.itertuples()}


def collate_slabs(slabs: list) -> dict:
    """Per-row slice tables padded to the batch's longest (positions +inf, bits ``SLAB_NO_LABEL``); {} when no row has
    one."""
    if not any(s is not None for s in slabs):
        return {}
    n, L = len(slabs), max(len(s[3]) for s in slabs if s is not None)
    g, c, sp = torch.zeros(n, 3), torch.zeros(n), torch.ones(n)
    pos, bits = torch.full((n, L), float("inf")), torch.full((n, L), SLAB_NO_LABEL, dtype=torch.int16)
    for i, s in enumerate(slabs):
        if s is not None:
            g[i], c[i], sp[i] = torch.from_numpy(s[0]), s[1], s[2]
            pos[i, :len(s[3])], bits[i, :len(s[4])] = torch.from_numpy(s[3]), torch.from_numpy(s[4])
    return {"slab_has": torch.tensor([s is not None for s in slabs]), "slab_g": g, "slab_c": c, "slab_sp": sp,
            "slab_s": pos, "slab_bits": bits}


def token_probes(device, b: int, shape_zyx, patch_xyz, flip: torch.Tensor | None,
                 theta: torch.Tensor | None) -> tuple[torch.Tensor, int, int]:
    """Each encoder token probed at its centre and 8 corners, mapped back through this batch's affine (``theta``) and
    left-right ``flip`` to the stored grid: (B, N * 9, 3) stored z, y, x indices, N, 9."""
    px, py, pz = (int(v) for v in patch_xyz)
    Z, Y, X = (int(v) for v in shape_zyx)
    iz, iy, ix = torch.meshgrid(torch.arange(Z // pz, device=device), torch.arange(Y // py, device=device),
                                torch.arange(X // px, device=device), indexing="ij")
    ctr = torch.stack([iz * pz + (pz - 1) / 2, iy * py + (py - 1) / 2, ix * px + (px - 1) / 2], -1).reshape(-1, 3)
    half = torch.tensor([pz / 2 - 0.5, py / 2 - 0.5, px / 2 - 0.5], device=device)
    sgn = torch.tensor([[0, 0, 0]] + [[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], device=device)
    n_tok, n_probe = ctr.shape[0], sgn.shape[0]
    pts = (ctr.unsqueeze(1) + sgn * half).reshape(-1, 3).float()                 # (M, 3) z, y, x output voxels
    size = torch.tensor([Z, Y, X], device=device, dtype=torch.float32)
    xyz = ((2 * pts + 1) / size - 1).flip(-1).unsqueeze(0).expand(b, -1, -1)    # normalised x, y, z
    if theta is not None:
        xyz = torch.einsum("bij,bmj->bmi", theta[:, :, :3], xyz) + theta[:, :, 3].unsqueeze(1)
    src = ((xyz.flip(-1) + 1) * size - 1) / 2                                     # z, y, x in the flipped input
    if flip is not None:
        src[..., 2] = torch.where(flip.view(-1, 1), (X - 1) - src[..., 2], src[..., 2])
    return src, n_tok, n_probe


def slab_labels(batch: dict, device, shape_zyx, patch_xyz, flip: torch.Tensor | None,
                theta: torch.Tensor | None) -> torch.Tensor | None:
    """Per row, slice-label subtype (``SLAB_TASKS``) and encoder token: (B, 6, N) int16, -1 unlabelled, 0 a token in
    slices without that finding, r >= 1 in its r-th run of consecutive positive slices. Each token's probes
    (``token_probes``) go to the slice axis (``s = g . zyx + c``); a probe takes the nearest slice within half the
    spacing. A token is in a run if any probe is, negative if every probe is. None when no row has slice labels."""
    if "slab_has" not in batch or not bool(batch["slab_has"].any()):
        return None
    has = batch["slab_has"].to(device)
    src, n_tok, n_probe = token_probes(device, has.shape[0], shape_zyx, patch_xyz, flip, theta)
    s = (src * batch["slab_g"].to(device).unsqueeze(1)).sum(-1) + batch["slab_c"].to(device).unsqueeze(1)
    pos, bits = batch["slab_s"].to(device), batch["slab_bits"].to(device).long()
    L = pos.shape[1]
    j = torch.searchsorted(pos, s.contiguous())
    lo, hi = (j - 1).clamp(0, L - 1), j.clamp(0, L - 1)
    d_lo, d_hi = (s - pos.gather(1, lo)).abs(), (pos.gather(1, hi) - s).abs()
    near = torch.where(d_lo <= d_hi, lo, hi)
    ok = (torch.minimum(d_lo, d_hi) <= batch["slab_sp"].to(device).unsqueeze(1) / 2 + 0.25) \
        & (bits.gather(1, near) != SLAB_NO_LABEL) & has.unsqueeze(1)
    labelled = bits != SLAB_NO_LABEL
    out = []
    for k in range(len(SLAB_TASKS)):
        p = ((bits >> k) & 1).bool() & labelled
        start = p & ~torch.cat([torch.zeros_like(p[:, :1]), p[:, :-1]], 1)
        run = torch.cumsum(start.long(), 1) * p                                    # (B, L) run id per slice
        r = torch.where(ok, run.gather(1, near), torch.full_like(near, -1)).view(-1, n_tok, n_probe)
        lab = torch.where((r > 0).any(-1), r.amax(-1), torch.where((r == 0).all(-1), 0, -1))
        out.append(lab)
    return torch.stack(out, 1).to(torch.int16)


# fastMRI+ boxes (brain.prep.fastmri_labels): channel c is the task BOX_TASKS[c]; token-MIL channels are the
# slice-label subtypes, then these (TOKEN_MIL_TASKS)
BOX_TASKS = ("mri_wm_lesion", "mri_postop", "mri_mass", "mri_edema", "mri_infarct")
TOKEN_MIL_TASKS = SLAB_TASKS + BOX_TASKS


def load_boxes(path: str | Path, volumes: pd.DataFrame) -> dict:
    """volume id -> ((3, 4) stored index -> native index affine, native shape (k, j, i), boxes (n, 7): channel, then
    k / j / i lower and upper bounds) from fastmri_boxes.parquet (keyed by case id)."""
    d = pd.read_parquet(path)
    vid = volumes[volumes["dataset"].astype(str).str.startswith("fastmri")].drop_duplicates("case_id") \
        .set_index("case_id")["volume_id"]
    return {str(vid[r.case_id]): (np.asarray(r.affine, np.float32).reshape(3, 4), np.asarray(r.shape, np.float32),
                                  np.asarray(r.boxes, np.float32).reshape(-1, 7))
            for r in d.itertuples() if r.case_id in vid.index}


def collate_boxes(boxes: list) -> dict:
    """Per-row box tables padded to the batch's most boxes (channel -1); {} when no row has one. ``box_rank``: each
    box's 1-based rank within its channel (its bag id)."""
    if not any(x is not None for x in boxes):
        return {}
    n, nb = len(boxes), max([len(x[2]) for x in boxes if x is not None] + [1])
    aff, shape = torch.zeros(n, 3, 4), torch.ones(n, 3)
    tab, rank = torch.full((n, nb, 7), -1.0), torch.zeros(n, nb, dtype=torch.long)
    for i, x in enumerate(boxes):
        if x is None:
            continue
        aff[i], shape[i] = torch.from_numpy(x[0]), torch.from_numpy(x[1])
        if len(x[2]):
            t = torch.from_numpy(x[2])
            tab[i, :len(t)] = t
            seen: dict[int, int] = {}
            for j, c in enumerate(t[:, 0].long().tolist()):
                seen[c] = seen.get(c, 0) + 1
                rank[i, j] = seen[c]
    return {"box_has": torch.tensor([x is not None for x in boxes]), "box_aff": aff, "box_shape": shape,
            "box_tab": tab, "box_rank": rank}


def box_labels(batch: dict, device, shape_zyx, patch_xyz, flip: torch.Tensor | None,
               theta: torch.Tensor | None) -> torch.Tensor | None:
    """Per row, box channel (``BOX_TASKS``) and encoder token: (B, 5, N) int16, r >= 1 when the token overlaps the
    channel's r-th box, 0 when the token lies inside the scan's native field of view (all its ``token_probes``) and in
    none, -1 otherwise (or no box table). A box's 8 corners go forward (native -> stored grid -> this batch's flip and
    affine) and every token their bounding region touches counts as overlapping, so boxes smaller than a token's probe
    spacing (4 mm white-matter lesions) are not missed. The trainer turns 0 into unlabelled on scans positive for the
    finding (boxes are not exhaustive). None when no row has a box table."""
    if "box_has" not in batch or not bool(batch["box_has"].any()):
        return None
    has = batch["box_has"].to(device)
    b = has.shape[0]
    px, py, pz = (int(v) for v in patch_xyz)
    Z, Y, X = (int(v) for v in shape_zyx)
    gz, gy, gx = Z // pz, Y // py, X // px
    src, n_tok, n_probe = token_probes(device, b, shape_zyx, patch_xyz, flip, theta)
    aff = batch["box_aff"].to(device)
    kji = torch.einsum("bij,bmj->bmi", aff[:, :, :3], src) + aff[:, :, 3].unsqueeze(1)        # (B, M, 3)
    shape = batch["box_shape"].to(device)
    inside = ((kji >= -0.5) & (kji <= shape.unsqueeze(1) - 0.5)).all(-1) & has.unsqueeze(1)
    base = torch.where(inside.view(b, n_tok, n_probe).all(-1), 0, -1)                         # (B, N)
    out = base.unsqueeze(1).repeat(1, len(BOX_TASKS), 1).view(b, len(BOX_TASKS), gz, gy, gx).clone()
    tab, rank = batch["box_tab"].cpu(), batch["box_rank"].cpu()
    size = torch.tensor([Z, Y, X], dtype=torch.float64)
    corners = torch.tensor([[a, c, e] for a in (0, 1) for c in (0, 1) for e in (0, 1)], dtype=torch.long)
    for i in torch.nonzero(has).flatten().tolist():
        rows = tab[i][tab[i, :, 0] >= 0]
        if not len(rows):
            continue
        a = batch["box_aff"][i].double()
        a_inv = torch.linalg.inv(a[:, :3])
        th = theta[i].double().cpu() if theta is not None else None
        for j, bx in enumerate(rows):
            lo, hi = bx[[1, 3, 5]].double(), bx[[2, 4, 6]].double()
            pts = torch.where(corners.bool(), hi, lo)                                          # (8, 3) native k, j, i
            zyx = (pts - a[:, 3]) @ a_inv.T                                                     # stored z, y, x
            if flip is not None and bool(flip[i]):
                zyx[:, 2] = (X - 1) - zyx[:, 2]
            if th is not None:                       # input -> output: invert the sampling affine (normalised x, y, z)
                n_in = ((2 * zyx + 1) / size - 1).flip(-1)
                n_out = (n_in - th[:, 3]) @ torch.linalg.inv(th[:, :3]).T
                zyx = (((n_out.flip(-1) + 1) * size) - 1) / 2
            t_lo = torch.floor(zyx.min(0).values / torch.tensor([pz, py, px])).long()
            t_hi = torch.floor((zyx.max(0).values - 1e-3) / torch.tensor([pz, py, px])).long()   # upper bound exclusive
            t_lo = torch.maximum(t_lo, torch.zeros(3, dtype=torch.long))
            t_hi = torch.minimum(t_hi, torch.tensor([gz - 1, gy - 1, gx - 1]))
            if (t_hi < t_lo).any():
                continue                                                                        # outside the grid
            c, r = int(bx[0]), int(rank[i][j])
            v = out[i, c, t_lo[0]:t_hi[0] + 1, t_lo[1]:t_hi[1] + 1, t_lo[2]:t_hi[2] + 1]
            out[i, c, t_lo[0]:t_hi[0] + 1, t_lo[1]:t_hi[1] + 1, t_lo[2]:t_hi[2] + 1] = torch.clamp(v, min=r) \
                if r > 0 else v
    return out.view(b, len(BOX_TASKS), -1).to(torch.int16)


def token_mil_labels(batch: dict, device, shape_zyx, patch_xyz, flip, theta) -> torch.Tensor | None:
    """(B, len(TOKEN_MIL_TASKS), N): the slice-label channels, then the box channels (-1 where a row has neither)."""
    slab = slab_labels(batch, device, shape_zyx, patch_xyz, flip, theta)
    box = box_labels(batch, device, shape_zyx, patch_xyz, flip, theta)
    if slab is None and box is None:
        return None
    ref = slab if slab is not None else box
    fill = lambda k: torch.full((ref.shape[0], k, ref.shape[2]), -1, dtype=torch.int16, device=device)  # noqa: E731
    return torch.cat([slab if slab is not None else fill(len(SLAB_TASKS)),
                      box if box is not None else fill(len(BOX_TASKS))], 1)


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
                      ct_hu=tuple(gate.get("ct_hu", (-500.0, 200.0))), mri_lo=float(gate.get("mri_lo", -5.5)),
                      min_frac=float(gate.get("min_frac", 0.25))).reshape(raw.shape[0], -1)   # gate on (B,1,X,Y,Z)


def gate_cfg(cfg: dict) -> dict:
    """The run's token gate: ``views.gate`` (what the frozen benchmark reads too), else the legacy ``model.gate``."""
    return (cfg.get("views") or {}).get("gate") or (cfg.get("model") or {}).get("gate") or {}


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
        tgt = torch.where(below[rows], torch.zeros((), device=seg.device), seg[:, :1])   # validity channel kept
        seg = torch.cat([tgt, seg[:, 1:]], 1)
    return raw, seg


def token_voxels(m: torch.Tensor, shape_zyx, patch_xyz) -> torch.Tensor:
    """Token mask (B, N) in the encoder's (z, y, x) token order -> voxel mask (B, Z, Y, X)."""
    px, py, pz = (int(v) for v in patch_xyz)
    z, y, x = (int(v) for v in shape_zyx)
    g = m.view(m.shape[0], z // pz, y // py, x // px)
    return g.repeat_interleave(pz, 1).repeat_interleave(py, 2).repeat_interleave(px, 3)


def dense_targets(batch: dict, x: torch.Tensor, raw_dense: torch.Tensor, hide: torch.Tensor | None, patch_xyz,
                  gate: dict | None, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Targets and voxel weights (D, 1, Z/s, Y/s, X/s) of the dense slots at the decoder stride. Reconstruction: the
    unblanked encoder input's first channel, weighted on the hidden voxels (``hide``, (B, Z, Y, X)). Synthesis: the
    partner contrast (``raw_dense``, native units, already moved with the image) in its modality's canonical window,
    weighted on its foreground tokens."""
    src = batch["dense_src"].to(x.device)
    rows = batch["dense_pair"][:, 0].to(x.device)
    pool = (lambda a: F.avg_pool3d(a, stride)) if stride > 1 else (lambda a: a)
    shape = (len(src), 1, *(d // stride for d in x.shape[2:]))
    tgt = torch.zeros(shape, device=x.device)
    w = torch.zeros(shape, device=x.device)
    syn = src >= 0
    if bool(syn.any()):
        mod = batch["dense_mod"].to(x.device)
        fg = token_voxels(foreground_tokens(raw_dense, mod, patch_xyz, gate), x.shape[2:], patch_xyz)
        tgt[syn] = pool(canonical_window(raw_dense, mod))[src[syn]]
        w[syn] = pool(fg.unsqueeze(1).float())[src[syn]]
    rec = ~syn
    if bool(rec.any()) and hide is not None:
        r = rows[rec]
        tgt[rec] = pool(x[r, :1].float())
        w[rec] = pool(hide[r].unsqueeze(1).float())
    return tgt, w


def sharp_tasks(reg, cfg: dict) -> torch.Tensor | None:
    """Per registry task, whether a scan asked it must stay sharp (``Task.fine`` under ``aug.task_aware``): no
    through-plane or isotropic blur and no field-of-view cut (``prepare``); None when task-aware augmentation is off."""
    if not (cfg.get("aug") or {}).get("task_aware"):
        return None
    return torch.tensor([t.fine for t in reg.tasks])


def protected_tasks(reg) -> tuple[torch.Tensor, torch.Tensor]:
    """Per registry task, two bool tensors for ``prepare``: ``fov_unsafe`` (the target needs the whole brain in view:
    alignment-dependent and live targets; no field-of-view cut) and ``aug_unsafe`` (a live target, measured by
    ``labels.live_targets`` on the unaugmented, template-aligned image: no rotation / shift, bias field or drift, which
    would move the image away from its label; the mirror flip is fine, the measures are flip-invariant). A synthesis
    target is ``fov_unsafe`` too: the cut would blank the input but not the partner contrast."""
    from brain.multitask.labels import registration_sensitive

    live = [any(s.get("source") == "live" for s in t.sources) for t in reg.tasks]
    synth = [any(s.get("source") == "synth" for s in t.sources) for t in reg.tasks]
    fov = [registration_sensitive(t.id) or lv or sy for t, lv, sy in zip(reg.tasks, live, synth)]
    return torch.tensor(fov), torch.tensor(live)


def prepare(batch: dict, cfg: dict, device, patch_xyz, train: bool, fov_unsafe: torch.Tensor | None = None,
            aug_unsafe: torch.Tensor | None = None, sharp: torch.Tensor | None = None) -> dict:
    """Raw batch -> encoder input (B, C, Z, Y, X) windowed, foreground keep (B, N), and float segmentation targets
    moved with their image. Training: flip, rotation / shift (not on rows asking a task in ``aug_unsafe``), a
    field-of-view cut on rows asking no task in ``fov_unsafe`` (n_tasks bool: targets that need the whole brain in
    view; see ``protected_tasks``; None = every row is unsafe, no cut), then appearance augmentation
    (``aug.appearance``: bias / through-plane blur / noise / blur / drift, ``brain.data.augment``) with the window
    pinned to each modality's canonical one (frozen evaluation's; CT density targets live in it). The foreground
    gate is taken before appearance augmentation, so noise cannot flip background tokens. ``aug_unsafe`` rows also
    get no bias field or drift. A three-channel encoder (``model.in_chans`` 3) gets the three role windows, unjittered,
    with every appearance augmentation in native units (``augment``'s roles path: no drift). Dense slots: the partner
    contrasts move with their image; masked rows (training) get ``tasks.mask_ratio`` of their foreground tokens blanked
    after the reconstruction target is taken (``dense_targets``). Rows asking a task in ``sharp`` (``sharp_tasks``) get
    no blur and no field-of-view cut."""
    in_chans = int((cfg.get("model") or {}).get("in_chans", 1))
    aug = (cfg.get("aug") or {}) if train else {}
    raw = batch["image"].to(device, non_blocking=True).float().unsqueeze(1)
    mod = batch["modality"].to(device)
    b = raw.shape[0]
    task = batch["task"].to(device)
    asked = task >= 0
    live_rows = (aug_unsafe.to(device)[task.clamp_min(0)] & asked).any(1) if aug_unsafe is not None else None
    sharp_rows = (sharp.to(device)[task.clamp_min(0)] & asked).any(1) if sharp is not None else None
    seg = batch["seg"].to(device, non_blocking=True)
    # (P, 1, Z, Y, X) target; with ignored voxels (value 2) a second, validity channel rides through every spatial op
    seg = torch.stack([seg == 1, seg != 2], 1).float() if (seg == 2).any() else seg.float().unsqueeze(1)
    rows = batch["seg_pair"][:, 0].to(device)
    dense_pair = batch.get("dense_pair", torch.zeros(0, 2, dtype=torch.long)).to(device)
    raw_dense = batch["dense"].to(device, non_blocking=True).float().unsqueeze(1) if len(dense_pair) else None
    drows = batch["dense_row"].to(device) if raw_dense is not None else None
    flip = None
    if train and float(aug.get("flip_lr", 0.0)) > 0:
        flip = torch.rand(b, device=device) < float(aug["flip_lr"])
        raw = torch.where(flip.view(-1, 1, 1, 1, 1), raw.flip(-1), raw)
        if len(seg):
            seg = torch.where(flip[rows].view(-1, 1, 1, 1, 1), seg.flip(-1), seg)
        if raw_dense is not None and len(raw_dense):
            raw_dense = torch.where(flip[drows].view(-1, 1, 1, 1, 1), raw_dense.flip(-1), raw_dense)
    theta = augment_theta(b, raw.shape[2:], aug.get("affine") or {}, device, skip=live_rows) if train else None
    if theta is not None:
        bg = raw.amin(dim=(1, 2, 3, 4), keepdim=True)
        grid = F.affine_grid(theta, list(raw.shape), align_corners=False)
        raw = F.grid_sample(raw - bg, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + bg
        if len(seg):
            seg = F.grid_sample(seg, grid[rows], mode="bilinear", padding_mode="zeros", align_corners=False)
        if raw_dense is not None and len(raw_dense):
            dbg = raw_dense.amin(dim=(1, 2, 3, 4), keepdim=True)
            raw_dense = F.grid_sample(raw_dense - dbg, grid[drows], mode="bilinear", padding_mode="zeros",
                                      align_corners=False) + dbg
        del grid
    if train and (aug.get("fov_cut") or {}).get("p", 0.0):
        unsafe = fov_unsafe.to(device)[task.clamp_min(0)] & (task >= 0) if fov_unsafe is not None \
            else torch.ones_like(task, dtype=torch.bool)
        allowed = ~unsafe.any(1) if sharp_rows is None else ~unsafe.any(1) & ~sharp_rows
        for key in ("slab_has", "box_has"):    # a cut could hide a labelled finding from its bag
            if key in batch:
                allowed &= ~batch[key].to(device)
        raw, seg = fov_cut(raw, seg, rows, mod, allowed, aug["fov_cut"])
    keep = foreground_tokens(raw, mod, patch_xyz, gate_cfg(cfg))
    slab = token_mil_labels(batch, device, raw.shape[2:], patch_xyz, flip, theta)
    app = aug.get("appearance")
    if train and app:
        thick = batch["thick"].to(device) if "thick" in batch else None
        window = {"mode": "roles", "jitter": 0.0} if in_chans == 3 else {"mode": "fixed"}
        x = appearance(raw, mod, {**app, "window": window},
                       thick=(thick[:, 0], thick[:, 1].long()) if thick is not None else None, protect=live_rows,
                       sharp=sharp_rows)
    else:
        x = encoder_window(raw, mod, in_chans)
    hide = dense_tgt = dense_w = None
    tk = cfg.get("tasks") or {}
    if train and "masked" in batch and bool(batch["masked"].any()):
        masked = batch["masked"].to(device)
        tok = keep & masked.unsqueeze(1) & (torch.rand(keep.shape, device=device) < float(tk.get("mask_ratio", 0.6)))
        hide = token_voxels(tok, x.shape[2:], patch_xyz)
    if len(dense_pair):
        stride = int((cfg.get("loss") or {}).get("seg_stride", 2))
        dense_tgt, dense_w = dense_targets(batch, x, raw_dense, hide, patch_xyz, gate_cfg(cfg),
                                           stride)
    if hide is not None:                      # after the reconstruction target: blank voxels (0 in every channel)
        x = x.masked_fill(hide.unsqueeze(1), 0.0)
    return {"x": x, "keep": keep, "modality": mod, "task": task, "y": batch["y"].to(device),
            "seg": seg, "seg_pair": batch["seg_pair"].to(device),
            "dense_pair": dense_pair, "dense_tgt": dense_tgt, "dense_w": dense_w,
            "meta": batch["meta"].to(device) if "meta" in batch else None, "slab": slab}


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


def drop_datasets(labels: pd.DataFrame, volumes: pd.DataFrame, prefixes) -> pd.DataFrame:
    """Labels without the scans of datasets named by ``data.exclude_datasets`` (dataset-name prefixes): those scans
    leave training and validation entirely (every task, SSL included)."""
    if not prefixes:
        return labels
    ds = volumes.drop_duplicates("volume_id").set_index("volume_id")["dataset"].astype(str)
    gone = labels["volume_id"].map(ds).fillna("").str.startswith(tuple(prefixes))
    print(f"[data] exclude_datasets {list(prefixes)}: {int(gone.sum())} label rows, "
          f"{labels.loc[gone, 'volume_id'].nunique()} scans dropped", flush=True)
    return labels[~gone]


def task_loaders(cfg: dict, reg: Registry, train_ids: list[str], *, batch: int, workers: int, seed: int, main: bool,
                 distributed: bool, synthetic: bool = False):
    """Training loader, fixed validation loader (main rank only; ``pick_val_volumes``), label statistics and the number
    of training scans, for the multitask and joint trainers. ``synthetic`` gives phantom volumes (CPU smoke runs)."""
    from torch.utils.data import DataLoader

    from brain.common.paths import resolve

    d, tk = cfg["data"], cfg["tasks"]
    if synthetic:
        shape = tuple(reversed(d["volume_size_xyz"]))
        ds = SyntheticMultitask(reg, shape, n=32, seed=seed, tasks=train_ids)
        val = DataLoader(SyntheticMultitask(reg, shape, n=4, seed=999, tasks=train_ids), batch_size=2,
                         collate_fn=collate)
        return DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate, drop_last=True), val, {}, len(ds)
    store = resolve(d["store_root"])
    labels, stats = load_labels(resolve(d["labels_dir"]))
    volumes = pd.read_csv(store / d.get("index_name", "volumes.csv"), low_memory=False)
    labels = drop_datasets(labels, volumes, d.get("exclude_datasets"))
    ds = MultitaskDataset(store, labels, volumes, reg, stats, split="train", train=True, tasks=train_ids,
                          k_max=int(tk["k_max"]), p_all=float(tk["p_all"]), rare_beta=float(tk["rare_beta"]),
                          max_seg=int(tk["max_seg"]), group_weights=tk.get("group_weights"),
                          meta_drop=float((cfg["model"].get("meta") or {}).get("drop", 0.0)),
                          skull_strip=float(((cfg.get("aug") or {}).get("skull_strip") or {}).get("p", 0.0)),
                          mask_p=float(tk.get("mask_p", 0.0)),
                          max_dense=None if tk.get("max_dense") is None else int(tk["max_dense"]),
                          task_aware=bool((cfg.get("aug") or {}).get("task_aware", False)),
                          slices=load_slices(resolve(d["rsna_slices"])) if d.get("rsna_slices") else None,
                          boxes=load_boxes(resolve(d["fastmri_boxes"]), volumes) if d.get("fastmri_boxes") else None)
    loader = make_loader(ds, batch_size=batch, shuffle=True, seed=seed, workers=workers,
                         balance_alpha=float(d.get("balance_alpha", 1.0)), balance_groups=d.get("balance_groups"),
                         distributed=distributed)
    val = None
    if main:
        # reconstruction needs masking, which validation never applies
        val_ids = [t for t in train_ids if not any(s.get("source") == "ssl_recon" for s in reg[t].sources)]
        pick = pick_val_volumes(labels, volumes, reg, val_ids, int(d.get("val_per_task", 64)))
        vds = MultitaskDataset(store, labels, volumes, reg, stats, split="val", train=False, tasks=val_ids, k_max=64,
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
    "brain" on a dark background, a cube "lesion" as the segmentation target; a reconstruction task masks the sample, a
    synthesis task's partner is the phantom with inverted contrast."""

    def __init__(self, reg: Registry, shape_zyx, n: int = 64, k_max: int = 4, seed: int = 0,
                 tasks: list[str] | None = None):
        self.reg, self.shape, self.n, self.k_max, self.seed = reg, tuple(shape_zyx), int(n), int(k_max), int(seed)
        self.tasks = [reg.index[t] for t in tasks] if tasks is not None else \
            [reg.index[t.id] for t in reg.trained if not t.ssl]
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
        dslots, dense, masked = [], [], False
        for i in pick:
            t = self.tasks[i]
            kind = self.reg[t].type
            slot, dslot = -1, NO_DENSE
            if kind == "dense":
                if sum(d != NO_DENSE for d in dslots) + len(segs) >= 2:
                    continue
                v = float("nan")
                if any(s.get("source") == "ssl_recon" for s in self.reg[t].sources):
                    dslot, masked = RECON, True
                else:
                    dslot = len(dense)
                    dense.append(torch.where(brain, -1.0, -2.0) + 0.1 * torch.randn(self.shape, generator=g))
            elif kind == "binary":
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
            dslots.append(dslot)
        return {"image": img.half(), "modality": 1, "task": torch.tensor(tasks, dtype=torch.long),
                "y": torch.tensor(ys, dtype=torch.float32), "seg_slot": torch.tensor(slots, dtype=torch.long),
                "seg": torch.stack(segs) if segs else torch.zeros(0, *self.shape, dtype=torch.uint8),
                "dense_slot": torch.tensor(dslots, dtype=torch.long),
                "dense": torch.stack(dense).half() if dense else torch.zeros(0, *self.shape, dtype=torch.float16),
                "dense_mod": torch.ones(len(dense), dtype=torch.long), "masked": masked,
                "volume_id": f"synthetic:{idx}", "thick": torch.tensor([1.0, -1.0])}
