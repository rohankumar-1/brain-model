"""Segmentation on frozen 3D BrainModel tokens.

The encoder runs once per volume (one pass over the native 256 x 256 x 160 grid); its 20 x 16 x 16 token
map (one 16 x 16 x 8 patch each) is cached. A small 3D conv decoder, trained on the cached tokens alone,
upsamples the map to one logit per voxel. Nothing but the final tokens reaches the decoder: no raw
voxels and no intermediate blocks, so the decoder sees exactly the resolution the encoder hands over.

Dice is scored on 64 bin-centre axial slices (``dice_*``, the slices the retired 2D reference decoder used) and on
the whole volume (``dice_full_*``). Several sequences (BraTS) are encoded one at a time and their tokens are
concatenated.

    python -m brain.downstream.seg3d --task wmh --ckpt runs/snap/predict_8k.pt
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.prep import store
from brain.downstream.embed3d import FrozenEncoder3D, read_volume_xyz
from brain.downstream.embeddings import _fp, cache_fresh, load_frozen_encoder, resolve_path
from brain.downstream.metrics import bootstrap_ci
from brain.downstream.results import eval_banner, finalize_run, format_rows, new_run_dir, write_json
from brain.downstream.tasks import DEFAULT_VOLUMES, TaskBundle, load_task, load_volumes
from brain.common.optim import lr_at
from brain.common.windowing import windows_for
from brain.common.grid import DEPTH
from brain.common.paths import REPO_ROOT
from brain.prep.store import FORMAT_VERSION
from brain.models.conv3d import Res, group_norm, up3

VOX = (DEPTH, 256, 256)  # voxels (Z, Y, X)
PATCH = (16, 16, 8)  # the encoder's native patch (x, y, z)
TILE = ((128, 128, 64), (64, 64, 32))  # window and stride in voxels (x, y, z): the training crop size


BRATS_SEQUENCES = ("t1n", "t1c", "t2w", "t2f")


def uniform_slice_indices(depth: int, n: int) -> np.ndarray:
    """Increasing z indices, one at the centre of each of ``n`` equal bins."""
    depth, n = int(depth), int(n)
    if depth <= 0 or n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n >= depth:
        return np.arange(depth, dtype=np.int64)
    return ((np.arange(n, dtype=np.float64) + 0.5) * depth / n).astype(np.int64)


def sequence_paths(rel: str, all_sequences: bool) -> list[str]:
    """Image zips of one case: the four BraTS sequences, or just ``rel``."""
    if all_sequences and rel.endswith("-t1n.zip"):
        return [rel.replace("-t1n.zip", f"-{s}.zip") for s in BRATS_SEQUENCES]
    return [rel]


def region_seg_loss(logits: torch.Tensor, target: torch.Tensor, regions: dict[str, tuple[int, ...]]) -> torch.Tensor:
    """Cross-entropy on the label map plus soft Dice per named region (fp32).

    Labels outside the head's classes are ignored instead of crashing CE.
    """
    logits, target = logits.float(), target.long()
    target = target.masked_fill((target < 0) | (target >= logits.shape[1]), 255)
    ce = F.cross_entropy(logits, target, ignore_index=255)
    probs = logits.softmax(dim=1)
    dice = []
    for ids in regions.values():
        idx = [i for i in ids if i < probs.shape[1]]
        pred = probs[:, idx].sum(1)
        gt = torch.isin(target, torch.tensor(idx, device=target.device)).float()
        dice.append(1 - (2 * (pred * gt).sum() + 1) / (pred.sum() + gt.sum() + 1))
    return ce + torch.stack(dice).mean()


def grid_for(patch_xyz) -> tuple[int, int, int]:
    """Token grid (Z, Y, X) of a full volume."""
    px, py, pz = patch_xyz
    return VOX[0] // pz, VOX[1] // py, VOX[2] // px
# the axial slices the 2D reference decoder is scored on (64 bin-centre slices of the volume)
EVAL_SLICES = torch.from_numpy(uniform_slice_indices(DEPTH, 64)).long()
# Training copies of each volume besides the original: (shift in voxels (x, y, z), flip left-right). A shift off
# the patch grid or a flip gives the encoder a new tokenization of the same brain; the mask moves with it.
AUG = (None, ((8, 8, 4), False), ((0, 0, 0), True), ((-8, 8, -4), True))


def shift_fill(t: torch.Tensor, shifts, fill) -> torch.Tensor:
    """Move the last three axes by ``shifts`` (content to higher index for positive), padding with ``fill``."""
    out = torch.full_like(t, fill)
    src, dst = [], []
    for sh, n in zip(shifts, t.shape[-3:]):
        src.append(slice(max(0, -sh), n - max(0, sh)))
        dst.append(slice(max(0, sh), n - max(0, -sh)))
    out[(..., *dst)] = t[(..., *src)]
    return out


def aug_volume(vol_xyz: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return vol_xyz
    shift, flip = AUG[k]
    v = vol_xyz.flip(0) if flip else vol_xyz
    return shift_fill(v, shift, v.min()) if any(shift) else v


def aug_mask(mask_zyx: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return mask_zyx
    shift, flip = AUG[k]
    m = mask_zyx.flip(-1) if flip else mask_zyx
    return shift_fill(m, shift[::-1], 0) if any(shift) else m


def token_aug(tok: torch.Tensor, mask: torch.Tensor, k: int, patch_xyz) -> tuple[torch.Tensor, torch.Tensor]:
    """Cheap training augs on the cached token grid: bit 0 = flip X, bit 1 = roll one token in X.

    The mask moves by the same flip / by one patch in X. No extra encoded copies on disk.
    """
    if k == 0:
        return tok, mask
    nz, ny, nx = grid_for(patch_xyz)
    x = tok.view(nz, ny, nx, -1)
    if k & 1:
        x, mask = x.flip(2), mask.flip(-1)
    if k & 2:
        x, mask = torch.roll(x, 1, 2), torch.roll(mask, int(patch_xyz[0]), -1)
    return x.reshape(tok.shape), mask


def sample_crop(mask, patch_xyz, crop, rng, p_fg: float = 0.8) -> tuple[int, int, int]:
    """Token-grid origin of a ``crop`` cube; 80% of the time the cube covers a lesion token."""
    nz, ny, nx = grid_for(patch_xyz)
    cz, cy, cx = crop
    pz, py, px = patch_xyz[2], patch_xyz[1], patch_xyz[0]
    m = np.asarray(mask)
    fg = m.reshape(nz, pz, ny, py, nx, px).any(axis=(1, 3, 5))
    if fg.any() and rng.random() < p_fg:
        coords = np.argwhere(fg)
        z, y, x = (int(v) for v in coords[int(rng.integers(len(coords)))])
    else:
        z, y, x = int(rng.integers(nz)), int(rng.integers(ny)), int(rng.integers(nx))
    return (int(np.clip(z - cz // 2, 0, nz - cz)),
            int(np.clip(y - cy // 2, 0, ny - cy)),
            int(np.clip(x - cx // 2, 0, nx - cx)))


def crop_tokens(tok, mask, origin, crop, patch_xyz):
    """Slice tokens and the matching voxel mask to a token-grid cube."""
    nz, ny, nx = grid_for(patch_xyz)
    cz, cy, cx = crop
    z0, y0, x0 = origin
    pz, py, px = patch_xyz[2], patch_xyz[1], patch_xyz[0]
    c = tok.shape[-1]
    t = tok.view(nz, ny, nx, c)[z0:z0 + cz, y0:y0 + cy, x0:x0 + cx].reshape(cz * cy * cx, c)
    m = mask[z0 * pz:(z0 + cz) * pz, y0 * py:(y0 + cy) * py, x0 * px:(x0 + cx) * px]
    return t, m


# ----------------------------------------------------------------------------- decoder

class TokenDecoder3D(nn.Module):
    """Token-grid U-Net, then upsample to voxel logits.

    Two stride-2 stages on the token grid (with skips) keep local structure that a pure upsample stack
    throws away. The remaining stages double in-plane size — and depth until full — to the 160 x 256 x 256
    voxel grid. ``pe_dim`` fuses pre-transformer patch embeddings after the token-grid U-Net (a UNETR-style
    skip at the encoder's native resolution). ``deep_sup`` adds 1x1 heads at the token grid and after the
    first voxel upsample.
    """

    def __init__(self, in_dim: int, n_classes: int, grid=grid_for(PATCH), widths=(128, 96, 64, 32, 16),
                 dropout: float = 0.0, deep_sup: bool = False, out_vox=VOX, pe_dim: int = 0):
        super().__init__()
        self.grid = tuple(grid)
        self.out_vox = tuple(out_vox)
        self.deep_sup = deep_sup
        self.pe_dim = int(pe_dim)
        w0 = widths[0]
        self.n_down = min(2, max(0, int(np.log2(min(self.grid))) - 1))
        n_z, n_xy = int(np.log2(out_vox[0] // grid[0])), int(np.log2(out_vox[1] // grid[1]))
        steps = [(2 if i < n_z else 1, 2, 2) for i in range(n_xy)]
        vox_w = [w0] + list(widths[len(widths) - n_xy:])
        self.drop = nn.Dropout(dropout)
        self.stem = nn.Sequential(nn.Conv3d(in_dim, w0, 1, bias=False), group_norm(w0), nn.GELU(), Res(w0, w0))
        self.downs = nn.ModuleList(
            [nn.Sequential(nn.Conv3d(w0, w0, 3, stride=2, padding=1, bias=False), group_norm(w0), nn.GELU(), Res(w0, w0))
             for _ in range(self.n_down)]
        )
        self.enc_ups = nn.ModuleList([up3(w0, w0, (2, 2, 2)) for _ in range(self.n_down)])
        self.fuses = nn.ModuleList([Res(2 * w0, w0) for _ in range(self.n_down)])
        self.pe_proj = nn.Sequential(nn.Conv3d(self.pe_dim, w0, 1, bias=False), group_norm(w0), nn.GELU()) if self.pe_dim else None
        self.pe_fuse = Res(2 * w0, w0) if self.pe_dim else None
        self.stages = nn.ModuleList(
            [nn.Sequential(up3(vox_w[i], vox_w[i + 1], s), Res(vox_w[i + 1], vox_w[i + 1])) for i, s in enumerate(steps)]
        )
        self.head = nn.Conv3d(vox_w[-1], n_classes, 1)
        self.aux_tok = nn.Conv3d(w0, n_classes, 1) if deep_sup else None
        self.aux_mid = nn.Conv3d(vox_w[1], n_classes, 1) if deep_sup else None

    def forward(self, tokens: torch.Tensor, grid=None):
        grid = tuple(grid or self.grid)
        if self.pe_dim:
            pe, tokens = tokens[..., -self.pe_dim:], tokens[..., :-self.pe_dim]
        else:
            pe = None
        b, n, c = tokens.shape
        x = self.drop(tokens).transpose(1, 2).reshape(b, c, *grid)
        x = self.stem(x)
        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x)
        for up, fuse, skip in zip(reversed(self.enc_ups), reversed(self.fuses), reversed(skips)):
            x = fuse(torch.cat([up(x), skip], 1))
        if pe is not None:
            x = self.pe_fuse(torch.cat([x, self.pe_proj(pe.transpose(1, 2).reshape(b, self.pe_dim, *grid))], 1))
        aux_tok = self.aux_tok(x) if self.aux_tok is not None else None
        aux_mid = None
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == 0 and self.aux_mid is not None:
                aux_mid = self.aux_mid(x)
        logits = self.head(x)
        if self.deep_sup:
            return logits, (aux_tok, aux_mid)
        return logits


class PatchFCN(nn.Module):
    """Classify each token, then nearest-upsample to voxels (PatchFCN / linear probe with a small FCN).

    The loss is the usual voxel CE+Dice on the expanded map, so Dice is comparable to the U-Net decoder.
    """

    def __init__(self, in_dim: int, n_classes: int, grid=grid_for(PATCH), patch_xyz=PATCH, dropout: float = 0.0,
                 width: int = 128, out_vox=VOX):
        super().__init__()
        self.grid = tuple(grid)
        self.patch_xyz = tuple(patch_xyz)
        self.out_vox = tuple(out_vox)
        self.drop = nn.Dropout(dropout)
        self.net = nn.Sequential(
            nn.Conv3d(in_dim, width, 1, bias=False), group_norm(width), nn.GELU(),
            Res(width, width), Res(width, width),
            nn.Conv3d(width, n_classes, 1),
        )

    def forward(self, tokens: torch.Tensor, grid=None):
        grid = tuple(grid or self.grid)
        b, n, c = tokens.shape
        x = self.drop(tokens).transpose(1, 2).reshape(b, c, *grid)
        logits = self.net(x)
        pz, py, px = self.patch_xyz[2], self.patch_xyz[1], self.patch_xyz[0]
        return F.interpolate(logits, size=(grid[0] * pz, grid[1] * py, grid[2] * px), mode="nearest")


def _decode_loss(out, y, regions):
    """CE+Dice on the final logits; optional deep-sup terms are resized to the target crop."""
    if isinstance(out, tuple):
        logits, aux = out
        loss = region_seg_loss(logits, y, regions)
        for a, w in zip(aux, (0.4, 0.4)):
            if a is None:
                continue
            a = F.interpolate(a.float(), size=y.shape[-3:], mode="trilinear", align_corners=False)
            loss = loss + w * region_seg_loss(a, y, regions)
        return logits, loss
    return out, region_seg_loss(out, y, regions)


# ----------------------------------------------------------------------------- cache

def _meta_ok(meta: Path, want: dict, data: Path, ckpt: Path) -> bool:
    """The cache's recorded settings match and it was built from the checkpoint's current weights."""
    got = json.loads(meta.read_text())
    fp = got.pop("ckpt_fingerprint", None)
    return got == want and cache_fresh(data, ckpt, fp)


def _wait_meta(path: Path, meta: Path, label: str, tries: int = 720) -> None:
    if path.exists() and not meta.exists():
        print(f"[seg3d-cache] waiting for {label} (another job is writing it)", flush=True)
        for _ in range(tries):
            time.sleep(10)
            if meta.exists():
                break
        else:
            raise SystemExit(f"[seg3d-cache] timed out waiting for {path}")


class Cache:
    """``feats[i]`` are the tokens of volume ``i``; ``aug[row[i], k - 1]`` those of its training copy ``k``.

    ``pe`` is the matching pre-transformer patch embedding (same grid), concatenated by ``tokens()``.
    """

    def __init__(self, ids, feats, masks, aug=None, row=None, pe=None, pe_aug=None):
        self.volume_ids, self.feats, self.masks, self.aug, self.row = ids, feats, masks, aug, row or {}
        self.pe, self.pe_aug = pe, pe_aug

    @property
    def n_aug(self) -> int:
        return 1 if self.aug is None else 1 + self.aug.shape[1]

    @property
    def pe_dim(self) -> int:
        return 0 if self.pe is None else int(self.pe.shape[-1])

    def tokens(self, i: int, k: int = 0):
        t = self.feats[i] if k == 0 else self.aug[self.row[i], k - 1]
        if self.pe is None:
            return t
        p = self.pe[i] if k == 0 else self.pe_aug[self.row[i], k - 1]
        return np.concatenate([np.asarray(t), np.asarray(p)], -1)


def _read(task: TaskBundle, vid: str, slices_root: Path, need_mask: bool):
    paths = sequence_paths(task.image_paths[vid], True) if task.name.startswith("brats") else [task.image_paths[vid]]
    vols = [read_volume_xyz(resolve_path(slices_root, p)) for p in paths]
    mask = None
    if need_mask:
        mask = store.read(resolve_path(slices_root, task.mask_paths[vid]), "mask")
        if mask.shape != VOX:
            raise SystemExit(f"{vid}: mask {mask.shape} is not the {VOX} grid")
    return vols, mask


def build_cache(task: TaskBundle, enc: FrozenEncoder3D, ckpt: Path, slices_root: Path, cache_dir: Path,
                device: torch.device, n_aug: int = len(AUG), threads: int = 6, tile: bool = False,
                pe: bool = False) -> Cache:
    ids = [str(v) for v in task.volume_ids]
    train_ids = [v for v in ids if task.splits[v] == "train"]
    row = {ids.index(v): j for j, v in enumerate(train_ids)} if n_aug > 1 else {}
    root = Path(cache_dir) / "_seg3d" / task.name
    root.mkdir(parents=True, exist_ok=True)
    taps = tuple(enc.tap_blocks)
    patch = tuple(enc.patch_xyz)
    n_tok = int(np.prod(grid_for(patch)))
    tag = (f"{ckpt.parent.name}_{ckpt.stem}" + ("_tap" + "-".join(map(str, taps)) if taps else "")
           + ("" if patch == PATCH else "_p" + "x".join(map(str, patch))) + ("_tiled" if tile else ""))
    pe_tag = (f"{ckpt.parent.name}_{ckpt.stem}"
              + ("" if patch == PATCH else "_p" + "x".join(map(str, patch))) + ("_tiled" if tile else ""))
    n_seq = 4 if task.name.startswith("brats") else 1
    mpath, fpath, apath, meta = root / "masks.npy", root / f"{tag}.npy", root / f"{tag}_aug.npy", root / f"{tag}.json"
    ppath, papath, pmeta = root / f"{pe_tag}_pe.npy", root / f"{pe_tag}_pe_aug.npy", root / f"{pe_tag}_pe.json"
    want = {"volume_ids": ids, "n_seq": n_seq, "dim": enc.dim, "taps": list(taps), "patch": list(patch), "tile": tile, "n_aug": n_aug, "train": train_ids}
    pe_want = {"volume_ids": ids, "n_seq": n_seq, "dim": enc.dim, "patch": list(patch), "tile": tile, "n_aug": n_aug, "train": train_ids, "kind": "patch"}
    have_masks = mpath.exists() and (root / "masks.json").exists() and json.loads((root / "masks.json").read_text()) == ids
    _wait_meta(fpath, meta, fpath.name)
    if pe:
        _wait_meta(ppath, pmeta, ppath.name)
    have_tok = fpath.exists() and meta.exists() and _meta_ok(meta, want, fpath, ckpt)
    have_pe = (not pe) or (ppath.exists() and pmeta.exists() and _meta_ok(pmeta, pe_want, ppath, ckpt))
    if have_masks and have_tok and have_pe:
        return Cache(ids, np.load(fpath, mmap_mode="r"), np.load(mpath, mmap_mode="r"),
                     np.load(apath, mmap_mode="r") if n_aug > 1 else None, row,
                     np.load(ppath, mmap_mode="r") if pe else None,
                     np.load(papath, mmap_mode="r") if pe and n_aug > 1 else None)
    need_tok, need_pe, need_mask = not have_tok, pe and not have_pe, not have_masks
    masks = None if not need_mask else np.lib.format.open_memmap(mpath, "w+", np.uint8, (len(ids), *VOX))
    dim = enc.dim * (1 + len(taps)) * n_seq  # taps (shallow to deep), then the final tokens, per sequence
    pe_dim = enc.dim * n_seq
    feats = np.lib.format.open_memmap(fpath, "w+", np.float16, (len(ids), n_tok, dim)) if need_tok else None
    aug = (np.lib.format.open_memmap(apath, "w+", np.float16, (len(train_ids), n_aug - 1, n_tok, dim))
           if need_tok and n_aug > 1 else None)
    pe_feats = np.lib.format.open_memmap(ppath, "w+", np.float16, (len(ids), n_tok, pe_dim)) if need_pe else None
    pe_aug = (np.lib.format.open_memmap(papath, "w+", np.float16, (len(train_ids), n_aug - 1, n_tok, pe_dim))
              if need_pe and n_aug > 1 else None)
    mod = task.modality_filter or ("mri" if task.name.startswith("brats") else "ct")
    c, w = windows_for("canonical", mod)[0]
    tile_kw = dict(window_xyz=TILE[0], stride_xyz=TILE[1]) if tile else {}

    def encode_tok(vols, k):
        vs = [aug_volume(v, k) for v in vols]
        if tile:
            toks = [enc.embed_tokens(v, c, w, modality=mod, **tile_kw)[0] for v in vs]
        elif len(vs) == 1:
            toks = [enc.embed_tokens(vs[0], c, w, modality=mod)[0]]
        else:
            toks = list(enc.embed_volumes(vs, c, w, mod))
        return torch.cat(toks, -1).half().cpu().numpy()

    def encode_pe(vols, k):
        vs = [aug_volume(v, k) for v in vols]
        if tile:
            pes = [enc.embed_patch(v, c, w, modality=mod, **tile_kw) for v in vs]
        elif len(vs) == 1:
            pes = [enc.embed_patch(vs[0], c, w, modality=mod)]
        else:
            pes = list(enc.embed_patches(vs, c, w, mod))
        return torch.cat(pes, -1).half().cpu().numpy()

    with ThreadPoolExecutor(threads) as pool:
        queue, nxt = deque(), 0
        for i in range(len(ids)):
            while nxt < len(ids) and len(queue) < 2 * threads:
                queue.append(pool.submit(_read, task, ids[nxt], slices_root, need_mask))
                nxt += 1
            vols, mask = queue.popleft().result()
            vols = [torch.from_numpy(np.ascontiguousarray(v)).to(device) for v in vols]
            if need_tok:
                feats[i] = encode_tok(vols, 0)
                if i in row:
                    for k in range(1, n_aug):
                        aug[row[i], k - 1] = encode_tok(vols, k)
            if need_pe:
                pe_feats[i] = encode_pe(vols, 0)
                if i in row:
                    for k in range(1, n_aug):
                        pe_aug[row[i], k - 1] = encode_pe(vols, k)
            if mask is not None:
                masks[i] = mask
            if (i + 1) % 100 == 0 or i + 1 == len(ids):
                kind = "+".join(n for n, on in (("tokens", need_tok), ("pe", need_pe)) if on) or "read"
                print(f"[seg3d-cache] {task.name} {kind}: {i + 1}/{len(ids)} volumes", flush=True)
    if feats is not None:
        feats.flush()
        meta.write_text(json.dumps({**want, "ckpt_fingerprint": _fp(ckpt)}))
    if aug is not None:
        aug.flush()
    if pe_feats is not None:
        pe_feats.flush()
        pmeta.write_text(json.dumps({**pe_want, "ckpt_fingerprint": _fp(ckpt)}))
    if pe_aug is not None:
        pe_aug.flush()
    if masks is not None:
        masks.flush()
        (root / "masks.json").write_text(json.dumps(ids))
    return Cache(ids, np.load(fpath, mmap_mode="r"), np.load(mpath, mmap_mode="r"),
                 np.load(apath, mmap_mode="r") if n_aug > 1 else None, row,
                 np.load(ppath, mmap_mode="r") if pe else None,
                 np.load(papath, mmap_mode="r") if pe and n_aug > 1 else None)


# ----------------------------------------------------------------------------- train / score

def n_classes_for(regions) -> int:
    return max(i for ids in regions.values() for i in ids) + 1


@torch.no_grad()
def score_volumes(dec: nn.Module, cache: Cache, vols: np.ndarray, regions: dict, device,
                  batch_size: int = 8) -> tuple[dict, pd.DataFrame]:
    dec.eval()
    ids = {name: torch.tensor(r, device=device) for name, r in regions.items()}
    rows = []
    vols = np.asarray(vols)
    bs = max(1, min(int(batch_size), len(vols) or 1))
    for start in range(0, len(vols), bs):
        chunk = vols[start:start + bs]
        tok = torch.stack([torch.from_numpy(np.array(cache.tokens(int(vi)))) for vi in chunk]).to(device).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = dec(tok)
            pred = (out[0] if isinstance(out, tuple) else out).argmax(1)
        for j, vi in enumerate(chunk):
            gt = torch.from_numpy(np.asarray(cache.masks[int(vi)])).to(device)
            row = {"volume_id": cache.volume_ids[int(vi)]}
            for tag, sl in (("", EVAL_SLICES.to(device)), ("full_", slice(None))):
                for name, r in ids.items():
                    g, p = torch.isin(gt[sl], r), torch.isin(pred[j, sl], r)
                    if not tag:
                        row[f"has_{name}"] = bool(g.any())
                    # an absent region scores 1 when nothing is predicted and 0 otherwise (before 2026-10-25: NaN,
                    # which never penalised a decoder that paints absent regions; as metrics.region_dice does)
                    row[f"{tag}{name}"] = (float(2 * (g & p).sum() / (g.sum() + p.sum())) if g.any()
                                           else (0.0 if p.any() else 1.0))
            rows.append(row)
    df = pd.DataFrame(rows)
    df["dice_macro"] = df[list(regions)].mean(axis=1)
    df["dice_full_macro"] = df[[f"full_{n}" for n in regions]].mean(axis=1)
    per = {n: float(df[n].mean()) for n in regions}
    full = {n: float(df[f"full_{n}"].mean()) for n in regions}
    out = {"task_type": "segmentation", "dice_per_class": per, "dice_macro": float(np.nanmean(list(per.values()))),
           "dice_full_per_class": full, "dice_full_macro": float(np.nanmean(list(full.values()))),
           "n_volumes": len(df), "n_positive": {n: int(df[f"has_{n}"].sum()) for n in regions}}
    out.update({f"dice_{n}": v for n, v in per.items()})
    return out, df


def _batch(cache, pick, ks, device, patch_xyz, encoded: bool):
    """Load one token/mask pair per volume; ``ks`` selects token-aug (n_aug=1) or an encoded copy."""
    xs, ys = [], []
    for i, k in zip(pick, ks):
        if encoded:
            t = torch.from_numpy(np.array(cache.tokens(int(i), int(k)))).to(device).float()
            m = aug_mask(torch.from_numpy(np.asarray(cache.masks[int(i)])).to(device), int(k))
        else:
            t = torch.from_numpy(np.array(cache.tokens(int(i)))).to(device).float()
            m = torch.from_numpy(np.array(cache.masks[int(i)])).to(device)
            t, m = token_aug(t, m, int(k), patch_xyz)
        xs.append(t)
        ys.append(m)
    return xs, ys


def train_decoder(dec, cache, task, split, *, steps, batch_size, lr, seed, eval_every, device, max_val=None,
                  patch_xyz=PATCH, patience: int = 8, min_delta: float = 0.001, crop=None) -> list[dict]:
    """Cosine AdamW until val Dice plateaus. ``steps`` is a budget; ``patience`` is evals without a real gain."""
    train, val = split("train"), split("val")
    if max_val is not None:
        val = val[:max_val]
    if len(train) == 0:
        raise SystemExit("segmentation task has no train volumes")
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    history, best, best_state, run, stale = [], -1.0, None, 0.0, 0
    warmup = min(200, max(1, steps // 10))
    dec.train()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, steps, lr, warmup=warmup)
        pick = np.sort(rng.choice(train, batch_size, replace=len(train) < batch_size))
        encoded = cache.n_aug > 1
        ks = rng.integers(0, cache.n_aug if encoded else 4, len(pick))
        xs, ys = _batch(cache, pick, ks, device, patch_xyz, encoded)
        if crop is not None:
            cropped = []
            for t, m in zip(xs, ys):
                origin = sample_crop(m.detach().cpu().numpy(), patch_xyz, crop, rng)
                cropped.append(crop_tokens(t, m, origin, crop, patch_xyz))
            xs, ys = zip(*cropped)
        x, y = torch.stack(list(xs)), torch.stack(list(ys))
        grid = crop if crop is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = _decode_loss(dec(x, grid=grid) if grid else dec(x), y, task.regions)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        run += float(loss)
        if step % eval_every == 0 or step == steps:
            row = {"step": step, "train_loss": run / min(eval_every, step), "lr": float(opt.param_groups[0]["lr"])}
            run = 0.0
            if len(val):
                m, _ = score_volumes(dec, cache, val, task.regions, device, batch_size=batch_size)
                row["val_dice_macro"] = m["dice_macro"]
                if m["dice_macro"] > best + min_delta:
                    best, best_state, stale = m["dice_macro"], {k: v.detach().clone() for k, v in dec.state_dict().items()}, 0
                else:
                    stale += 1
            print(f"[seg3d] step {step}/{steps} " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "step"), flush=True)
            history.append(row)
            dec.train()
            if patience and stale >= patience:
                print(f"[seg3d] early stop at step {step} (best val dice {best:.4f}, patience {patience})", flush=True)
                break
    if best_state is not None:
        dec.load_state_dict(best_state)
        print(f"[seg3d] restored best val dice {best:.4f} after {history[-1]['step'] if history else 0} steps")
    return history


def main() -> None:
    from brain.downstream.records import now_utc, pretrain_overlap, ssl_exclude

    started = now_utc()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--slices-root", type=Path, default=DEFAULT_VOLUMES.parent)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "runs" / "downstream")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "runs" / "downstream" / "_cache" / f"v{FORMAT_VERSION}")
    ap.add_argument("--steps", type=int, default=6000,
                    help="training budget (cosine horizon). Stops earlier when val Dice plateaus")
    ap.add_argument("--batch", type=int, default=None,
                    help="decoder batch (default 8, or 4 on the 32^3 fine grid)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=8, help="evals without a min-delta val gain before stop (0 = run the budget)")
    ap.add_argument("--min-delta", type=float, default=0.002, help="val Dice improvement that resets patience")
    ap.add_argument("--dropout", type=float, default=0.1, help="dropout on the input token channels")
    ap.add_argument("--fine", action="store_true",
                    help="8x8x4 patches with tiled training-size windows (the resolution that matches 2D in-plane)")
    ap.add_argument("--patch", default="", help="x:y:z patch for the encoder (default: the trained global patch, 16:16:8)")
    ap.add_argument("--tile", action="store_true", help="encode overlapping training-size windows (needed for small patches)")
    ap.add_argument("--taps", default="", help="encoder blocks (separated by : or ,) to feed the decoder besides the final tokens")
    ap.add_argument("--aug", type=int, default=None,
                    help="encoded copies per training volume (default 1 with --fine, else 4). 1 uses token-grid flips/rolls instead")
    ap.add_argument("--crop", type=int, default=None,
                    help="token-grid crop side (default 16 on the fine 32^3 grid, 8 on 16^3; 0 = full volume)")
    ap.add_argument("--deep-sup", action="store_true",
                    help="auxiliary CE+Dice at the token grid and after the first voxel upsample")
    ap.add_argument("--pe-skip", action="store_true",
                    help="fuse pre-transformer patch embeddings as a skip after the token-grid U-Net")
    ap.add_argument("--patch-fcn", action="store_true",
                    help="classify on the token grid and nearest-upsample to voxels (no voxel U-Net)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--keep-cache", action="store_true", help="keep this checkpoint's token cache after the run")
    args = ap.parse_args()
    if args.fine:
        args.patch = args.patch or "8:8:4"
        args.tile = True
    if args.aug is None:
        args.aug = 1 if args.fine else len(AUG)
    if args.batch is None:
        args.batch = 4 if args.fine and args.crop == 0 else 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = load_frozen_encoder(args.ckpt, device)
    if not isinstance(enc, FrozenEncoder3D):
        raise SystemExit("seg3d needs a 3D BrainModel checkpoint; 2D ViTs use brain.downstream.run_suite")
    if args.patch:
        enc.patch_xyz = tuple(int(v) for v in re.split("[,:]", args.patch))
        if len(enc.patch_xyz) != 3:
            raise SystemExit(f"--patch needs three integers, got {enc.patch_xyz} (do not put commas in sbatch EXTRA_ARGS; Slurm splits on them)")
    enc.tap_blocks = tuple(sorted(int(t) for t in re.split("[,:]", args.taps) if t != ""))
    volumes = load_volumes(DEFAULT_VOLUMES)
    task = load_task(args.task, volumes)
    if not task.regions:
        raise SystemExit(f"{task.name} has no segmentation regions")
    eval_banner(task.name, f"segmentation  ·  {args.ckpt}")
    if args.patch_fcn and args.deep_sup:
        raise SystemExit("--deep-sup is for the U-Net decoder, not --patch-fcn")
    cache = build_cache(task, enc, args.ckpt, args.slices_root, args.cache_dir, device, args.aug,
                        tile=args.tile, pe=args.pe_skip)
    if args.cache_only:
        return
    split = lambda s: np.array([i for i, v in enumerate(cache.volume_ids) if task.splits[v] == s], dtype=np.int64)  # noqa: E731
    grid = grid_for(enc.patch_xyz)
    if args.crop is None:
        args.crop = 16 if min(grid) >= 32 else 8
    crop = None if args.crop == 0 else tuple(min(args.crop, s) for s in grid)
    n_cls = n_classes_for(task.regions)
    tok_dim, pe_dim = cache.feats.shape[-1], cache.pe_dim
    if args.patch_fcn:
        dec = PatchFCN(tok_dim + pe_dim, n_cls, grid, patch_xyz=enc.patch_xyz, dropout=args.dropout).to(device)
        head = "tok_pfcn"
    else:
        dec = TokenDecoder3D(tok_dim, n_cls, grid, dropout=args.dropout, deep_sup=args.deep_sup,
                             pe_dim=pe_dim).to(device)
        head = "tok_unet3d"
    print(f"[seg3d] {task.name} decoder {sum(p.numel() for p in dec.parameters()) / 1e6:.2f}M params "
          f"in_dim={tok_dim} pe_dim={pe_dim} head={head} "
          f"train/val/test={len(split('train'))}/{len(split('val'))}/{len(split('test'))} "
          f"patch={list(enc.patch_xyz)} tile={args.tile} taps={list(enc.tap_blocks)} copies={cache.n_aug} "
          f"crop={list(crop) if crop else 'full'} deep_sup={args.deep_sup} dropout={args.dropout} "
          f"budget={args.steps} eval_every={args.eval_every} patience={args.patience} "
          f"min_delta={args.min_delta} batch={args.batch} lr={args.lr}", flush=True)
    history = train_decoder(dec, cache, task, split, steps=args.steps, batch_size=args.batch, lr=args.lr,
                            seed=args.seed, eval_every=args.eval_every, device=device, patch_xyz=enc.patch_xyz,
                            patience=args.patience, min_delta=args.min_delta, crop=crop)
    test = split("test")
    metrics, per_volume = score_volumes(dec, cache, test, task.regions, device)
    d, f = per_volume["dice_macro"].to_numpy(), per_volume["dice_full_macro"].to_numpy()
    lo, hi = bootstrap_ci(lambda i: float(np.nanmean(d[i])), len(d))
    flo, fhi = bootstrap_ci(lambda i: float(np.nanmean(f[i])), len(f))
    metrics.update(head=head, dice_macro_ci=[lo, hi], dice_full_macro_ci=[flo, fhi], history=history,
                   n_test=len(test), n_train=len(split("train")), n_val=len(split("val")))
    vals = [r["val_dice_macro"] for r in history if r.get("val_dice_macro") is not None]
    if vals:                                          # the restored decoder is the best-val one
        metrics["val"] = {"dice_macro": float(np.nanmax(vals))}
    print(f"[seg3d] test dice_macro={metrics['dice_macro']:.4f} [{lo:.4f}, {hi:.4f}] "
          f"full={metrics['dice_full_macro']:.4f} per_class={metrics['dice_per_class']}")
    eval_banner(task.name, "summary")
    rows = [
        ["dice_macro", f"{metrics['dice_macro']:.4f}", f"[{lo:.4f}, {hi:.4f}]"],
        ["dice_full", f"{metrics['dice_full_macro']:.4f}", f"[{flo:.4f}, {fhi:.4f}]"],
    ]
    for cls, d in (metrics.get("dice_per_class") or {}).items():
        rows.append([str(cls), f"{float(d):.4f}", ""])
    print(format_rows(["metric", "value", "95% CI"], rows, left=1), flush=True)
    print(flush=True)
    run_dir = new_run_dir(args.out_root, task.name, args.ckpt, args.run_name)
    write_json(run_dir / "heads" / f"{head}.json", metrics)
    per_volume.to_csv(run_dir / "predictions" / f"{head}_volumes.csv", index=False)
    torch.save({"model": dec.state_dict(), "regions": {k: list(v) for k, v in task.regions.items()}},
               run_dir / "seg_decoder.pt")
    config = {"task": task.name, "task_type": "segmentation", "head": head, "ckpt": str(args.ckpt),
              "regions": {k: list(v) for k, v in task.regions.items()}, "steps": args.steps, "batch_size": args.batch,
              "lr": args.lr, "seed": args.seed, "taps": list(enc.tap_blocks), "patch": list(enc.patch_xyz),
              "tile": args.tile, "aug_copies": cache.n_aug, "dropout": args.dropout,
              "crop": list(crop) if crop else None, "deep_sup": args.deep_sup, "pe_skip": args.pe_skip,
              "patch_fcn": args.patch_fcn, "pe_dim": pe_dim,
              "n_train": metrics["n_train"], "n_test": metrics["n_test"],
              "patience": args.patience, "min_delta": args.min_delta, "stopped_step": history[-1]["step"] if history else 0,
              "eval_every": args.eval_every, "fine": args.fine, "started_utc": started,
              "slices_root": str(args.slices_root), "volumes_csv": str(DEFAULT_VOLUMES), "n_val": metrics["n_val"],
              "eval_slices": "64 bin-centre axial slices (dice_*) and the whole volume (dice_full_*)",
              "pretrain_overlap": pretrain_overlap(task.volume_ids, volumes, ssl_exclude(args.ckpt))}
    print(f"[seg3d] summary -> {finalize_run(run_dir, config, {head: metrics}, suite='seg3d')}")
    if not args.keep_cache:
        drop_token_cache(args.cache_dir, task.name, args.ckpt)


def drop_token_cache(cache_dir: Path, task: str, ckpt: Path) -> None:
    """Delete this checkpoint's cached token maps for ``task`` (tens of GB per task; only a rerun of the same
    checkpoint and task would reuse them). The masks are shared across checkpoints and stay."""
    root = Path(cache_dir) / "_seg3d" / task
    prefix = f"{ckpt.parent.name}_{ckpt.stem}"
    freed = 0
    for f in root.glob(prefix + "*"):
        if f.is_file() and f.suffix in (".npy", ".json"):
            freed += f.stat().st_size
            f.unlink()
    print(f"[seg3d] dropped token cache {root}/{prefix}* ({freed / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
