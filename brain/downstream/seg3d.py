"""Segmentation on frozen 3D BrainModel tokens.

The encoder runs once per volume (one pass over the native 256 x 256 x 160 grid); its 20 x 16 x 16 token
map (one 16 x 16 x 8 patch each) is cached, with 3 shifted / flipped training copies of each training volume
(``AUG``). A small 3D conv decoder (``TokenDecoder3D``, head ``tok_unet3d``), trained on token-grid crops of the
cached tokens alone, upsamples the map to one logit per voxel. Nothing but the final tokens reaches the decoder: no
raw voxels and no intermediate blocks, so the decoder sees exactly the resolution the encoder hands over.

Dice is scored on 64 bin-centre axial slices (``dice_*``, the slices the retired 2D reference decoder used) and on
the whole volume (``dice_full_*``). Several sequences (BraTS) are encoded one at a time and their tokens are
concatenated.

    python -m brain.downstream.seg3d --task wmh --ckpt runs/snap/predict_8k.pt
"""

from __future__ import annotations

import argparse
import json
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
from brain.downstream.records import fingerprint
from brain.downstream.embed3d import FrozenEncoder3D, read_volume_xyz
from brain.downstream.embeddings import resolve_path
from brain.downstream.metrics import bootstrap_ci
from brain.downstream.results import eval_banner, finalize_run, format_rows, new_run_dir, write_json
from brain.downstream.tasks import DEFAULT_VOLUMES, TaskBundle, load_task, load_volumes
from brain.common.optim import lr_at
from brain.common.windowing import windows_for
from brain.common.grid import DEPTH, PATCH_XYZ, TOKEN_GRID
from brain.common.paths import REPO_ROOT
from brain.prep.store import FORMAT_VERSION
from brain.models.conv3d import Res, group_norm, up3

VOX = (DEPTH, 256, 256)  # voxels (Z, Y, X)
PATCH = PATCH_XYZ  # the encoder's patch (x, y, z), 16 x 16 x 8
GRID = TOKEN_GRID  # its token grid over a full volume (Z, Y, X), 20 x 16 x 16


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


def sample_crop(mask, crop, rng, p_fg: float = 0.8) -> tuple[int, int, int]:
    """Token-grid origin of a ``crop`` cube; 80% of the time the cube covers a lesion token."""
    nz, ny, nx = GRID
    cz, cy, cx = crop
    px, py, pz = PATCH
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


def crop_tokens(tok, mask, origin, crop):
    """Slice tokens and the matching voxel mask to a token-grid cube."""
    nz, ny, nx = GRID
    cz, cy, cx = crop
    z0, y0, x0 = origin
    px, py, pz = PATCH
    c = tok.shape[-1]
    t = tok.view(nz, ny, nx, c)[z0:z0 + cz, y0:y0 + cy, x0:x0 + cx].reshape(cz * cy * cx, c)
    m = mask[z0 * pz:(z0 + cz) * pz, y0 * py:(y0 + cy) * py, x0 * px:(x0 + cx) * px]
    return t, m


# ----------------------------------------------------------------------------- decoder

class TokenDecoder3D(nn.Module):
    """Token-grid U-Net, then upsample to voxel logits.

    Two stride-2 stages on the token grid (with skips) keep local structure that a pure upsample stack
    throws away. The remaining stages double in-plane size — and depth until full — to the 160 x 256 x 256
    voxel grid.
    """

    def __init__(self, in_dim: int, n_classes: int, grid=GRID, widths=(128, 96, 64, 32, 16),
                 dropout: float = 0.0, out_vox=VOX):
        super().__init__()
        self.grid = tuple(grid)
        self.out_vox = tuple(out_vox)
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
        self.stages = nn.ModuleList(
            [nn.Sequential(up3(vox_w[i], vox_w[i + 1], s), Res(vox_w[i + 1], vox_w[i + 1])) for i, s in enumerate(steps)]
        )
        self.head = nn.Conv3d(vox_w[-1], n_classes, 1)

    def forward(self, tokens: torch.Tensor, grid=None):
        grid = tuple(grid or self.grid)
        b, n, c = tokens.shape
        x = self.drop(tokens).transpose(1, 2).reshape(b, c, *grid)
        x = self.stem(x)
        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x)
        for up, fuse, skip in zip(reversed(self.enc_ups), reversed(self.fuses), reversed(skips)):
            x = fuse(torch.cat([up(x), skip], 1))
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


# ----------------------------------------------------------------------------- cache

def _meta_ok(meta: Path, want: dict, data: Path, ckpt: Path) -> bool:
    """The cache's recorded settings match and it was built from the checkpoint's current weights."""
    got = json.loads(meta.read_text())
    fp = got.pop("ckpt_fingerprint", None)
    return got == want and bool(fp) and str(fp) == fingerprint(ckpt)


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
    """``feats[i]`` are the tokens of volume ``i``; ``aug[row[i], k - 1]`` those of its training copy ``k``."""

    def __init__(self, ids, feats, masks, aug=None, row=None):
        self.volume_ids, self.feats, self.masks, self.aug, self.row = ids, feats, masks, aug, row or {}

    @property
    def n_aug(self) -> int:
        return 1 if self.aug is None else 1 + self.aug.shape[1]

    def tokens(self, i: int, k: int = 0):
        return self.feats[i] if k == 0 else self.aug[self.row[i], k - 1]


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
                device: torch.device, n_aug: int = len(AUG), threads: int = 6) -> Cache:
    ids = [str(v) for v in task.volume_ids]
    train_ids = [v for v in ids if task.splits[v] == "train"]
    row = {ids.index(v): j for j, v in enumerate(train_ids)} if n_aug > 1 else {}
    root = Path(cache_dir) / "_seg3d" / task.name
    root.mkdir(parents=True, exist_ok=True)
    n_tok = int(np.prod(GRID))
    tag = f"{ckpt.parent.name}_{ckpt.stem}"
    n_seq = 4 if task.name.startswith("brats") else 1
    mpath, fpath, apath, meta = root / "masks.npy", root / f"{tag}.npy", root / f"{tag}_aug.npy", root / f"{tag}.json"
    want = {"volume_ids": ids, "n_seq": n_seq, "dim": enc.dim, "n_aug": n_aug, "train": train_ids}
    have_masks = mpath.exists() and (root / "masks.json").exists() and json.loads((root / "masks.json").read_text()) == ids
    _wait_meta(fpath, meta, fpath.name)
    have_tok = fpath.exists() and meta.exists() and _meta_ok(meta, want, fpath, ckpt)

    def opened() -> Cache:
        return Cache(ids, np.load(fpath, mmap_mode="r"), np.load(mpath, mmap_mode="r"),
                     np.load(apath, mmap_mode="r") if n_aug > 1 else None, row)

    if have_masks and have_tok:
        return opened()
    need_tok, need_mask = not have_tok, not have_masks
    masks = None if not need_mask else np.lib.format.open_memmap(mpath, "w+", np.uint8, (len(ids), *VOX))
    dim = enc.dim * n_seq                       # the sequences' tokens side by side (BraTS)
    feats = np.lib.format.open_memmap(fpath, "w+", np.float16, (len(ids), n_tok, dim)) if need_tok else None
    aug = (np.lib.format.open_memmap(apath, "w+", np.float16, (len(train_ids), n_aug - 1, n_tok, dim))
           if need_tok and n_aug > 1 else None)
    mod = task.modality_filter or ("mri" if task.name.startswith("brats") else "ct")
    c, w = windows_for("canonical", mod)[0]

    def encode_tok(vols, k):
        vs = [aug_volume(v, k) for v in vols]
        toks = [enc.embed_tokens(vs[0], c, w, mod)[0]] if len(vs) == 1 else list(enc.embed_volumes(vs, c, w, mod))
        return torch.cat(toks, -1).half().cpu().numpy()

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
            if mask is not None:
                masks[i] = mask
            if (i + 1) % 100 == 0 or i + 1 == len(ids):
                print(f"[seg3d-cache] {task.name} {'tokens' if need_tok else 'masks'}: {i + 1}/{len(ids)} volumes",
                      flush=True)
    if feats is not None:
        feats.flush()
        meta.write_text(json.dumps({**want, "ckpt_fingerprint": fingerprint(ckpt)}))
    if aug is not None:
        aug.flush()
    if masks is not None:
        masks.flush()
        (root / "masks.json").write_text(json.dumps(ids))
    return opened()


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
            pred = dec(tok).argmax(1)
        for j, vi in enumerate(chunk):
            gt = torch.from_numpy(np.asarray(cache.masks[int(vi)])).to(device)
            row = {"volume_id": cache.volume_ids[int(vi)]}
            for tag, sl in (("", EVAL_SLICES.to(device)), ("full_", slice(None))):
                for name, r in ids.items():
                    g, p = torch.isin(gt[sl], r), torch.isin(pred[j, sl], r)
                    if not tag:
                        row[f"has_{name}"] = bool(g.any())
                    # an absent region scores 1 when nothing is predicted and 0 otherwise (before 2026-10-25: NaN,
                    # which never penalised a decoder that paints absent regions)
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


def _batch(cache, pick, ks, device):
    """One (tokens, voxel mask) pair per volume: encoded copy ``k`` (0 = the volume itself) with its moved mask."""
    xs, ys = [], []
    for i, k in zip(pick, ks):
        xs.append(torch.from_numpy(np.array(cache.tokens(int(i), int(k)))).to(device).float())
        ys.append(aug_mask(torch.from_numpy(np.asarray(cache.masks[int(i)])).to(device), int(k)))
    return xs, ys


def train_decoder(dec, cache, task, split, *, steps, batch_size, lr, seed, eval_every, device, max_val=None,
                  patience: int = 8, min_delta: float = 0.001, crop=None) -> list[dict]:
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
        ks = rng.integers(0, cache.n_aug, len(pick))
        xs, ys = _batch(cache, pick, ks, device)
        if crop is not None:
            xs, ys = zip(*[crop_tokens(t, m, sample_crop(m.detach().cpu().numpy(), crop, rng), crop)
                           for t, m in zip(xs, ys)])
        x, y = torch.stack(list(xs)), torch.stack(list(ys))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = region_seg_loss(dec(x, grid=crop), y, task.regions)
        if not torch.isfinite(loss):
            # 2026-11-05: AISD trained on NaN from the start for every encoder (not reproducible in memory); name the
            # batch and skip the step instead of poisoning the decoder
            bad = [(int(i), int(k)) for i, k in zip(pick, ks)]
            print(f"[seg3d] non-finite loss at step {step}: x finite={bool(torch.isfinite(x).all())} "
                  f"y range=({int(y.min())},{int(y.max())}) volumes/copies={bad}", flush=True)
        else:
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
    ap.add_argument("--batch", type=int, default=8, help="decoder batch")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=8, help="evals without a min-delta val gain before stop (0 = run the budget)")
    ap.add_argument("--min-delta", type=float, default=0.002, help="val Dice improvement that resets patience")
    ap.add_argument("--dropout", type=float, default=0.1, help="dropout on the input token channels")
    ap.add_argument("--crop", type=int, default=8, help="token-grid crop side for training (0 = full volume)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--keep-cache", action="store_true", help="keep this checkpoint's token cache after the run")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from brain.downstream.models import resolve_model

    enc = resolve_model(args.ckpt).load(device)
    if not isinstance(enc, FrozenEncoder3D):
        raise SystemExit("seg3d needs one of our 3D encoder checkpoints")
    volumes = load_volumes(DEFAULT_VOLUMES)
    task = load_task(args.task, volumes)
    if not task.regions:
        raise SystemExit(f"{task.name} has no segmentation regions")
    eval_banner(task.name, f"segmentation  ·  {args.ckpt}")
    cache = build_cache(task, enc, args.ckpt, args.slices_root, args.cache_dir, device)
    if args.cache_only:
        return
    split = lambda s: np.array([i for i, v in enumerate(cache.volume_ids) if task.splits[v] == s], dtype=np.int64)  # noqa: E731
    crop = None if args.crop == 0 else tuple(min(args.crop, s) for s in GRID)
    n_cls = n_classes_for(task.regions)
    tok_dim = cache.feats.shape[-1]
    dec = TokenDecoder3D(tok_dim, n_cls, GRID, dropout=args.dropout).to(device)
    head = "tok_unet3d"
    print(f"[seg3d] {task.name} decoder {sum(p.numel() for p in dec.parameters()) / 1e6:.2f}M params "
          f"in_dim={tok_dim} head={head} "
          f"train/val/test={len(split('train'))}/{len(split('val'))}/{len(split('test'))} copies={cache.n_aug} "
          f"crop={list(crop) if crop else 'full'} dropout={args.dropout} "
          f"budget={args.steps} eval_every={args.eval_every} patience={args.patience} "
          f"min_delta={args.min_delta} batch={args.batch} lr={args.lr}", flush=True)
    history = train_decoder(dec, cache, task, split, steps=args.steps, batch_size=args.batch, lr=args.lr,
                            seed=args.seed, eval_every=args.eval_every, device=device,
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
              "lr": args.lr, "seed": args.seed, "patch": list(PATCH), "aug_copies": cache.n_aug,
              "dropout": args.dropout, "crop": list(crop) if crop else None,
              "n_train": metrics["n_train"], "n_test": metrics["n_test"],
              "patience": args.patience, "min_delta": args.min_delta, "stopped_step": history[-1]["step"] if history else 0,
              "eval_every": args.eval_every, "started_utc": started,
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
