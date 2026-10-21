"""Frozen encoders and cached per-volume slice embeddings.

A ViT checkpoint contributes ``[class token, mean patch, max patch]``
(3 x embed_dim); a 3D BrainEncoder checkpoint is read by ``embed3d``. Downstream tasks use a
uniform axial subset: 32 slices by default, 64 on focal disease (ICH, tumor, lesion).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from brain.models.vit2d import VIT_SIZES, VisionTransformer
from brain.common.windowing import asinh_window, windows_for
from brain.common.grid import DEPTH

UNIFORM_SLICES = 32  # non-focal tasks (age, CDR, ASD): centers of 32 bins
FOCAL_SLICES = 64  # ICH / tumor / lesion tasks: finer axial grid
VOLUME_DEPTH = DEPTH
CACHE_VERSION = "v4"  # v3 was cls+mean; v4 is cls+mean+amax and stores z_index
CACHE_VERSION_3D = "v5_3d_tok"  # native-grid tokens + xyz; v4_3d was packed axial planes


def uniform_slice_indices(depth: int, n: int) -> np.ndarray:
    """Increasing z indices, one at the center of each equal bin."""
    depth, n = int(depth), int(n)
    if depth <= 0 or n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n >= depth:
        return np.arange(depth, dtype=np.int64)
    return ((np.arange(n, dtype=np.float64) + 0.5) * depth / n).astype(np.int64)


def resolve_path(slices_root: Path, rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else Path(slices_root) / path


def read_uniform(path: Path, name: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """One zip open, one read: (z indices, ``image`` or ``mask`` rows at those z)."""
    import zarr
    from zarr.storage import ZipStore

    with ZipStore(path, mode="r") as store:
        arr = zarr.open_group(store, mode="r")[name]
        zs = uniform_slice_indices(arr.shape[0], n)
        return zs, np.asarray(arr[zs])


class FrozenEncoder:
    """Eval-mode encoder. ``embed`` windows native units then returns (B, dim) float32."""

    def __init__(self, model: torch.nn.Module, kind: str, representation: str, dim: int):
        self.model, self.kind, self.representation, self.dim = model, kind, representation, int(dim)

    @torch.no_grad()
    def embed(self, x: torch.Tensor, c: float = 40.0, w: float = 40.0) -> torch.Tensor:
        x = asinh_window(x, c, w)
        if self.kind == "vit":
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
                return self.model.features(x).float()
        return self.model.encode(x)[0]


def peek_is_brain3d(ckpt_path: Path) -> bool:
    """True if this checkpoint is a 3D BrainEncoder (not the 2D slice ViT)."""
    from brain.downstream.embed3d import is_brain3d_state

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)  # header only, no full read
    cfg = ckpt.get("config") or {}
    if isinstance(cfg.get("views"), dict) and ({"patch_xyz", "global_patch_xyz"} & set(cfg["views"])):
        return True
    state = {str(k).removeprefix("_orig_mod.").removeprefix("module."): v for k, v in ckpt["model"].items()}
    return is_brain3d_state(state)


def peek_n_registers(ckpt_path: Path) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    return int(((ckpt.get("config") or {}).get("model") or {}).get("n_registers", 0) or 0)


def load_frozen_encoder(ckpt_path: Path, device: torch.device):
    """Load a ViT or 3D BrainEncoder checkpoint (LeJEPA or BrainDINO) and freeze it."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {str(k).removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    from brain.downstream.embed3d import is_brain3d_state, load_brain3d

    views = (ckpt.get("config") or {}).get("views")
    if is_brain3d_state(state) or (isinstance(views, dict) and ({"patch_xyz", "global_patch_xyz"} & set(views))):
        enc = load_brain3d(ckpt, device)
        if "multitask" in ckpt:                    # multitask checkpoints also expose their task slots
            from brain.multitask.slots import attach

            attach(enc, ckpt, device)
        return enc
    cfg = (ckpt.get("config") or {}).get("model") or {}
    if "patch_embed.weight" in state:
        dim = int(state["patch_embed.weight"].shape[0])
        size = str(cfg.get("size", ""))
        if size not in VIT_SIZES:
            size = next(k for k, v in VIT_SIZES.items() if int(v["embed_dim"]) == dim)
        model = VisionTransformer(
            size=size,
            img_size=int(cfg.get("img_size", 256)),
            patch_size=int(cfg.get("patch_size", state["patch_embed.weight"].shape[-1])),
            proj_dim=int(cfg.get("proj_dim", 128)),
            proj_hidden=cfg.get("proj_hidden"),
            registers=int(cfg.get("registers", 0)),
            qk_norm=bool(cfg.get("qk_norm", False)),
        ).to(device)
        kind, representation, out_dim = "vit", "cls+mean+amax", 3 * dim
    else:
        raise ValueError(f"Unrecognized checkpoint {ckpt_path} (first key {next(iter(state))})")
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    return FrozenEncoder(model, kind, representation, out_dim)


@torch.no_grad()
def extract_volume_embeddings(
    enc: FrozenEncoder,
    image_paths: dict[str, str],
    volume_ids: list[str],
    slices_root: Path,
    device: torch.device,
    max_slices: int = UNIFORM_SLICES,
    batch_size: int = 64,
    modalities: dict[str, str] | None = None,
    window: str = "canonical",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, S, D) embeddings, (N, S) validity mask, (N, S) source z. One zip read per volume."""
    if getattr(enc, "kind", "") == "brain3d":
        from brain.downstream.embed3d import extract_volume_embeddings_3d

        return extract_volume_embeddings_3d(
            enc, image_paths, volume_ids, slices_root, device, max_slices,
            modalities=modalities, window=window,
        )
    modalities = modalities or {}
    sample_mod = modalities.get(volume_ids[0], "mri") if volume_ids else "mri"
    wins = windows_for(window, sample_mod)
    dim = enc.dim * len(wins)
    embs = np.zeros((len(volume_ids), max_slices, dim), dtype=np.float32)
    mask = np.zeros((len(volume_ids), max_slices), dtype=bool)
    z_index = np.zeros((len(volume_ids), max_slices), dtype=np.int16)
    for i, vid in enumerate(volume_ids):
        zs, vol = read_uniform(resolve_path(slices_root, image_paths[vid]), "image", max_slices)
        if vol.ndim == 3:
            vol = vol[:, None]
        wins_i = windows_for(window, modalities.get(vid, sample_mod))
        for s in range(0, len(zs), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(vol[s : s + batch_size])).to(device, dtype=torch.float32)
            feats = [enc.embed(x, c, w) for c, w in wins_i]
            embs[i, s : s + len(x)] = torch.cat(feats, dim=-1).cpu().numpy()
        mask[i, : len(zs)] = True
        z_index[i, : len(zs)] = zs
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(volume_ids):
            print(f"[embed] {i + 1}/{len(volume_ids)} volumes window={window} n_win={len(wins_i)}")
    return embs, mask, z_index


def load_or_build_embeddings(
    *,
    ckpt: Path,
    slices_root: Path,
    image_paths: dict[str, str],
    volume_ids: list[str],
    cache_dir: Path,
    task: str,
    max_slices: int = UNIFORM_SLICES,
    force: bool = False,
    window: str = "canonical",
    modalities: dict[str, str] | None = None,
) -> dict:
    is3d = peek_is_brain3d(ckpt)
    tag = "" if is3d or max_slices == UNIFORM_SLICES else f"_s{max_slices}"
    if is3d and os.environ.get("BRAIN_FINE_GRID", "") == "1":
        tag += "_finegrid"
    elif is3d and os.environ.get("BRAIN_FINE_POOL", "") == "1":
        tag += "_finepool"
    if is3d and os.environ.get("BRAIN_EVAL_TILE", "") == "1":
        tag += "_tiled"
    eval_taps = os.environ.get("BRAIN_EVAL_TAPS", "").strip()     # e.g. "3": block 4's output (0-based index)
    taps_only = os.environ.get("BRAIN_EVAL_TAPS_ONLY", "") == "1"
    if is3d and eval_taps:
        tag += f"_tap{eval_taps.replace(',', '-')}" + ("only" if taps_only else "")
    win = (window or "canonical").lower()
    ver = CACHE_VERSION_3D if is3d else CACHE_VERSION
    path = Path(cache_dir) / f"{ckpt.parent.name}_{ckpt.stem}" / task / f"embeddings_{ver}_{win}{tag}.npz"
    ids = [str(v) for v in volume_ids]
    if path.exists() and not force:
        data = np.load(path, allow_pickle=False)
        same_ids = [str(x) for x in data["volume_ids"]] == ids
        spatial = str(data["representation"]) == "tokens"
        # caches from before register capture lack the register outputs of register models: rebuild those
        has_regs = "registers" in data.files or not (is3d and peek_n_registers(ckpt))
        if is3d and "slots" not in data.files:     # multitask checkpoints: caches from before the slot readout
            from brain.multitask.slots import is_multitask

            has_regs = has_regs and not is_multitask(ckpt)
        if same_ids and has_regs and (spatial or int(data["max_slices"]) == max_slices):
            z = data["z"] if "z" in data.files else slot_z(len(ids), max_slices)
            z = np.asarray(z, dtype=np.float32 if getattr(z, "ndim", 2) == 3 else np.int16)
            n_tok = int(data["mus"].shape[1])
            mus = data["mus"]
            mus = mus if (n_tok > 8192 or mus.dtype == np.float32) else mus.astype(np.float32)
            return {
                "volume_ids": ids, "mus": mus, "mask": data["mask"],
                "z": z, "latent_dim": int(data["latent_dim"]),
                "max_slices": n_tok if spatial else max_slices,
                "representation": str(data["representation"]), "cache_path": path,
                "cls": data["cls"] if "cls" in data.files else None,
                "registers": data["registers"] if "registers" in data.files else None,
                "slots": data["slots"] if "slots" in data.files else None,
                "taskout": data["taskout"] if "taskout" in data.files else None,
            }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = load_frozen_encoder(ckpt, device)
    if is3d and eval_taps:
        enc.tap_blocks = tuple(int(t) for t in eval_taps.split(","))
        enc.tap_norm, enc.taps_only = True, taps_only
    print(f"[encoder] {enc.kind} representation={enc.representation} dim={enc.dim} window={win} ckpt={ckpt}")
    mus, mask, z = extract_volume_embeddings(
        enc, image_paths, ids, slices_root, device, max_slices,
        modalities=modalities, window=win,
    )
    spatial = getattr(enc, "representation", "") == "tokens"
    z_save = np.asarray(z, dtype=np.float16 if spatial else np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        volume_ids=np.asarray(ids), mus=mus.astype(np.float16, copy=False), mask=mask, z=z_save,
        latent_dim=np.int32(mus.shape[-1]), max_slices=np.int32(mus.shape[1] if spatial else max_slices),
        representation=np.asarray(enc.representation), window=np.asarray(win),
    )
    cls = getattr(enc, "cls_bank", None)                     # CLS models: (N, D * n_windows)
    if cls is not None:
        payload["cls"] = cls.astype(np.float32)
    regs = getattr(enc, "reg_bank", None)                    # register models: (N, R * D * n_windows)
    if regs is not None:
        payload["registers"] = regs.astype(np.float32)
    slots, taskout = getattr(enc, "slot_bank", None), getattr(enc, "taskout_bank", None)   # multitask checkpoints
    if slots is not None:
        payload["slots"], payload["taskout"] = slots.astype(np.float32), taskout.astype(np.float32)
    # Uncompressed for the 32^3 grid: compressing ~20 GB of embeddings is slower than the encode.
    (np.savez if mus.shape[1] > 8192 else np.savez_compressed)(path, **payload)
    return {
        "volume_ids": ids, "mus": mus, "mask": mask,
        "z": np.asarray(z, dtype=np.float32 if spatial else np.int16),
        "latent_dim": int(mus.shape[-1]),
        "max_slices": int(mus.shape[1]) if spatial else max_slices,
        "representation": enc.representation, "cache_path": path,
        "cls": cls,
        "registers": regs,
        "slots": slots,
        "taskout": taskout,
    }


def slot_z(n_vol: int, n_slices: int, depth: int = VOLUME_DEPTH) -> np.ndarray:
    """Broadcast uniform bin centers to (N, S) when a cache has no stored z."""
    zs = uniform_slice_indices(depth, n_slices)
    return np.broadcast_to(zs, (n_vol, n_slices)).copy()
