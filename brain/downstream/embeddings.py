"""Frozen encoders and the per-task embedding cache.

Any model the suite evaluates is a ``brain.downstream.models.ModelRef`` (one of our 3D checkpoints, or a registry
model: a published encoder or a baseline). ``load_or_build_embeddings`` encodes a task's volumes once per model and
window setting and caches the tokens (N, T, D), their foreground mask and positions, and the global CLS / register
banks; a cache is reused only while the model's fingerprint matches.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

CACHE_VERSION_3D = "v5_3d_tok"  # native-grid tokens + xyz; v4_3d was packed axial planes
BIG_CACHE_BYTES = 8e9  # token arrays above this: stored as a memory-mapped .npy sidecar, read back as float16


def resolve_path(slices_root: Path, rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else Path(slices_root) / path


def load_frozen_encoder(ckpt_path: Path, device: torch.device):
    """One of our 3D BrainEncoder checkpoints (SSL, multitask or joint), frozen."""
    from brain.downstream.models import resolve_model

    return resolve_model(ckpt_path).load(device)


def _fp(ckpt: Path) -> str:
    from brain.downstream.records import fingerprint

    return fingerprint(Path(ckpt))


def cache_fresh(cache_path: Path, ckpt: Path, recorded: str | None) -> bool:
    """Whether a cache was built from the checkpoint's current weights (its recorded fingerprint matches). Caches are
    keyed by checkpoint path, and ``best.pt`` / ``last.pt`` are overwritten during training."""
    return bool(recorded) and str(recorded) == _fp(ckpt)


def _readable(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as d:
            files = set(d.files)
        if "mus" not in files:                              # big caches keep their tokens in a memory-mapped sidecar
            np.load(path.with_suffix(".mus.npy"), mmap_mode="r")
            files.add("mus")
        return {"volume_ids", "representation", "mus", "mask"} <= files
    except Exception:
        return False


def cache_tag() -> str:
    """Encoding variants of our 3D encoders set by environment (``BRAIN_FINE_GRID`` / ``FINE_POOL`` / ``EVAL_TILE`` /
    ``EVAL_TAPS``): part of the cache name, and recorded with every result (``records.ENV_FLAGS``)."""
    tag = ""
    if os.environ.get("BRAIN_FINE_GRID", "") == "1":
        tag += "_finegrid"
    elif os.environ.get("BRAIN_FINE_POOL", "") == "1":
        tag += "_finepool"
    if os.environ.get("BRAIN_EVAL_TILE", "") == "1":
        tag += "_tiled"
    taps = os.environ.get("BRAIN_EVAL_TAPS", "").strip()       # e.g. "3": block 4's output (0-based index)
    if taps:
        tag += f"_tap{taps.replace(',', '-')}" + ("only" if os.environ.get("BRAIN_EVAL_TAPS_ONLY", "") == "1" else "")
    return tag


def load_or_build_embeddings(
    *,
    model,
    slices_root: Path,
    image_paths: dict[str, str],
    volume_ids: list[str],
    cache_dir: Path,
    task: str,
    force: bool = False,
    window: str = "canonical",
    modalities: dict[str, str] | None = None,
) -> dict:
    """``model``: a ``ModelRef`` (or a checkpoint path). Returns ``mus`` (N, T, D), ``mask`` (N, T), ``z`` (N, T, 3)
    positions, ``cls`` / ``registers`` banks or None, and the cache path."""
    from brain.downstream.models import ModelRef, resolve_model

    model = model if isinstance(model, ModelRef) else resolve_model(model)
    tag = cache_tag() if model.is_checkpoint else ""
    win = (window or "canonical").lower() if model.is_checkpoint else "native"   # registry models do their own
    path = Path(cache_dir) / model.cache_key / task / f"embeddings_{CACHE_VERSION_3D}_{win}{tag}.npz"
    ids = [str(v) for v in volume_ids]
    if path.exists() and not force and not _readable(path):
        print(f"[cache] unreadable (truncated write?), rebuilding: {path}")
        force = True
    if path.exists() and not force:
        data = np.load(path, allow_pickle=False)
        same_ids = [str(x) for x in data["volume_ids"]] == ids
        fresh = "ckpt_fingerprint" in data.files and str(data["ckpt_fingerprint"]) == model.fingerprint
        if not fresh:
            print(f"[cache] stale (the model changed after it was built), rebuilding: {path}", flush=True)
        if same_ids and fresh:
            mus = np.load(path.with_suffix(".mus.npy"), mmap_mode="r") if "mus" not in data.files else data["mus"]
            # small caches as float32; big ones stay float16 (a float32 copy of RSNA's tokens alone was ~220 GB):
            # the heads cast per chunk
            big = mus.shape[1] > 8192 or mus.nbytes > BIG_CACHE_BYTES
            mus = mus if (big or mus.dtype == np.float32) else mus.astype(np.float32)
            return {"volume_ids": ids, "mus": mus, "mask": data["mask"], "z": np.asarray(data["z"], dtype=np.float32),
                    "latent_dim": int(data["latent_dim"]), "max_slices": int(mus.shape[1]),
                    "representation": str(data["representation"]), "cache_path": path,
                    "cls": data["cls"] if "cls" in data.files else None,
                    "registers": data["registers"] if "registers" in data.files else None}
    from brain.downstream.embed3d import extract_volume_embeddings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = model.load(device)
    taps = os.environ.get("BRAIN_EVAL_TAPS", "").strip()
    if taps and model.is_checkpoint:
        enc.tap_blocks = tuple(int(t) for t in taps.split(","))
        enc.tap_norm, enc.taps_only = True, os.environ.get("BRAIN_EVAL_TAPS_ONLY", "") == "1"
    print(f"[encoder] {model.model_id} ({model.training_class}, {getattr(enc, 'kind', '?')}) window={win}")
    e = extract_volume_embeddings(enc, image_paths, ids, slices_root, modalities=modalities, window=win)
    mus = e["mus"]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(volume_ids=np.asarray(ids), mus=mus.astype(np.float16, copy=False), mask=e["mask"],
                   z=e["z"].astype(np.float16), latent_dim=np.int32(mus.shape[-1]), max_slices=np.int32(mus.shape[1]),
                   representation=np.asarray("tokens"), window=np.asarray(win),
                   ckpt_fingerprint=np.asarray(model.fingerprint))
    for k in ("cls", "registers"):
        if e[k] is not None:
            payload[k] = e[k].astype(np.float32)
    # Uncompressed for the big grids: compressing ~20 GB of embeddings is slower than the encode.
    tmp = path.with_name(path.stem + ".tmp.npz")             # write then rename: a full disk leaves no truncated cache
    if payload["mus"].nbytes > BIG_CACHE_BYTES:             # tokens to an uncompressed .npy, read back memory-mapped
        side = path.with_suffix(".mus.npy")
        np.save(side.with_name(side.stem + ".tmp.npy"), payload.pop("mus"))
        os.replace(side.with_name(side.stem + ".tmp.npy"), side)
        np.savez(tmp, **payload)
    else:
        (np.savez if mus.shape[1] > 8192 else np.savez_compressed)(tmp, **payload)
    os.replace(tmp, path)
    return {"volume_ids": ids, "mus": mus, "mask": e["mask"], "z": e["z"], "latent_dim": int(mus.shape[-1]),
            "max_slices": int(mus.shape[1]), "representation": "tokens", "cache_path": path,
            "cls": e["cls"], "registers": e["registers"]}
