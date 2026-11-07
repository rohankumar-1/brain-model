"""Volume codec of the pool archives (JPEG-LS / JPEG-XL lossless slices) and unpacking an archive back to NIfTI.

The FOMO / ABIDE ``*.jls.zip`` archives and the image members of ``<name>.raw.zip`` packs (``brain.prep.rawpack``)
hold one encoded volume per ``.jls`` member plus its ``.json`` header (affine, scaling). The packing runs that wrote
the FOMO archives are finished (2026-10); only reading them back is kept::

    python -m brain.prep.archive --unpack POOL/fomo260k/PT018_HBN.jls.zip --dest $TMPDIR/extracted [--subset 0/4]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import imagecodecs
import nibabel as nib
import numpy as np


MAGIC_JLS = b"FOMOJLS1"
MAGIC_JXL = b"FOMOJXL1"  # same container, JPEG-XL lossless slices: ~40% smaller than JPEG-LS on our MRI/CT, ~3x slower
#                          to decode. Member names keep the ``.jls`` suffix; ``decode_volume`` reads either codec.
JXL_EFFORT = 7
CODECS = {
    "jls": (MAGIC_JLS, imagecodecs.jpegls_encode, imagecodecs.jpegls_decode),
    "jxl": (MAGIC_JXL, lambda a: imagecodecs.jpegxl_encode(a, lossless=True, effort=JXL_EFFORT, numthreads=1),
            imagecodecs.jpegxl_decode),
}
DEFAULT_CODEC = "jxl"
KIND_U16, KIND_I16, KIND_U8 = 1, 2, 3
def slice_axis(shape: tuple[int, ...]) -> int:
    return int(np.argmin(shape))


def as_uint16_volume(arr: np.ndarray) -> tuple[np.ndarray, dict]:
    """JPEG-LS stores unsigned 16-bit. Integer NIfTIs keep exact samples; floats are scaled."""
    arr = np.squeeze(np.asanyarray(arr))
    meta: dict = {"orig_dtype": str(arr.dtype)}
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0)
        meta["nonfinite"] = True
    if np.issubdtype(arr.dtype, np.floating):
        rounded = np.round(arr)
        if float(np.max(np.abs(arr.astype(np.float64) - rounded))) < 1e-2:
            arr = rounded.astype(np.int32)
            meta["was_float_int"] = True
    if np.issubdtype(arr.dtype, np.integer):
        amin, amax = int(arr.min()), int(arr.max())
        if amin >= 0 and amax <= 65535:
            return np.ascontiguousarray(arr, dtype=np.uint16), {**meta, "kind": "uint16", "bias": 0}
        return np.ascontiguousarray(arr.astype(np.int32) + 32768, dtype=np.uint16), {
            **meta, "kind": "int16", "bias": 32768}
    lo, hi = float(np.min(arr)), float(np.max(arr))
    meta.update({"kind": "uint16_quant", "bias": 0, "quant_lo": lo, "quant_hi": hi})
    if hi <= lo:
        return np.zeros(arr.shape, np.uint16), meta
    scaled = np.round((arr.astype(np.float64) - lo) / (hi - lo) * 65535.0)
    return np.ascontiguousarray(np.clip(scaled, 0, 65535).astype(np.uint16)), meta


def encode_volume(vol: np.ndarray, codec: str = DEFAULT_CODEC) -> bytes:
    vol = np.ascontiguousarray(vol)
    if vol.dtype == np.uint8:
        kind = KIND_U8
    elif vol.dtype == np.uint16:
        kind = KIND_U16
    else:
        raise TypeError(vol.dtype)
    axis = slice_axis(vol.shape)
    n = int(vol.shape[axis])
    buf = bytearray()
    magic, encode, _ = CODECS[codec]
    buf += magic
    buf += struct.pack("<B", vol.ndim)
    buf += struct.pack("<" + "I" * vol.ndim, *[int(s) for s in vol.shape])
    buf += struct.pack("<BBI", axis, kind, n)
    for i in range(n):
        sl = np.squeeze(np.take(vol, i, axis=axis))
        sl = np.ascontiguousarray(sl)
        if sl.ndim == 1:
            sl = sl.reshape(1, -1)
        elif sl.ndim > 2:
            sl = sl.reshape(sl.shape[0], -1)
        enc = encode(sl)
        buf += struct.pack("<I", len(enc))
        buf += enc
    return bytes(buf)


def restore_array(vol: np.ndarray, meta: dict) -> np.ndarray:
    """Invert ``as_uint16_volume`` so the NIfTI holds the original storage values (scl_slope/inter apply)."""
    kind = meta.get("kind")
    if kind == "int16":
        return np.ascontiguousarray(vol.astype(np.int32) - int(meta.get("bias") or 32768), dtype=np.int16)
    if kind == "uint16_quant":
        lo, hi = float(meta["quant_lo"]), float(meta["quant_hi"])
        return np.ascontiguousarray(vol.astype(np.float32) / 65535.0 * (hi - lo) + lo)
    return np.ascontiguousarray(vol, dtype=np.uint16)


def in_subset(member: str, subset: tuple[int, int] | None) -> bool:
    """Shard ``i`` of ``n`` by subject folder (``PT/<subject>/...``): a subject's sessions and sequences stay together."""
    if subset is None:
        return True
    i, n = subset
    parts = member.split("/")
    key = parts[1] if len(parts) > 2 else member
    return zlib.crc32(key.encode()) % n == i


def unpack_jls_zip(zpath: Path, dest_root: Path, *, workers: int = 8,
                   subset: tuple[int, int] | None = None) -> tuple[int, int]:
    """Write ``.nii.gz`` under ``dest_root`` from a JPEG-LS zip (skip existing); ``subset=(i, n)`` takes one
    subject shard."""
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        members = [n for n in zf.namelist() if n.endswith(".jls") and not n.endswith("/") and in_subset(n, subset)]
    jobs = []
    for name in members:
        out = dest_root / Path(name[: -4] + ".nii.gz")  # ``a/b/mprage.jls`` -> ``a/b/mprage.nii.gz``
        if out.exists() and out.stat().st_size > 1000:
            continue
        jobs.append((str(zpath), name, str(out)))
    print(f"[archive] unpack {zpath.name}: {len(members)} volumes, {len(jobs)} to write -> {dest_root}", flush=True)
    if not jobs:
        return 0, 0
    n_ok = n_err = 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_unpack_one, *job) for job in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
                n_ok += 1
            except Exception as e:
                n_err += 1
                if n_err <= 8:
                    print(f"[archive] unpack ERR {type(e).__name__}: {e}", flush=True)
            if i % 100 == 0 or i == len(futs):
                print(f"[archive] unpack {i}/{len(futs)} ok={n_ok} err={n_err}", flush=True)
    return n_ok, n_err


def _unpack_one(zpath: str, member: str, dest: str) -> None:
    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        blob = zf.read(member)
        raw_meta = zf.read(member + ".json")
    meta = json.loads(raw_meta)
    vol = restore_array(decode_volume(blob), meta)
    affine = np.array(meta["affine"], dtype=np.float64)
    # nib.save keys off the suffix; ``.nii.gz.partial.PID`` is not a NIfTI path.
    tmp = dest_p.parent / f"{dest_p.name}.partial.{os.getpid()}.nii.gz"
    img = nib.Nifti1Image(vol, affine)
    slope, inter = meta.get("slope"), meta.get("inter")
    if slope is not None or inter is not None:
        img.header.set_slope_inter(float(slope or 1.0), float(inter or 0.0))
    try:
        nib.save(img, str(tmp))
        os.replace(tmp, dest_p)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def decode_volume(blob: bytes) -> np.ndarray:
    decode = next((d for magic, _, d in CODECS.values() if blob[:8] == magic), None)
    if decode is None:
        raise ValueError(f"unknown volume container {blob[:8]!r}")
    off = 8
    ndim = blob[off]
    off += 1
    shape = struct.unpack_from("<" + "I" * ndim, blob, off)
    off += 4 * ndim
    axis, kind, n = struct.unpack_from("<BBI", blob, off)
    off += 6
    dtype = {KIND_U16: np.uint16, KIND_I16: np.int16, KIND_U8: np.uint8}[kind]
    slices = []
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", blob, off)
        off += 4
        slices.append(decode(blob[off:off + ln]))
        off += ln
    vol = np.stack(slices, axis=axis)
    return np.ascontiguousarray(vol, dtype=dtype).reshape(shape)


def encode_file(path: str) -> tuple[str, bytes, dict]:
    p = Path(path)
    img = nib.load(str(p))
    raw = img.dataobj
    arr = np.asanyarray(raw.get_unscaled() if hasattr(raw, "get_unscaled") else raw)
    vol, meta = as_uint16_volume(arr)
    slope = getattr(raw, "slope", None)
    inter = getattr(raw, "inter", None)
    hdr = img.header
    meta.update({
        "shape": [int(s) for s in vol.shape],
        "affine": img.affine.astype(np.float32).tolist(),
        "zooms": [float(z) for z in hdr.get_zooms()],
        "slope": None if slope is None else float(slope),
        "inter": None if inter is None else float(inter),
        "source": p.name,
    })
    return path, encode_volume(vol), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unpack", type=Path, required=True, help="archive zip to restore as NIfTI under --dest")
    ap.add_argument("--dest", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--subset", default=None, help="i/n subject shard (crc32 of the subject folder)")
    a = ap.parse_args()
    dest = a.dest or (Path(a.unpack).with_suffix("").with_suffix("") / "unpacked")
    subset = tuple(int(v) for v in a.subset.split("/")) if a.subset else None
    n_ok, n_err = unpack_jls_zip(a.unpack, dest, workers=a.workers, subset=subset)
    print(f"[archive] unpack done ok={n_ok} err={n_err}", flush=True)
    if n_err:
        raise SystemExit(n_err)


if __name__ == "__main__":
    main()
