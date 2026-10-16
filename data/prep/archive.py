"""Pack extracted FOMO NIfTIs onto pool with JPEG-LS, then drop scratch extracts.

Session zips were deleted after extract. Once a family is in ``slices_zarr``, the
extracted ``.nii.gz`` tree is only a scratch copy. This rewrites it as JPEG-LS
(DICOM lossless) uint16 slices, one zip per family or a small-family group::

    python -m data.prep.archive --families fomo_nigerian,fomo_m4raw --delete-extracted
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import imagecodecs
import nibabel as nib
import numpy as np

from data.prep.fomo import FOMO_ROOT, PT_SHORT, pt_from_family
from data.prep.sources import DATASETS as POOL_DATASETS

MAGIC_JLS = b"FOMOJLS1"
KIND_U16, KIND_I16, KIND_U8 = 1, 2, 3
# Tiny families share one zip so pool is not one archive per PT.
GROUP_SMALL = ("PT002_Nigerian_Clinical", "PT024_M4Raw")
_STOP = False


def _request_stop(signum, frame):
    global _STOP
    _STOP = True
    print(f"[archive] signal {signum} — stop after this member", flush=True)


def pool_fomo(root: Path | None = None) -> Path:
    return Path(root) if root else (POOL_DATASETS / "fomo260k")


def archive_for_pt(pt: str, dest_root: Path) -> Path:
    if pt in GROUP_SMALL:
        return dest_root / "fomo_group_small.jls.zip"
    return dest_root / f"{pt}.jls.zip"


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


def encode_volume(vol: np.ndarray) -> bytes:
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
    buf += MAGIC_JLS
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
        enc = imagecodecs.jpegls_encode(sl)
        buf += struct.pack("<I", len(enc))
        buf += enc
    return bytes(buf)


def decode_volume(blob: bytes) -> np.ndarray:
    if blob[:8] != MAGIC_JLS:
        raise ValueError("not FOMOJLS1")
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
        slices.append(imagecodecs.jpegls_decode(blob[off:off + ln]))
        off += ln
    vol = np.stack(slices, axis=axis)
    return np.ascontiguousarray(vol, dtype=dtype).reshape(shape)


def _encode_one(path: str) -> tuple[str, bytes, dict]:
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


def nifti_paths(extract_root: Path, pt: str) -> list[Path]:
    root = extract_root / pt
    if not root.is_dir():
        return []
    return imaging_paths(root)


def imaging_paths(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*"):
        n = p.name.lower()
        if n.endswith(".nii.gz") or n.endswith(".nii") or n.endswith(".img"):
            out.append(p)
    return sorted(out)


def member_name(pt: str, extract_root: Path, path: Path) -> str:
    rel = path.relative_to(extract_root / pt).as_posix()
    if rel.endswith(".nii.gz"):
        rel = rel[: -len(".nii.gz")]
    elif rel.endswith(".nii"):
        rel = rel[: -len(".nii")]
    return f"{pt}/{rel}.jls"


def _staging(zpath: Path) -> Path:
    return zpath.with_name(zpath.name + ".d")


def _existing_jls(staging: Path) -> set[str]:
    if not staging.is_dir():
        return set()
    return {p.relative_to(staging).as_posix() for p in staging.rglob("*.jls")}


def _write_member(staging: Path, name: str, blob: bytes, meta: dict) -> None:
    dest = staging / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    dest.with_name(dest.name + ".json").write_text(json.dumps(meta, separators=(",", ":")))


def salvage_zip(zpath: Path, staging: Path) -> int:
    """Copy zip members into staging. Never unlinks the zip (append-kill corrupts the CD)."""
    if not zpath.exists():
        return 0
    staging.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zpath, "r") as zf:
            names = zf.namelist()
            n = 0
            for name in names:
                dest = staging / name
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(name))
                n += 1
        print(f"[archive] salvaged {n}/{len(names)} from {zpath.name}", flush=True)
        return n
    except (zipfile.BadZipFile, struct.error, ValueError) as e:
        try:
            n = _salvage_local_headers(zpath, staging)
            print(f"[archive] salvaged {n} local-headers from {zpath.name} ({type(e).__name__})", flush=True)
            return n
        except Exception as e2:
            print(f"[archive] skip salvage {zpath.name}: {type(e2).__name__}: {e2}", flush=True)
            return 0


def _salvage_local_headers(zpath: Path, staging: Path) -> int:
    sig = b"PK\x03\x04"
    n = 0
    with zpath.open("rb") as f:
        while True:
            pos = f.tell()
            hdr = f.read(30)
            if len(hdr) < 30:
                break
            if hdr[:4] != sig:
                chunk = hdr[1:] + f.read(1 << 20)
                idx = chunk.find(sig)
                if idx < 0:
                    if len(chunk) < (1 << 20) + 29:
                        break
                    f.seek(pos + 1 + len(chunk) - 3)
                    continue
                f.seek(pos + 1 + idx)
                continue
            _ver, flag, _method, _t, _d, _crc, csize, _usz, fnlen, extlen = struct.unpack_from(
                "<HHHHHIIIHH", hdr, 4)
            name = f.read(fnlen)
            f.read(extlen)
            if flag & 0x8:
                print(f"[archive] data descriptor at {pos}, stop salvage", flush=True)
                break
            data = f.read(csize)
            if len(data) < csize:
                break
            fname = name.decode("utf-8", "replace")
            if not fname or fname.endswith("/"):
                continue
            dest = staging / fname
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                n += 1
            if n and n % 500 == 0:
                print(f"[archive] salvage {n} @ {f.tell() / 1e9:.1f}G", flush=True)
    return n


def _zip_staging(staging: Path, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".partial")
    files = sorted(p for p in staging.rglob("*") if p.is_file())
    print(f"[archive] zipping {len(files)} files -> {dest.name}", flush=True)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for i, p in enumerate(files, 1):
            zf.write(p, p.relative_to(staging).as_posix())
            if i % 500 == 0:
                print(f"[archive] zip {i}/{len(files)}", flush=True)
            if _STOP:
                print("[archive] stop while zipping; zip not finalized", flush=True)
                return
    os.replace(tmp, dest)
    print(f"[archive] ready {dest} ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)


def _encode_to_staging(jobs: list[tuple[str, str]], staging: Path, workers: int) -> tuple[int, int]:
    if not jobs:
        return 0, 0
    n_ok = n_err = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_encode_one, path): name for path, name in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            name = futs[fut]
            try:
                _src, blob, meta = fut.result()
            except Exception as e:
                n_err += 1
                if n_err <= 8:
                    print(f"[archive] ERR {name} {type(e).__name__}: {e}", flush=True)
                continue
            _write_member(staging, name, blob, meta)
            n_ok += 1
            if i % 50 == 0:
                print(f"[archive] {i}/{len(jobs)} ok={n_ok} err={n_err} {name}", flush=True)
            if _STOP:
                print(f"[archive] stop after {n_ok} this slot", flush=True)
                break
    return n_ok, n_err


def pack_pts(pts: list[str], extract_root: Path, dest_root: Path, workers: int,
             delete_extracted: bool) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    by_zip: dict[Path, list[str]] = {}
    for pt in pts:
        by_zip.setdefault(archive_for_pt(pt, dest_root), []).append(pt)

    for zpath, group in by_zip.items():
        staging = _staging(zpath)
        staging.mkdir(parents=True, exist_ok=True)
        for cand in (zpath, zpath.with_name(zpath.name + ".partial")):
            salvage_zip(cand, staging)
        have = _existing_jls(staging)
        jobs = []
        want: list[str] = []
        for pt in group:
            for p in nifti_paths(extract_root, pt):
                name = member_name(pt, extract_root, p)
                want.append(name)
                if name not in have:
                    jobs.append((str(p), name))
            mp = FOMO_ROOT / pt / "mapping.tsv"
            if mp.exists():
                dest = staging / pt / "mapping.tsv"
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(mp.read_bytes())
        print(f"[archive] {staging.name}: {len(have)} have, {len(jobs)} to encode pts={group}", flush=True)
        n_ok, n_err = _encode_to_staging(jobs, staging, workers)
        print(f"[archive] wrote staging ok={n_ok} err={n_err}", flush=True)
        if n_err and not _STOP:
            raise SystemExit(f"encode errors: {n_err}")
        have = _existing_jls(staging)
        missing = [n for n in want if n not in have]
        if missing or _STOP:
            print(f"[archive] partial {len(have)}/{len(want)} missing={len(missing)}", flush=True)
            continue
        _zip_staging(staging, zpath)
        if not zpath.exists() or _STOP:
            continue
        _rmtree(staging)
        for cand in (zpath.with_name(zpath.name + ".partial"),):
            if cand.exists() and cand != zpath:
                cand.unlink()
        if delete_extracted:
            for pt in group:
                target = extract_root / pt
                print(f"[archive] delete extracted {target}", flush=True)
                _rmtree(target)


def _rmtree(path: Path) -> None:
    import shutil
    if path.is_dir():
        shutil.rmtree(path)


def families_to_pts(families: list[str]) -> list[str]:
    out = []
    for f in families:
        f = f.strip()
        if not f:
            continue
        if f in PT_SHORT:
            out.append(f)
        else:
            out.append(pt_from_family(f))
    return out


def pack_tree(src: Path, dest: Path, workers: int, delete_source: bool) -> None:
    """JPEG-LS pack every NIfTI/Analyze under ``src`` into one zip at ``dest``."""
    src = src.resolve()
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    paths = imaging_paths(src)
    print(f"[archive] tree {src} n={len(paths)} -> {dest}", flush=True)
    staging = _staging(dest)
    staging.mkdir(parents=True, exist_ok=True)
    for cand in (dest, dest.with_name(dest.name + ".partial")):
        salvage_zip(cand, staging)
    have = _existing_jls(staging)
    jobs = []
    want: list[str] = []
    for p in paths:
        rel = p.relative_to(src).as_posix()
        for suf in (".nii.gz", ".nii", ".img"):
            if rel.endswith(suf):
                rel = rel[: -len(suf)]
                break
        name = rel + ".jls"
        want.append(name)
        if name not in have:
            jobs.append((str(p), name))
    print(f"[archive] {staging.name}: {len(have)} have, {len(jobs)} to encode", flush=True)
    n_ok, n_err = _encode_to_staging(jobs, staging, workers)
    print(f"[archive] wrote staging ok={n_ok} err={n_err}", flush=True)
    if n_err and not _STOP:
        raise SystemExit(f"encode errors: {n_err}")
    have = _existing_jls(staging)
    missing = [n for n in want if n not in have]
    if missing or _STOP:
        print(f"[archive] partial {len(have)}/{len(want)} missing={len(missing)}", flush=True)
        return
    _zip_staging(staging, dest)
    if not dest.exists() or _STOP:
        return
    _rmtree(staging)
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists() and partial != dest:
        partial.unlink()
    if delete_source:
        print(f"[archive] delete source {src}", flush=True)
        _rmtree(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="", help="comma list of fomo_<short> or PT names")
    ap.add_argument("--tree", type=Path, default=None, help="pack an arbitrary image tree (NIfTI/Analyze)")
    ap.add_argument("--extract-root", type=Path, default=FOMO_ROOT / "extracted")
    ap.add_argument("--dest", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--delete-extracted", action="store_true")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    if a.tree:
        dest = a.dest or (a.tree.parent / (a.tree.name + ".jls.zip"))
        print(f"[archive] tree={a.tree} dest={dest} workers={a.workers}", flush=True)
        pack_tree(a.tree, dest, a.workers, a.delete_extracted)
        return
    families = [x for x in a.families.split(",") if x.strip()]
    if not families:
        raise SystemExit("pass --families or --tree")
    pts = families_to_pts(families)
    dest = a.dest or pool_fomo()
    print(f"[archive] pts={pts} dest={dest} workers={a.workers}", flush=True)
    pack_pts(pts, a.extract_root, dest, a.workers, a.delete_extracted)


if __name__ == "__main__":
    main()
