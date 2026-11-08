"""One compressed Zarr ZipStore per volume (Blosc/zstd chunks, zip stored so a chunk reads alone).

Format 7: the image zip holds ``image`` (Z, 1, Y, X) int16 and, when one was computed or supplied, ``brain``
(Z, Y, X) uint8, so a training sample is one file open. Dataset label masks (seg, ICH, lesion, ...) are downstream-only
and stay in ``<case>.<name>.zip`` next to the image.

Image encoding (format 7, attrs ``dtype_rev: 7`` and ``scale``): CT is int16 Hounsfield units rounded to 1 HU with
everything below -500 HU set to -1024 (air: the token gate cuts at -500 and every window saturates there); MRI is the
robust z as int16 ``round(z * 1000)`` with ``scale: 0.001`` (max error 0.0005 z). ``read`` / ``read_image`` /
``read_many`` decode every image array (``kind`` ct_hu / mri_z) to float (float32 unless ``dtype`` says otherwise), so
callers see native units whatever the zip's version. Format-6 zips (float16 image, no ``dtype_rev``) still read the
same way during the re-encode (brain.prep.reencode); masks and ``brain`` arrays are stored as before.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import zarr

FORMAT_VERSION = 7  # 7: int16 image + scale (6: float16 image; 256 x 256 x 160 grid, brain array in the image zip)
READ_VERSIONS = (6, 7)  # image zips the readers decode
CHUNK_Z = 8
CLEVEL = 9              # Blosc zstd level: 9 vs 5 is ~7% smaller for ~0.8 s more per volume at write; same decode

DTYPE_REV = 7                       # attr marking an int16-encoded image array
IMAGE_KINDS = ("ct_hu", "mri_z")    # the image array's ``kind`` attr
CT_AIR_BELOW, CT_AIR = -500.0, -1024  # HU below CT_AIR_BELOW are stored as CT_AIR
STEP = {"ct_hu": 1.0, "mri_z": 0.001}  # stored integer unit per kind (HU; z)


def encode_image(vol: np.ndarray, kind: str, step: float | None = None,
                 air_below: float | None = CT_AIR_BELOW) -> tuple[np.ndarray, dict]:
    """Native-unit float volume -> (int16 array, attrs to add). CT: values below ``air_below`` HU set to -1024 (None:
    no clamp), then 1 HU steps; MRI: ``step`` z (0.001)."""
    if kind not in IMAGE_KINDS:
        raise ValueError(f"unknown image kind {kind!r}")
    step = float(STEP[kind] if step is None else step)
    v = np.asarray(vol, dtype=np.float32)
    if not np.isfinite(v).all():
        raise ValueError("image has non-finite values")
    extra = {}
    if kind == "ct_hu" and air_below is not None:
        v = np.where(v < np.float32(air_below), np.float32(CT_AIR), v)
        extra["air_below"] = float(air_below)
    q = np.rint(v / np.float32(step)) if step != 1.0 else np.rint(v)
    lo, hi = float(q.min(initial=0)), float(q.max(initial=0))
    if lo < -32768 or hi > 32767:
        raise ValueError(f"image range [{lo * step}, {hi * step}] does not fit int16 at step {step}")
    return q.astype(np.int16), {"dtype_rev": DTYPE_REV, "scale": step, **extra}


def is_image(attrs: dict) -> bool:
    """An image array (decoded on read) rather than a mask / brain array."""
    return attrs.get("kind") in IMAGE_KINDS or int(attrs.get("dtype_rev") or 0) >= DTYPE_REV


def decode(arr: np.ndarray, attrs: dict, dtype=np.float32) -> np.ndarray:
    """A stored array in native units: an int16 image times its ``scale``, a format-6 float16 image as is, both cast
    to ``dtype``. Masks and ``brain`` arrays (no image ``kind``) are returned unchanged."""
    if not is_image(attrs):
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        scale = float(attrs.get("scale", 1.0))
        inv = round(1.0 / scale) if scale > 0 else 0
        div = inv if scale != 1.0 and inv and abs(inv * scale - 1.0) < 1e-9 else None
        if np.dtype(dtype) == np.float16:          # the training loaders: torch's half cast is ~5x numpy's
            import torch

            t = torch.from_numpy(np.ascontiguousarray(arr))
            if scale == 1.0:
                return t.half().numpy()                # integers: exact below 2048 HU, round-to-even above (as before)
            t = t.float()
            return (t.div_(div) if div else t.mul_(scale)).half().numpy()
        if div:                                         # integer divisor: correctly rounded, so -6000 -> -6.0 exactly
            out = np.divide(arr, np.float32(div), dtype=np.float32)
        else:
            out = arr.astype(np.float32)
            if scale != 1.0:
                out *= np.float32(scale)
        return out if np.dtype(dtype) == np.float32 else out.astype(dtype)
    return arr.astype(dtype, copy=False)


def _safe(name: str) -> str:
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name)).strip("._")
    return (stem or "volume")[:180]


def path_for(root: Path, folder: str, case_id: str, mask: str | None = None) -> Path:
    return Path(root) / folder / (f"{_safe(case_id)}.{mask}.zip" if mask else f"{_safe(case_id)}.zip")


def paths(root: Path, case) -> tuple[Path, dict[str, Path]]:
    """Image zip and mask zips for a case, using the original store's folder and file names."""
    return (path_for(root, case.folder, case.case_id),
            {name: path_for(root, case.folder, case.mask_id, name) for name, _ in case.masks})


def write_case(root: Path, case, res) -> None:
    """One image zip (``image`` + ``brain``) and the dataset label-mask zips."""
    image, masks = paths(root, case)
    kind = "ct_hu" if case.modality == "ct" else "mri_z"
    vol, enc = encode_image(res.vol, kind)
    arrays = {"image": (vol[:, None], {**res.info, "kind": kind, **enc})}
    if res.brain is not None:
        arrays["brain"] = (res.brain.astype(np.uint8), {k: res.info[k] for k in BRAIN_ATTRS if k in res.info})
    write_arrays(image, arrays)
    for name, m in res.masks.items():
        if name != "brain":                            # a shipped brain mask is the ``brain`` array above
            write(masks[name], "mask", m)


BRAIN_ATTRS = ("brain_method", "brain_frac", "brain_ok", "brain_dice_tpl", "brain_rev", "stripped_input")


def remove_case(root: Path, case) -> None:
    image, masks = paths(root, case)
    for p in (image, *masks.values()):
        p.unlink(missing_ok=True)


def plain(v):
    """A JSON-safe attribute / manifest value: numpy scalars to Python, non-finite floats to None, arrays to lists."""
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer, int, bool, np.bool_)):
        return v.item() if hasattr(v, "item") else v
    if isinstance(v, (list, tuple, np.ndarray)):          # geometry vectors / matrices stay JSON arrays
        return [plain(x) for x in (v.tolist() if isinstance(v, np.ndarray) else v)]
    return v if isinstance(v, (str, type(None))) else str(v)


def write(path: Path, name: str, arr: np.ndarray, attrs: dict | None = None) -> None:
    """Atomic write of one array. ``arr`` is (Z, ...) chunked CHUNK_Z slices deep."""
    write_arrays(path, {name: (arr, attrs)})


def write_arrays(path: Path, arrays: dict[str, tuple[np.ndarray, dict | None]]) -> None:
    """Atomic write of several arrays into one zip, each (Z, ...) chunked CHUNK_Z slices deep."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    write_zip(tmp, arrays)
    os.replace(tmp, path)


def write_zip(path: Path, arrays: dict[str, tuple[np.ndarray, dict | None]], clevel: int | None = None) -> None:
    """Write ``arrays`` into a new zip at ``path`` (not atomic: callers write a temp name and rename). Attrs get
    ``format_version`` unless they carry their own."""
    from zarr.codecs import BloscCodec
    from zarr.storage import ZipStore

    path = Path(path)
    path.unlink(missing_ok=True)
    store = ZipStore(path, mode="w", compression=0)
    try:
        group = zarr.open_group(store, mode="w")
        for name, (arr, attrs) in arrays.items():
            arr = np.ascontiguousarray(arr)
            group.create_array(
                name, data=arr, overwrite=True,
                chunks=(min(CHUNK_Z, arr.shape[0]),) + arr.shape[1:],
                compressors=BloscCodec(cname="zstd", clevel=CLEVEL if clevel is None else clevel,
                                       shuffle="bitshuffle", typesize=arr.dtype.itemsize),
                attributes={"format_version": FORMAT_VERSION, **{k: plain(v) for k, v in (attrs or {}).items()}})
    except Exception:
        store.close()
        path.unlink(missing_ok=True)
        raise
    store.close()


def _get(a, dtype) -> np.ndarray:
    """One zarr array read in native units (images decoded, see ``decode``)."""
    return decode(np.asarray(a[:]), a.attrs, dtype)


def read(path: Path, name: str, dtype=np.float32) -> np.ndarray:
    """One array; an image comes back decoded to ``dtype`` (native units), any other array as stored."""
    from zarr.storage import ZipStore

    store = ZipStore(path, mode="r")
    try:
        return _get(zarr.open_group(store, mode="r")[name], dtype)
    finally:
        store.close()


def squeeze_channel(arr: np.ndarray) -> np.ndarray:
    """Stored ``(Z, 1, Y, X)`` image / mask arrays -> ``(Z, Y, X)``; 3D arrays pass through."""
    arr = np.asarray(arr)
    return arr[:, 0] if arr.ndim == 4 else arr


def read_image(path: Path, name: str = "image", dtype=np.float32) -> np.ndarray:
    """One stored volume array as ``(Z, Y, X)``: an image in native units as ``dtype``, a mask as stored."""
    return squeeze_channel(read(path, name, dtype))


def read_many(path: Path, names: tuple[str, ...] | list[str], dtype=np.float32) -> dict[str, np.ndarray]:
    """Several arrays from one zip with a single open; names the zip lacks are omitted. The image is decoded to
    ``dtype`` (native units); other arrays come back as stored."""
    from zarr.storage import ZipStore

    store = ZipStore(path, mode="r")
    try:
        g = zarr.open_group(store, mode="r")
        have = set(g.array_keys())
        return {n: _get(g[n], dtype) for n in names if n in have}
    finally:
        store.close()


def read_all(path: Path) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Every array in a zip with its attrs, AS STORED (no decoding: an int16 image stays int16 with its ``scale``), so
    a zip can be rewritten with one array replaced. Use ``decode`` for native units."""
    from zarr.storage import ZipStore

    store = ZipStore(path, mode="r")
    try:
        g = zarr.open_group(store, mode="r")
        names = sorted(g.array_keys())
        return {n: np.asarray(g[n][:]) for n in names}, {n: dict(g[n].attrs) for n in names}
    finally:
        store.close()


def read_attrs(path: Path, name: str) -> dict:
    from zarr.storage import ZipStore

    store = ZipStore(path, mode="r")
    try:
        return dict(zarr.open_group(store, mode="r")[name].attrs)
    finally:
        store.close()


def is_current(path: Path, name: str, shape: tuple[int, ...], dtype) -> bool:
    """Resume check: readable, right shape and STORED dtype (int16 for a format-7 image), and written by this format
    version."""
    if not Path(path).is_file():
        return False
    try:
        from zarr.storage import ZipStore

        store = ZipStore(path, mode="r")
        try:
            a = zarr.open_group(store, mode="r")[name]
            return (tuple(a.shape) == tuple(shape) and np.dtype(a.dtype) == np.dtype(dtype)
                    and a.attrs.get("format_version") == FORMAT_VERSION)
        finally:
            store.close()
    except Exception:
        return False
