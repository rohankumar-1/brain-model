"""One compressed Zarr ZipStore per volume (Blosc/zstd chunks, zip stored so a chunk reads alone).

Format 6: the image zip holds ``image`` (Z, 1, Y, X) float16 and, when one was computed or supplied, ``brain``
(Z, Y, X) uint8, so a training sample is one file open. Dataset label masks (seg, ICH, lesion, ...) are downstream-only
and stay in ``<case>.<name>.zip`` next to the image.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import zarr

FORMAT_VERSION = 6  # 6: 256 x 256 x 160 grid, brain mask inside the image zip (5: 128 deep, separate .brain.zip)
CHUNK_Z = 8


def _safe(name: str) -> str:
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name)).strip("._")
    return (stem or "volume")[:180]


def path_for(root: Path, folder: str, case_id: str, mask: str | None = None) -> Path:
    return Path(root) / folder / (f"{_safe(case_id)}.{mask}.zip" if mask else f"{_safe(case_id)}.zip")


def paths(root: Path, case) -> tuple[Path, dict[str, Path]]:
    """Image zip and mask zips for a case, using the original store's folder and file names."""
    return (path_for(root, case.folder, case.case_id),
            {name: path_for(root, case.folder, case.mask_id, name) for name, _ in case.masks})


def brain_path(root: Path, case) -> Path:
    """Format-5 brain mask zip (``<case_id>.brain.zip``); format 6 keeps the mask inside the image zip."""
    return path_for(root, case.folder, case.case_id, "brain")


def write_case(root: Path, case, res) -> None:
    """One image zip (``image`` + ``brain``) and the dataset label-mask zips. A stale format-5 brain zip is removed."""
    image, masks = paths(root, case)
    arrays = {"image": (res.vol[:, None], {**res.info, "kind": "ct_hu" if case.modality == "ct" else "mri_z"})}
    if res.brain is not None:
        arrays["brain"] = (res.brain.astype(np.uint8), {k: res.info[k] for k in BRAIN_ATTRS if k in res.info})
    write_arrays(image, arrays)
    for name, m in res.masks.items():
        if name != "brain":                            # a shipped brain mask is the ``brain`` array above
            write(masks[name], "mask", m)
    brain_path(root, case).unlink(missing_ok=True)


BRAIN_ATTRS = ("brain_method", "brain_frac", "brain_ok", "brain_dice_tpl", "brain_rev", "stripped_input")


def remove_case(root: Path, case) -> None:
    image, masks = paths(root, case)
    for p in (image, *masks.values(), brain_path(root, case)):
        p.unlink(missing_ok=True)


def _plain(v):
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer, int, bool, np.bool_)):
        return v.item() if hasattr(v, "item") else v
    if isinstance(v, (list, tuple, np.ndarray)):          # geometry vectors / matrices stay JSON arrays
        return [_plain(x) for x in (v.tolist() if isinstance(v, np.ndarray) else v)]
    return v if isinstance(v, (str, type(None))) else str(v)


def write(path: Path, name: str, arr: np.ndarray, attrs: dict | None = None) -> None:
    """Atomic write of one array. ``arr`` is (Z, ...) chunked CHUNK_Z slices deep."""
    write_arrays(path, {name: (arr, attrs)})


def write_arrays(path: Path, arrays: dict[str, tuple[np.ndarray, dict | None]]) -> None:
    """Atomic write of several arrays into one zip, each (Z, ...) chunked CHUNK_Z slices deep."""
    from zarr.codecs import BloscCodec
    from zarr.storage import ZipStore

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    tmp.unlink(missing_ok=True)
    store = ZipStore(tmp, mode="w", compression=0)
    try:
        group = zarr.open_group(store, mode="w")
        for name, (arr, attrs) in arrays.items():
            arr = np.ascontiguousarray(arr)
            group.create_array(
                name, data=arr, overwrite=True,
                chunks=(min(CHUNK_Z, arr.shape[0]),) + arr.shape[1:],
                compressors=BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle", typesize=arr.dtype.itemsize),
                attributes={"format_version": FORMAT_VERSION, **{k: _plain(v) for k, v in (attrs or {}).items()}})
    except Exception:
        store.close()
        tmp.unlink(missing_ok=True)
        raise
    store.close()
    os.replace(tmp, path)


def read(path: Path, name: str) -> np.ndarray:
    from zarr.storage import ZipStore

    store = ZipStore(path, mode="r")
    try:
        return np.asarray(zarr.open_group(store, mode="r")[name][:])
    finally:
        store.close()


def squeeze_channel(arr: np.ndarray) -> np.ndarray:
    """Stored ``(Z, 1, Y, X)`` image / mask arrays -> ``(Z, Y, X)``; 3D arrays pass through."""
    arr = np.asarray(arr)
    return arr[:, 0] if arr.ndim == 4 else arr


def read_image(path: Path, name: str = "image") -> np.ndarray:
    """One stored volume array as ``(Z, Y, X)`` in the stored dtype."""
    return squeeze_channel(read(path, name))


def read_many(path: Path, names: tuple[str, ...] | list[str]) -> dict[str, np.ndarray]:
    """Several arrays from one zip with a single open; names the zip lacks are omitted."""
    from zarr.storage import ZipStore

    store = ZipStore(path, mode="r")
    try:
        g = zarr.open_group(store, mode="r")
        have = set(g.array_keys())
        return {n: np.asarray(g[n][:]) for n in names if n in have}
    finally:
        store.close()


def read_all(path: Path) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Every array in a zip with its attrs (used to rewrite a zip with one array replaced)."""
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
    """Resume check: readable, right shape and dtype, and written by this format version."""
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
