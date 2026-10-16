"""One compressed Zarr ZipStore per array (Blosc/zstd chunks, zip stored so a chunk reads alone)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import zarr

FORMAT_VERSION = 5  # 5: template-registered grid, native-geometry attrs, optional brain-mask zip
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
    """Per-volume brain mask zip: ``<case_id>.brain.zip`` next to the image."""
    return path_for(root, case.folder, case.case_id, "brain")


def write_case(root: Path, case, res) -> None:
    """Image zip, dataset-supplied mask zips, and (when computed) the brain-mask zip."""
    image, masks = paths(root, case)
    write(image, "image", res.vol[:, None], {**res.info, "kind": "ct_hu" if case.modality == "ct" else "mri_z"})
    for name, m in res.masks.items():
        write(masks[name], "mask", m)
    if res.brain is not None and "brain" not in res.masks:
        write(brain_path(root, case), "mask", res.brain)


def remove_case(root: Path, case) -> None:
    image, masks = paths(root, case)
    for p in (image, *masks.values(), brain_path(root, case)):
        p.unlink(missing_ok=True)


def _plain(v):
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer, int, bool, np.bool_)):
        return v.item() if hasattr(v, "item") else v
    return v if isinstance(v, (str, type(None))) else str(v)


def write(path: Path, name: str, arr: np.ndarray, attrs: dict | None = None) -> None:
    """Atomic write. ``arr`` is (Z, ...) chunked CHUNK_Z slices deep."""
    from zarr.codecs import BloscCodec
    from zarr.storage import ZipStore

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    tmp.unlink(missing_ok=True)
    store = ZipStore(tmp, mode="w", compression=0)
    try:
        zarr.open_group(store, mode="w").create_array(
            name, data=np.ascontiguousarray(arr), overwrite=True,
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
