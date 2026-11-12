"""Native-geometry ("ngeo") CT copies: the reader and their geometry (runs/diagnostics/ngeo/CONTRACT.md, s. 2-3).

Next to each CT image zip ``<case>.zip`` (the 1 mm template grid) the prep writes ``<case>.ngeo.zip``: the scan on its
own slices (axis 0 along the acquisition's slice normal, inferior -> superior, native spacing, never interpolated
through-plane) and 0.5 mm in-plane on a fixed 512 x 512 field, plus ``<case>.<mask>.ngeo.zip`` label masks on the
same grid. ``volumes_ngeo.csv`` in the store root lists them under the 1 mm rows' ``volume_id``, so the label tables
work unchanged. They are a data product only: since 2026-11-11 the model reads every scan, CT included, from the
1 mm store (native-geometry input broke CT / MRI transfer).

Frames. The ngeo zip's ``affine_template`` maps its voxel index (s, r, c) to the template grid's PHYSICAL mm
(ITK/LPS); ``template_index_affine`` composes it with the inverse of the 1 mm grid's index -> physical map
(``grid_origin`` / ``grid_spacing`` / ``grid_direction``), giving the 1 mm grid's voxel index (x, y, z) (x = the stored
array's last axis) of every native voxel.

Only ``read_ngeo`` / ``read_ngeo_mask`` know the zip layout; adjust them if the prep's format changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

AIR_HU, AIR_BELOW_HU = -1024, -500.0          # the store's CT air clamp (re-encode, contract s. 1)
GRID_KEYS = ("grid_origin", "grid_spacing", "grid_direction")


# ------------------------------------------------------------------------------------------------ files
def ngeo_path(path: str | Path) -> Path:
    """``<case>.zip`` -> ``<case>.ngeo.zip``; ``<case>.<mask>.zip`` -> ``<case>.<mask>.ngeo.zip`` (same folder)."""
    p = Path(path)
    if p.name.endswith(".ngeo.zip"):
        return p
    if not p.name.endswith(".zip"):
        raise ValueError(f"{p}: not a store zip")
    return p.with_name(p.name[:-4] + ".ngeo.zip")


def clamp_air(img: np.ndarray) -> np.ndarray:
    """int16 Hounsfield units with everything below -500 HU at -1024 (the gate's air floor)."""
    a = np.asarray(img)
    if a.dtype != np.int16:
        a = np.rint(np.clip(a.astype(np.float32), -32768, 32767)).astype(np.int16)
    return np.where(a < AIR_BELOW_HU, np.int16(AIR_HU), a)


def read_ngeo(path: str | Path, brain: bool = True) -> dict:
    """One ngeo image zip (contract s. 2) -> ``image`` (S, R, C) int16 HU (air clamped), ``brain`` (S, R, C) uint8 or
    None, ``spacing`` (3,) mm in array order (slice, row, col), ``affine_template`` (4, 4) voxel (s, r, c) -> template
    physical mm, ``grid`` (the 1 mm grid's index -> physical attrs when the zip carries them, else None) and ``attrs``
    (everything on ``image``)."""
    import zarr
    from zarr.storage import ZipStore

    store = ZipStore(Path(path), mode="r")
    try:
        g = zarr.open_group(store, mode="r")
        have = set(g.array_keys())
        img = np.asarray(g["image"][:])
        attrs = dict(g["image"].attrs)
        br = np.asarray(g["brain"][:]) if brain and "brain" in have else None
    finally:
        store.close()
    if img.ndim == 4:                                            # (S, 1, R, C) as written
        img = img[:, 0]
    scale = attrs.get("scale")
    if scale is not None and float(scale) != 1.0:                # store format 7 convention (CT HU: scale 1, none)
        img = img.astype(np.float32) * float(scale)
    sp = attrs.get("spacing_mm") or [attrs.get("slice_mm", 5.0), 0.5, 0.5]
    return {"image": clamp_air(img), "brain": None if br is None else (br > 0).astype(np.uint8),
            "spacing": np.asarray(sp, dtype=np.float32).reshape(3),
            "affine_template": np.asarray(attrs["affine_template"], dtype=np.float64).reshape(4, 4),
            "grid": {k: attrs[k] for k in GRID_KEYS} if all(attrs.get(k) is not None for k in GRID_KEYS) else None,
            # exact per-slice template position (physical mm) of pixel (0, 0) (prep: tpl_slice_origin); None if absent
            "slice_origin": (np.asarray(attrs["tpl_slice_origin"], dtype=np.float64).reshape(-1, 3)
                             if attrs.get("tpl_slice_origin") is not None else None),
            "attrs": attrs}


def read_ngeo_mask(path: str | Path) -> np.ndarray | None:
    """A label mask on the ngeo grid, (S, R, C) in its stored dtype; None when the prep wrote none."""
    from brain.prep import store

    p = Path(path)
    if not p.is_file():
        return None
    return store.read_image(p, "mask")


def grid_attrs(image_zip: str | Path) -> dict:
    """The 1 mm template grid's index -> physical attrs from a store image zip."""
    from brain.prep import store

    a = store.read_attrs(Path(image_zip), "image")
    missing = [k for k in GRID_KEYS if a.get(k) is None]
    if missing:
        raise KeyError(f"{image_zip}: no {missing} (needed to place ngeo tokens in the template frame)")
    return {k: a[k] for k in GRID_KEYS}


# ------------------------------------------------------------------------------------------------ geometry
def template_index_affine(affine_template: np.ndarray, grid: dict) -> np.ndarray:
    """(4, 4): ngeo voxel index (s, r, c) -> 1 mm template grid voxel index (x, y, z) (ITK order), from the zip's
    ``affine_template`` (-> template physical mm) and the grid's ``grid_origin`` / ``grid_spacing`` /
    ``grid_direction`` (index -> physical: o + D diag(sp) i)."""
    d = np.asarray(grid["grid_direction"], dtype=np.float64).reshape(3, 3)
    sp = np.asarray(grid["grid_spacing"], dtype=np.float64).reshape(3)
    g = np.eye(4)
    g[:3, :3] = d @ np.diag(sp)
    g[:3, 3] = np.asarray(grid["grid_origin"], dtype=np.float64).reshape(3)
    return np.linalg.inv(g) @ np.asarray(affine_template, dtype=np.float64).reshape(4, 4)


def template_index_slices(slice_origin: np.ndarray, grid: dict) -> np.ndarray:
    """(S, 3) per-slice template physical origins (``tpl_slice_origin``) -> 1 mm template voxel index (x, y, z)."""
    d = np.asarray(grid["grid_direction"], dtype=np.float64).reshape(3, 3)
    sp = np.asarray(grid["grid_spacing"], dtype=np.float64).reshape(3)
    o = np.asarray(grid["grid_origin"], dtype=np.float64).reshape(3)
    return (np.linalg.inv(d @ np.diag(sp)) @ (np.asarray(slice_origin, dtype=np.float64) - o).T).T


def to_template_index(aff: torch.Tensor, src: torch.Tensor, slice_origin: torch.Tensor | None = None) -> torch.Tensor:
    """Points ``src`` (B, M, 3) in ngeo voxel (s, r, c) -> template index (x, y, z) (B, M, 3). With ``slice_origin``
    (B, S, 3), each point uses its nearest slice's exact origin plus the in-plane steps (columns 1, 2 of ``aff``) and
    the mean slice step for its fractional slice offset: exact on the slices for unevenly spaced series, where the
    least-squares ``aff`` is not (prep ``tpl_resid_mm`` > 3)."""
    a = aff.double()
    src = src.double()
    if slice_origin is None:
        return torch.einsum("bij,bmj->bmi", a[:, :3, :3], src) + a[:, :3, 3].unsqueeze(1)
    o = slice_origin.double()
    s0 = src[..., 0].round().clamp(0, o.shape[1] - 1)
    base = o.gather(1, s0.long().unsqueeze(-1).expand(-1, -1, 3))
    rest = torch.stack([src[..., 0] - s0, src[..., 1], src[..., 2]], -1)
    return base + torch.einsum("bij,bmj->bmi", a[:, :3, :3], rest)
