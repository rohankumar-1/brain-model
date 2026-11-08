"""Native-geometry ("ngeo") CT: the reader, geometry and batching helpers (runs/diagnostics/ngeo/CONTRACT.md, s. 2-3).

Next to each CT image zip ``<case>.zip`` (the 1 mm template grid) the prep writes ``<case>.ngeo.zip``: the scan on its
own slices (axis 0 along the acquisition's slice normal, inferior -> superior, native spacing, never interpolated
through-plane) and 0.5 mm in-plane on a fixed 512 x 512 field, plus ``<case>.<mask>.ngeo.zip`` label masks on the
same grid. ``volumes_ngeo.csv`` in the store root lists them under the 1 mm rows' ``volume_id``, so the label tables
work unchanged.

Frames. Every encoder token carries the position of its centre in TEMPLATE mm: the 1 mm grid's voxel index (x, y, z)
(x = the stored array's last axis) + 0.5, i.e. exactly the coordinates ``BrainQuery`` has always given the 1 mm tokens
of a whole stored volume (``crop_bbox_mm`` at origin 0, spacing 1). The ngeo zip's ``affine_template`` maps its voxel
index (s, r, c) to the template grid's PHYSICAL mm (ITK/LPS); ``template_index_affine`` composes it with the inverse of
the 1 mm grid's index -> physical map (``grid_origin`` / ``grid_spacing`` / ``grid_direction``), so a 1 mm token and
an ngeo token at the same place in the head get the same rotary / position code.

Only ``read_ngeo`` / ``read_ngeo_mask`` know the zip layout; adjust them if the prep's format changes.
"""

from __future__ import annotations

import math
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


def token_centres_mm(aff: torch.Tensor, shape_zyx, patch_xyz) -> torch.Tensor:
    """(B, N, 3) template mm (x, y, z) of each token centre, in the encoder's (s, r, c) token order. ``aff`` (B, 4, 4)
    or (4, 4): ngeo voxel (s, r, c) -> template index (``template_index_affine``); ``shape_zyx`` the (padded) grid
    (S, R, C); ``patch_xyz`` the token's extent in input voxels (col, row, slice)."""
    px, py, pz = (int(v) for v in patch_xyz)
    s_n, r_n, c_n = (int(v) for v in shape_zyx)
    a = aff if aff.dim() == 3 else aff.unsqueeze(0)
    dev = a.device
    s = torch.arange(s_n // pz, device=dev, dtype=torch.float64) * pz + (pz - 1) / 2
    r = torch.arange(r_n // py, device=dev, dtype=torch.float64) * py + (py - 1) / 2
    c = torch.arange(c_n // px, device=dev, dtype=torch.float64) * px + (px - 1) / 2
    src = torch.stack(torch.meshgrid(s, r, c, indexing="ij"), -1).reshape(-1, 3)        # (N, 3) s, r, c
    a = a.double()
    idx = torch.einsum("bij,nj->bni", a[:, :3, :3], src) + a[:, :3, 3].unsqueeze(1)
    return (idx + 0.5).float()                                                           # voxel centre i -> i + 0.5 mm


def token_size_mm(spacing_zyx: torch.Tensor, patch_xyz) -> torch.Tensor:
    """(B, 3) token extent in mm along the encoder's (x, y, z) = (col, row, slice) axes, for ``ScaleEmbed``."""
    px, py, pz = (float(v) for v in patch_xyz)
    sp = spacing_zyx.float()
    return torch.stack([sp[:, 2] * px, sp[:, 1] * py, sp[:, 0] * pz], -1)


def inplane_theta(b: int, shape_zyx, spacing_zyx: torch.Tensor, aug: dict, device,
                  skip: torch.Tensor | None = None) -> torch.Tensor | None:
    """Per-row affine (B, 3, 4) for ``F.affine_grid`` on an ngeo batch (normalised x = col, y = row, z = slice):
    in-plane only, a rotation about the slice normal (``rot_deg_inplane``, default ``rot_deg[2]``: the 1 mm grid's
    axial angle), an in-plane scale (``scale`` [lo, hi], default none) and an in-plane shift (``shift_mm``), with
    probability ``p`` (never on ``skip`` rows). The slice axis maps to itself: no out-of-plane rotation, no
    through-plane resampling. None when nothing is drawn."""
    p = float(aug.get("p", 0.0))
    if p <= 0:
        return None
    on = torch.rand(b, device=device) < p
    if skip is not None:
        on &= ~skip.to(device)
    onf = on.float()
    rot = float(aug.get("rot_deg_inplane", (aug.get("rot_deg") or [0, 0, 10])[2]))
    ang = torch.deg2rad((torch.rand(b, device=device) * 2 - 1) * rot * onf)
    lo, hi = aug.get("scale") or (1.0, 1.0)
    scale = torch.exp(torch.empty(b, device=device).uniform_(math.log(float(lo)), math.log(float(hi))))
    scale = torch.where(on, scale, torch.ones_like(scale))
    shift = (torch.rand(b, 2, device=device) * 2 - 1) * float(aug.get("shift_mm", 0.0)) * onf.unsqueeze(-1)
    sp = spacing_zyx.to(device).float()
    sp = sp.expand(b, 3) if sp.dim() == 2 else sp.view(1, 3).expand(b, 3)
    s_n, r_n, c_n = (int(v) for v in shape_zyx)
    half = torch.stack([c_n * sp[:, 2], r_n * sp[:, 1]], -1) / 2                      # (B, 2) half extent x, y in mm
    c, s = ang.cos() / scale, ang.sin() / scale
    theta = torch.zeros(b, 3, 4, device=device)
    # diag(1/h) R diag(h) / scale in the x-y plane (mm-true rotation on a non-square field); z unchanged
    theta[:, 0, 0], theta[:, 0, 1] = c, -s * half[:, 1] / half[:, 0]
    theta[:, 1, 0], theta[:, 1, 1] = s * half[:, 0] / half[:, 1], c
    theta[:, 2, 2] = 1.0
    theta[:, :2, 3] = shift / half
    return theta


# ------------------------------------------------------------------------------------------------ batching
def pad_slices(item: dict, n: int) -> dict:
    """An ngeo sample padded along the slice axis to ``n`` slices: image with air (-1024), masks / dense targets with
    0; ``slice_keep`` (n,) marks the real slices."""
    s = int(item["image"].shape[0])
    out = dict(item)
    out["slice_keep"] = torch.arange(n) < s
    if s == n:
        return out
    if s > n:
        raise ValueError(f"{s} slices do not fit a batch padded to {n}")

    def pad(t: torch.Tensor, value, axis: int) -> torch.Tensor:
        shape = list(t.shape)
        shape[axis] = n - s
        return torch.cat([t, torch.full(shape, value, dtype=t.dtype)], axis)

    out["image"] = pad(item["image"], AIR_HU, 0)
    for k in ("seg", "dense"):
        if k in item and item[k] is not None and item[k].dim() == 4:
            out[k] = pad(item[k], 0, 1)
    return out


def batch_slices(sizes, multiple: int = 4) -> int:
    """The batch's padded slice count: its longest scan, rounded up to ``multiple`` (fewer distinct shapes for the
    allocator / cuDNN autotuning; the slice axis has no stride, so any count works)."""
    m = max(int(v) for v in sizes)
    k = max(1, int(multiple))
    return int(math.ceil(m / k) * k)
