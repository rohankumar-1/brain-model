"""Crops on the volume's own token grid, cut in the loader workers.

Every crop starts on a coarse-patch boundary, so a token of one crop is the same voxels as the token of
another crop that covers the same place. Each view therefore carries ``pos``: the flat index, in the
volume's coarse grid, of each position it covers, and ``parent``: which of those positions each of its
tokens belongs to. Coarse crops have one token per position; fine-patch crops have several.

Groups: ``global`` (large), ``local`` (small, same patch), ``fine`` (small, finer patch), ``win``
(one large crop that is windowed twice later). Axis order is (Z, Y, X) for grids, (x, y, z) for vectors
that mirror the config.
"""

from __future__ import annotations

from typing import Sequence

import torch

from brain.data.gate import token_keep

GROUPS = ("global", "local", "fine", "win")


def _integral(grid: torch.Tensor) -> torch.Tensor:
    """Summed-area table of a bool grid (B, Z, Y, X)."""
    b, gz, gy, gx = grid.shape
    sat = grid.new_zeros(b, gz + 1, gy + 1, gx + 1, dtype=torch.int32)
    sat[:, 1:, 1:, 1:] = grid
    return sat.cumsum(1).cumsum(2).cumsum(3)


def _box_mean(sat: torch.Tensor, o: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Foreground fraction of the boxes at origins ``o`` (B, 3), sizes ``size`` (Z, Y, X)."""
    i = torch.arange(o.shape[0])
    z0, y0, x0 = o.unbind(1)
    z1, y1, x1 = z0 + size[0], y0 + size[1], x0 + size[2]
    at = lambda z, y, x: sat[i, z, y, x]  # noqa: E731
    total = (at(z1, y1, x1) - at(z0, y1, x1) - at(z1, y0, x1) - at(z1, y1, x0)
             + at(z0, y0, x1) + at(z0, y1, x0) + at(z1, y0, x0) - at(z0, y0, x0))
    return total.float() / float(size[0] * size[1] * size[2])


def _place(center: torch.Tensor, length: int, lo: torch.Tensor, hi: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Start of a crop of ``length`` that contains ``center`` and lies in [lo, hi) if it fits."""
    first = torch.maximum(lo, center - length + 1)
    last = torch.minimum(center, hi - length)
    first = torch.where(last < first, lo, first)  # region smaller than the crop: pin to its start
    last = torch.maximum(last, first)
    u = torch.rand(center.shape[0], generator=gen)
    return first + (u * (last - first + 1)).long().clamp(max=(last - first))


def _sample(
    grid: torch.Tensor, sat: torch.Tensor, size: Sequence[int], region: tuple[torch.Tensor, torch.Tensor],
    bounds: tuple[torch.Tensor, torch.Tensor], fg_min: float, retries: int, gen: torch.Generator,
) -> torch.Tensor:
    """Crop origins (B, 3) in (z, y, x) tokens. A foreground token inside ``region`` is the crop's anchor.

    The crop stays inside ``bounds``. The best of ``retries`` draws is kept, or the first that has at
    least ``fg_min`` foreground.
    """
    b, gz, gy, gx = grid.shape
    zz, yy, xx = (torch.arange(n) for n in (gz, gy, gx))
    lo, hi = region
    inside = (
        ((zz[None] >= lo[:, 0:1]) & (zz[None] < hi[:, 0:1]))[:, :, None, None]
        & ((yy[None] >= lo[:, 1:2]) & (yy[None] < hi[:, 1:2]))[:, None, :, None]
        & ((xx[None] >= lo[:, 2:3]) & (xx[None] < hi[:, 2:3]))[:, None, None, :]
    )
    probs = (grid & inside).reshape(b, -1).float()
    probs = torch.where(probs.sum(1, keepdim=True) > 0, probs, inside.reshape(b, -1).float())
    best_frac, best = torch.full((b,), -1.0), torch.zeros(b, 3, dtype=torch.long)
    for _ in range(max(int(retries), 1)):
        idx = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        center = torch.stack([idx // (gy * gx), (idx // gx) % gy, idx % gx], 1)
        o = torch.stack([_place(center[:, a], size[a], bounds[0][:, a], bounds[1][:, a], gen) for a in range(3)], 1)
        frac = _box_mean(sat, o, size)
        take = frac > best_frac
        best_frac, best = torch.where(take, frac, best_frac), torch.where(take[:, None], o, best)
        if bool((best_frac >= fg_min).all()):
            break
    return best


def _box(o: torch.Tensor, size: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    return o, o + o.new_tensor(list(size))


def _flat_index(o: torch.Tensor, size: Sequence[int], grid_zyx: Sequence[int]) -> torch.Tensor:
    """Flat coarse-grid index (B, N) of every position of a box, in (Z, Y, X) order."""
    gz, gy, gx = grid_zyx
    z, y, x = torch.meshgrid(*(torch.arange(n) for n in size), indexing="ij")
    return ((o[:, 0, None] + z.reshape(-1)) * gy + (o[:, 1, None] + y.reshape(-1))) * gx + (o[:, 2, None] + x.reshape(-1))


def _gather(volume: torch.Tensor, o_vox: torch.Tensor, size_vox: Sequence[int]) -> torch.Tensor:
    """One crop per row of a (B, C, Z, Y, X) volume, copied along the contiguous X axis."""
    b, c = volume.shape[:2]
    sz, sy, sx = size_vox
    out = volume.new_empty(b, c, sz, sy, sx)
    for i in range(b):
        z0, y0, x0 = (int(v) for v in o_vox[i])
        out[i] = volume[i, :, z0:z0 + sz, y0:y0 + sy, x0:x0 + sx]
    return out


def _coords(o_vox: torch.Tensor, size_tok: Sequence[int], patch_xyz: Sequence[int], spacing: torch.Tensor,
            origin_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Token centers in mm (B, N, 3 as x, y, z) in (Z, Y, X) token order, and the patch size in mm (B, 3)."""
    px, py, pz = (int(v) for v in patch_xyz)
    z, y, x = torch.meshgrid(*(torch.arange(n) + 0.5 for n in size_tok), indexing="ij")
    vox = torch.stack([
        o_vox[:, 2, None] + x.reshape(-1) * px,
        o_vox[:, 1, None] + y.reshape(-1) * py,
        o_vox[:, 0, None] + z.reshape(-1) * pz,
    ], -1).float()
    return origin_mm[:, None, :] + vox * spacing[:, None, :], spacing * spacing.new_tensor([px, py, pz])


def _snap(size_xyz: Sequence[int], patch_xyz: Sequence[int], volume_xyz: Sequence[int]) -> tuple[int, int, int]:
    """Crop size in tokens (Z, Y, X): whole patches, at least one, no bigger than the volume."""
    t = [max(1, min(int(s), int(n)) // int(p)) for s, p, n in zip(size_xyz, patch_xyz, volume_xyz)]
    return t[2], t[1], t[0]


def _rotation_zyx(deg: Sequence[float]) -> torch.Tensor:
    """3x3 rotation in (z, y, x) index space from angles (degrees) about the z, y and x axes."""
    az, ay, ax = (torch.deg2rad(torch.tensor(float(d), dtype=torch.float64)) for d in deg)
    c, s = torch.cos, torch.sin
    rz = torch.tensor([[1, 0, 0], [0, c(az), -s(az)], [0, s(az), c(az)]], dtype=torch.float64)   # in the y-x plane
    ry = torch.tensor([[c(ay), 0, s(ay)], [0, 1, 0], [-s(ay), 0, c(ay)]], dtype=torch.float64)   # in the z-x plane
    rx = torch.tensor([[c(ax), -s(ax), 0], [s(ax), c(ax), 0], [0, 0, 1]], dtype=torch.float64)   # in the z-y plane
    return rz @ ry @ rx


def draw_geometry(geo: dict, b: int, gen: torch.Generator | None) -> torch.Tensor:
    """(B, 3, 3) maps from a view's voxel offsets to source-volume offsets, (z, y, x): a random rotation (uniform in
    +-``rot_deg`` per axis), isotropic scale (``scale`` range; > 1 zooms in) and left-right flip (``flip_lr``), each
    applied with probability ``p``. Physical voxel size is unchanged: a scaled view looks like a larger or smaller
    head, which stays within normal anatomical variation for the default +-15%."""
    rot = [float(r) for r in geo.get("rot_deg", (15.0, 15.0, 15.0))]
    lo, hi = (float(v) for v in geo.get("scale", (0.85, 1.15)))
    p, p_flip = float(geo.get("p", 1.0)), float(geo.get("flip_lr", 0.5))
    out = torch.empty(b, 3, 3, dtype=torch.float64)
    for i in range(b):
        u = torch.rand(6, generator=gen, dtype=torch.float64)
        if float(u[0]) >= p:
            out[i] = torch.eye(3, dtype=torch.float64)
            continue
        ang = [(2 * float(u[1 + a]) - 1) * rot[a] for a in range(3)]
        scale = lo + (hi - lo) * float(u[4])
        flip = torch.diag(torch.tensor([1.0, 1.0, -1.0 if float(u[5]) < p_flip else 1.0], dtype=torch.float64))
        out[i] = _rotation_zyx(ang) @ flip / scale
    return out


def _theta(m: torch.Tensor, center: torch.Tensor, size_vox: Sequence[int], vol_zyx: Sequence[int]) -> torch.Tensor:
    """``affine_grid`` thetas (B, 3, 4), (x, y, z) order, align_corners=False, for views of ``size_vox`` (Z, Y, X)
    whose voxel offset from the view center maps through ``m`` (B, 3, 3) to ``center`` + offset in the volume."""
    d = torch.tensor([float(v) for v in vol_zyx], dtype=torch.float64)
    half = torch.tensor([float(v) / 2 for v in size_vox], dtype=torch.float64)
    lin = (2.0 / d)[None, :, None] * m * half[None, None, :]                     # zyx -> zyx
    trans = 2.0 * center / d - 1.0
    rev = [2, 1, 0]
    return torch.cat([lin[:, rev][:, :, rev], trans[:, rev, None]], dim=2).float()


def _map_tokens(m, center, size_vox, patch_zyx, size_tok) -> torch.Tensor:
    """Source-volume voxel position (B, N, 3; z, y, x) of every token center of a view, token order (Z, Y, X)."""
    zz, yy, xx = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) + 0.5 for n in size_tok), indexing="ij")
    u = torch.stack([zz, yy, xx], -1).reshape(-1, 3) * torch.tensor([float(p) for p in patch_zyx], dtype=torch.float64)
    off = u - torch.tensor([float(s) / 2 for s in size_vox], dtype=torch.float64)
    return center[:, None, :] + torch.einsum("bij,nj->bni", m, off)


def _lookup(src: torch.Tensor, cell_zyx: Sequence[int], grid: torch.Tensor):
    """Cell index and inside-flag of source positions ``src`` (B, N, 3) on a (B, Z, Y, X) grid of cells."""
    cell = torch.tensor([float(c) for c in cell_zyx], dtype=torch.float64)
    idx = torch.floor(src / cell).long()
    shape = torch.tensor(grid.shape[1:])
    inside = ((idx >= 0) & (idx < shape)).all(-1)
    idx = torch.minimum(idx.clamp_min(0), shape - 1)
    b = torch.arange(grid.shape[0])[:, None]
    flat = (idx[..., 0] * shape[1] + idx[..., 1]) * shape[2] + idx[..., 2]
    return flat, inside, grid[b, idx[..., 0], idx[..., 1], idx[..., 2]] & inside


def _group_from_boxes(
    volume, origins_tok, size_coarse, patch, coarse_patch, grid_zyx, keep, spacing, origin_mm,
    stripped=None, drop=None, keep_stripped=None, geo=None, vol_zyx=None, coarse_keep=None, lesion_grid=None,
) -> dict:
    """Stack the views of one group. ``origins_tok`` are coarse (z, y, x) token origins, one (B, 3) per view.

    With ``geo`` (one (B, 3, 3) map per view, ``draw_geometry``) a view is not cut here: it is sampled on the GPU
    (``batch3d.prepare``) from the whole volume through ``theta``, rotated / scaled / flipped about the crop's
    center. Its tokens are then matched to the volume by where their centers land: ``pos`` / ``tok_pos`` are the
    coarse / native cells of those points, and ``fg`` the volume's token gate there, so views of the same anatomy
    still meet in the dense terms."""
    ratio = [c // p for c, p in zip(coarse_patch, patch)]  # per (x, y, z); 1 for coarse groups
    size_tok = (size_coarse[0] * ratio[2], size_coarse[1] * ratio[1], size_coarse[2] * ratio[0])
    cpx, cpy, cpz = coarse_patch
    views, coords, fg, pmm, parent, pos, tok_pos = [], [], [], [], [], [], []
    lesion = []
    n_coarse = size_coarse[0] * size_coarse[1] * size_coarse[2]
    native_grid = tuple(keep.shape[1:])
    zz, yy, xx = torch.meshgrid(*(torch.arange(n) for n in size_tok), indexing="ij")
    slot = ((zz // ratio[2]) * size_coarse[1] + yy // ratio[1]) * size_coarse[2] + xx // ratio[0]
    size_vox = tuple(t * q for t, q in zip(size_tok, (patch[2], patch[1], patch[0])))
    thetas = []
    for o in origins_tok:
        o_vox = o * o.new_tensor([cpz, cpy, cpx])
        o_native = o * o.new_tensor([ratio[2], ratio[1], ratio[0]])
        if geo is not None:
            k = len(thetas)
            dropped = drop is not None and drop[k]
            m = geo[k]
            center = o_vox.double() + torch.tensor([float(s) / 2 for s in size_vox], dtype=torch.float64)
            thetas.append(_theta(m, center, size_vox, vol_zyx))
            src_tok = _map_tokens(m, center, size_vox, (patch[2], patch[1], patch[0]), size_tok)
            gate = keep_stripped if dropped and keep_stripped is not None else keep
            nat, _, f = _lookup(src_tok, (patch[2], patch[1], patch[0]), gate)
            if lesion_grid is not None:
                lesion.append(_lookup(src_tok, (patch[2], patch[1], patch[0]), lesion_grid)[2])
            slot_src = _map_tokens(m, center, size_vox, (cpz, cpy, cpx), size_coarse)
            cpos, cin, _ = _lookup(slot_src, (cpz, cpy, cpx), coarse_keep)
            # a token whose slot center left the volume has no partner position: drop it from every term
            f = f & cin.gather(1, slot.reshape(1, -1).expand(o.shape[0], -1))
            c, p = _coords(o_vox, size_tok, patch, spacing, origin_mm)
            coords.append(c)
            pmm.append(p)
            pos.append(cpos)
            tok_pos.append(nat)
            parent.append(slot.reshape(1, -1).expand(o.shape[0], -1))
            fg.append(f)
            continue
        src = stripped if (stripped is not None and drop is not None and drop[len(views)]) else volume
        views.append(_gather(src, o_vox, size_vox))
        if lesion_grid is not None:
            lesion.append(_crop_fg(lesion_grid, o_native, size_tok))
        c, p = _coords(o_vox, size_tok, patch, spacing, origin_mm)
        coords.append(c)
        pmm.append(p)
        pos.append(_flat_index(o, size_coarse, grid_zyx))
        tok_pos.append(_flat_index(o_native, size_tok, native_grid))
        parent.append(slot.reshape(1, -1).expand(o.shape[0], -1))
        dropped = stripped is not None and drop is not None and drop[len(views) - 1]
        fg.append(_crop_fg(keep_stripped if dropped and keep_stripped is not None else keep, o_native, size_tok))
    out = {"coords": torch.stack(coords), "patch_mm": torch.stack(pmm),
           "parent": torch.stack(parent).to(torch.int32), "pos": torch.stack(pos).to(torch.int32),
           "tok_pos": torch.stack(tok_pos).to(torch.int32), "fg": torch.stack(fg), "n_par": n_coarse}
    if lesion:
        out["lesion"] = torch.stack(lesion)
    if geo is not None:
        out["theta"] = torch.stack(thetas)
        out["drop"] = torch.tensor([bool(drop is not None and drop[k]) for k in range(len(thetas))])[:, None] \
            .expand(-1, origins_tok[0].shape[0]).clone()
        out["vshape"] = tuple(int(v) for v in size_vox)
    else:
        out["view"] = torch.stack(views)
    return out


def _crop_fg(keep: torch.Tensor, o: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Foreground of a token box, (B, N), from a (B, Z, Y, X) bool grid."""
    b = keep.shape[0]
    zz, yy, xx = torch.meshgrid(*(torch.arange(n) for n in size), indexing="ij")
    z, y, x = o[:, 0, None] + zz.reshape(-1), o[:, 1, None] + yy.reshape(-1), o[:, 2, None] + xx.reshape(-1)
    return keep[torch.arange(b)[:, None], z, y, x]


def token_lesion(lesion: torch.Tensor, patch_xyz, min_frac: float = 0.01) -> torch.Tensor:
    """(B, Z', Y', X') bool: tokens of ``patch_xyz`` whose voxels are at least ``min_frac`` lesion. ``lesion`` is
    (B, 1, Z, Y, X) bool on the volume grid."""
    px, py, pz = (int(p) for p in patch_xyz)
    b, _, z, y, x = lesion.shape
    z, y, x = z - z % pz, y - y % py, x - x % px
    pooled = torch.nn.functional.avg_pool3d(lesion[:, :, :z, :y, :x].float(), (pz, py, px))
    return pooled[:, 0] >= min_frac


def raw_views(batch: dict, cfg: dict, gen: torch.Generator | None = None) -> dict:
    """CPU crops of a (B, 1, Z, Y, X) volume in any float dtype.

    Per group ``g``: ``g_view`` (V, B, 1, Z, Y, X), ``g_coords`` (V, B, N, 3), ``g_fg`` (V, B, N),
    ``g_patch_mm`` (V, B, 3), ``g_parent`` (V, B, N), ``g_pos`` (V, B, P) on the coarse volume grid, and
    ``g_tok_pos`` (V, B, N) on this group's native volume grid (fine patches: 8 cells per coarse
    position). The window pair is one crop.
    """
    volume = batch["volume"].detach()
    modality = batch["modality"].detach().cpu()
    spacing = batch["spacing_mm"].detach().float().cpu()
    origin_mm = batch["origin_mm"].detach().float().cpu()
    v = cfg["views"]
    gate = v.get("gate") or {}
    gate = dict(ct_hu=tuple(gate.get("ct_hu", (-500.0, 200.0))), mri_lo=float(gate.get("mri_lo", -5.5)))
    patch = tuple(int(p) for p in v["patch_xyz"])
    b, _, nz, ny, nx = volume.shape
    xyz = volume.permute(0, 1, 4, 3, 2)
    grid = token_keep(xyz, modality, patch, **gate)
    grid_zyx = tuple(grid.shape[1:])
    sat = _integral(grid)
    if gen is None:
        gen = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,))))
    kw = dict(fg_min=float(v.get("fg_min", 0.15)), retries=int(v.get("retries", 8)), gen=gen)
    whole = (torch.zeros(b, 3, dtype=torch.long), torch.tensor(grid_zyx).expand(b, 3))
    nested = bool(v.get("nested", True))

    def spec(name: str) -> dict:
        return v.get(name) or {}

    n_g, n_l, n_f = (int(spec(k).get("n", 0)) for k in ("global", "local", "fine"))
    size_g = _snap(spec("global")["size_xyz"], patch, (nx, ny, nz))
    origins: dict[str, list[torch.Tensor]] = {"global": [], "local": [], "fine": [], "win": []}
    for i in range(n_g):
        region = _box(origins["global"][0], size_g) if (nested and i) else whole
        origins["global"].append(_sample(grid, sat, size_g, region, whole, **kw))
    for name, n in (("local", n_l), ("fine", n_f)):
        if n == 0:
            continue
        size = _snap(spec(name)["size_xyz"], patch, (nx, ny, nz))
        for k in range(n):
            if nested:
                parent = _box(origins["global"][k % n_g], size_g)
                origins[name].append(_sample(grid, sat, size, parent, parent, **kw))
            else:
                origins[name].append(_sample(grid, sat, size, whole, whole, **kw))
    if bool(v.get("win", True)):                  # the window-pair crop (LeJEPA); MAE does not use it
        first = _box(origins["global"][0], size_g) if nested else whole
        origins["win"].append(_sample(grid, sat, size_g, first, whole, **kw))

    # Per-view skull dropout (``aug.skull.per_view``): each view independently is cut to the brain mask, so views of
    # one scan with and without skull meet in the invariance terms. A skull-free view gates its tokens on the cut
    # scan, so positions outside the brain drop out like air (and leave the dense consensus) instead of asking
    # skull tokens to match background.
    brain = batch.get("brain")
    stripped, skull_p = None, float(((cfg.get("aug") or {}).get("skull") or {}).get("p", 0.0))
    if brain is not None and skull_p > 0:
        floor = torch.where(modality.view(-1, 1, 1, 1, 1) == 0, -1024.0, -6.0).to(volume.dtype)
        stripped = torch.where(brain.bool(), volume, floor)

    def drops(n: int) -> list[bool]:
        return [stripped is not None and float(torch.rand(1, generator=gen)) < skull_p for _ in range(n)]

    # Geometric augmentation (``aug.geometric``): every view gets its own rotation / scale / flip. Views are then
    # sampled on the GPU from the whole volume, which ships once instead of the cut crops.
    geo_cfg = (cfg.get("aug") or {}).get("geometric")
    geo_on = bool(geo_cfg) and float(geo_cfg.get("p", 1.0)) > 0 and bool(batch.get("geometric", True))

    thick, axis = batch.get("thick_mm"), batch.get("thick_axis")
    out: dict = {"modality": modality,
                 "contrast_id": batch.get("contrast_id", torch.full((b,), -1, dtype=torch.long)),
                 "tissue_q": batch.get("tissue_q", torch.full((b, 32), float("nan"))),
                 "thick_mm": thick.detach().float().cpu() if thick is not None else torch.ones(b),
                 "thick_axis": axis.detach().long().cpu() if axis is not None else torch.full((b,), -1, dtype=torch.long),
                 "patch": {}, "n_pos": int(grid_zyx[0] * grid_zyx[1] * grid_zyx[2]),
                 "n_par": {}, "groups": []}
    for name in GROUPS:
        if not origins[name]:
            continue
        gp = tuple(int(p) for p in spec(name).get("patch_xyz", patch)) if name == "fine" else patch
        size = size_g if name in ("global", "win") else _snap(spec(name)["size_xyz"], patch, (nx, ny, nz))
        keep = token_keep(xyz, modality, gp, **gate) if gp != patch else grid
        keep_s = None
        if stripped is not None:
            keep_s = token_keep(stripped.permute(0, 1, 4, 3, 2), modality, gp, **gate)
        n_views = len(origins[name])
        geo = None
        if geo_on:
            m = draw_geometry(geo_cfg, b, gen)
            geo = [m] * n_views if name == "win" else [draw_geometry(geo_cfg, b, gen) for _ in range(n_views)]
        les_grid = None
        if batch.get("lesion") is not None:     # token holds lesion if >= 1% of its voxels are lesion
            les_grid = token_lesion(batch["lesion"], gp)
        g = _group_from_boxes(volume, origins[name], size, gp, patch, grid_zyx, keep, spacing, origin_mm,
                              stripped, drops(n_views), keep_s, geo=geo, vol_zyx=(nz, ny, nx), coarse_keep=grid,
                              lesion_grid=les_grid)
        out["groups"].append(name)
        out["patch"][name] = gp
        out["n_par"][name] = g.pop("n_par")
        if "vshape" in g:
            out.setdefault("vshape", {})[name] = g.pop("vshape")
        out.update({f"{name}_{k}": t for k, t in g.items()})
        if name == "fine":
            out["n_pos_fine"] = int(keep[0].numel())
    if geo_on:
        out["volume_full"] = volume.half() if volume.dtype != torch.float16 else volume
        skull = (cfg.get("aug") or {}).get("skull") or {}
        if bool(skull.get("per_view", False)) and float(skull.get("p", 0.0)) > 0:
            # every volume of a chunk ships a mask so they batch together; without a usable one it never drops
            out["brain_full"] = brain.bool() if stripped is not None else torch.ones_like(volume, dtype=torch.bool)
    return out
