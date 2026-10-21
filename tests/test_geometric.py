"""Per-view geometric augmentation (brain.data.crops3d / batch3d.sample_views)."""

from __future__ import annotations

import copy

import torch

from brain.data.batch3d import sample_views
from brain.data.crops3d import raw_views
from brain.train.defaults import default_config


def _setup(seed: int = 0):
    torch.manual_seed(seed)
    z, y, x = 48, 64, 64
    zz, yy, xx = torch.meshgrid(*(torch.arange(n).float() for n in (z, y, x)), indexing="ij")
    vol = (torch.sin(zz / 5) + torch.cos(yy / 7) * xx / 30 + (xx - 20).clamp(min=0) / 40)[None, None]
    cfg = default_config("lejepa3d")
    cfg["aug"]["skull"] = {"p": 0.0}
    for k, v in (("global", [32, 32, 16]), ("local", [16, 16, 8]), ("fine", [16, 16, 8])):
        cfg["views"][k]["size_xyz"] = v
    batch = {"volume": vol, "modality": torch.tensor([1]), "spacing_mm": torch.ones(1, 3),
             "origin_mm": torch.zeros(1, 3), "thick_mm": torch.ones(1), "thick_axis": torch.tensor([-1])}
    return vol, cfg, batch


def test_identity_geometry_reproduces_the_grid_crops():
    _, cfg, batch = _setup()
    old = raw_views(dict(batch), cfg, torch.Generator().manual_seed(5))
    geo = copy.deepcopy(cfg)
    geo["aug"]["geometric"] = {"p": 1.0, "rot_deg": [0, 0, 0], "scale": [1, 1], "flip_lr": 0.0}
    new = raw_views(dict(batch), geo, torch.Generator().manual_seed(5))
    views = sample_views([new], new["groups"], torch.device("cpu"))
    for g in old["groups"]:
        # float32 grid coordinates land on voxel centers up to rounding, so trilinear weights can be ~1e-3 off on O(1) values
        assert torch.allclose(views[f"{g}_view"].float(), old[f"{g}_view"].float(), atol=5e-3), g
        for k in ("pos", "fg", "parent", "tok_pos", "coords"):
            assert torch.equal(old[f"{g}_{k}"], new[f"{g}_{k}"]), (g, k)


def test_rotated_views_keep_shapes_and_overlap():
    _, cfg, batch = _setup(1)
    cfg["aug"]["geometric"] = {"p": 1.0, "rot_deg": [15, 15, 15], "scale": [0.85, 1.15], "flip_lr": 0.5}
    out = raw_views(dict(batch), cfg, torch.Generator().manual_seed(3))
    views = sample_views([out], out["groups"], torch.device("cpu"))
    for g in out["groups"]:
        assert views[f"{g}_view"].shape[-3:] == tuple(out["vshape"][g])
        assert torch.isfinite(views[f"{g}_view"]).all()
    # the two global views still share coarse positions, so the dense consensus has partners
    a = set(out["global_pos"][0][0][out["global_fg"][0][0]].tolist())
    b = set(out["global_pos"][1][0][out["global_fg"][1][0]].tolist())
    assert len(a & b) > 0


def test_validation_crops_are_never_warped():
    _, cfg, batch = _setup(2)
    cfg["aug"]["geometric"] = {"p": 1.0, "rot_deg": [15, 15, 15], "scale": [0.85, 1.15], "flip_lr": 0.5}
    out = raw_views({**batch, "geometric": False}, cfg, torch.Generator().manual_seed(1))
    assert "volume_full" not in out and "global_view" in out
