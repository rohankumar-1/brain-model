"""Budgeted token refinement (brain.models.refine, BrainEncoder ``refine``): geometry, selection, and the coarse-grid
contract of a refined encoding."""

from __future__ import annotations

import torch

from brain.models.refine import child_index, choose_cells, detail_scores
from brain.models.vit3d import BrainEncoder

PATCH = (8, 8, 4)                  # x, y, z voxels
SHAPE_ZYX = (16, 32, 32)           # coarse grid (4, 4, 4)
SPACING = (1.0, 1.0, 1.0)


def _encoder() -> BrainEncoder:
    torch.manual_seed(0)
    return BrainEncoder(size="test", patch_xyz=PATCH, n_cls=1, qk_norm=True, proj_dim=16, proj_hidden=32,
                        checkpoint_blocks=False).eval()


def _coords(patch_xyz, rows: int = 2) -> torch.Tensor:
    """Token centres in mm, (rows, N, 3) in (x, y, z), tokens in (Z, Y, X) order: index i -> (i + 0.5) * patch."""
    px, py, pz = patch_xyz
    gz, gy, gx = SHAPE_ZYX[0] // pz, SHAPE_ZYX[1] // py, SHAPE_ZYX[2] // px
    z, y, x = torch.meshgrid(torch.arange(gz), torch.arange(gy), torch.arange(gx), indexing="ij")
    c = torch.stack([(x + 0.5) * px, (y + 0.5) * py, (z + 0.5) * pz], -1).float().reshape(1, -1, 3)
    return c.expand(rows, -1, -1).clone()


def _plan(rows: int, n: int):
    return torch.arange(rows), [rows], [n]


def test_children_sit_at_the_half_size_patch_centres():
    enc = _encoder()
    view = torch.randn(2, 1, *SHAPE_ZYX)
    coarse = _coords(PATCH)
    pmm = torch.tensor([PATCH], dtype=torch.float32).expand(2, 3)
    keep = torch.ones(2, coarse.shape[1], dtype=torch.bool)
    refine = torch.zeros_like(keep)
    refine[:, [0, 21, 63]] = True
    _, coords, keep_all, grid = enc._refined(view, coarse, pmm, None, keep, refine, zyx=True)
    fine = _coords(tuple(p // 2 for p in PATCH))
    idx, _ = child_index(grid, coords.device)
    n = coarse.shape[1]
    for cell in (0, 21, 63):
        got = coords[0, n + 8 * cell: n + 8 * cell + 8]
        assert torch.allclose(got, fine[0, idx[cell]])
    assert int(keep_all.sum()) == 2 * (n - 3 + 24)


def test_nothing_refined_is_the_plain_encoding():
    enc = _encoder()
    view = torch.randn(2, 1, *SHAPE_ZYX)
    coords, n = _coords(PATCH), 64
    pmm = torch.tensor([PATCH], dtype=torch.float32).expand(2, 3)
    keep = torch.rand(2, n) > 0.3
    with torch.no_grad():
        a = enc.encode_rows(view, coords, pmm, keep, _plan(2, n), tokens=True)
        b = enc.encode_rows(view, coords, pmm, keep, _plan(2, n), tokens=True, refine=torch.zeros_like(keep))
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


def test_refined_encoding_keeps_the_coarse_grid_and_trains_the_fine_path():
    enc = _encoder().train()
    view = torch.randn(2, 1, *SHAPE_ZYX)
    coords, n = _coords(PATCH), 64
    pmm = torch.tensor([PATCH], dtype=torch.float32).expand(2, 3)
    keep = torch.ones(2, n, dtype=torch.bool)
    keep[1, :10] = False
    refine = choose_cells(detail_scores(view, PATCH), keep, 0.25)
    pooled, dense = enc.encode_rows(view, coords, pmm, keep, _plan(2, n), tokens=True, refine=refine)
    assert pooled.shape == (2, enc.embed_dim) and dense.shape == (2, n, enc.embed_dim)
    assert torch.all(dense[1, :10] == 0)                                   # dropped cells stay zero
    with torch.no_grad():
        _, plain = enc.encode_rows(view, coords, pmm, keep, _plan(2, n), tokens=True)
    assert not torch.allclose(dense[refine], plain[refine], atol=1e-4)     # refined cells come from their children
    dense.sum().backward()
    assert enc.patch_embed.proj.weight.grad is not None


def test_forward_refined_matches_encode_rows():
    enc = _encoder()
    view_zyx = torch.randn(1, 1, *SHAPE_ZYX)
    coords, n = _coords(PATCH, rows=1), 64
    pmm = torch.tensor([PATCH], dtype=torch.float32)
    keep = torch.ones(1, n, dtype=torch.bool)
    refine = choose_cells(detail_scores(view_zyx, PATCH), keep, 0.2, random_frac=0.0)
    with torch.no_grad():
        out = enc(view_zyx.permute(0, 1, 4, 3, 2), coords, pmm, keep=keep, refine=refine)
        pooled, dense = enc.encode_rows(view_zyx, coords, pmm, keep, _plan(1, n), tokens=True, refine=refine)
    assert torch.allclose(out["tokens"], dense, atol=1e-5) and torch.allclose(out["pooled"], pooled, atol=1e-5)
    assert int(out["n_refined"]) == int(refine.sum())


def test_choose_cells_respects_budget_mask_and_detail():
    scores = torch.arange(100, dtype=torch.float32).repeat(3, 1)
    keep = torch.ones(3, 100, dtype=torch.bool)
    keep[1, 50:] = False
    sel = choose_cells(scores, keep, torch.tensor([0.1, 0.1, 0.0]), random_frac=0.0)
    assert sel.sum(1).tolist() == [10, 5, 0]
    assert sel[0, 90:].all() and not (sel[1] & ~keep[1]).any()           # the top by detail, kept cells only
    mixed = choose_cells(scores, keep, 0.2, random_frac=0.5, generator=torch.Generator().manual_seed(0))
    assert mixed.sum(1).tolist() == [20, 10, 20] and mixed[0, 90:].all()  # half by detail, half at random


def test_lejepa_step_with_refined_global_crops():
    """A pretraining step with refinement on: same outputs as without (coarse-grid dense entries), finite loss, the
    encoder gets gradient."""
    from brain.data.batch3d import lejepa_batch
    from brain.models.viewnet3d import ViewNet, batch_meta
    from brain.objectives.lejepa3d import SIGReg, lejepa_loss
    from tests.test_lejepa3d import _batch, _cfg, _enc

    torch.manual_seed(0)
    enc = _enc()
    views = lejepa_batch(_batch(), _cfg(), torch.device("cpu"), gen=torch.Generator().manual_seed(0))
    net = ViewNet(enc, local_pooled=False, dense=True, refine={"groups": ["global"], "frac_max": 0.5})
    out = net(views)
    plain = ViewNet(enc, local_pooled=False, dense=True)(views)
    assert out["proj"].shape == plain["proj"].shape
    assert [e.shape for e in out["dense"]["emb"]] == [e.shape for e in plain["dense"]["emb"]]
    parts = lejepa_loss(out, batch_meta(views), SIGReg(n_proj=16, knots=9), lamb=0.02, win_weight=1.0,
                        dense_weight=1.0, tokens_per_view=3, gather_ddp=False)
    assert all(torch.isfinite(v) for v in parts.values())
    parts["loss"].backward()
    assert float(enc.patch_embed.proj.weight.grad.abs().sum()) > 0
