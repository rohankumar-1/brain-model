"""Unit tests for the 3D LeJEPA encoder. Run: python -m tests.test_brainmodel"""

from __future__ import annotations

import torch

from brainmodel.data.coords import crop_bbox_mm, patch_size_mm, token_centers_mm
from brainmodel.data.dataset import make_synthetic_volume
from brainmodel.losses.sigreg import SIGReg
from brainmodel.models.rope3d import RoPE3D
from brainmodel.models.vit3d import BrainEncoder


def test_crop_bbox_token_centers_roundtrip_anisotropic():
    spacing = torch.tensor([1.0, 1.0, 2.0])
    origin = torch.tensor([10.0, 20.0, 30.0])
    bbox = crop_bbox_mm((4, 2, 2), (16, 16, 8), spacing, origin)
    assert torch.allclose(bbox[:3], torch.tensor([14.0, 22.0, 34.0]))
    assert torch.allclose(bbox[3:], torch.tensor([30.0, 38.0, 50.0]))
    coords = token_centers_mm(bbox, (16, 16, 8), (8, 8, 4))
    pmm = patch_size_mm(bbox, (16, 16, 8), (8, 8, 4))
    assert coords.shape == (8, 3)
    assert torch.allclose(pmm, torch.tensor([8.0, 8.0, 8.0]))
    assert torch.allclose(coords[0], torch.tensor([18.0, 26.0, 38.0]))


def test_sigreg_gaussian_low_collapsed_high():
    torch.manual_seed(0)
    sig = SIGReg(n_proj=64, knots=9)
    gauss = torch.randn(512, 32)
    collapsed = torch.ones(512, 32)
    anisotropic = torch.randn(512, 32) * torch.linspace(0.01, 3.0, 32)
    g, c, a = float(sig(gauss)), float(sig(collapsed)), float(sig(anisotropic))
    assert g < 5.0, g
    assert c > g * 5, (c, g)
    assert a > g, (a, g)


def test_rope_translation_preserves_relative_attention():
    torch.manual_seed(0)
    rope = RoPE3D(head_dim=32, base_mm=100.0)
    b, h, n, d = 2, 2, 6, 32
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    coords = torch.randn(b, n, 3) * 20
    q1, k1 = rope.rotate(q, k, coords)
    q2, k2 = rope.rotate(q, k, coords + 17.0)
    logits1 = torch.matmul(q1, k1.transpose(-1, -2))
    logits2 = torch.matmul(q2, k2.transpose(-1, -2))
    assert torch.allclose(logits1, logits2, atol=1e-4, rtol=1e-4)


def test_asinh_window_is_finite_and_independent_per_view():
    from brainmodel.data.augment import window_view

    x = torch.linspace(-1024, 400, 8 * 8 * 4).reshape(1, 1, 8, 8, 4).expand(2, -1, -1, -1, -1).contiguous()
    mod = torch.zeros(2, dtype=torch.long)
    aug = {"window": {"mode": "random", "mix": "brain"}}
    torch.manual_seed(0)
    a = window_view(x, mod, aug)
    torch.manual_seed(1)
    b = window_view(x, mod, aug)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not torch.allclose(a, b, atol=1e-5)
    from lib.windowing import asinh_window

    floor = torch.full((1, 1, 8, 8, 4), -1024.0)
    y = asinh_window(floor, 40.0, 40.0)
    assert torch.isfinite(y).all()
    assert float(y.std()) < 1e-6


def test_token_gate_is_a_threshold_at_the_patch_center():
    from brainmodel.data.brain_mask import token_keep

    vol = torch.full((2, 1, 8, 8, 4), -1000.0)
    vol[0, :, 2:6, 2:6, 1:3] = 40.0
    vol[0, :, 0, 0, 0] = 1500.0  # bone stays
    vol[1] = -6.0
    vol[1, :, 2:6, 2:6, 1:3] = 0.0
    mod = torch.tensor([0, 1])
    grid = token_keep(vol, mod, (4, 4, 2), ct_hu=(-500.0, 200.0), mri_lo=-5.5)
    assert grid.shape == (2, 2, 2, 2)  # (B, Z, Y, X)
    assert bool(grid[0].any()) and not bool(grid[0].all()) and bool(grid[1].any()) and not bool(grid[1].all())


def _cfg(**views):
    from brainmodel.config import DEFAULT_CONFIG, smoke_config
    from copy import deepcopy

    cfg = smoke_config(deepcopy(DEFAULT_CONFIG))
    cfg["views"].update(views)
    return cfg


def _batch(n=2, size=(32, 32, 16)):
    from brainmodel.data.dataset import collate_volumes

    return collate_volumes([
        make_synthetic_volume(size_xyz=size, spacing_mm=(1.0, 1.0, 2.0), modality=i % 2, seed=3 + i) for i in range(n)
    ])


def _raw(cfg, n=2, seed=0):
    from brainmodel.data.crops import raw_views

    b = _batch(n)
    b["volume"] = b["volume"].permute(0, 1, 4, 3, 2)
    return b, raw_views(b, cfg, torch.Generator().manual_seed(seed))


def test_crops_cut_the_voxels_their_positions_name():
    cfg = _cfg()
    b, raw = _raw(cfg)
    vol = b["volume"]  # (B, 1, Z, Y, X)
    gz, gy, gx = 16 // 4, 32 // 8, 32 // 8
    assert raw["n_pos"] == gz * gy * gx
    for g, (pz, py, px) in (("global", (4, 8, 8)), ("local", (4, 8, 8)), ("fine", (4, 8, 8)), ("win", (4, 8, 8))):
        view, pos = raw[f"{g}_view"], raw[f"{g}_pos"]
        assert view.shape[:3] == (pos.shape[0], 2, 1) and raw[f"{g}_fg"].shape[-1] == raw[f"{g}_parent"].shape[-1]
        for v in range(view.shape[0]):
            for i in range(2):
                first = int(pos[v, i, 0])
                z0, y0, x0 = first // (gy * gx) * pz, (first // gx) % gy * py, first % gx * px
                sz, sy, sx = view.shape[3:]
                assert torch.equal(view[v, i, 0], vol[i, 0, z0:z0 + sz, y0:y0 + sy, x0:x0 + sx])
    assert raw["groups"] == ["global", "local", "fine", "win"]


def test_nested_crops_share_tokens_and_fine_tokens_point_at_their_position():
    cfg = _cfg()
    for seed in range(5):
        _, raw = _raw(cfg, seed=seed)
        for i in range(2):
            g0, g1 = set(raw["global_pos"][0, i].tolist()), set(raw["global_pos"][1, i].tolist())
            assert g0 & g1, "later globals are centered inside the first, so they overlap it"
            for k in range(2):
                parent = set(raw["global_pos"][k % 2, i].tolist())
                assert set(raw["local_pos"][k, i].tolist()) <= parent
                assert set(raw["fine_pos"][k, i].tolist()) <= parent
        par = raw["fine_parent"]
        assert par.shape[-1] == 4 * 4 * 4 * 1 * 1 and int(par.max()) < raw["n_par"]["fine"]
        assert int(par.max()) == raw["n_par"]["fine"] - 1
        # eight fine tokens per coarse position when the ratio is 2 x 2 x 2
        assert torch.bincount(par[0, 0].long()).unique().tolist() == [8]
        # native fine-grid index: one slot per token, eight distinct children under each parent
        assert raw["n_pos_fine"] == 8 * 8 * 8
        assert raw["fine_tok_pos"].shape[-1] == raw["fine_parent"].shape[-1]
        assert torch.equal(raw["global_tok_pos"], raw["global_pos"])
        children = raw["fine_tok_pos"][0, 0][raw["fine_parent"][0, 0] == 0]
        assert children.unique().numel() == 8
    _, free = _raw(_cfg(nested=False), seed=1)
    assert free["global_view"].shape == raw["global_view"].shape


def test_consensus_is_zero_for_agreeing_views_and_ignores_unshared_positions():
    from brainmodel.losses.dense import consensus, parent_mean

    torch.manual_seed(0)
    emb = torch.randn(2, 5, 4, requires_grad=True)
    ok = torch.ones(2, 5, dtype=torch.bool)
    pos = torch.arange(5).repeat(2, 1)
    zero, cov = consensus([emb, emb.detach()], [ok, ok], [pos, pos], 8)
    assert float(zero) < 1e-10 and float(cov) == 1.0
    other = emb.detach() + 1.0
    loss, _ = consensus([emb, other], [ok, ok], [pos, pos], 8)
    assert abs(float(loss) - 0.25) < 1e-5  # each view is 0.5 from the mean in every dimension
    loss.backward()
    assert float(emb.grad.abs().sum()) > 0
    apart, cov = consensus([emb, other], [ok, ok], [pos, pos + 3], 16)
    assert float(cov) < 1.0
    none, cov = consensus([emb, other], [ok, ok], [pos, pos + 5], 16)
    assert float(none) == 0.0 and float(cov) == 0.0
    live = torch.randn(2, 5, 4, requires_grad=True)
    tgt = live.detach() + 1.0
    stopped, _ = consensus([live, tgt], [ok, ok], [pos, pos], 8, stopgrad_mean=True)
    stopped.backward()
    assert float(live.grad.abs().sum()) > 0
    # fine tokens average into their coarse position, and a dropped token does not count
    tok = torch.arange(8.0).view(1, 8, 1).expand(1, 8, 2).clone()
    keep = torch.ones(1, 8, dtype=torch.bool)
    keep[0, 7] = False
    m, v = parent_mean(tok, keep, torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]), 3)
    assert torch.allclose(m[0, :, 0], torch.tensor([1.5, 5.0, 0.0])) and v[0].tolist() == [True, True, False]


def test_degrade_changes_crops_only_when_drawn():
    from brainmodel.data.augment import degrade

    x = torch.zeros(4, 1, 16, 16, 16)
    off = {"noise": {"p": 0.0}, "blur": {"p": 0.0}, "drift": {"p": 0.0}}
    assert torch.equal(degrade(x, off), x)
    noisy = degrade(x, {"noise": {"p": 1.0, "sigma": (0.1, 0.1)}})
    assert 0.05 < float(noisy.std()) < 0.2
    texture = torch.randn(4, 1, 16, 16, 16)
    blurred = degrade(texture, {"blur": {"p": 1.0, "sigma_vox": (1.5, 1.5)}})
    assert float(blurred.std()) < 0.5 * float(texture.std())
    drift = degrade(x, {"drift": {"p": 1.0, "amp": 0.3}})
    assert float(drift.abs().max()) > 0.05 and float((drift[:, :, 1:] - drift[:, :, :-1]).abs().max()) < 0.2


def _enc(dense=True):
    return BrainEncoder(
        size="test", patch_xyz=(8, 8, 4), checkpoint_blocks=False, proj_dim=16, proj_hidden=32,
        qk_norm=True, proj_norm="batch", dense_dim=16 if dense else 0, dense_hidden=32,
    )


def test_dense_step_trains_encoder_and_token_head():
    from brainmodel.data.lejepa import lejepa_batch
    from brainmodel.losses.lejepa import lejepa_loss
    from brainmodel.models.viewnet import ViewNet, batch_meta

    cfg = _cfg()
    enc = _enc()
    views = lejepa_batch(_batch(), cfg, torch.device("cpu"), gen=torch.Generator().manual_seed(0))
    assert views["global_view"].shape[:2] == (2, 2) and views["win_view"].shape[:2] == (2, 2)
    assert not torch.equal(views["win_view"][0], views["win_view"][1])
    out = ViewNet(enc, local_pooled=False, dense=True)(views)
    assert out["proj"].shape[:2] == (2, 2) and out["win"].shape[:2] == (2, 2)
    assert len(out["dense"]["emb"]) == 2 + 2 + 2 + 2
    meta = batch_meta(views)
    sig = SIGReg(n_proj=16, knots=9)
    parts = lejepa_loss(out, meta, sig, lamb=0.02, win_weight=1.0, dense_weight=1.0, tokens_per_view=3, gather_ddp=False)
    expect = (0.98 * parts["inv"] + 0.02 * parts["sigreg"] + parts["win"]
              + 0.98 * parts["dense_inv"] + 0.02 * parts["dense_sig"])
    assert torch.allclose(parts["loss"], expect, rtol=1e-4, atol=1e-4)
    assert all(torch.isfinite(v) for v in parts.values()) and 0.0 < float(parts["overlap"]) <= 1.0
    parts["loss"].backward()
    assert float(enc.token_head.net[0].weight.grad.abs().sum()) > 0
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in enc.blocks.parameters())
    torch.optim.AdamW(enc.parameters(), lr=1e-3).step()


def test_pooled_only_recipe_matches_the_2d_objective():
    from brainmodel.data.lejepa import lejepa_batch
    from brainmodel.losses.lejepa import lejepa_loss
    from brainmodel.models.viewnet import ViewNet, batch_meta

    views = lejepa_batch(_batch(), _cfg(), torch.device("cpu"), gen=torch.Generator().manual_seed(0))
    out = ViewNet(_enc(dense=False), local_pooled=True, dense=False)(views)
    assert out["dense"] is None and out["proj"].shape[:2] == (4, 2)  # 2 global + 2 local
    parts = lejepa_loss(out, batch_meta(views), SIGReg(n_proj=16, knots=9), lamb=0.02, win_weight=1.0, gather_ddp=False)
    assert set(parts) == {"loss", "inv", "win", "sigreg"}
    assert torch.allclose(parts["loss"], 0.98 * parts["inv"] + 0.02 * parts["sigreg"] + parts["win"], atol=1e-5)


def test_length_buckets_do_not_change_pooled_or_dense_values():
    from brainmodel.data.lejepa import length_plan

    torch.manual_seed(0)
    enc = _enc().eval()
    rows, n = 10, 2 * 2 * 4  # a (16, 16, 16) crop of (8, 8, 4) patches
    view = torch.randn(rows, 1, 16, 16, 16)
    coords = torch.randn(rows, n, 3) * 20
    pmm = torch.ones(rows, 3) * 8
    keep = torch.rand(rows, n) > torch.linspace(0.1, 0.8, rows).unsqueeze(1)
    keep[3] = False  # an empty row falls back to the full mean
    counts = keep.sum(1)
    with torch.no_grad():
        ref, ref_tok = enc.encode_rows(view, coords, pmm, keep, length_plan(counts, 1), (8, 8, 4), tokens=True)
        for k in (2, 3, rows):
            got, got_tok = enc.encode_rows(view, coords, pmm, keep, length_plan(counts, k), (8, 8, 4), tokens=True)
            assert torch.allclose(ref, got, atol=1e-4, rtol=1e-4), k
            assert torch.allclose(ref_tok, got_tok, atol=1e-4, rtol=1e-4), k
    assert ref_tok.shape == (rows, n, enc.embed_dim) and float(ref_tok[~keep].abs().sum()) == 0.0
    plan = length_plan(counts, 3)
    assert sum(plan[1]) == rows and plan[2] == sorted(plan[2], reverse=True)


def _predict_cfg():
    cfg = _cfg()
    cfg["loss"]["predict"] = {"weight": 1.0, "pixel_weight": 0.5, "pixel_cells": [2, 2, 2], "blocks": 1,
                              "block_tokens": [1, 1], "depth": 1, "ramp_steps": 0}
    return cfg


def _predict_enc():
    return BrainEncoder(
        size="test", patch_xyz=(8, 8, 4), checkpoint_blocks=False, proj_dim=16, proj_hidden=32, qk_norm=True,
        proj_norm="batch", dense_dim=16, dense_hidden=32, predict_depth=1, predict_pixel_dim=8,
    )


def test_block_mask_hides_a_union_of_cuboids_inside_the_grid():
    from brainmodel.data.lejepa import block_mask

    torch.manual_seed(0)
    m = block_mask(64, (8, 8, 8), blocks=3, lo=3, hi=6)
    assert m.shape == (64, 512) and m.dtype == torch.bool
    frac = m.float().mean()
    assert 0.25 < float(frac) < 0.75, float(frac)
    one = block_mask(16, (4, 6, 8), blocks=1, lo=2, hi=3).view(16, 4, 6, 8)
    for r in range(16):  # one cuboid: the hidden voxels fill their own bounding box
        idx = one[r].nonzero()
        lo, hi = idx.min(0).values, idx.max(0).values
        assert int(one[r].sum()) == int(((hi - lo) + 1).prod()) and bool((hi - lo + 1).min() >= 2)
    tiny = block_mask(4, (2, 2, 2), blocks=2, lo=3, hi=6)  # sides clamp to the grid
    assert tiny.all()


def test_cell_means_follow_token_order_and_cell_order():
    from brainmodel.models.viewnet import cell_means

    view = torch.zeros(1, 1, 8, 8, 16)  # (Z, Y, X); patches (z=4, y=8, x=8) give a 2 x 1 x 2 token grid
    for tz in range(2):
        for tx in range(2):
            for cz in range(2):
                for cy in range(2):
                    for cx in range(2):
                        view[0, 0, tz * 4 + cz * 2:tz * 4 + cz * 2 + 2, cy * 4:cy * 4 + 4,
                             tx * 8 + cx * 4:tx * 8 + cx * 4 + 4] = 100 * (2 * tz + tx) + 4 * cz + 2 * cy + cx
    got = cell_means(view, (8, 8, 4), (2, 2, 2))
    expect = torch.tensor([[100.0 * t + c for c in range(8)] for t in range(4)])
    assert got.shape == (1, 4, 8) and torch.allclose(got[0], expect)


def test_masked_prediction_partitions_the_crop_and_leaves_its_target_alone():
    from brainmodel.data.lejepa import lejepa_batch
    from brainmodel.losses.lejepa import lejepa_loss
    from brainmodel.models.viewnet import ViewNet, batch_meta

    cfg, enc = _predict_cfg(), _predict_enc()
    torch.manual_seed(0)
    views = lejepa_batch(_batch(), cfg, torch.device("cpu"), gen=torch.Generator().manual_seed(0))
    ctx, tgt, fg = views["ctx_keep"], views["tgt_mask"], views["global_fg"]
    assert not bool((ctx & tgt).any()) and torch.equal(ctx | tgt, fg) and bool(tgt.any()) and bool(ctx.any())
    assert sum(views["plans"]["ctx"][1]) == ctx.shape[0] * ctx.shape[1]
    out = ViewNet(enc, dense=True, predict=True)(views)
    pr = out["predict"]
    assert pr["latent"].shape == pr["target"].shape == (4, 8, 16) and pr["pixel"].shape == pr["pixel_target"].shape == (4, 8, 8)
    assert not pr["target"].requires_grad
    parts = lejepa_loss(out, batch_meta(views), SIGReg(n_proj=16, knots=9), lamb=0.02, win_weight=1.0, dense_weight=1.0,
                        tokens_per_view=3, predict_weight=1.0, pixel_weight=0.5, gather_ddp=False)
    assert all(torch.isfinite(v) for v in parts.values()) and {"pred", "pix"} <= set(parts)
    expect = (0.98 * parts["inv"] + 0.02 * parts["sigreg"] + parts["win"] + 0.98 * parts["dense_inv"]
              + 0.02 * parts["dense_sig"] + parts["pred"] + 0.5 * parts["pix"])
    assert torch.allclose(parts["loss"], expect, rtol=1e-4, atol=1e-4)
    enc.zero_grad()
    (parts["pred"] + parts["pix"]).backward()
    assert float(enc.predictor.latent.weight.grad.abs().sum()) > 0 and float(enc.predictor.pixel.weight.grad.abs().sum()) > 0
    assert float(enc.patch_embed.proj.weight.grad.abs().sum()) > 0  # the encoder learns through the visible tokens
    assert enc.token_head.net[0].weight.grad is None  # the target is a constant for this term


def test_multiscale_uses_native_fine_tokens_and_a_second_predict():
    from brainmodel.data.lejepa import lejepa_batch
    from brainmodel.losses.lejepa import lejepa_loss
    from brainmodel.models.viewnet import ViewNet, batch_meta

    cfg, enc = _predict_cfg(), _predict_enc()
    cfg["loss"]["dense"]["fine_weight"] = 1.0
    cfg["loss"]["predict"]["fine_weight"] = 1.0
    cfg["loss"]["predict"]["fine_blocks"] = 1
    cfg["loss"]["predict"]["fine_block_tokens"] = [1, 1]
    torch.manual_seed(0)
    views = lejepa_batch(_batch(), cfg, torch.device("cpu"), gen=torch.Generator().manual_seed(0))
    ctx, tgt, fg = views["fine_ctx_keep"], views["fine_tgt_mask"], views["fine_fg"]
    assert not bool((ctx & tgt).any()) and torch.equal(ctx | tgt, fg) and bool(tgt.any())
    out = ViewNet(enc, dense=True, predict=True, predict_fine=True)(views)
    assert out["dense_fine"] is not None and out["predict_fine"] is not None
    assert out["dense_fine"]["emb"][0].shape[1] == views["fine_tok_pos"].shape[-1]
    pf = out["predict_fine"]
    assert pf["latent"].shape == pf["target"].shape and not pf["target"].requires_grad
    parts = lejepa_loss(
        out, batch_meta(views), SIGReg(n_proj=16, knots=9), lamb=0.02, win_weight=1.0, dense_weight=1.0,
        tokens_per_view=3, predict_weight=1.0, pixel_weight=0.5, gather_ddp=False,
        fine_weight=1.0, fine_predict_weight=1.0)
    assert {"fine_inv", "fine_overlap", "fine_pred", "fine_pix"} <= set(parts)
    expect = (0.98 * parts["inv"] + 0.02 * parts["sigreg"] + parts["win"] + 0.98 * parts["dense_inv"]
              + 0.02 * parts["dense_sig"] + 0.98 * parts["fine_inv"] + parts["pred"] + 0.5 * parts["pix"]
              + parts["fine_pred"] + 0.5 * parts["fine_pix"])
    assert torch.allclose(parts["loss"], expect, rtol=1e-4, atol=1e-4)
    enc.zero_grad()
    (parts["fine_inv"] + parts["fine_pred"] + parts["fine_pix"]).backward()
    assert float(enc.predictor.latent.weight.grad.abs().sum()) > 0
    assert float(enc.token_head.net[0].weight.grad.abs().sum()) > 0


def test_masked_prediction_is_independent_of_length_buckets():
    from brainmodel.data.lejepa import lejepa_batch
    from brainmodel.models.viewnet import ViewNet

    cfg, enc = _predict_cfg(), _predict_enc().eval()
    got = []
    for k in (1, 3):
        torch.manual_seed(0)
        views = lejepa_batch(_batch(4), cfg, torch.device("cpu"), gen=torch.Generator().manual_seed(0), buckets=k)
        with torch.no_grad():
            got.append(ViewNet(enc, dense=True, predict=True)(views)["predict"])
    for key in ("latent", "pixel"):
        assert torch.allclose(got[0][key], got[1][key], atol=1e-4, rtol=1e-4), key


def test_block_taps_leave_the_final_tokens_alone():
    enc = _enc(dense=False).eval()
    view = torch.randn(1, 1, 32, 32, 8)
    coords = torch.rand(1, 32, 3) * 100
    pmm = torch.tensor([[32.0, 32.0, 8.0]])
    with torch.no_grad():
        base = enc(view, coords, pmm, patch_xyz=(8, 8, 4))
        tap = enc(view, coords, pmm, patch_xyz=(8, 8, 4), taps=(0, 1))
    assert base["taps"] == [] and len(tap["taps"]) == 2 and torch.equal(base["tokens"], tap["tokens"])
    assert tap["taps"][0].shape == tap["tokens"].shape and not torch.allclose(tap["taps"][0], tap["taps"][1])
    assert base["patch"].shape == base["tokens"].shape and torch.equal(base["patch"], tap["patch"])


def test_family_sampler_balances_families_and_splits_ranks():
    import pandas as pd
    from collections import Counter

    from brainmodel.data.dataset import FamilySampler

    fam = ["big"] * 900 + ["mid"] * 90 + ["small"] * 10
    ds = type("D", (), {"df": pd.DataFrame({"dataset_family": fam}), "__len__": lambda self: len(fam)})()
    one = FamilySampler(ds, 0.5, seed=1)
    drawn = Counter(fam[i] for i in list(one) * 20)
    share = {k: v / sum(drawn.values()) for k, v in drawn.items()}
    expect = {"big": 900 ** 0.5, "mid": 90 ** 0.5, "small": 10 ** 0.5}
    tot = sum(expect.values())
    for k in expect:
        assert abs(share[k] - expect[k] / tot) < 0.05, (k, share[k])
    ranks = [FamilySampler(ds, 0.5, seed=1, num_replicas=2, rank=r) for r in (0, 1)]
    assert len(ranks[0]) == len(ranks[1]) == 500
    assert sorted(list(ranks[0]) + list(ranks[1])) == sorted(one)
    assert list(ranks[0]) == list(FamilySampler(ds, 0.5, seed=1, num_replicas=2, rank=0))
    ranks[0].set_epoch(1)
    assert list(ranks[0]) != list(FamilySampler(ds, 0.5, seed=1, num_replicas=2, rank=0))


def test_is_new_best_keeps_lower_val_loss():
    from brainmodel.train.train_ssl import is_new_best

    assert is_new_best({"loss": 1.2}, "loss", 1.5) == 1.2
    assert is_new_best({"loss": 1.7}, "loss", 1.5) is None
    assert is_new_best({"loss": float("nan")}, "loss", 1.5) is None
    assert is_new_best({"loss": float("inf")}, "loss", 1.5) is None
    assert is_new_best({"loss": 0.4, "inv": 0.1}, "inv", 0.2) == 0.1
    assert is_new_best({"inv": 0.05}, "loss", 1.0) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
