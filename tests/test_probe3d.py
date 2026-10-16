"""Unit tests for the frozen-token probes. Run: python -m tests.test_probe3d"""

import numpy as np
import torch

from downstream import probe3d as P
from downstream.embed3d import token_gate


def _vol(device="cpu"):
    g = torch.Generator().manual_seed(0)
    v = torch.randn(256, 256, 128, generator=g) * 0.3 - 6.0  # MRI background
    v[48:208, 48:208, 16:112] = torch.randn(160, 160, 96, generator=g) * 0.5 + 1.0
    return v.to(device)


def test_lesion_grid_counts_patch_fraction():
    m = np.zeros((128, 256, 256), np.uint8)
    m[0:8, 0:16, 0:16] = 1  # exactly the first patch
    m[8:12, 16:32, 0:16] = 1  # half of the patch at (z=1, y=1, x=0)
    g = P.lesion_grid(m, "atlas")
    assert g.shape == (4096,)
    assert g[0] == 1.0 and abs(g[16 * 16] - 0.0) < 1e-9 and abs(g[1 * 256 + 1 * 16] - 0.5) < 1e-6
    w = np.full((128, 256, 256), 2, np.uint8)
    assert P.lesion_grid(w, "wmh").sum() == 0 and P.lesion_grid(w, "atlas").min() == 1
    inv = np.full((128, 256, 256), 4, np.uint8)
    assert P.lesion_grid(inv, "aisd").sum() == 0
    inv[0, 0, 0] = 2
    assert P.lesion_grid(inv, "aisd").sum() > 0
    assert "instance" in P.LESION_FAMILIES
    assert P.lesion_grid(m, "instance")[0] == 1.0


def test_token_positions_follow_zyx_order():
    pos = P.token_positions()
    assert pos.shape == (4096, 3)
    assert pos[1, 2] > pos[0, 2] and pos[16, 1] > pos[0, 1] and pos[256, 0] > pos[0, 0]


def test_pixel_provider_matches_gate_and_shape():
    v = _vol()
    feat, fg = P.Pixel().tokens(v, "mri")
    assert feat.shape == (4096, 4) and fg.shape == (4096,)
    assert 0.1 < fg.float().mean() < 0.9
    assert torch.equal(fg, token_gate(v, "mri", P.PATCH_XYZ))
    assert feat[fg][:, 0].mean() > feat[~fg][:, 0].mean()  # brain is brighter than background


def test_slice2d_provider_lands_on_the_3d_grid():
    from models.vit import VisionTransformer

    class _Enc:
        model = VisionTransformer(size="tiny", img_size=256, patch_size=16).eval()

    feat, fg = P.Slice2D(_Enc(), "x", batch=64).tokens(_vol(), "mri")
    assert feat.shape == (4096, _Enc.model.embed_dim) and fg.shape == (4096,)


def test_a_new_checkpoint_loads_and_embeds_a_full_volume(tmp_path=None):
    import tempfile
    from pathlib import Path

    from brainmodel.config import DEFAULT_CONFIG
    from brainmodel.train.train_ssl import build_encoder
    from downstream.embeddings import load_frozen_encoder

    cfg = {**DEFAULT_CONFIG, "model": {**DEFAULT_CONFIG["model"], "size": "test", "dense_dim": 16, "proj_hidden": 32},
           "loss": {**DEFAULT_CONFIG["loss"], "predict": {**DEFAULT_CONFIG["loss"]["predict"], "weight": 1.0, "depth": 1}}}
    enc = build_encoder(cfg, torch.device("cpu"))
    assert enc.token_head is not None and enc.predictor is not None  # training-only parts must not block loading
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.pt"
        torch.save({"model": enc.state_dict(), "config": cfg, "step": 1}, path)
        loaded = load_frozen_encoder(path, torch.device("cpu"))
        assert loaded.kind == "brain3d" and loaded.model.token_head is not None
        feat, fg = P.Brain3D(loaded, "new").tokens(_vol(), "mri")
    assert feat.shape == (4096, 64) and fg.shape == (4096,)
    assert float(feat[~fg].abs().sum()) == 0.0 and float(feat[fg].abs().sum()) > 0.0


def test_tiled_embedding_matches_one_pass_for_a_full_size_window_and_covers_every_token():
    from brainmodel.config import DEFAULT_CONFIG
    from brainmodel.train.train_ssl import build_encoder
    from downstream.embed3d import FrozenEncoder3D

    cfg = {**DEFAULT_CONFIG, "model": {**DEFAULT_CONFIG["model"], "size": "test", "dense_dim": 0, "proj_hidden": 32}}
    enc = FrozenEncoder3D(build_encoder(cfg, torch.device("cpu")).eval(), (16, 16, 8), (256, 256, 128), (1.0, 1.0, 1.0))
    vol = _vol()
    one, _, fg = enc.embed_tokens(vol, 0.0, 1.0, "mri")
    same, _, _ = enc.embed_tokens(vol, 0.0, 1.0, "mri", window_xyz=(256, 256, 128))
    assert torch.allclose(one.float(), same.float(), atol=1e-4), float((one - same).abs().max())
    tiled, xyz, fg2 = enc.embed_tokens(vol, 0.0, 1.0, "mri", window_xyz=(128, 128, 64), stride_xyz=(64, 64, 32))
    assert tiled.shape == one.shape and xyz.shape == (4096, 3) and bool((fg == fg2).all())
    assert float(tiled[fg].abs().sum(-1).min()) > 0.0 and float(tiled[~fg].abs().sum()) == 0.0
    assert not torch.allclose(tiled, one, atol=1e-3)  # windows see less context, so the tokens differ
    import os
    os.environ["BRAIN_EMBED_BATCH"] = "1"
    tiled1, _, _ = enc.embed_tokens(vol, 0.0, 1.0, "mri", window_xyz=(128, 128, 64), stride_xyz=(64, 64, 32))
    os.environ["BRAIN_EMBED_BATCH"] = "8"
    tiled8, _, _ = enc.embed_tokens(vol, 0.0, 1.0, "mri", window_xyz=(128, 128, 64), stride_xyz=(64, 64, 32))
    assert torch.allclose(tiled1, tiled8, atol=1e-5)
    os.environ.pop("BRAIN_EMBED_BATCH", None)


def _volumes(signal: float, n_vol=6, n_tok=300, d=16, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_vol):
        frac = (rng.random(n_tok) < 0.05).astype(np.float32)
        x = rng.normal(size=(n_tok, d)).astype(np.float32)
        x[:, 0] += signal * frac
        case = P.Case(f"v{i}", "atlas", "mri", None, None, "train")
        out.append(P.Volume(case, torch.from_numpy(x).half(), frac, np.arange(n_tok), x.mean(0)))
    return out


def test_lesion_probe_separates_signal_from_noise():
    good = P.lesion_probe(_volumes(4.0), _volumes(4.0, seed=1), hidden=0)
    bad = P.lesion_probe(_volumes(0.0), _volumes(0.0, seed=1), hidden=0)
    assert good["auroc"] > 0.95 and good["lift"] > 5
    assert abs(bad["auroc"] - 0.5) < 0.15 and bad["lift"] < 3


def test_location_probe_reads_position_when_it_is_encoded():
    pos = P.token_positions()
    rng = np.random.default_rng(0)
    vols = []
    for i in range(6):
        idx = np.sort(rng.choice(4096, 800, replace=False))
        x = rng.normal(size=(800, 8)).astype(np.float32)
        x[:, :3] += 12 * pos[idx]
        case = P.Case(f"v{i}", "atlas", "mri", None, None, "test")
        vols.append(P.Volume(case, torch.from_numpy(x), None, idx, x.mean(0)))
    assert P.location_probe(vols)["r2_mean"] > 0.7
    for v in vols:
        v.feat = torch.randn_like(v.feat)
    assert P.location_probe(vols)["r2_mean"] < 0.1


def _fingerprint_volumes(signal: float, seed=0):
    pos, rng, out = P.token_positions(), np.random.default_rng(seed), []
    for i in range(16):
        idx = np.sort(rng.choice(4096, 400, replace=False))
        x = rng.normal(size=(400, 8)).astype(np.float32)
        x[:, :3] += signal * pos[idx]
        x[:, 3] += 3.0 * (i % 2)  # a per-dataset offset the fingerprint probe can read
        case = P.Case(f"f{i}", "atlas" if i % 2 else "wmh", "mri", None, None, "test")
        out.append(P.Volume(case, torch.from_numpy(x), None, idx, x.mean(0)))
    return out


def test_probe_intervals_cover_the_estimate_and_pair_across_encoders():
    good = P.run_probes("good", _volumes(4.0), _volumes(4.0, seed=1), _fingerprint_volumes(12.0), reps=60)
    bad = P.run_probes("bad", _volumes(0.0), _volumes(0.0, seed=1), _fingerprint_volumes(0.0), reps=60)
    assert set(good["ci"]) == set(P.METRICS) and len(good["boot"]["pr"]) == 60
    for m, est in (("lift_lin", good["lesion"]["mean_linear_lift"]), ("loc_r2", good["location"]["r2_mean"]),
                   ("dataset_acc", good["fingerprint"]["dataset_bal_acc"])):
        assert good["ci"][m]["lo"] <= est + 0.05 * abs(est) and est - 0.05 * abs(est) <= good["ci"][m]["hi"], m
    assert good["ci"]["loc_r2"]["lo"] > bad["ci"]["loc_r2"]["hi"]
    table = P.paired_table({"good": good, "bad": bad}, ref="bad")
    assert "paired difference vs bad" in table and "*" in table
    assert "_scores" not in str(good["lesion"]) and "_volumes" not in good["location"]
    assert "[" in P.format_summary({"good": good})


def test_fine_patch_tiled_tokens_with_block_taps_cover_the_fine_grid():
    from brainmodel.config import DEFAULT_CONFIG
    from brainmodel.train.train_ssl import build_encoder
    from downstream.embed3d import FrozenEncoder3D

    cfg = {**DEFAULT_CONFIG, "model": {**DEFAULT_CONFIG["model"], "size": "test", "dense_dim": 0, "proj_hidden": 32}}
    model = build_encoder(cfg, torch.device("cpu")).eval()
    enc = FrozenEncoder3D(model, (8, 8, 4), (256, 256, 128), (1.0, 1.0, 1.0))
    enc.tap_blocks = (0,)
    tok, _, fg = enc.embed_tokens(_vol(), 0.0, 1.0, "mri", window_xyz=(128, 128, 64), stride_xyz=(64, 64, 32))
    assert tok.shape == (32 ** 3, 2 * model.embed_dim) and fg.shape == (32 ** 3,)
    assert float(tok[~fg].abs().sum()) == 0.0 and float(tok[fg].abs().sum()) > 0.0


def test_fine_pooled_tokens_land_on_the_coarse_grid():
    from brainmodel.config import DEFAULT_CONFIG
    from brainmodel.train.train_ssl import build_encoder
    from downstream.embed3d import FrozenEncoder3D

    cfg = {**DEFAULT_CONFIG, "model": {**DEFAULT_CONFIG["model"], "size": "test", "dense_dim": 0, "proj_hidden": 32}}
    enc = FrozenEncoder3D(build_encoder(cfg, torch.device("cpu")).eval(), (16, 16, 8), (256, 256, 128), (1.0, 1.0, 1.0))
    vol = _vol()
    coarse, xyz, fg = enc.embed_tokens(vol, 0.0, 1.0, "mri", fine_pool=False)
    pooled, xyz2, fg2 = enc.embed_tokens(vol, 0.0, 1.0, "mri", fine_pool=True)
    assert pooled.shape == coarse.shape and bool((fg == fg2).all()) and torch.allclose(xyz, xyz2)
    assert float(pooled[~fg].abs().sum()) == 0.0 and float(pooled[fg].abs().sum()) > 0.0
    assert enc.patch_xyz == (16, 16, 8)  # restored after the fine pass
    assert not torch.allclose(pooled, coarse, atol=1e-3)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
