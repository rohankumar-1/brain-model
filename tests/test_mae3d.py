"""3D MAE: masking, voxel-reconstruction loss, and Primus-style encoder checkpoints loading for frozen evaluation."""

from __future__ import annotations

import copy

import torch

from brain.data.batch3d import lejepa_batch, random_mask
from brain.data.volumes import collate_volumes, make_synthetic_volume
from brain.models.mae3d import MAE3D
from brain.train.defaults import default_config, smoke_config
from brain.train.mae3d import build_decoder, build_encoder


def _cfg():
    cfg = smoke_config(default_config("mae3d"))
    cfg["views"]["local"]["n"] = cfg["views"]["fine"]["n"] = 0
    return cfg


def test_random_mask_hides_the_ratio_of_foreground_only():
    torch.manual_seed(0)
    fg = torch.rand(8, 200) > 0.4
    m = random_mask(fg, 0.75)
    assert not (m & ~fg).any()
    assert torch.allclose(m.sum(1).float(), (fg.sum(1).float() * 0.75).floor())


def test_mae_forward_backward_and_hidden_share():
    torch.manual_seed(0)
    cfg = _cfg()
    size, sp = tuple(cfg["data"]["volume_size_xyz"]), tuple(cfg["data"]["default_spacing_mm"])
    batch = collate_volumes([make_synthetic_volume(size, sp, modality=i % 2, seed=i) for i in range(2)])
    b = lejepa_batch(batch, cfg, torch.device("cpu"))
    enc = build_encoder(cfg, torch.device("cpu"))
    model = MAE3D(enc, build_decoder(cfg, enc, torch.device("cpu")))
    out = model(b)
    assert torch.isfinite(out["loss"]) and 0 < float(out["hidden"]) <= 0.75
    out["loss"].backward()
    assert enc.patch_embed.proj.weight.grad is not None and float(enc.patch_embed.proj.weight.grad.abs().sum()) > 0


def test_primus_encoder_round_trips_through_the_eval_loader(tmp_path):
    from brain.downstream.embeddings import load_frozen_encoder

    cfg = default_config("mae3d")
    cfg["model"]["size"] = "test"
    enc = build_encoder(cfg, torch.device("cpu"))
    path = tmp_path / "mae.pt"
    torch.save({"model": enc.state_dict(), "decoder": {}, "config": cfg, "step": 0}, path)
    fe = load_frozen_encoder(path, torch.device("cpu"))
    assert fe.force_fine_pool and fe.patch_xyz == (16, 16, 8)
    for a, b in zip(fe.model.state_dict().values(), enc.state_dict().values()):
        assert torch.equal(a, b)


def test_split_target_and_lesion_weight_run():
    torch.manual_seed(1)
    cfg = _cfg()
    cfg["loss"]["predict"].update(target="split", lesion={"visible": [0.3, 0.5], "weight": 2.0})
    size, sp = tuple(cfg["data"]["volume_size_xyz"]), tuple(cfg["data"]["default_spacing_mm"])
    batch = collate_volumes([make_synthetic_volume(size, sp, modality=1, seed=i) for i in range(2)])
    batch["lesion"] = torch.zeros(2, 1, size[2], size[1], size[0], dtype=torch.bool)
    batch["lesion"][:, :, 4:12, 8:24, 8:24] = True
    b = lejepa_batch(batch, cfg, torch.device("cpu"))
    assert "global_lesion" in b
    enc = build_encoder(cfg, torch.device("cpu"))
    dec = build_decoder(cfg, enc, torch.device("cpu"))
    assert dec.out.out_features == 8 * 8 * 4 + 2
    out = MAE3D(enc, dec, target="split", lesion_weight=2.0)(b)
    assert torch.isfinite(out["loss"]) and torch.isfinite(out["r2"])


def test_lesion_tokens_are_kept_partly_visible():
    from brain.data.batch3d import _mask_group

    torch.manual_seed(0)
    fg = torch.ones(1, 4, 512, dtype=torch.bool)
    les = torch.zeros(1, 4, 512, dtype=torch.bool)
    les[..., :20] = True
    chunk = {"global_fg": fg, "global_lesion": les, "patch": {"global": (8, 8, 4)}, "vshape": {"global": (32, 64, 64)}}
    d = {"plans": {}}
    _mask_group(d, [chunk], torch.device("cpu"), 1, group="global", ctx_key="ctx", tgt_key="tgt", plan="p",
                blocks=0, block_tokens=(2, 4), ratio=[0.9, 0.9], lesion_visible=[0.3, 0.5])
    visible = (d["ctx"][0][:, :20]).float().mean(1)
    assert bool((visible >= 0.3).all())                         # >= 30% of the lesion stays visible at 90% masking
