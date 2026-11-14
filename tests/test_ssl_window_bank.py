"""The LeJEPA / joint SSL branch with the encoder's learned window bank (model.window_bank): native-unit views, the
bank inside the view network, no window pair, the bank's own optimizer group; bank off = the old path."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from brain.data.volumes import collate_volumes, make_synthetic_volume


def _cfg(bank: bool, win_weight: float = 0.0):
    from brain.train.defaults import LEJEPA3D, smoke_config

    cfg = smoke_config(deepcopy(LEJEPA3D))
    cfg["loss"]["win_weight"] = win_weight
    cfg["loss"].setdefault("dense", {})["weight"] = 1.0
    cfg["model"]["window_bank"] = {"enabled": bank, "k": 6, "lr_mult": 0.1}
    return cfg


def _views(cfg, seed=0):
    from brain.data.batch3d import lejepa_batch

    torch.manual_seed(seed)
    batch = collate_volumes([make_synthetic_volume(size_xyz=(32, 32, 16), spacing_mm=(1.0, 1.0, 2.0), modality=i % 2,
                                                   seed=3 + i) for i in range(2)])           # one CT row, one MRI row
    return lejepa_batch(batch, cfg, torch.device("cpu"), gen=torch.Generator().manual_seed(seed))


def test_bank_off_keeps_the_windowed_views_and_the_window_pair():
    from brain.data.batch3d import native_aug
    from brain.models.viewnet3d import ViewNet
    from brain.train.lejepa3d import build_encoder

    cfg = _cfg(bank=False, win_weight=1.0)
    assert native_aug(cfg) is None
    v = _views(cfg)
    assert "win" in v["groups"] and float(v["global_view"].abs().max()) < 10      # asinh-windowed, not HU
    enc = build_encoder(cfg, torch.device("cpu"))
    assert enc.window_bank is None
    x = torch.randn(2, 1, 8, 16, 16)
    assert ViewNet(enc)._input(x, torch.zeros(2, dtype=torch.long)) is x           # nothing between view and encoder


def test_bank_on_views_are_native_one_channel_and_the_pair_is_left_out():
    cfg = _cfg(bank=True)
    v = _views(cfg)
    assert "win" not in v["groups"] and "win_view" not in v
    g = v["global_view"]                                       # (V, B, 1, Z, Y, X)
    assert g.shape[2] == 1
    ct = v["modality"] == 0
    assert float(g[:, ct].abs().max()) > 100                   # Hounsfield units reach the encoder unwindowed


def test_bank_on_window_pair_weight_is_refused():
    with pytest.raises(ValueError, match="win_weight"):
        _views(_cfg(bank=True, win_weight=1.0))


def test_bank_on_forward_is_finite_and_trains_the_windows():
    from brain.models.viewnet3d import ViewNet, batch_meta
    from brain.objectives.lejepa3d import lejepa_loss
    from brain.objectives.sigreg import SIGReg
    from brain.train.lejepa3d import build_encoder

    cfg = _cfg(bank=True)
    enc = build_encoder(cfg, torch.device("cpu"))
    assert enc.window_bank is not None and enc.in_chans == 6
    v = _views(cfg)
    out = ViewNet(enc, dense=True)(v)
    assert out["win"] is None and torch.isfinite(out["proj"]).all()
    parts = lejepa_loss(out, batch_meta(v), SIGReg(n_proj=16, knots=9), lamb=0.02, win_weight=0.0, dense_weight=1.0,
                        tokens_per_view=3, gather_ddp=False)
    assert all(torch.isfinite(p) for p in parts.values()) and float(parts["win"]) == 0.0
    parts["loss"].backward()
    for p in (enc.window_bank.c_raw, enc.window_bank.w_raw):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0


def test_bank_gets_its_own_group_in_both_optimizers():
    from brain.common.optim import param_groups
    from brain.train.lejepa3d import build_encoder

    enc = build_encoder(_cfg(bank=True), torch.device("cpu"))
    bank = {id(enc.window_bank.c_raw), id(enc.window_bank.w_raw)}
    groups = param_groups(enc, 0.05, bank_mult=0.1)
    g = [x for x in groups if {id(p) for p in x["params"]} & bank]
    assert len(g) == 1 and {id(p) for p in g[0]["params"]} == bank
    assert g[0]["lr_scale"] == 0.1 and g[0]["weight_decay"] == 0.0
    off = build_encoder(_cfg(bank=False), torch.device("cpu"))       # no bank: grouping as before (no extra group)
    assert sorted((x["lr_scale"], x["weight_decay"]) for x in param_groups(off, 0.05, bank_mult=0.1)) == \
        sorted((x["lr_scale"], x["weight_decay"]) for x in param_groups(off, 0.05))
    from brain.models.window_bank import is_bank_param                # the joint trainer groups BrainQuery's names
    assert is_bank_param("backbone.window_bank.c_raw") and not is_bank_param("backbone.patch_embed.proj.weight")
