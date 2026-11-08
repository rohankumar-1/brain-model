"""Shared learned window bank (brain.models.window_bank, runs/diagnostics/ngeo/CONTRACT.md 4): calibration to tissue
units, init = the asinh presets, gradients, the model / frozen-eval wiring, the native augmentation path, and bank off =
today's input bit for bit."""

from __future__ import annotations

import pytest
import torch

from brain.common.windowing import CANONICAL, PRESETS, ROLES, asinh_window, role_windows
from brain.models.window_bank import (CALIBRATION, INITS, WindowBank, attach_bank, bank_cfg, bank_of, calibrate,
                                      encoder_in_chans, is_bank_param)

CPU = torch.device("cpu")
BANK = {"enabled": True, "k": 6, "init": "contract6", "per_modality": False, "lr_mult": 0.1}


def _native(u: torch.Tensor, mod: int) -> torch.Tensor:
    m, s = CALIBRATION[mod]
    return m + s * u


def test_bank_off_by_default_and_config_keys():
    assert bank_cfg({}) is None and bank_cfg({"window_bank": {"enabled": False, "k": 6}}) is None
    assert encoder_in_chans({"in_chans": 3}) == 3 and encoder_in_chans({"window_bank": BANK}) == 6
    assert bank_cfg({"window_bank": {"enabled": True}})["k"] == 6          # defaults filled in
    assert bank_cfg({"window_bank": {"enabled": True}})["lr_mult"] == 0.1


def test_init_windows_reproduce_the_asinh_presets():
    """Window (c, w) in u is asinh_window(x, m + s c, s w) in native units: tissue = CT brain / MRI tissue, narrow
    tissue = CT stroke, very bright = CT bone, exactly (up to the logit / softplus round trip)."""
    bank = WindowBank(k=6, init="contract6").eval()
    hu = torch.linspace(-1024, 2000, 4001).view(1, 1, -1)
    z = torch.linspace(-6, 20, 4001).view(1, 1, -1)
    y_ct = bank(hu, torch.tensor([0]))[0]
    y_mr = bank(z, torch.tensor([1]))[0]
    assert y_ct.shape == (6, 4001)
    tol = dict(atol=2e-4, rtol=1e-4)
    assert torch.allclose(y_ct[1], asinh_window(hu[0, 0], *PRESETS["ct"]["brain"]), **tol)
    assert torch.allclose(y_mr[1], asinh_window(z[0, 0], *PRESETS["mri"]["tissue"]), **tol)
    assert torch.allclose(y_ct[0], asinh_window(hu[0, 0], *PRESETS["ct"]["stroke"]), **tol)
    assert torch.allclose(y_ct[5], asinh_window(hu[0, 0], *PRESETS["ct"]["bone"]), **tol)
    # every window, both modalities: the native-unit preset the calibration implies
    for k, (c, w) in enumerate(INITS["contract6"]):
        for mod, x, y in ((0, hu, y_ct), (1, z, y_mr)):
            m, s = CALIBRATION[mod]
            assert torch.allclose(y[k], asinh_window(x[0, 0], m + s * c, s * w), **tol)
    nw = bank.native_windows()
    assert nw["ct"][1] == pytest.approx(CANONICAL["ct"], rel=1e-5)
    assert nw["mri"][1] == pytest.approx(CANONICAL["mri"], rel=1e-5)
    assert nw["ct"][5] == pytest.approx(ROLES["ct"][2], rel=1e-5)


def test_ct_and_mri_with_the_same_tissue_units_give_the_same_channels():
    torch.manual_seed(0)
    bank = WindowBank(k=6).eval()
    with torch.no_grad():                                     # move off the init so the check is not a special case
        bank.c_raw.add_(0.3 * torch.randn_like(bank.c_raw))
        bank.w_raw.add_(0.3 * torch.randn_like(bank.w_raw))
    u = torch.randn(1, 1, 4, 5, 6) * 3
    x = torch.cat([_native(u, 0), _native(u, 1)])
    y = bank(x, torch.tensor([0, 1]))
    assert y.shape == (2, 6, 4, 5, 6)
    assert torch.allclose(y[0], y[1], atol=1e-5)
    assert torch.allclose(calibrate(x, torch.tensor([0, 1]))[0], u[0], atol=1e-5)
    # per_modality (fallback): same module, separate tables, identical at init
    pm = WindowBank(k=6, per_modality=True).eval()
    assert pm.c_raw.shape == (2, 6)
    assert torch.allclose(pm(x, torch.tensor([0, 1]))[0], pm(x, torch.tensor([0, 1]))[1], atol=1e-5)


@pytest.mark.parametrize("per_modality", [False, True])
def test_gradient_reaches_centres_and_widths(per_modality):
    bank = WindowBank(k=6, per_modality=per_modality).train()
    x = torch.cat([torch.randn(1, 1, 6, 6, 6) * 60 + 40, torch.randn(1, 1, 6, 6, 6)])
    mod = torch.tensor([0, 1])
    y = bank(x, mod)                                          # training: recomputed in the backward (checkpoint)
    (y * torch.randn_like(y)).sum().backward()
    for p in (bank.c_raw, bank.w_raw):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert (p.grad.abs() > 0).all()                       # every window (and with per_modality every row) learns
    c, w = bank.centres_widths()
    assert (w > 0).all() and (c > bank.c_lo).all() and (c < bank.c_hi).all()


def test_bounds_hold_for_extreme_parameters():
    bank = WindowBank(k=2, init=[[0.0, 1.0], [1.0, 2.0]])
    with torch.no_grad():
        bank.c_raw.fill_(1e4)
        bank.w_raw.fill_(1e4)
    c, w = bank.centres_widths()
    assert torch.all(c <= bank.c_hi) and torch.all(w <= bank.w_hi)
    with torch.no_grad():
        bank.w_raw.fill_(-1e4)
    assert torch.all(bank.centres_widths()[1] >= bank.w_lo)
    with pytest.raises(ValueError):
        WindowBank(k=1)


# ---------------------------------------------------------------------------------------------------- wiring
def _mt_cfg(bank: dict | None = None, **aug):
    from brain.train.defaults import default_config, smoke_config

    cfg = smoke_config(default_config("multitask"))
    if aug:
        cfg["aug"] = aug
    if bank is not None:
        cfg["model"]["window_bank"] = bank
    return cfg


def _raw(n=4):
    from brain.multitask.data import SyntheticMultitask, collate
    from brain.multitask.registry import load_registry

    ds = SyntheticMultitask(load_registry(), (16, 32, 32), n=n)
    return collate([ds[i] for i in range(n)])


def _old_canonical(raw, mod):
    """The encoder input before the bank existed (multitask.data.canonical_window at 3f4c95a), verbatim."""
    cw = torch.tensor([CANONICAL["ct"], CANONICAL["mri"]], dtype=torch.float32)[mod]
    return asinh_window(raw, cw[:, 0].view(-1, 1, 1, 1, 1), cw[:, 1].view(-1, 1, 1, 1, 1))


def test_bank_off_prepare_is_bit_identical_to_the_old_input():
    from brain.multitask.data import prepare

    raw = _raw()
    img, mod = raw["image"].float().unsqueeze(1), raw["modality"]
    for cfg in (_mt_cfg(), _mt_cfg({"enabled": False, "k": 6})):
        b = prepare(raw, cfg, CPU, (8, 8, 4), train=False)
        assert torch.equal(b["x"], _old_canonical(img, mod)) and b["hide"] is None
    cfg = _mt_cfg()
    cfg["model"]["in_chans"] = 3
    assert torch.equal(prepare(raw, cfg, CPU, (8, 8, 4), train=False)["x"], role_windows(img, mod))


def test_bank_on_prepare_returns_native_units_and_brainquery_windows_inside():
    from brain.multitask.data import prepare
    from brain.multitask.model import BrainQuery
    from brain.multitask.registry import load_registry
    from brain.train.multitask import build_encoder

    reg = load_registry()
    cfg = _mt_cfg(dict(BANK))
    raw = _raw()
    b = prepare(raw, cfg, CPU, (8, 8, 4), train=False)
    assert torch.equal(b["x"], raw["image"].float().unsqueeze(1))       # native units: the model windows
    off = prepare(raw, _mt_cfg(), CPU, (8, 8, 4), train=False)
    assert torch.equal(b["keep"], off["keep"])                          # the gate reads raw values either way
    torch.manual_seed(0)
    enc = build_encoder(cfg, CPU, synthetic=True)
    assert enc.in_chans == 6 and isinstance(bank_of(enc), WindowBank)
    model = BrainQuery(enc, reg, cfg["model"]).eval()
    model.set_token_tasks([reg.index[t] for t in reg.train_ids(cfg["tasks"])])
    out = model(b["x"], b["keep"], b["task"], b["seg_pair"], modality=b["modality"])
    assert torch.isfinite(out["binary"]).all()
    if len(b["seg_pair"]):
        assert out["seg"].shape == (len(b["seg_pair"]), 1, 8, 16, 16)
    # pre-windowed input (K channels) passes the bank: same answers
    xw = bank_of(enc)(b["x"], b["modality"])
    out2 = model(xw, b["keep"], b["task"], b["seg_pair"])
    assert torch.allclose(out["binary"], out2["binary"], atol=1e-5)
    with pytest.raises(ValueError):                                     # native input without its modality
        model(b["x"], b["keep"], b["task"], b["seg_pair"])


def test_tiny_forward_backward_with_the_bank_trains_the_windows():
    """CPU training step on phantoms with masking and appearance augmentation: the loss reaches the bank, and the
    bank sits in its own optimizer group at lr_mult with no weight decay."""
    from brain.multitask.data import prepare
    from brain.multitask.losses import slot_losses
    from brain.multitask.model import BrainQuery
    from brain.multitask.registry import load_registry
    from brain.train.multitask import build_encoder, param_groups

    reg = load_registry()
    app = {"bias": {"p": 1.0, "amp": 0.2}, "through_plane": {"p": 1.0, "fwhm_vox": [2.0, 6.0]},
           "noise": {"p": 1.0, "kind": "rician", "snr": [15, 100], "sigma": [0.02, 0.1]},
           "ct_physical": {"p": 1.0, "offset_hu": 10.0, "scale": 0.03},
           "kernel": {"p": 1.0, "sigma_vox": [0.3, 1.0], "sharpen": 0.5, "amount": [0.2, 1.0]},
           "blur": {"p": 0.5, "sigma_vox": [0.5, 1.0]}, "drift": {"p": 1.0, "amp": 0.1}}
    cfg = _mt_cfg(dict(BANK), flip_lr=0.5, affine={"p": 0.0}, appearance=app)
    torch.manual_seed(0)
    model = BrainQuery(build_encoder(cfg, CPU, synthetic=True), reg, cfg["model"]).train()
    model.set_token_tasks([reg.index[t] for t in reg.train_ids(cfg["tasks"])])
    b = prepare(_raw(), cfg, CPU, (8, 8, 4), train=True)
    assert b["x"].shape[1] == 1 and torch.isfinite(b["x"]).all()
    out = model(b["x"], b["keep"], b["task"], b["seg_pair"], modality=b["modality"], hide=b["hide"],
                dense_pair=b["dense_pair"])
    loss, has = slot_losses(out, b["task"], b["y"], model.task_type, b["seg"], b["seg_pair"], 2, dense=b)
    loss[has].mean().backward()
    bank = model.backbone.window_bank
    assert bank.c_raw.grad is not None and bank.c_raw.grad.abs().sum() > 0
    assert bank.w_raw.grad is not None and bank.w_raw.grad.abs().sum() > 0
    groups = param_groups(model, [], 0.05, 1.0, 1.0, bank_mult=0.1)
    bg = [g for g in groups if g.get("name") == "window_bank"]
    assert len(bg) == 1 and bg[0]["lr_scale"] == 0.1 and bg[0]["weight_decay"] == 0.0 and len(bg[0]["params"]) == 2
    assert all(not any(p is q for q in bg[0]["params"]) for g in groups if g is not bg[0] for p in g["params"])
    assert is_bank_param("backbone.window_bank.c_raw") and not is_bank_param("backbone.blocks.0.attn.qkv.weight")


def test_masked_reconstruction_target_is_the_fixed_canonical_window():
    """With the bank, the masked rows are blanked inside the model and the reconstruction target is the canonical
    window of the native input (fixed: the learned windows cannot shape their own target)."""
    from brain.multitask.data import dense_targets

    raw = torch.randn(2, 1, 16, 32, 32) * 30 + 40
    mod = torch.tensor([0, 1])
    batch = {"dense_src": torch.tensor([-1, -1]), "dense_pair": torch.tensor([[0, 0], [1, 0]])}
    hide = torch.zeros(2, 16, 32, 32, dtype=torch.bool)
    hide[:, :8] = True
    tgt, w = dense_targets(batch, raw, None, hide, (8, 8, 4), None, 1, rec_src=_old_canonical(raw, mod))
    assert torch.equal(tgt, _old_canonical(raw, mod)) and w.sum() == hide.sum()


def test_frozen_eval_windows_with_the_learned_bank(tmp_path):
    """load_brain3d rebuilds the bank from the checkpoint's config and loads its (trained) windows with the weights;
    encode_volume then encodes once through the bank, whatever window mode is asked."""
    from brain.downstream.embed3d import load_brain3d
    from brain.train.multitask import build_encoder

    cfg = _mt_cfg(dict(BANK))
    torch.manual_seed(0)
    enc = build_encoder(cfg, CPU, synthetic=True).eval()
    with torch.no_grad():
        enc.window_bank.c_raw.add_(0.5)                      # "trained": not the init any more
    ck = {"model": enc.state_dict(), "config": cfg}
    frozen = load_brain3d(ck, CPU)
    assert frozen.single_pass and not frozen.role_channels and frozen.bank is not None
    assert torch.equal(frozen.bank.c_raw, enc.window_bank.c_raw)
    vol = torch.randn(32, 32, 16) * 40 + 40                   # (X, Y, Z) HU
    toks_a, _, fg = frozen.embed_windows(vol, [(40.0, 40.0)], modality="ct")
    toks_b, _, _ = frozen.embed_windows(vol, [(500.0, 400.0)], modality="ct")
    assert torch.equal(toks_a, toks_b)                        # (c, w) ignored: the bank windows
    x = enc.window_bank(vol[None, None], torch.tensor([0]))
    assert torch.allclose(frozen._window(vol, 0.0, 1.0, "ct"), x[0])
    # an encoder without a bank: unchanged (one asinh window per pass)
    pcfg = _mt_cfg()
    plain = build_encoder(pcfg, CPU, synthetic=True).eval()             # records its architecture in pcfg
    f2 = load_brain3d({"model": plain.state_dict(), "config": pcfg}, CPU)
    assert f2.bank is None and not f2.single_pass
    assert torch.equal(f2._window(vol, 40.0, 40.0, "ct"), asinh_window(vol, 40.0, 40.0))


def test_attach_checks_the_encoder_channels():
    from brain.models.vit3d import BrainEncoder

    enc = BrainEncoder(size="test", patch_xyz=(8, 8, 4), in_chans=1, proj_dim=16, proj_hidden=32,
                       checkpoint_blocks=False)
    with pytest.raises(ValueError):
        attach_bank(enc, {"window_bank": BANK})
    assert attach_bank(enc, {}).window_bank is None


# ---------------------------------------------------------------------------------------------------- augmentation
def test_native_appearance_keeps_ct_physical_and_mri_unchanged_in_kind():
    from brain.data.augment import appearance

    torch.manual_seed(0)
    x = torch.cat([torch.randn(2, 1, 8, 16, 16) * 30 + 40, torch.randn(2, 1, 8, 16, 16)])
    x[:2, :, :, :2] = -1024.0                                 # CT air
    mod = torch.tensor([0, 0, 1, 1])
    # MRI-only augmentation (bias field): CT rows untouched, output stays native units with one channel
    y = appearance(x, mod, {"window": {"mode": "native"}, "bias": {"p": 1.0, "amp": 0.3}})
    assert y.shape == x.shape and torch.equal(y[:2], x[:2]) and not torch.allclose(y[2:], x[2:])
    # CT physical: offset / scale on the foreground only, air stays at the floor; MRI rows untouched
    y = appearance(x, mod, {"window": {"mode": "native"}, "ct_physical": {"p": 1.0, "offset_hu": 10, "scale": 0.03}})
    assert torch.equal(y[2:], x[2:]) and torch.equal(y[:2, :, :, :2], x[:2, :, :, :2])
    d = (y[:2] - x[:2])[:, :, :, 2:]
    assert d.abs().max() <= 10 + 0.03 * x[:2].abs().max() + 1e-3 and d.abs().max() > 0
    # kernel blur / sharpen: CT only, in-plane only (a z-constant volume stays z-constant)
    zc = x[:, :, :1].expand_as(x).clone()
    y = appearance(zc, mod, {"window": {"mode": "native"}, "kernel": {"p": 1.0, "sharpen": 0.5}})
    assert torch.equal(y[2:], zc[2:]) and not torch.allclose(y[:2], zc[:2])
    assert torch.allclose(y[:2], y[:2, :, :1].expand_as(y[:2]), atol=1e-4)
    # no augmentation at all: the input, unwindowed
    assert torch.equal(appearance(x, mod, {"window": {"mode": "native"}}), x)


def test_trainer_saves_the_bank_with_the_encoder_and_frozen_eval_loads_it(tmp_path, monkeypatch):
    """Two synthetic steps with the bank on: the checkpoint's encoder (``model``) holds the windows, the config records
    the bank, and load_brain3d rebuilds the encoder with them."""
    from brain.downstream.embed3d import load_brain3d
    from brain.train import engine

    monkeypatch.setenv("BRAIN_PERSIST_SYNTHETIC", "1")
    engine.run(["multitask", "--synthetic", "--max-steps", "2", "--log-every", "1", "--out-dir", str(tmp_path),
                "--set", "model.window_bank.enabled=true", "--set", "train.ckpt_every=2"])
    ck = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert {"window_bank.c_raw", "window_bank.w_raw"} <= set(ck["model"])
    assert bank_cfg(ck["config"]["model"]) is not None
    frozen = load_brain3d(ck, CPU)
    assert frozen.bank is not None and torch.equal(frozen.bank.c_raw, ck["model"]["window_bank.c_raw"])
