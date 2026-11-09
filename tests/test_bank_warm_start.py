"""Window-bank warm start from a 1-channel (canonical window) checkpoint: identical outputs at the bank's init, and
the trainer resumes such a checkpoint with the bank on."""

from __future__ import annotations

import copy

import torch

from brain.models.window_bank import add_bank, bank_of, tissue_channel, warm_start_state, WindowBank
from brain.multitask.data import SyntheticMultitask, collate, prepare
from brain.multitask.model import BrainQuery
from brain.multitask.registry import load_registry
from brain.train.defaults import default_config, smoke_config
from brain.train.multitask import build_encoder

CPU = torch.device("cpu")
REG = load_registry()
BANK = {"enabled": True, "k": 6, "init": "contract6", "per_modality": False, "lr_mult": 0.1, "decoder_grad": False}


def test_tissue_channel_is_the_canonical_window():
    bank = WindowBank(6)
    ch = tissue_channel(bank)
    x = torch.linspace(-1100, 1500, 64).view(1, 1, 4, 4, 4)
    ct = bank(x, torch.tensor([0]))[:, ch]
    assert torch.allclose(ct, torch.asinh((x[:, 0] - 40.0) / 40.0), atol=1e-5)
    z = torch.linspace(-6, 10, 64).view(1, 1, 4, 4, 4)
    mr = bank(z, torch.tensor([1]))[:, ch]
    assert torch.allclose(mr, torch.asinh((z[:, 0] - 0.5) / 1.5), atol=1e-5)


def _batch():
    ds = SyntheticMultitask(REG, (16, 32, 32), n=4, seed=3, k_max=6)
    items = [ds[i] for i in range(4)]
    for it in items[:2]:                                       # two CT rows (HU), two MRI rows (z)
        it["image"] = (it["image"].float() * 300.0).half()
        it["modality"] = 0
    return collate(items)


def test_warm_started_bank_model_answers_like_the_one_channel_model():
    torch.manual_seed(0)
    cfg = smoke_config(default_config("multitask"))
    plain = BrainQuery(build_encoder(copy.deepcopy(cfg), CPU, synthetic=True), REG, cfg["model"]).eval()
    plain.set_token_tasks([REG.index[t] for t in REG.train_ids(cfg["tasks"])])
    bcfg = copy.deepcopy(cfg)
    bcfg["model"]["window_bank"] = dict(BANK)
    torch.manual_seed(1)
    banked = BrainQuery(build_encoder(bcfg, CPU, synthetic=True), REG, bcfg["model"]).eval()
    bank = bank_of(banked.backbone)
    state = warm_start_state(plain.state_dict(), banked.state_dict(), tissue_channel(bank))
    res = banked.load_state_dict(state, strict=False)
    assert all(k.startswith("backbone.window_bank.") for k in res.missing_keys) and not res.unexpected_keys
    banked.set_token_tasks([REG.index[t] for t in REG.train_ids(cfg["tasks"])])
    raw = _batch()
    a = prepare(raw, cfg, CPU, (8, 8, 4), train=False)
    b = prepare(raw, bcfg, CPU, (8, 8, 4), train=False)
    assert a["x"].shape[1] == 1 and b["x"].shape[1] == 1 and torch.equal(a["keep"], b["keep"])
    with torch.no_grad():
        oa = plain(a["x"], a["keep"], a["task"], a["seg_pair"], meta=a["meta"])
        ob = banked(b["x"], b["keep"], b["task"], b["seg_pair"], meta=b["meta"], modality=b["modality"])
    for k in ("binary", "regression"):
        assert torch.allclose(oa[k], ob[k], atol=1e-4), k
    assert len(a["seg_pair"]) and torch.allclose(oa["seg"], ob["seg"], atol=1e-4)


def test_add_bank_keeps_a_trained_encoders_tokens():
    torch.manual_seed(0)
    cfg = smoke_config(default_config("multitask"))
    enc = build_encoder(copy.deepcopy(cfg), CPU, synthetic=True).eval()
    ref = copy.deepcopy(enc)
    enc = add_bank(enc, {"window_bank": dict(BANK)}).eval()
    assert enc.in_chans == 6 and bank_of(enc) is not None
    raw = _batch()
    a = prepare(raw, cfg, CPU, (8, 8, 4), train=False)
    x = a["x"]
    z = raw["image"].float().unsqueeze(1)
    from brain.multitask.model import BrainQuery as _BQ  # noqa: F401  (coords helper below)
    from brain.data.coords import crop_bbox_mm, patch_size_mm, token_centers_mm

    size = (32, 32, 16)
    bbox = crop_bbox_mm((0, 0, 0), size, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    c, p = token_centers_mm(bbox, size, (8, 8, 4)), patch_size_mm(bbox, size, (8, 8, 4))
    n = x.shape[0]
    with torch.no_grad():
        t0 = ref(x.permute(0, 1, 4, 3, 2), c.expand(n, -1, -1), p.expand(n, -1), keep=a["keep"])["tokens"]
        t1 = enc(bank_of(enc)(z, raw["modality"]).permute(0, 1, 4, 3, 2), c.expand(n, -1, -1), p.expand(n, -1),
                 keep=a["keep"])["tokens"]
    assert torch.allclose(t0, t1, atol=1e-4)


def test_trainer_resumes_a_bankless_checkpoint_with_the_bank(tmp_path, monkeypatch):
    from brain.train import engine

    monkeypatch.setenv("BRAIN_PERSIST_SYNTHETIC", "1")
    src = tmp_path / "plain"
    engine.run(["multitask", "--synthetic", "--max-steps", "2", "--log-every", "1", "--out-dir", str(src),
                "--set", "train.eval_every=100", "--set", "train.ckpt_every=2", "--set", "train.ema_decay=0.9"])
    dst = tmp_path / "bank"
    engine.run(["multitask", "--synthetic", "--max-steps", "4", "--log-every", "1", "--out-dir", str(dst),
                "--set", f"output.resume_from={src / 'last.pt'}", "--reset-schedule",
                "--set", "model.window_bank={enabled: true, k: 6}", "--set", "train.ema_decay=0.9",
                "--set", "train.eval_every=100", "--set", "train.ckpt_every=2"])
    ck = torch.load(dst / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 4 and ck["multitask"]["backbone.patch_embed.proj.weight"].shape[1] == 6
    assert any(".window_bank." in k for k in ck["multitask"])
