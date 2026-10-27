"""OMNI-style task tokens and mixture-of-experts top blocks (``model.task_tokens``, ``model.moe``)."""

from __future__ import annotations

import torch

from brain.models.vit3d import BrainEncoder, MoEMLP, _task_mask
from brain.multitask.data import SyntheticMultitask, collate, prepare
from brain.multitask.model import BrainQuery
from brain.multitask.registry import load_registry
from brain.train.defaults import default_config, smoke_config
from brain.train.multitask import build_encoder

SHAPE = (16, 32, 32)
CPU = torch.device("cpu")
NO_SEG = torch.zeros(0, 2, dtype=torch.long)
REG = load_registry()


def _encoder(**kw) -> BrainEncoder:
    torch.manual_seed(0)
    return BrainEncoder(size="test", patch_xyz=(8, 8, 4), n_cls=1, proj_dim=16, proj_hidden=32, **kw)


def _inputs(b=2, n=40):
    torch.manual_seed(1)
    view = torch.randn(b, 1, 32, 32, 16)
    coords = torch.randn(b, 64, 3) * 20
    psize = torch.full((b, 3), 8.0)
    keep = torch.ones(b, 64, dtype=torch.bool)
    keep[1, n:] = False
    return view, coords, psize, keep


def _model(tokens=True, moe=0):
    torch.manual_seed(0)
    cfg = smoke_config(default_config("multitask"))
    cfg["model"]["task_tokens"] = {"enabled": tokens}
    cfg["model"]["moe"] = {"blocks": moe, "experts": 3, "top_k": 2}
    model = BrainQuery(build_encoder(cfg, CPU, synthetic=True), REG, cfg["model"]).eval()
    if tokens:
        model.set_token_tasks([REG.index[t] for t in REG.train_ids(cfg["tasks"])])
    return cfg, model


def _prepared(cfg, n=2):
    ds = SyntheticMultitask(REG, SHAPE, n=n)
    return prepare(collate([ds[i] for i in range(n)]), cfg, CPU, (8, 8, 4), train=False)


@torch.no_grad()
def test_moe_from_a_trained_mlp_changes_nothing():
    enc = _encoder()
    enc.eval()
    view, coords, psize, keep = _inputs()
    before = enc(view, coords, psize, keep=keep)["tokens"]
    enc.enable_moe(1, n_experts=4, top_k=2)
    assert isinstance(enc.blocks[-1].mlp, MoEMLP) and not isinstance(enc.blocks[0].mlp, MoEMLP)
    assert torch.allclose(enc(view, coords, psize, keep=keep)["tokens"], before, atol=1e-5)


def test_moe_balance_loss_reaches_the_router_through_checkpointed_blocks():
    enc = _encoder()
    enc.enable_moe(1, n_experts=3)
    enc.train()
    assert enc.checkpoint_blocks
    view, coords, psize, keep = _inputs()
    out = enc(view, coords, psize, keep=keep)
    aux = enc.moe_aux()
    assert 0.9 < float(aux) < 3.0
    (out["tokens"].pow(2).mean() + aux).backward()
    router = enc.blocks[-1].mlp.router.weight
    assert router.grad is not None and router.grad.abs().sum() > 0


def test_task_mask_isolates_task_tokens_and_keeps_read_only_ones_out():
    pad = torch.tensor([[True, True, True, True, True, True, False]])        # 1 prefix, 3 task, 3 patches (1 pad)
    m = _task_mask(pad, 1, 3, torch.tensor([True, True, False]))[0, 0]
    tt = slice(1, 4)
    assert torch.equal(m[tt, tt], torch.eye(3, dtype=torch.bool))            # task tokens: only themselves
    assert m[tt, 0].all() and m[tt, 4:6].all() and not m[tt, 6].any()       # ... plus prefix and kept patches
    assert m[4:6, 1:3].all() and not m[4:6, 3].any()                          # patches read written tokens only
    assert not m[0, 3]                                                        # the prefix too


@torch.no_grad()
def test_read_only_token_changes_nothing_else():
    enc = _encoder()
    enc.eval()
    view, coords, psize, keep = _inputs()
    tt = torch.randn(3, enc.embed_dim)
    write = torch.tensor([True, True, False])
    a = enc(view, coords, psize, keep=keep, task_tokens=tt, task_write=write)
    b = enc(view, coords, psize, keep=keep, task_tokens=torch.cat([tt[:2], torch.randn(1, enc.embed_dim)]),
            task_write=write)
    assert torch.allclose(a["tokens"], b["tokens"], atol=1e-5)
    assert torch.allclose(a["task_tokens"][:, :2], b["task_tokens"][:, :2], atol=1e-5)
    assert not torch.allclose(a["task_tokens"][:, 2], b["task_tokens"][:, 2])


@torch.no_grad()
def test_an_answer_does_not_depend_on_the_other_tasks_asked():
    cfg, model = _model(moe=1)
    b = _prepared(cfg)
    n = b["x"].shape[0]
    trained = [REG.index[t] for t in REG.train_ids(cfg["tasks"])]
    held = next(i for i, t in enumerate(REG.tasks) if not t.trained and t.type == "binary")
    t0 = next(i for i in trained if REG.tasks[i].type == "binary")
    others = [i for i in trained if i != t0][:3]
    alone, many, with_held = (model(b["x"], b["keep"], torch.tensor([t] * n), NO_SEG)["binary"][:, 0]
                              for t in ([t0], [t0, *others], [t0, held]))
    assert torch.allclose(alone, many, atol=1e-5) and torch.allclose(alone, with_held, atol=1e-5)
    out = model(b["x"], b["keep"], torch.tensor([[t0, -1]] * n), NO_SEG, alone=True)
    assert torch.allclose(out["binary"][:, 0], alone, atol=1e-5)


def test_token_model_trains_tokens_and_heads():
    cfg, model = _model(moe=1)
    model.train()
    b = _prepared(cfg)
    t0 = REG.index[REG.train_ids(cfg["tasks"])[0]]
    out = model(b["x"], b["keep"], torch.tensor([[t0]] * b["x"].shape[0]), NO_SEG)
    (out["binary"].sum() + out["regression"].sum() + model.backbone.moe_aux()).backward()
    assert model.task_emb.weight.grad[t0].abs().sum() > 0
    assert model.tt_proj.weight.grad.abs().sum() > 0


def test_moe_checkpoint_reloads_through_the_frozen_loader():
    from brain.downstream.embed3d import load_brain3d

    cfg, model = _model(moe=2)
    enc = model.backbone.eval()
    m = {**cfg["model"], "size": "test", "qk_norm": True, "proj_dim": 16, "proj_hidden": 32,   # the phantom encoder
         "n_cls": enc.n_cls}
    ck = {"model": enc.state_dict(), "config": {"model": m, "views": {"patch_xyz": [8, 8, 4]}}}
    net = load_brain3d(ck, CPU).model
    assert len(net.moe_blocks) == 2 and len(net.moe_blocks[0].experts) == 3
    view, coords, psize, keep = _inputs()
    with torch.no_grad():
        assert torch.allclose(net(view, coords, psize, keep=keep)["tokens"],
                              enc(view, coords, psize, keep=keep)["tokens"], atol=1e-5)


def test_trainer_runs_task_tokens_with_moe_on_phantoms(tmp_path):
    from brain.train import engine

    engine.run(["multitask", "--synthetic", "--max-steps", "2", "--log-every", "1", "--out-dir", str(tmp_path),
                "--set", "optim.freeze_steps=0", "--set", "model.task_tokens.enabled=true",
                "--set", "model.moe.blocks=1"])


@torch.no_grad()
def test_write_bias_keeps_a_warm_encoder_unchanged_until_it_opens():
    enc = _encoder()
    enc.eval()
    view, coords, psize, keep = _inputs()
    plain = enc(view, coords, psize, keep=keep)["tokens"]
    tt = torch.randn(3, enc.embed_dim)
    enc.task_write_bias = 30.0
    damped = enc(view, coords, psize, keep=keep, task_tokens=tt)
    assert torch.allclose(damped["tokens"], plain, atol=1e-4)
    enc.task_write_bias = 0.0
    assert not torch.allclose(enc(view, coords, psize, keep=keep, task_tokens=tt)["tokens"], plain, atol=1e-3)
