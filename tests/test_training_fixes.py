"""Regressions for the 2026-10-25 review fixes: gathered-term gradients, fp32 RoPE / residual stream, protected
live targets, dense consensus by views, balancer clamp and data term, parameter groups. python tests/run.py
tests/test_training_fixes.py"""

from __future__ import annotations

import torch

from brain.common.optim import param_groups
from brain.models.rope3d import RoPE3D
from brain.multitask.data import SyntheticMultitask, collate, prepare, protected_tasks
from brain.multitask.losses import Balancer
from brain.multitask.registry import load_registry
from brain.objectives.dense import consensus
from brain.objectives.sigreg import gather_batch, scale_grad
from brain.train.defaults import default_config, smoke_config

CPU = torch.device("cpu")
REG = load_registry()


def test_scale_grad_keeps_the_value_and_scales_the_gradient():
    z = torch.randn(3, requires_grad=True)
    y = scale_grad(z, 4.0)
    assert torch.equal(y, z)
    y.sum().backward()
    assert torch.allclose(z.grad, torch.full((3,), 4.0))
    z2 = torch.randn(2, 3, 4, requires_grad=True)          # single process: world_ref reproduces the old 1/ref weighting
    gather_batch(z2, True, world_ref=2).sum().backward()
    assert torch.allclose(z2.grad, torch.full_like(z2, 0.5))


def test_fp32_rope_is_accurate_under_bf16_and_legacy_is_not():
    c = torch.rand(1, 500, 3) * torch.tensor([256.0, 256.0, 160.0])
    q = torch.randn(1, 2, 500, 64)
    ref, _ = RoPE3D(64).rotate(q, q, c)                    # fp32 everywhere
    fixed, _ = RoPE3D(64, fp32=True).rotate(q.bfloat16(), q.bfloat16(), c)
    legacy, _ = RoPE3D(64).rotate(q.bfloat16(), q.bfloat16(), c)
    err_fixed, err_legacy = (fixed.float() - ref).abs().max(), (legacy.float() - ref).abs().max()
    assert err_fixed < 0.1 < err_legacy


def test_fp32_residual_stream_under_autocast():
    from brain.models.vit3d import BrainEncoder

    seen = []
    for flag in (False, True):
        enc = BrainEncoder(size="test", patch_xyz=(8, 8, 4), qk_norm=True, n_cls=1, proj_dim=16, proj_hidden=32,
                           checkpoint_blocks=False, fp32_residual=flag).eval()
        h = enc.blocks[1].register_forward_pre_hook(lambda m, a: seen.append(a[0].dtype))
        from brain.multitask.explain import encode_backbone

        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            encode_backbone(enc, torch.randn(1, 1, 16, 32, 32), torch.ones(1, 4 * 4 * 4, dtype=torch.bool))
        h.remove()
    assert seen == [torch.bfloat16, torch.float32]


def test_live_target_rows_get_no_rotation_bias_or_drift():
    fov, live = protected_tasks(REG)
    live_ids = {REG.tasks[i].id for i in torch.nonzero(live).flatten().tolist()}
    assert {"brain_asymmetry", "intensity_asymmetry", "ct_brain_density"} <= live_ids and bool((fov | ~live).all())
    raw = collate([SyntheticMultitask(REG, (16, 32, 32), n=4)[i] for i in range(4)])
    li = REG.index["brain_asymmetry"]
    raw["task"][0] = -1
    raw["task"][0, 0] = li                                   # row 0 asks a live target; the others do not
    raw["task"][1:] = torch.where(live[raw["task"][1:].clamp_min(0)], torch.full_like(raw["task"][1:], -1), raw["task"][1:])
    cfg = smoke_config(default_config("multitask"))
    cfg["aug"] = {"flip_lr": 0.0, "affine": {"p": 1.0, "rot_deg": [10, 10, 10], "shift_mm": 4.0},
                  "appearance": {"bias": {"p": 1.0, "amp": 0.3}, "drift": {"p": 1.0, "amp": 0.3}}}
    plain = prepare(raw, cfg, CPU, (8, 8, 4), train=False)
    aug = prepare(raw, cfg, CPU, (8, 8, 4), train=True, aug_unsafe=live)
    assert torch.allclose(aug["x"][0], plain["x"][0], atol=1e-5)            # protected row untouched
    assert not torch.allclose(aug["x"][1:], plain["x"][1:])


def test_dense_consensus_counts_views_not_tokens():
    d = 4
    one = torch.randn(1, 2, d)                               # view A puts both its tokens on position 0
    a_pos, a_ok = torch.tensor([[0, 0]]), torch.ones(1, 2, dtype=torch.bool)
    b_emb, b_pos, b_ok = torch.randn(1, 1, d), torch.tensor([[1]]), torch.ones(1, 1, dtype=torch.bool)
    loss, cov = consensus([one, b_emb], [a_ok, b_ok], [a_pos, b_pos], n_pos=2)
    assert float(cov) == 0.0 and float(loss) == 0.0          # no position has two views
    b_pos2 = torch.tensor([[0]])                             # now view B covers position 0 too: mean over views
    loss2, _ = consensus([one, b_emb], [a_ok, b_ok], [a_pos, b_pos2], n_pos=2)
    mean = (one[0].mean(0) + b_emb[0, 0]) / 2
    want = ((one[0] - mean).square().sum() + (b_emb[0, 0] - mean).square().sum()) / (3 * d)
    assert torch.allclose(loss2, want)


def test_uncertainty_weights_are_clamped_and_the_data_term_stays_positive():
    bal = Balancer(3, "uncertainty", clamp=1.0)
    with torch.no_grad():
        bal.log_var.copy_(torch.tensor([-5.0, 0.0, 5.0]))
    w = bal.weights()
    assert float(w.max()) <= torch.e + 1e-4 and float(w.min()) >= 1 / torch.e - 1e-4
    loss = torch.tensor([[0.05, 0.05, 0.05]])
    total, _ = bal(loss, torch.ones_like(loss, dtype=torch.bool), torch.tensor([[0, 1, 2]]))
    assert float(bal.data_term) > 0 and float(total) < float(bal.data_term)    # the +s part can push the total down


def test_final_norm_gets_the_deepest_lr_and_task_tables_no_decay():
    from brain.multitask.model import MultitaskModel
    from brain.train.multitask import build_encoder

    cfg = smoke_config(default_config("multitask"))
    model = MultitaskModel(build_encoder(cfg, CPU, synthetic=True), REG, cfg["model"])
    groups = param_groups(model, 0.05, layer_decay=0.8)
    where = {id(p): g for g in groups for p in g["params"]}
    named = dict(model.named_parameters())
    assert where[id(named["backbone.norm.weight"])]["lr_scale"] == 1.0
    assert where[id(named["backbone.patch_embed.proj.weight"])]["lr_scale"] < 1.0
    assert where[id(named["task_emb.weight"])]["weight_decay"] == 0.0
