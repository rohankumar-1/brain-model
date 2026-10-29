"""CPU tests for the shared training pieces: schedule, parameter groups, resume discovery, best tracking."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from brain.common.checkpoint import atomic_torch_save, find_resume, load_ckpt, state_is_finite
from brain.common.metrics import effective_rank
from brain.common.optim import finite_params, lr_at, param_groups, set_lr
from brain.train.engine import build_config, is_new_best, parse_args


def test_lr_schedule_warms_up_then_decays_to_the_floor():
    peak = 1.0
    assert lr_at(0, 100, peak, 10) == peak / 10
    assert lr_at(9, 100, peak, 10) == peak
    assert lr_at(100, 100, peak, 10) < 0.011 * peak
    assert lr_at(55, 100, peak, 10) < lr_at(20, 100, peak, 10)


def test_param_groups_skip_decay_on_biases_norms_and_tokens():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.norm = nn.LayerNorm(4)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 4))

    m = M()
    g = param_groups(m, 0.05)
    decayed = {id(p) for grp in g if grp["weight_decay"] > 0 for p in grp["params"]}
    assert id(m.lin.weight) in decayed
    assert id(m.lin.bias) not in decayed and id(m.cls_token) not in decayed and id(m.norm.weight) not in decayed


def test_layer_decay_scales_deeper_blocks_higher_and_set_lr_applies_them():
    class Backbone(nn.Module):
        depth = 3

        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
            self.patch_embed = nn.Linear(4, 4)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.head = nn.Linear(4, 4)

    net = Net()
    groups = param_groups(net, 0.05, layer_decay=0.5, lr=1.0)
    scales = sorted({g["lr_scale"] for g in groups})
    assert scales == [0.125, 0.25, 0.5, 1.0]  # embeddings 0.5**3 ... head 1.0
    opt = torch.optim.SGD(groups, lr=1.0)
    set_lr(opt, 0.1)
    assert all(abs(g["lr"] - 0.1 * g["lr_scale"]) < 1e-9 for g in opt.param_groups)


def test_is_new_best_keeps_lower_val_loss_and_rejects_nan():
    assert is_new_best({"loss": 1.2}, "loss", 1.5) == 1.2
    assert is_new_best({"loss": 1.7}, "loss", 1.5) is None
    assert is_new_best({"loss": float("nan")}, "loss", 1.5) is None
    assert is_new_best({"erank": 3.0}, "loss", 1.5) is None
    assert is_new_best({"loss": 0.5}, "missing_metric", 1.5) == 0.5  # falls back to loss


def test_find_resume_prefers_last_then_the_configured_fallback(tmp_path):
    td = tmp_path
    out, fallback = Path(td) / "run", Path(td) / "old.pt"
    assert find_resume(out, fallback) is None
    atomic_torch_save({"step": 1}, fallback)
    assert find_resume(out, fallback) == fallback
    atomic_torch_save({"step": 2}, out / "last.pt")
    assert find_resume(out, fallback) == out / "last.pt"
    assert load_ckpt(out / "last.pt")["step"] == 2 and not list(out.glob("*.tmp"))


def test_nan_guards():
    assert state_is_finite({"w": torch.ones(2), "n": torch.tensor(3)})
    assert not state_is_finite({"w": torch.tensor([1.0, float("nan")])})
    m = nn.Linear(2, 2)
    assert finite_params(m)
    m.weight.data[0, 0] = float("inf")
    assert not finite_params(m)


def test_effective_rank_separates_collapsed_from_spread_features():
    torch.manual_seed(0)
    assert effective_rank(torch.randn(256, 16)) > 12
    collapsed = torch.randn(256, 1) @ torch.randn(1, 16)
    assert effective_rank(collapsed) < 2


def test_build_config_layers_defaults_file_and_overrides():
    args = parse_args(["lejepa3d", "--out-dir", "runs/x", "--set", "optim.lr=1e-4", "--set", "aug.curve.p=0"])
    cfg = build_config(args)
    assert cfg["output"]["out_dir"] == "runs/x" and cfg["optim"]["lr"] == 1e-4 and cfg["aug"]["curve"]["p"] == 0
    assert cfg["aug"]["bias"]["p"] > 0                     # untouched defaults survive
    smoke = build_config(parse_args(["lejepa3d", "--synthetic"]))
    assert smoke["data"]["volume_size_xyz"] == [32, 32, 16]
    for method in ("mae3d", "multitask", "joint"):
        assert build_config(parse_args([method]))["optim"]["batch_size"] > 0


def test_is_new_best_maximises_the_validation_score_without_a_loss_fallback():
    assert is_new_best({"score": 0.6}, "score", 0.5) == 0.6
    assert is_new_best({"score": 0.4}, "score", 0.5) is None
    assert is_new_best({"score": 0.1}, "score", float("-inf")) == 0.1
    assert is_new_best({"loss": 0.1}, "score", 0.5) is None          # a loss never stands in for the score


def test_wsd_holds_the_peak_then_decays_to_the_floor_so_a_run_can_be_extended():
    peak = 1.0
    assert lr_at(0, 100, peak, 10, kind="wsd") == peak / 10
    assert lr_at(10, 100, peak, 10, kind="wsd") == peak == lr_at(79, 100, peak, 10, kind="wsd")   # last 20% decays
    assert lr_at(90, 100, peak, 10, kind="wsd") < lr_at(81, 100, peak, 10, kind="wsd") < peak
    assert abs(lr_at(100, 100, peak, 10, floor=0.01, kind="wsd") - 0.01) < 1e-12
    # extending a run that is still constant changes nothing up to the old decay start
    assert all(lr_at(s, 100, peak, 10, kind="wsd") == lr_at(s, 200, peak, 10, kind="wsd") for s in range(80))
    assert lr_at(95, 100, peak, 10, kind="wsd", decay_steps=10) < peak == lr_at(89, 100, peak, 10, kind="wsd", decay_steps=10)
