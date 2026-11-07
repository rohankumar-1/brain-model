"""Task-conditioned multitask model: registry, task sampling, batch preparation, model outputs, losses, trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from brain.common.optim import param_groups
from brain.multitask.data import MultitaskDataset, SyntheticMultitask, collate, prepare, protected_tasks
from brain.multitask.labels import live_targets
from brain.multitask.losses import Balancer, seg_loss, slot_losses
from brain.multitask.model import BrainQuery
from brain.multitask.registry import load_registry
from brain.train.defaults import default_config, smoke_config
from brain.train.multitask import build_encoder

SHAPE = (16, 32, 32)                              # (Z, Y, X): the smoke grid
CPU = torch.device("cpu")
NO_SEG = torch.zeros(0, 2, dtype=torch.long)      # no segmentation slots
REG = load_registry()


def _cfg(**aug):
    cfg = smoke_config(default_config("multitask"))
    if aug:
        cfg["aug"] = aug
    return cfg


def _model(tokens: bool = True):
    """The default (task-token) model, or with ``tokens=False`` the slot readout (the ResNet specialists')."""
    torch.manual_seed(0)
    cfg = _cfg()
    cfg["model"]["task_tokens"]["enabled"] = tokens
    model = BrainQuery(build_encoder(cfg, CPU, synthetic=True), REG, cfg["model"]).eval()
    if tokens:
        model.set_token_tasks([REG.index[t] for t in REG.train_ids(cfg["tasks"])])
    return cfg, model


def _raw(n=4):
    ds = SyntheticMultitask(REG, SHAPE, n=n)
    return collate([ds[i] for i in range(n)])


def _prepared(cfg, n=4, train=False, patch=(8, 8, 4)):
    return prepare(_raw(n), cfg, CPU, patch, train=train)


def _tables(tasks_by_volume: dict[str, list[tuple[str, float, str]]]):
    """Label and volume tables for volumes ``{vid: [(task, value, ref), ...]}`` (all train)."""
    rows = [(v, t, y, r) for v, ts in tasks_by_volume.items() for t, y, r in ts]
    labels = pd.DataFrame(rows, columns=["volume_id", "task", "value", "ref"]).assign(split="train", use="train")
    vols = pd.DataFrame({"volume_id": list(tasks_by_volume), "image_path": "a.zip", "modality": "mri",
                         "dataset_family": [f"f{i}" for i in range(len(tasks_by_volume))]})
    return labels, vols


def test_registry_tasks_are_complete():
    assert REG.trained and len(REG.trained) < len(REG)
    for t in REG.tasks:
        assert t.sources, t.id
        assert t.type != "multiclass" or len(t.classes) >= 2
        assert t.analog is None or REG[t.analog].trained, t.id


def test_collate_pads_slots_and_pairs_segmentation_masks():
    b = _raw()
    assert b["task"].shape[1] == int((b["task"] >= 0).sum(1).max())
    assert torch.isnan(b["y"][b["task"] < 0]).all()
    assert all(REG[int(b["task"][i, j])].type == "segmentation" for i, j in b["seg_pair"].tolist())
    assert len(b["seg"]) == len(b["seg_pair"])


@pytest.mark.parametrize("tokens", [True, False])
def test_forward_gives_every_head_and_one_mask_per_seg_slot(tokens):
    cfg, model = _model(tokens=tokens)
    b = _prepared(cfg)
    out = model(b["x"], b["keep"], b["task"], b["seg_pair"])
    bk = tuple(b["task"].shape)
    assert out["binary"].shape == bk and out["regression"].shape == bk
    assert out["multiclass"].shape == (*bk, REG.max_classes())
    if len(b["seg_pair"]):
        assert out["seg"].shape == (len(b["seg_pair"]), 1, *(s // 2 for s in SHAPE))


@torch.no_grad()
def test_slots_are_permutation_equivariant():
    """Slots carry no order: permuting the requested tasks permutes the outputs and changes nothing else."""
    cfg, model = _model()
    b = _prepared(cfg)
    task, perm = torch.tensor([[0, 1, 4, 5]] * b["x"].shape[0]), torch.tensor([2, 0, 3, 1])
    a, p = (model(b["x"], b["keep"], t, NO_SEG) for t in (task, task[:, perm]))
    for head in ("binary", "regression"):
        assert torch.allclose(a[head][:, perm], p[head], atol=1e-5)


@torch.no_grad()
@torch.no_grad()
def test_multiclass_logits_cover_only_the_tasks_classes():
    cfg, model = _model()
    b = _prepared(cfg)
    t = REG.index["neuro_diagnosis"]
    lg = model(b["x"], b["keep"], torch.full((b["x"].shape[0], 1), t), NO_SEG)["multiclass"]
    k = len(REG[t].classes)
    assert torch.isfinite(lg[..., :k]).all() and torch.isinf(lg[..., k:]).all()


def test_losses_are_masked_per_slot_and_balanced_per_task():
    cfg, model = _model()
    model.train()
    b = _prepared(cfg, n=6, train=True)
    out = model(b["x"], b["keep"], b["task"], b["seg_pair"])
    loss, has = slot_losses(out, b["task"], b["y"], model.task_type, b["seg"], b["seg_pair"])
    assert not has[b["task"] < 0].any() and (loss[~has] == 0).all()
    total, per_task = Balancer(len(REG))(loss, has, b["task"])
    assert torch.isfinite(total) and (per_task >= 0).all()
    total.backward()
    assert model.task_emb.weight.grad is not None


def test_balancer_weighs_groups_not_task_counts():
    bal = Balancer(len(REG), group_of=[REG.groups.index(t.group) for t in REG.tasks])
    many = [REG.index[t.id] for t in REG.trained if t.group == "anatomy_volume"][:10]
    task = torch.tensor([many + [REG.index["age"]]])
    total, _ = bal(torch.tensor([[1.0] * len(many) + [3.0]]), torch.ones_like(task, dtype=torch.bool), task)
    assert abs(float(total) - 2.0) < 1e-5            # (group means 1 and 3) / 2, not (10 + 3) / 11 (s = 0)


def test_seg_loss_prefers_the_right_mask():
    t = torch.zeros(1, 1, 8, 8, 8)
    t[..., 2:6, 2:6, 2:6] = 1
    assert seg_loss(t * 20 - 10, t) < 0.1 < seg_loss(10 - t * 20, t)


def test_flip_moves_image_and_mask_together():
    cfg = _cfg(flip_lr=1.0, affine={"p": 0.0})        # every row flips
    raw = _raw()
    if len(raw["seg_pair"]):
        b, plain = (prepare(raw, cfg, CPU, (8, 8, 4), train=t) for t in (True, False))
        assert torch.equal(b["seg"], plain["seg"].flip(-1)) and torch.allclose(b["x"], plain["x"].flip(-1))


def test_rotation_keeps_mask_mass():
    cfg = _cfg(flip_lr=0.0, affine={"p": 1.0, "rot_deg": [10, 10, 10], "shift_mm": 0.0})
    raw = _raw()
    if len(raw["seg_pair"]):
        b = prepare(raw, cfg, CPU, (8, 8, 4), train=True)
        assert torch.allclose(raw["seg"].float().sum(dim=(1, 2, 3)), b["seg"].sum(dim=(1, 2, 3, 4)), rtol=0.1)


def test_live_targets_measure_left_right_asymmetry():
    zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, s) for s in (32, 64, 64)], indexing="ij")
    brain = (zz**2 + yy**2 + xx**2) < 0.7
    img = np.where(brain, 30.0, -1000.0).astype(np.float32)
    sym = live_targets(img, brain, "ct")
    assert sym["mask_asym"] < 1e-6 and sym["intensity_asym"] < 1e-6
    assert abs(sym["ct_brain_hu"] - 30.0) < 1e-3 and sym["ct_csf_frac"] == 0.0
    lesion = np.where((xx > 0.2) & (xx < 0.6) & brain, 70.0, img).astype(np.float32)   # right-sided hyperdensity
    les = live_targets(lesion, brain, "ct")
    assert les["intensity_asym"] > 0.05 and les["mask_asym"] < 1e-6
    assert np.isnan(live_targets(img, brain, "mri")["ct_csf_frac"])


def test_dataset_draws_subsets_of_available_tasks(tmp_path):
    labels, vols = _tables({"v1": [("sex", 1.0, ""), ("age", 40.0, ""), ("brain_mask", np.nan, "@brain")],
                            "v2": [("sex", 0.0, "")]})
    ds = MultitaskDataset(tmp_path, labels, vols, REG, {"age": {"mean": 30.0, "std": 10.0}}, split="train",
                          k_max=8, p_all=0.0, max_seg=0)
    avail = ds.items[0][1]
    assert len(ds) == 2 and {REG[t].id for t, _, _ in avail} == {"sex", "age", "brain_mask"}
    assert {REG[t].id: y for t, y, _ in avail}["age"] == 1.0          # standardised
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = ds._draw(avail, rng)
        assert 1 <= len(d) <= 3 and all(REG[t].type != "segmentation" for t, _, _ in d)
    ev = MultitaskDataset(tmp_path, labels, vols, REG, {}, split="train", train=False, max_seg=1)
    assert len(ev._draw(ev.items[0][1], rng)) == 3


def test_group_sampling_gives_each_group_its_share(tmp_path):
    """18 tasks of one group and 2 of another, all available: draws split ~evenly between the groups."""
    ids = [t.id for t in REG.trained if t.group in ("anatomy_volume", "demographics")]
    labels, vols = _tables({"v1": [(t, 1.0, "") for t in ids]})
    ds = MultitaskDataset(tmp_path, labels, vols, REG, {}, split="train", k_max=1, p_all=0.0)
    rng = np.random.default_rng(0)
    share = np.mean([REG[ds._draw(ds.items[0][1], rng)[0][0]].group == "demographics" for _ in range(2000)])
    assert 0.45 < share < 0.55, share


def test_fewshot_fits_only_the_held_out_row():
    from brain.multitask.fewshot import fit_embedding

    cfg, model = _model()
    model.requires_grad_(False)
    t = next(i for i, task in enumerate(REG.tasks) if not task.trained and task.type == "binary")
    ds = SyntheticMultitask(REG, SHAPE, n=4)
    items = [{**ds[i], "task": torch.tensor([t]), "y": torch.tensor([float(i % 2)]), "seg_slot": torch.tensor([-1]),
              "seg": torch.zeros(0, *SHAPE, dtype=torch.uint8)} for i in range(4)]
    before = model.task_emb.weight.detach().clone()
    amp = dict(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    saved = fit_embedding(model, REG, cfg, [collate(items[:2]), collate(items[2:])], t, CPU, amp, steps=5, lr=1e-2)
    after = model.task_emb.weight.detach()
    others = torch.arange(len(REG)) != t
    assert torch.equal(before[others], after[others]) and not torch.equal(before[t], after[t])
    assert torch.equal(saved["task"], before[t]) and not model.task_emb.weight.requires_grad


def test_trainer_runs_on_phantoms(tmp_path):
    """Three steps of the slot readout on phantoms (raises on any failure); the task-token trainer has its own test
    (test_task_tokens)."""
    from brain.train import engine

    engine.run(["multitask", "--synthetic", "--max-steps", "3", "--log-every", "1", "--out-dir", str(tmp_path),
                "--set", "model.task_tokens.enabled=false"])


def test_weight_average_is_saved_evaluated_and_resumed(tmp_path, monkeypatch):
    """train.ema_decay: last_ema.pt holds averaged weights (not the live ones), every eval logs the average too, and a
    resume restores the average with its update count."""
    from brain.train import engine

    monkeypatch.setenv("BRAIN_PERSIST_SYNTHETIC", "1")
    args = ["multitask", "--synthetic", "--log-every", "1", "--out-dir", str(tmp_path), "--set", "train.ema_decay=0.9",
            "--set", "train.eval_every=2", "--set", "train.ckpt_every=2", "--set", "train.keep_every=2",
            "--set", "aug.task_aware=true"]
    engine.run(args + ["--max-steps", "4"])
    last, avg = (torch.load(tmp_path / n, map_location="cpu", weights_only=False) for n in ("last.pt", "last_ema.pt"))
    assert last["ema"]["n"] == 4 and "ema" not in avg
    k = next(k for k, v in last["multitask"].items() if v.is_floating_point() and "patch_embed" in k)
    assert not torch.equal(last["multitask"][k], avg["multitask"][k])
    assert torch.allclose(last["ema"]["avg"][k], avg["multitask"][k].float())
    assert all("ema" in h for h in last["history"])
    engine.run(args + ["--max-steps", "6", "--reset-schedule"])
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert last["ema"]["n"] == 6 and last["schedule"]["start"] == 4


def test_hl_gauss_target_and_loss():
    from brain.multitask.losses import hl_gauss_loss

    edges = torch.linspace(-5, 5, 81)
    centres = (edges[1:] + edges[:-1]) / 2
    y = torch.tensor([-1.3, 0.0, 2.2, 9.0])                         # the last is clipped to the range
    near = -((centres - y.clamp(-5, 5)[:, None]) ** 2) / (2 * 0.094 ** 2)
    good, flat = hl_gauss_loss(near, y, edges, 0.094), hl_gauss_loss(torch.zeros(4, 80), y, edges, 0.094)
    assert (good < 0.05).all() and (flat > 2).all()
    mean = (near.softmax(-1) * centres).sum(-1)
    assert torch.allclose(mean[:3], y[:3], atol=0.02)


def test_switch_resumed_run_to_hl_gauss(tmp_path, monkeypatch):
    """A scalar-regression run resumed with model.regression.kind=hlgauss gains the bin head (fresh optimizer)."""
    from brain.train import engine

    monkeypatch.setenv("BRAIN_PERSIST_SYNTHETIC", "1")
    args = ["multitask", "--synthetic", "--log-every", "1", "--out-dir", str(tmp_path), "--set", "train.eval_every=100",
            "--set", "train.ckpt_every=2", "--set", "train.ema_decay=0.9"]
    engine.run(args + ["--max-steps", "2"])
    engine.run(args + ["--max-steps", "4", "--reset-schedule", "--set", "model.regression.kind=hlgauss"])
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert last["schedule"]["start"] == 2 and any(k.startswith("regress_bins.") for k in last["ema"]["avg"])
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert last["step"] == 4 and any(k.startswith("regress_bins.") for k in last["multitask"])


def test_bins_from_scalar_keeps_predictions():
    from brain.multitask.model import BrainQuery
    from brain.multitask.registry import load_registry
    from brain.train.multitask import build_encoder
    from brain.train.defaults import default_config

    cfg = default_config("multitask")
    cfg["model"]["regression"]["kind"] = "hlgauss"
    reg = load_registry(None)
    m = BrainQuery(build_encoder(cfg, torch.device("cpu"), synthetic=True), reg, cfg["model"])
    z = torch.randn(3, 2, m.binary[0].in_features)
    m.bins_from_scalar()
    scalar = m.regress(z)[..., 0]
    out = m.heads(z, torch.zeros(3, 2, dtype=torch.long))
    assert torch.allclose(out["regression"], scalar.clamp(-4, 4), atol=1e-3)


def test_reset_schedule_rewarms_from_resume_step():
    from brain.train.engine import lr_of

    s = {"lr": 1e-3, "warmup_steps": 10, "max_steps": 60, "floor": 0.01, "kind": "wsd", "decay_steps": 10, "start": 30}
    assert lr_of(30, s) == 1e-4                         # first warmup step, not the peak
    assert lr_of(45, s) == 1e-3                         # stable phase
    assert abs(lr_of(60, s) - 1e-5) < 1e-12             # cooled down at max_steps
    assert lr_of(5, {**s, "start": 0}) == lr_of(35, s)  # same shape, shifted


def test_reset_schedule_is_idempotent_across_requeues():
    from brain.train.engine import resume_schedule

    old = {"lr": 6e-4, "warmup_steps": 1500, "max_steps": 30000, "floor": 0.01, "kind": "cosine", "decay_steps": None}
    new = {"lr": 3e-4, "warmup_steps": 1000, "max_steps": 60000, "floor": 0.01, "kind": "wsd", "decay_steps": 6000}
    first = resume_schedule(old, new, 30000, reset=True)
    assert first == {**new, "start": 30000}
    assert resume_schedule(first, new, 41000, reset=True) == first        # requeued chunk: no second re-warm
    assert resume_schedule(first, new, 41000, reset=False) == first
    assert resume_schedule(old, new, 30000, reset=False) == old


def test_sharp_rows_skip_blur_and_fov_cut():
    from brain.data.augment import degrade, through_plane

    x = torch.zeros(2, 1, 9, 9, 9)
    x[:, :, 4, 4, 4] = 1.0
    sharp = torch.tensor([True, False])
    tp = through_plane(x.clone(), {"p": 1.0, "fwhm_vox": [6.0, 6.0]}, None, skip=sharp)
    bl = degrade(x.clone(), {"blur": {"p": 1.0, "sigma_vox": [1.0, 1.0]}}, skip_blur=sharp)
    for y in (tp, bl):
        assert torch.equal(y[0], x[0]) and y[1, 0, 4, 4, 4] < 1.0


def test_registration_qc_only_drops_alignment_dependent_targets():
    from brain.multitask.labels import registration_sensitive

    assert all(registration_sensitive(t) for t in ("brain_volume", "brain_asymmetry", "vol_hippocampus",
                                                    "vol_cortex_asym", "tumor_volume"))
    assert not any(registration_sensitive(t) for t in ("hemorrhage_any", "age", "brain_mask", "seg_hippocampus",
                                                        "whole_tumor", "ct_csf_fraction"))


def test_fov_cut_blanks_image_and_masks_only_on_allowed_rows():
    from brain.multitask.data import fov_cut

    raw = torch.zeros(2, 1, 20, 8, 8)
    seg = torch.ones(2, 1, 20, 8, 8)
    mod = torch.tensor([0, 1])
    out, s = fov_cut(raw, seg, torch.tensor([0, 1]), mod, torch.tensor([True, False]), {"p": 1.0, "mm": [5, 5]})
    assert (out[0, :, :5] == -1024).all() and (out[0, :, 5:] == 0).all() and (s[0, :, :5] == 0).all()
    assert torch.equal(out[1], raw[1]) and torch.equal(s[1], seg[1])            # row 1 not allowed


def test_appearance_augmentation_keeps_the_foreground_gate():
    app = {"bias": {"p": 1.0, "amp": 0.2}, "through_plane": {"p": 1.0, "fwhm_vox": [2.0, 6.0]},
           "noise": {"p": 1.0, "kind": "rician", "snr": [15, 100], "sigma": [0.02, 0.1]},
           "blur": {"p": 1.0, "sigma_vox": [0.5, 1.0]}, "drift": {"p": 1.0, "amp": 0.1}}
    raw = _raw()
    plain = prepare(raw, _cfg(), CPU, (8, 8, 4), train=False)
    aug = prepare(raw, _cfg(flip_lr=0.0, affine={"p": 0.0}, appearance=app), CPU, (8, 8, 4), train=True)
    assert torch.equal(aug["keep"], plain["keep"]) and torch.isfinite(aug["x"]).all()
    assert not torch.allclose(aug["x"], plain["x"])


def test_a_three_channel_encoder_gets_the_role_windows_and_segments():
    """``model.in_chans`` 3: the three role windows in training and evaluation (unaugmented: exactly ``role_windows``),
    and the model runs on them, segmentation decoder included."""
    from brain.common.windowing import role_windows

    app = {"bias": {"p": 1.0, "amp": 0.2}, "noise": {"p": 1.0, "kind": "rician", "snr": [15, 100]},
           "blur": {"p": 1.0, "sigma_vox": [0.5, 1.0]}}
    cfg = _cfg(flip_lr=0.0, affine={"p": 0.0}, appearance=app)
    cfg["model"]["in_chans"] = 3
    raw = _raw()
    plain, aug = (prepare(raw, cfg, CPU, (8, 8, 4), train=t) for t in (False, True))
    assert plain["x"].shape[1] == 3 and aug["x"].shape == plain["x"].shape and torch.isfinite(aug["x"]).all()
    assert torch.allclose(plain["x"], role_windows(raw["image"].float().unsqueeze(1), raw["modality"]))
    assert torch.equal(aug["keep"], plain["keep"])
    torch.manual_seed(0)
    model = BrainQuery(build_encoder(cfg, CPU, synthetic=True), REG, cfg["model"]).eval()
    model.set_token_tasks([REG.index[t] for t in REG.train_ids(cfg["tasks"])])
    out = model(aug["x"], aug["keep"], aug["task"], aug["seg_pair"])
    assert torch.isfinite(out["binary"]).all()
    if len(aug["seg_pair"]):
        assert out["seg"].shape == (len(aug["seg_pair"]), 1, *(s // 2 for s in SHAPE))


def test_tasks_asked_alone_do_not_see_each_other_and_match_a_single_task_query():
    cfg, model = _model()
    b = _prepared(cfg)
    task = b["task"]
    with torch.no_grad():
        alone = model(b["x"], b["keep"], task, NO_SEG, alone=True)
        first = model(b["x"], b["keep"], task[:, :1], NO_SEG)                       # each volume's first task only
        other = task.clone()
        other[:, 1:] = -1                                                           # drop every other task
        dropped = model(b["x"], b["keep"], other, NO_SEG, alone=True)
    for head in ("binary", "regression"):
        assert torch.allclose(alone[head][:, 0], first[head][:, 0], atol=1e-5)
        assert torch.allclose(alone[head][:, 0], dropped[head][:, 0], atol=1e-5)


def test_sparse_seg_loss_does_not_reward_an_empty_map():
    """An all-background map scores near zero on a scan without a target but pays the voxel, cell and scan terms on one
    with a target; the plain loss prices the two cases alike enough to collapse (R10's aneurysm_location)."""
    from brain.multitask.losses import sparse_seg_loss

    empty = torch.full((2, 1, 8, 16, 16), -12.0)
    tgt = torch.zeros(2, 1, 8, 16, 16)
    tgt[1, 0, 3:5, 6:9, 6:9] = 1.0                                  # a small ball on the second scan only
    loss = sparse_seg_loss(empty, tgt, (4, 8, 8))
    assert loss[0] < 1e-3 and loss[1] > 5.0
    hot = empty.clone()
    hot[1, 0, 3:5, 6:9, 6:9] = 8.0                                  # the map lights up where the target is
    assert sparse_seg_loss(hot, tgt, (4, 8, 8))[1] < 0.5 * loss[1]


def test_slot_losses_route_sparse_tasks():
    from brain.multitask.data import TYPE_ID
    from brain.multitask.losses import seg_loss, slot_losses, sparse_seg_loss

    task = torch.tensor([[0, 1]])
    task_type = torch.tensor([TYPE_ID["segmentation"], TYPE_ID["segmentation"]])
    seg = torch.zeros(2, 1, 16, 32, 32)
    seg[0, 0, 4:8, 8:12, 8:12] = 1.0
    logits = torch.randn(2, 1, 8, 16, 16)
    out = {"seg": logits}
    pair = torch.tensor([[0, 0], [0, 1]])
    loss, has = slot_losses(out, task, torch.full((1, 2), float("nan")), task_type, seg, pair, 2,
                            sparse=torch.tensor([False, True]), cell_zyx=(4, 8, 8))
    t = torch.nn.functional.avg_pool3d(seg, 2)
    assert torch.allclose(loss[0, 0], seg_loss(logits[:1], t[:1])[0])
    assert torch.allclose(loss[0, 1], sparse_seg_loss(logits[1:], t[1:], (4, 8, 8))[0]) and has.all()


def test_synthrad_ct_mirrors_its_mr_rows():
    import pandas as pd

    from brain.multitask.labels import _mirror_synthrad_ct

    v = pd.DataFrame({"volume_id": ["mr1", "ct1", "mr2", "ct9"], "case_id": ["1B001", "1B001", "1B002", "2B009"],
                      "dataset": ["synthrad_t1w", "synthrad_ct", "synthrad_t1w", "synthrad_ct"]})
    f = pd.DataFrame({"volume_id": ["mr1", "mr2", "other"], "value": [1.0, 2.0, 3.0], "ref": ["a", "b", "c"],
                      "split": ["train", "val", "train"]})
    out = _mirror_synthrad_ct(f, v)
    ct = out[out.volume_id == "ct1"]
    assert len(out) == 4 and len(ct) == 1 and ct.value.iloc[0] == 1.0 and ct.ref.iloc[0] == "a"   # mr2 has no CT


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


def test_final_norm_gets_the_deepest_lr_and_task_tables_no_decay():
    from brain.multitask.model import BrainQuery
    from brain.train.multitask import build_encoder

    cfg = smoke_config(default_config("multitask"))
    model = BrainQuery(build_encoder(cfg, CPU, synthetic=True), REG, cfg["model"])
    groups = param_groups(model, 0.05, layer_decay=0.8)
    where = {id(p): g for g in groups for p in g["params"]}
    named = dict(model.named_parameters())
    assert where[id(named["backbone.norm.weight"])]["lr_scale"] == 1.0
    assert where[id(named["backbone.patch_embed.proj.weight"])]["lr_scale"] < 1.0
    assert where[id(named["task_emb.weight"])]["weight_decay"] == 0.0


def test_metadata_ids_harmonise_contrast_names():
    from brain.data.volumes import META, meta_id

    assert META[meta_id("t1w", "mri")] == META[meta_id("t1n", 1)] == META[meta_id(None, "mri", "oasis1")] == "t1"
    assert META[meta_id("t2f", "mri")] == "flair" and META[meta_id(float("nan"), "mri", "brainmetshare_bravo")] == "t1c"
    assert META[meta_id(None, "ct", "rsna_ich")] == "ct" and META[meta_id("cta", "ct")] == "cta"
    assert meta_id("weird", "mri") == 0                       # unknown label -> unknown


def test_skull_strip_never_touches_scans_asked_a_skull_task(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd

    from brain.data.volumes import META
    from brain.multitask.data import MultitaskDataset

    shape = (8, 12, 12)
    brain = np.zeros(shape, bool)
    brain[2:6, 3:9, 3:9] = True
    img = np.where(brain, 1.0, 5.0).astype(np.float32)      # "skull" brighter than brain
    img[0] = -6.0                                             # background floor
    monkeypatch.setattr("brain.prep.store.read_many", lambda path, keys: {"image": img, "brain": brain.astype(np.uint8)})
    rows = [("v_age", "age", 40.0, ""), ("v_mask", "brain_mask", np.nan, "@brain")]
    labels = pd.DataFrame([(v, t, y, r) for v, t, y, r in rows], columns=["volume_id", "task", "value", "ref"]) \
        .assign(split="train", use="train")
    vols = pd.DataFrame({"volume_id": ["v_age", "v_mask"], "image_path": "a.zip", "modality": "mri",
                         "dataset_family": ["f0", "f1"], "dataset": "d", "contrast": "t1w", "brain_ok": True,
                         "brain_method": "synthstrip", "stripped_input": False})
    ds = MultitaskDataset(tmp_path, labels, vols, REG, {"age": {"mean": 0.0, "std": 1.0}}, split="train",
                          k_max=1, p_all=0.0, max_seg=1, skull_strip=1.0)
    i_age = [v for v, _ in ds.items].index("v_age")
    a = ds[i_age]["image"].numpy()
    assert (a[~brain] == -6.0).all() and (a[brain] == 1.0).all()          # stripped to the background floor
    b = ds[1 - i_age]["image"].numpy()
    assert (b[~brain & (img == 5.0)] == 5.0).all()                        # brain_mask asked: skull kept
    assert ds[i_age]["meta"] == META.index("t1")
