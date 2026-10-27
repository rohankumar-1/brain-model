"""Task-conditioned multitask model: registry, task sampling, batch preparation, model outputs, losses, trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from brain.multitask.data import MultitaskDataset, SyntheticMultitask, collate, prepare
from brain.multitask.labels import live_targets
from brain.multitask.losses import Balancer, seg_loss, slot_losses
from brain.multitask.model import MultitaskModel
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


def _model(tokens: bool = True, **slot):
    """The default (task-token) model, or with ``tokens=False`` the slot readout older checkpoints use."""
    torch.manual_seed(0)
    cfg = _cfg()
    cfg["model"]["slot"].update(slot)
    cfg["model"]["task_tokens"]["enabled"] = tokens
    model = MultitaskModel(build_encoder(cfg, CPU, synthetic=True), REG, cfg["model"]).eval()
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
def test_cross_task_off_makes_slots_independent():
    cfg, model = _model(tokens=False, cross_task=False)
    b = _prepared(cfg)
    n = b["x"].shape[0]
    alone, both = (model(b["x"], b["keep"], torch.tensor([t] * n), NO_SEG)["binary"][:, 0] for t in ([1], [1, 4, 5]))
    assert torch.allclose(alone, both, atol=1e-5)


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
    for mode in ("uniform", "ema", "uncertainty"):
        total, per_task = Balancer(len(REG), mode)(loss, has, b["task"])
        assert torch.isfinite(total) and (per_task >= 0).all()
    total.backward()
    assert model.task_emb.weight.grad is not None


def test_balancer_weighs_groups_not_task_counts():
    bal = Balancer(len(REG), "uniform", group_of=[REG.groups.index(t.group) for t in REG.tasks])
    many = [REG.index[t.id] for t in REG.trained if t.group == "anatomy_volume"][:10]
    task = torch.tensor([many + [REG.index["age"]]])
    total, _ = bal(torch.tensor([[1.0] * len(many) + [3.0]]), torch.ones_like(task, dtype=torch.bool), task)
    assert abs(float(total) - 2.0) < 1e-5            # (group means 1 and 3) / 2, not (10 + 3) / 11


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
    """Three steps of the slot readout across its encoder-freeze boundary, on phantoms (raises on any failure); the
    task-token trainer has its own test (test_task_tokens)."""
    from brain.train import engine

    engine.run(["multitask", "--synthetic", "--max-steps", "3", "--log-every", "1", "--out-dir", str(tmp_path),
                "--set", "optim.freeze_steps=1", "--set", "model.task_tokens.enabled=false"])


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
