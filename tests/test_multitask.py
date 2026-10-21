"""Task-conditioned multitask model: registry, task sampling, batch preparation, model outputs, losses, trainer."""

from __future__ import annotations

import numpy as np
import torch

from brain.multitask.data import TYPE_ID, MultitaskDataset, SyntheticMultitask, collate, prepare
from brain.multitask.labels import live_targets
from brain.multitask.losses import Balancer, seg_loss, slot_losses
from brain.multitask.model import MultitaskModel
from brain.multitask.registry import load_registry
from brain.train.defaults import default_config, smoke_config
from brain.train.multitask import build_encoder

SHAPE = (16, 32, 32)          # (Z, Y, X): the smoke grid


def _setup():
    torch.manual_seed(0)
    cfg = smoke_config(default_config("multitask"))
    reg = load_registry()
    enc = build_encoder(cfg, torch.device("cpu"), synthetic=True)
    model = MultitaskModel(enc, reg, cfg["model"]).eval()
    return cfg, reg, model


def _batch(reg, n=4, seed=0):
    ds = SyntheticMultitask(reg, SHAPE, n=n, seed=seed)
    return collate([ds[i] for i in range(n)])


def test_registry_tasks_are_complete():
    reg = load_registry()
    assert reg.trained and len(reg.trained) < len(reg)
    for t in reg.tasks:
        assert t.sources, t.id
        assert t.type != "multiclass" or len(t.classes) >= 2
        assert t.analog is None or reg[t.analog].trained, t.id


def test_collate_pads_slots_and_pairs_segmentation_masks():
    reg = load_registry()
    b = _batch(reg)
    k = (b["task"] >= 0).sum(1)
    assert b["task"].shape[1] == int(k.max())
    assert torch.isnan(b["y"][b["task"] < 0]).all()
    for i, j in b["seg_pair"].tolist():
        assert reg[int(b["task"][i, j])].type == "segmentation"
    assert len(b["seg"]) == len(b["seg_pair"])


def test_forward_gives_every_head_and_one_mask_per_seg_slot():
    cfg, reg, model = _setup()
    raw = _batch(reg)
    b = prepare(raw, cfg, torch.device("cpu"), model.backbone.patch_xyz, train=False)
    out = model(b["x"], b["keep"], b["task"], b["seg_pair"])
    bk = tuple(b["task"].shape)
    assert out["binary"].shape == bk and out["regression"].shape == bk
    assert out["multiclass"].shape == (*bk, reg.max_classes())
    if len(b["seg_pair"]):
        assert out["seg"].shape == (len(b["seg_pair"]), 1, SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2)


def test_slots_are_permutation_equivariant():
    """Slots carry no order: permuting the requested tasks permutes the outputs and changes nothing else."""
    cfg, reg, model = _setup()
    b = prepare(_batch(reg), cfg, torch.device("cpu"), model.backbone.patch_xyz, train=False)
    task = torch.tensor([[0, 1, 4, 5]] * b["x"].shape[0])
    perm = torch.tensor([2, 0, 3, 1])
    none = torch.zeros(0, 2, dtype=torch.long)
    with torch.no_grad():
        a = model(b["x"], b["keep"], task, none)
        p = model(b["x"], b["keep"], task[:, perm], none)
    assert torch.allclose(a["binary"][:, perm], p["binary"], atol=1e-5)
    assert torch.allclose(a["regression"][:, perm], p["regression"], atol=1e-5)


def test_cross_task_off_makes_slots_independent():
    cfg, reg, _ = _setup()
    cfg["model"]["slot"]["cross_task"] = False
    enc = build_encoder(cfg, torch.device("cpu"), synthetic=True)
    model = MultitaskModel(enc, reg, cfg["model"]).eval()
    b = prepare(_batch(reg), cfg, torch.device("cpu"), model.backbone.patch_xyz, train=False)
    none = torch.zeros(0, 2, dtype=torch.long)
    n = b["x"].shape[0]
    with torch.no_grad():
        alone = model(b["x"], b["keep"], torch.tensor([[1]] * n), none)["binary"][:, 0]
        both = model(b["x"], b["keep"], torch.tensor([[1, 4, 5]] * n), none)["binary"][:, 0]
    assert torch.allclose(alone, both, atol=1e-5)


def test_multiclass_logits_cover_only_the_tasks_classes():
    cfg, reg, model = _setup()
    b = prepare(_batch(reg), cfg, torch.device("cpu"), model.backbone.patch_xyz, train=False)
    t = reg.index["neuro_diagnosis"]
    n = b["x"].shape[0]
    with torch.no_grad():
        lg = model(b["x"], b["keep"], torch.full((n, 1), t), torch.zeros(0, 2, dtype=torch.long))["multiclass"]
    k = len(reg[t].classes)
    assert torch.isfinite(lg[..., :k]).all() and torch.isinf(lg[..., k:]).all()


def test_losses_are_masked_per_slot_and_balanced_per_task():
    cfg, reg, model = _setup()
    model.train()
    b = prepare(_batch(reg, n=6), cfg, torch.device("cpu"), model.backbone.patch_xyz, train=True)
    out = model(b["x"], b["keep"], b["task"], b["seg_pair"])
    loss, has = slot_losses(out, b["task"], b["y"], model.task_type, b["seg"], b["seg_pair"])
    assert not has[b["task"] < 0].any()
    assert (loss[~has] == 0).all()
    for mode in ("uniform", "ema", "uncertainty"):
        total, per_task = Balancer(len(reg), mode)(loss, has, b["task"])
        assert torch.isfinite(total) and (per_task >= 0).all()
    total.backward()
    assert model.task_emb.weight.grad is not None


def test_seg_loss_prefers_the_right_mask():
    t = torch.zeros(1, 1, 8, 8, 8)
    t[..., 2:6, 2:6, 2:6] = 1
    good, bad = (t * 20 - 10), -(t * 20 - 10)
    assert seg_loss(good, t) < 0.1 < seg_loss(bad, t)


def test_flip_moves_image_and_mask_together():
    reg = load_registry()
    cfg = smoke_config(default_config("multitask"))
    cfg["aug"] = {"flip_lr": 1.0, "affine": {"p": 0.0}}
    raw = _batch(reg, n=4)
    if not len(raw["seg_pair"]):
        return
    b = prepare(raw, cfg, torch.device("cpu"), (8, 8, 4), train=True)            # flip_lr 1: every row flips
    plain = prepare(raw, cfg, torch.device("cpu"), (8, 8, 4), train=False)
    assert torch.equal(b["seg"], plain["seg"].flip(-1))
    assert torch.allclose(b["x"], plain["x"].flip(-1))


def test_rotation_keeps_mask_mass():
    reg = load_registry()
    cfg = smoke_config(default_config("multitask"))
    cfg["aug"] = {"flip_lr": 0.0, "affine": {"p": 1.0, "rot_deg": [10, 10, 10], "shift_mm": 0.0}}
    raw = _batch(reg, n=4)
    if not len(raw["seg_pair"]):
        return
    b = prepare(raw, cfg, torch.device("cpu"), (8, 8, 4), train=True)
    before = raw["seg"].float().sum(dim=(1, 2, 3))
    after = b["seg"].sum(dim=(1, 2, 3, 4))
    assert torch.allclose(before, after, rtol=0.1)


def test_live_targets_measure_left_right_asymmetry():
    z, y, x = 32, 64, 64
    zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, s) for s in (z, y, x)], indexing="ij")
    brain = (zz**2 + yy**2 + xx**2) < 0.7
    img = np.where(brain, 30.0, -1000.0).astype(np.float32)
    sym = live_targets(img, brain, "ct")
    assert sym["mask_asym"] < 1e-6 and sym["intensity_asym"] < 1e-6
    img2 = img.copy()
    img2[(xx > 0.2) & (xx < 0.6) & brain] = 70.0                     # a right-sided hyperdensity
    les = live_targets(img2, brain, "ct")
    assert les["intensity_asym"] > 0.05 and les["mask_asym"] < 1e-6
    assert abs(sym["ct_brain_hu"] - 30.0) < 1e-3 and sym["ct_csf_frac"] == 0.0
    assert np.isnan(live_targets(img, brain, "mri")["ct_csf_frac"])


def test_dataset_draws_subsets_of_available_tasks(tmp_path):
    import pandas as pd

    reg = load_registry()
    bin_t, reg_t, seg_t = "sex", "age", "brain_mask"
    labels = pd.DataFrame({
        "volume_id": ["v1"] * 3 + ["v2"],
        "task": [bin_t, reg_t, seg_t, bin_t],
        "value": [1.0, 40.0, np.nan, 0.0],
        "ref": ["", "", "@brain", ""],
        "split": ["train"] * 4, "use": ["train"] * 4,
    })
    vols = pd.DataFrame({"volume_id": ["v1", "v2"], "image_path": ["a.zip", "b.zip"], "modality": ["mri", "ct"],
                         "dataset_family": ["f1", "f2"]})
    ds = MultitaskDataset(tmp_path, labels, vols, reg, {"age": {"mean": 30.0, "std": 10.0}}, split="train",
                          k_max=8, p_all=0.0, max_seg=0)
    assert len(ds) == 2
    avail = ds.items[0][1]
    assert {reg[t].id for t, _, _ in avail} == {bin_t, reg_t, seg_t}
    assert dict((reg[t].id, y) for t, y, _ in avail)[reg_t] == 1.0          # standardised
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = ds._draw(avail, rng)
        assert 1 <= len(d) <= 3 and all(reg[t].type != "segmentation" for t, _, _ in d)
    ev = MultitaskDataset(tmp_path, labels, vols, reg, {}, split="train", train=False, max_seg=1)
    assert len(ev._draw(ev.items[0][1], rng)) == 3


def test_trainer_runs_on_phantoms(tmp_path):
    from brain.train import engine

    engine.run(["multitask", "--synthetic", "--max-steps", "3", "--log-every", "1", "--out-dir", str(tmp_path),
                "--set", "optim.freeze_steps=1"])
    assert TYPE_ID["segmentation"] == 3


def test_fewshot_fits_only_the_held_out_row():
    from brain.multitask.fewshot import fit_embedding
    from brain.multitask.data import SyntheticMultitask

    cfg, reg, model = _setup()
    model.requires_grad_(False)
    t = next(i for i, task in enumerate(reg.tasks) if not task.trained and task.type == "binary")
    ds = SyntheticMultitask(reg, SHAPE, n=4)
    items = []
    for i in range(4):
        it = ds[i]
        it.update(task=torch.tensor([t]), y=torch.tensor([float(i % 2)]), seg_slot=torch.tensor([-1]),
                  seg=torch.zeros(0, *SHAPE, dtype=torch.uint8))
        items.append(it)
    before = model.task_emb.weight.detach().clone()
    amp = dict(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    saved = fit_embedding(model, reg, cfg, [collate(items[:2]), collate(items[2:])], t, torch.device("cpu"), amp,
                          steps=5, lr=1e-2)
    after = model.task_emb.weight.detach()
    others = torch.arange(len(reg)) != t
    assert torch.equal(before[others], after[others])
    assert not torch.equal(before[t], after[t]) and torch.equal(saved["task"], before[t])
    assert not model.task_emb.weight.requires_grad


def test_group_sampling_gives_each_group_its_share():
    """30 tasks of one group and 2 of another, all available: draws should split ~evenly between the groups."""
    import pandas as pd

    reg = load_registry()
    many = [t.id for t in reg.trained if t.group == "anatomy_volume"]
    few = [t.id for t in reg.trained if t.group == "demographics"]
    ids = many + few
    labels = pd.DataFrame({"volume_id": "v1", "task": ids, "value": 1.0, "ref": "", "split": "train", "use": "train"})
    vols = pd.DataFrame({"volume_id": ["v1"], "image_path": ["a.zip"], "modality": ["mri"], "dataset_family": ["f"]})
    ds = MultitaskDataset(".", labels, vols, reg, {}, split="train", k_max=1, p_all=0.0)
    rng = np.random.default_rng(0)
    groups = [reg[ds._draw(ds.items[0][1], rng)[0][0]].group for _ in range(2000)]
    share = groups.count("demographics") / len(groups)
    assert 0.45 < share < 0.55, share


def test_balancer_weighs_groups_not_task_counts():
    reg = load_registry()
    groups = reg.groups
    group_of = [groups.index(t.group) for t in reg.tasks]
    bal = Balancer(len(reg), "uniform", group_of=group_of)
    many = [reg.index[t.id] for t in reg.trained if t.group == "anatomy_volume"][:10]
    one = reg.index["age"]
    task = torch.tensor([many + [one]])
    loss = torch.tensor([[1.0] * len(many) + [3.0]])
    total, _ = bal(loss, torch.ones_like(task, dtype=torch.bool), task)
    assert abs(float(total) - 2.0) < 1e-5            # (mean of group 1 = 1, group 2 = 3) / 2, not (10 + 3) / 11
