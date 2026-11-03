"""Self-supervised tasks on the multitask platform (2026-10-29, phase 1): masked reconstruction and contrast synthesis
as ``dense`` registry tasks (``tasks.ssl``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from brain.multitask.data import (MultitaskDataset, SyntheticMultitask, collate, foreground_tokens, prepare,
                                  token_voxels)
from brain.multitask.losses import slot_losses
from brain.multitask.model import BrainQuery
from brain.multitask.registry import load_registry
from brain.train.defaults import default_config, smoke_config
from brain.train.multitask import build_encoder

SHAPE = (16, 32, 32)
PATCH = (8, 8, 4)
CPU = torch.device("cpu")
REG = load_registry()
SSL_IDS = REG.train_ids({"ssl": True})


def _cfg():
    cfg = smoke_config(default_config("multitask"))
    cfg["model"]["task_tokens"] = {"enabled": True, "share": True}
    cfg["tasks"].update(ssl=True, mask_p=0.5, mask_ratio=0.6)
    return cfg


def test_ssl_tasks_only_train_when_enabled():
    plain, ssl = set(REG.train_ids({})), set(REG.train_ids({"ssl": True}))
    assert not any(REG[t].ssl for t in plain)
    assert ssl - plain == {"masked_recon", "synth_t1", "synth_t1c", "synth_t2", "synth_flair", "synth_ct"}
    assert REG["synth_t2"].type == "dense" and REG["synth_t2"].token == "contrast_synthesis"


def test_token_voxels_follow_the_encoder_token_order():
    """A foreground block in one corner: its gated tokens map back onto exactly that block."""
    raw = torch.full((1, 1, *SHAPE), -10.0)
    raw[0, 0, 8:16, 0:8, 16:32] = 5.0                       # z 8:16, y 0:8, x 16:32
    keep = foreground_tokens(raw, torch.tensor([1]), PATCH)
    vox = token_voxels(keep, SHAPE, PATCH)[0]
    assert vox.shape == SHAPE
    assert bool(vox[8:16, 0:8, 16:32].all()) and int(vox.sum()) == 8 * 8 * 16


def _dataset(rows, mask_p=1.0):
    vols = pd.DataFrame({"volume_id": ["a", "b"], "image_path": ["a.zip", "b.zip"], "modality": ["mri", "mri"],
                         "dataset_family": ["x", "x"], "dataset": ["x_t1", "x_t2"]})
    labels = pd.DataFrame(rows, columns=["volume_id", "task", "value", "ref", "split", "use"])
    return MultitaskDataset("/nonexistent", labels, vols, REG, {}, split="train", tasks=SSL_IDS, mask_p=mask_p)


def test_masked_samples_ask_only_self_supervised_tasks():
    ds = _dataset([("a", "masked_recon", np.nan, "@recon", "train", "train"),
                   ("a", "synth_t2", np.nan, "@synth:b.zip|mri", "train", "train"),
                   ("a", "brain_mask", np.nan, "@brain", "train", "train"),
                   ("a", "age", 40.0, "", "train", "train")])
    rng = np.random.default_rng(0)
    seen_masked = seen_plain = False
    for _ in range(50):
        drawn, masked = ds._draw_sample(ds.items[0][1], rng)
        ids = {REG[t].id for t, _, _ in drawn}
        if masked:
            seen_masked = True
            assert "masked_recon" in ids and ids <= {"masked_recon", "synth_t2"}
        else:
            seen_plain = True
            assert "masked_recon" not in ids
    ds.mask_p = 0.5
    for _ in range(50):
        drawn, masked = ds._draw_sample(ds.items[0][1], rng)
        seen_plain |= not masked
    assert seen_masked and seen_plain
    only = _dataset([("a", "masked_recon", np.nan, "@recon", "train", "train")], mask_p=0.01)
    assert all(only._draw_sample(only.items[0][1], rng)[1] for _ in range(20))     # nothing else to ask: masked


def _batch(seed=0, n=4, tasks=None):
    torch.manual_seed(seed)
    ds = SyntheticMultitask(REG, SHAPE, n=n, k_max=2, seed=seed, tasks=tasks or ["masked_recon", "synth_t2"])
    return collate([ds[i] for i in range(n)])


def test_prepare_blanks_hidden_tokens_and_keeps_the_target():
    """Every foreground token hidden: the input is blank there, and the reconstruction target is the unblanked input."""
    import torch.nn.functional as F

    cfg = _cfg()
    cfg["aug"] = {}
    cfg["tasks"]["mask_ratio"] = 1.0
    raw = _batch()
    raw["masked"][:] = False
    plain = prepare(raw, cfg, CPU, PATCH, train=True)
    raw["masked"][:] = True
    b = prepare(raw, cfg, CPU, PATCH, train=True)
    fg = token_voxels(b["keep"], SHAPE, PATCH)
    assert bool(fg.any()) and bool((b["x"][:, 0][fg] == 0).all())
    assert torch.equal(b["x"][:, 0][~fg], plain["x"][:, 0][~fg])
    rec = raw["dense_src"] < 0
    assert bool(rec.any()) and bool((raw["dense_src"] >= 0).any())
    want = F.avg_pool3d(plain["x"][b["dense_pair"][rec, 0], :1], 2)
    assert torch.allclose(b["dense_tgt"][rec], want)
    assert bool((b["dense_w"][rec] > 0).any())
    syn = ~rec                                            # synthesis: the partner (inverted phantom), not the input
    assert not torch.allclose(b["dense_tgt"][syn], F.avg_pool3d(plain["x"][b["dense_pair"][syn, 0], :1], 2))


def test_forward_and_loss_with_dense_slots():
    cfg = _cfg()
    torch.manual_seed(0)
    model = BrainQuery(build_encoder(cfg, CPU, synthetic=True), REG, cfg["model"])
    model.set_token_tasks([REG.index[t] for t in SSL_IDS])
    raw = _batch(seed=3)
    b = prepare(raw, cfg, CPU, PATCH, train=True)
    assert len(b["dense_pair"]), "the synthetic batch should ask a dense task"
    out = model(b["x"], b["keep"], b["task"], b["seg_pair"], meta=b["meta"], dense_pair=b["dense_pair"])
    assert out["dense"].shape[0] == len(b["dense_pair"])
    loss, has = slot_losses(out, b["task"], b["y"], model.task_type, b["seg"], b["seg_pair"], dense=b)
    dp = b["dense_pair"]
    assert bool(has[dp[:, 0], dp[:, 1]].all()) and bool(torch.isfinite(loss).all())
    loss[has].sum().backward()
    assert model.seg.dense_head.weight.grad is not None


def test_old_checkpoints_load_into_a_model_with_dense_tasks():
    cfg = _cfg()
    old_reg = REG.subset(REG.train_ids({}))
    torch.manual_seed(0)
    old = BrainQuery(build_encoder(cfg, CPU, synthetic=True), old_reg, cfg["model"])
    state = old.state_dict()
    state["type_emb.weight"] = state["type_emb.weight"][:5]          # a checkpoint from before the dense type
    old2 = BrainQuery(build_encoder(cfg, CPU, synthetic=True), old_reg, cfg["model"])
    old2.load_state_dict(state)                                       # strict: no dense head, padded type table
    assert old2.seg.dense_head is None
    assert torch.equal(old2.type_emb.weight[:5], state["type_emb.weight"])


def test_synthesis_pairs_share_a_grid_and_cbct_asks_for_its_ct(tmp_path):
    from brain.multitask import labels

    v = pd.DataFrame({
        "volume_id": ["b_t1", "b_t2", "b_fl", "s_mr", "s_ct", "t_ct", "t_cb", "x_t1", "x_t2"],
        "dedup_key": ["brats:1"] * 3 + ["synthrad:1BA001"] * 2 + ["synthrad:2BA001"] * 2 + ["fomo:9"] * 2,
        "folder": ["brats"] * 3 + ["synthrad", "synthrad_ct", "synthrad_ct", "synthrad_cbct", "fomo", "fomo"],
        "dataset": ["brats_t1n", "brats_t2w", "brats_t2f", "synthrad_t1w", "synthrad_ct", "synthrad_ct",
                    "synthrad_cbct", "fomo_t1w", "fomo_t2w"],
        "contrast_std": ["t1", "t2", "flair", "t1", "ct", "ct", "ct", "t1", "t2"],
        "modality": ["mri"] * 4 + ["ct"] * 3 + ["mri"] * 2,
        "image_path": [f"{i}.zip" for i in range(9)]})
    # BraTS and each SynthRAD patient share one affine; the FOMO session was registered scan by scan
    aff = ["A", "A", "A", "B", "B", "C", "C", "D", "E"]
    pd.DataFrame({"volume_id": v["volume_id"], "affine": aff}).to_parquet(tmp_path / "pairs_affine.parquet")
    task = REG["synth_t2"]
    labels._PAIRS.clear()
    t2 = labels.src_synth(v, task, {"contrast": "t2"}, workers=1, cache_dir=tmp_path)
    assert set(t2["volume_id"]) == {"b_t1", "b_fl"} and set(t2["ref"]) == {"@synth:1.zip|mri"}
    ct = labels.src_synth(v, REG["synth_ct"], {"contrast": "ct"}, workers=1, cache_dir=tmp_path)
    assert dict(zip(ct["volume_id"], ct["ref"])) == {"s_mr": "@synth:4.zip|ct", "t_cb": "@synth:5.zip|ct"}
    t1 = labels.src_synth(v, REG["synth_t1"], {"contrast": "t1"}, workers=1, cache_dir=tmp_path)
    assert set(t1["volume_id"]) == {"b_t2", "b_fl", "s_ct"}          # never x_t2: no shared grid
