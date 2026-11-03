"""Token maps (2026-11-01): mask tasks' token occupancy and RSNA slice labels as coarse localisation targets."""
import torch


def test_token_occupancy_and_balanced_loss():
    from brain.multitask.losses import token_occupancy, token_occupancy_loss

    seg = torch.zeros(1, 1, 16, 32, 32)
    seg[0, 0, 9:11, 3:5, 20:22] = 1                                  # inside token (z 1, y 0, x 1) of 8 x 16 x 16
    occ, valid = token_occupancy(seg, (8, 16, 16))
    assert occ.shape == (1, 8) and occ[0].nonzero().flatten().tolist() == [5] and valid.all()
    right = torch.full((1, 8), -6.0)
    right[0, 5] = 6.0
    assert token_occupancy_loss(right, occ, valid) < 0.01
    # all-off is the trap of plain BCE (7 of 8 tokens right): the balanced loss keeps it expensive
    assert token_occupancy_loss(torch.full((1, 8), -6.0), occ, valid) > 2.5
    seg2 = torch.stack([seg[:, 0], torch.zeros_like(seg[:, 0])], 1)  # validity 0 everywhere: nothing counts
    assert not token_occupancy(seg2, (8, 16, 16))[1].any()


def test_slab_mil_loss():
    from brain.multitask.losses import slab_mil_loss

    lab = torch.tensor([[-1, 0, 0, 1, 1, 0, 2, 2]])
    keep = torch.ones_like(lab, dtype=torch.bool)
    good = torch.tensor([[0.0, -6, -6, 6, -6, -6, -6, 6]])           # one hit per run is enough
    assert slab_mil_loss(good, lab, keep) < 0.01
    miss = good.clone()
    miss[0, 7] = -6.0                                                 # run 2 has no firing token
    assert slab_mil_loss(miss, lab, keep) > 1.0
    assert slab_mil_loss(good, torch.full_like(lab, -1), keep) == 0   # unlabelled: no loss


def _slab_batch():
    from brain.multitask.data import collate_slabs

    s = torch.arange(0, 160, 5, dtype=torch.float32)                 # slice positions = stored z (g = (1, 0, 0))
    bits = torch.zeros(len(s), dtype=torch.int16)
    bits[(s >= 50) & (s <= 60)] = 1 | 4                              # any + intraparenchymal in slices 50, 55, 60
    tab = (torch.tensor([1.0, 0, 0]).numpy(), 0.0, 5.0, s.numpy(), bits.numpy())
    return collate_slabs([tab, None])


def test_slab_labels_follow_flip_and_affine():
    from brain.multitask.data import slab_labels

    b = _slab_batch()
    shape, patch = (160, 32, 32), (16, 16, 8)
    lab = slab_labels(b, "cpu", shape, patch, None, None)
    assert lab.shape == (2, 6, 20 * 2 * 2)
    rows = lab[0, 0].view(20, 4)[:, 0]
    assert rows[6:8].tolist() == [1, 1] and (rows[:6] == 0).all() and (rows[8:19] == 0).all()
    assert rows[19] == -1                                  # z 152-159 runs past the last slice (155 + 2.5 mm)
    assert (lab[0, 2] == lab[0, 0]).all() and (lab[0, 1] <= 0).all()          # intraparenchymal yes, epidural no
    assert (lab[1] == -1).all()                                                 # a row without slice labels
    flip = torch.tensor([True, False])
    assert torch.equal(slab_labels(b, "cpu", shape, patch, flip, None), lab)    # left-right: same slices
    theta = torch.zeros(2, 3, 4)
    theta[:, :, :3] = torch.eye(3)
    theta[:, 2, 3] = 2 * 16 / 160                                               # sample 16 voxels higher
    rows = slab_labels(b, "cpu", shape, patch, None, theta)[0, 0].view(20, 4)[:, 0]
    assert rows[4:6].tolist() == [1, 1] and rows[6] == 0


def test_token_map_trains_and_joins_a_resumed_run(tmp_path, monkeypatch):
    from brain.train import engine

    monkeypatch.setenv("BRAIN_PERSIST_SYNTHETIC", "1")
    args = ["multitask", "--synthetic", "--log-every", "1", "--out-dir", str(tmp_path), "--set", "train.eval_every=100",
            "--set", "train.ckpt_every=2"]
    engine.run(args + ["--max-steps", "2"])
    engine.run(args + ["--max-steps", "4", "--reset-schedule", "--set", "model.token_map.enabled=true",
                       "--set", "loss.token_weight=0.5"])
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert last["step"] == 4 and "tok_bias" in last["multitask"]


def test_trainer_adds_slice_losses_to_rsna_slots():
    from brain.multitask.data import SLAB_TASKS, slab_labels
    from brain.multitask.registry import load_registry
    from brain.train.multitask import Trainer

    reg = load_registry(None)
    tr = object.__new__(Trainer)
    from brain.multitask.data import TOKEN_MIL_TASKS

    tr.tok_w, tr.slice_w, tr.box_w, tr.patch = 0.0, 1.0, 0.0, (16, 16, 8)
    tr.slab_sub = torch.full((len(reg),), -1, dtype=torch.long)
    for k, t in enumerate(TOKEN_MIL_TASKS):
        if t in reg.index:
            tr.slab_sub[reg.index[t]] = k
    tr.mil_w = torch.tensor([1.0] * len(SLAB_TASKS) + [0.0] * (len(TOKEN_MIL_TASKS) - len(SLAB_TASKS)))
    tr.aux_sum, tr.aux_cnt = torch.zeros(2), torch.zeros(2)
    b = _slab_batch()
    b["slab"] = slab_labels(b, "cpu", (160, 32, 32), (16, 16, 8), None, None)
    b["task"] = torch.tensor([[reg.index["hemorrhage_any"], reg.index["age"]],
                              [reg.index["hemorrhage_any"], -1]])               # row 1 has no slice labels
    b["seg_pair"] = torch.zeros(0, 2, dtype=torch.long)
    b["keep"] = torch.ones(2, 80, dtype=torch.bool)
    b["y"] = torch.ones(2, 2)
    seg, slab = tr._token_pairs(b)
    assert len(seg) == 0 and slab.tolist() == [[0, 0]]
    tok = torch.full((1, 80), -6.0)
    loss = tr._token_losses(b, tok, torch.zeros(2, 2), 0, slab)
    assert loss[0, 0] > 1.0 and loss[0, 1] == 0 and (loss[1] == 0).all()     # the positive run has no hit
    tok.view(20, 4)[6, 0] = 6.0
    assert tr._token_losses(b, tok, torch.zeros(2, 2), 0, slab)[0, 0] < 0.01


def test_frac_gate_keeps_edge_tokens_but_not_noise():
    from brain.data.gate import token_keep

    vol = torch.full((1, 1, 32, 32, 16), -1000.0)              # CT air, (B, 1, X, Y, Z)
    vol[:, :, :12, :, :] = 40.0                                 # tissue fills 12 of token (0, *, *)'s 16 x-columns ...
    vol[:, :, 20, 20, 4] = 40.0                                 # ... and one stray voxel sits in token x=1
    mod = torch.tensor([0])
    centre = token_keep(vol, mod, (16, 16, 8), mode="center")  # centre voxel (x=8) is tissue; x=24 is air
    frac = token_keep(vol, mod, (16, 16, 8), mode="frac", min_frac=0.25)
    mx = token_keep(vol, mod, (16, 16, 8), mode="max")
    assert frac[0, :, :, 0].all() and not frac[0, :, :, 1].any()   # 75% tissue kept, one voxel is not enough
    assert mx[0, :, :, 1].any()                                     # max mode keeps the stray voxel's token
    vol[:, :, :12, :, :] = -1000.0
    vol[:, :, :6, :, :] = 40.0                                      # 37.5% tissue, centre (x=8) in air
    assert not token_keep(vol, mod, (16, 16, 8), mode="center")[0, :, :, 0].any()
    assert token_keep(vol, mod, (16, 16, 8), mode="frac", min_frac=0.25)[0, :, :, 0].all()


def test_drop_datasets_removes_every_row_of_a_prefix():
    import pandas as pd
    from brain.multitask.data import drop_datasets

    labels = pd.DataFrame({"volume_id": ["a", "a", "b", "c"], "task": ["x", "y", "x", "x"]})
    vols = pd.DataFrame({"volume_id": ["a", "b", "c"], "dataset": ["rsna_ica_cta", "rsna_ich", "ixi_t1"]})
    assert drop_datasets(labels, vols, ["rsna_ica"])["volume_id"].tolist() == ["b", "c"]
    assert len(drop_datasets(labels, vols, [])) == 4


def test_box_labels_map_boxes_through_the_scan_affine():
    from brain.multitask.data import BOX_TASKS, box_labels, collate_boxes

    import numpy as np
    # native (k, j, i) = stored (z, y, x) shifted by (-40, 0, 0): native slices 0..15 cover stored z 40..55
    aff = np.array([[1, 0, 0, -40], [0, 1, 0, 0], [0, 0, 1, 0]], np.float32)
    shape = np.array([16, 32, 32], np.float32)
    boxes = np.array([[0, 3.5, 6.5, 0, 16, 0, 16],             # wm lesion: native k 4-6, j/i 0-15 -> z 44-46
                      [2, 9.5, 10.5, 16, 32, 16, 32]], np.float32)   # mass: native k 10, j/i 16-31 -> z 50
    b = collate_boxes([(aff, shape, boxes), None])
    lab = box_labels(b, "cpu", (160, 32, 32), (16, 16, 8), None, None)
    assert lab.shape == (2, len(BOX_TASKS), 80) and (lab[1] == -1).all()
    wm = lab[0, 0].view(20, 2, 2)
    assert (wm[5, 0, 0] == 1) and (wm[5, 1, 1] == 0)          # token z 40-47, y/x 0-15: in the box; y/x 16-31: not
    assert (wm[0] == -1).all() and (wm[7] == -1).all()        # z 0-7 / 56-63: outside the native field of view
    assert (wm[6] == 0).all()                                 # z 48-55: inside the field of view, no lesion box
    mass = lab[0, 2].view(20, 2, 2)
    assert mass[6, 1, 1] == 1 and mass[6, 0, 0] == 0 and (lab[0, 1][lab[0, 1] >= 0] == 0).all()


def test_box_bags_ignore_outside_tokens_on_positive_scans():
    from brain.multitask.data import TOKEN_MIL_TASKS, SLAB_TASKS
    from brain.train.multitask import Trainer

    tr = object.__new__(Trainer)
    tr.tok_w, tr.patch = 0.0, (16, 16, 8)
    tr.slab_sub = torch.tensor([len(SLAB_TASKS), -1])               # task 0 -> the first box channel
    tr.mil_w = torch.tensor([0.0] * len(SLAB_TASKS) + [1.0] * (len(TOKEN_MIL_TASKS) - len(SLAB_TASKS)))
    tr.aux_sum, tr.aux_cnt = torch.zeros(2), torch.zeros(2)
    slab = torch.full((2, len(TOKEN_MIL_TASKS), 4), -1, dtype=torch.int16)
    slab[:, len(SLAB_TASKS)] = torch.tensor([1, 0, 0, -1], dtype=torch.int16)
    b = {"task": torch.tensor([[0], [0]]), "y": torch.tensor([[1.0], [0.0]]), "slab": slab,
         "keep": torch.ones(2, 4, dtype=torch.bool), "seg_pair": torch.zeros(0, 2, dtype=torch.long)}
    pairs = torch.tensor([[0, 0], [1, 0]])
    tok = torch.tensor([[6.0, 6.0, 6.0, 6.0], [6.0, 6.0, 6.0, 6.0]])   # everything fires
    loss = tr._token_losses(b, tok, torch.zeros(2, 1), 0, pairs)
    assert loss[0, 0] < 0.01        # positive scan: the box fired, tokens outside it are not penalised
    assert loss[1, 0] > 3.0         # negative scan: firing tokens inside the field of view are


def test_box_labels_follow_the_batch_affine():
    import numpy as np
    from brain.multitask.data import box_labels, collate_boxes

    aff = np.array([[1, 0, 0, -40], [0, 1, 0, 0], [0, 0, 1, 0]], np.float32)
    boxes = np.array([[0, 3.5, 6.5, 0, 16, 0, 16]], np.float32)            # stored z 44-46 -> token z 5
    b = collate_boxes([(aff, np.array([16, 32, 32], np.float32), boxes)])
    theta = torch.zeros(1, 3, 4)
    theta[:, :, :3] = torch.eye(3)
    theta[:, 2, 3] = 2 * 16 / 160                                           # output z samples input z + 16
    lab = box_labels(b, "cpu", (160, 32, 32), (16, 16, 8), None, theta)[0, 0].view(20, 2, 2)
    assert lab[3, 0, 0] == 1 and lab[5, 0, 0] != 1                          # the box moved 16 voxels down


def test_remap_tasks_keeps_rows_by_id():
    from brain.train.multitask import remap_tasks

    old = {"task_emb.weight": torch.tensor([[1.0], [2.0], [3.0]]), "log_var": torch.tensor([0.1, 0.2, 0.3]),
           "other": torch.ones(5)}
    tpl = {"task_emb.weight": torch.full((4, 1), 9.0), "log_var": torch.zeros(4), "other": torch.zeros(5)}
    out = remap_tasks(old, ["a", "b", "c"], ["a", "new", "c", "b"], tpl)
    assert out["task_emb.weight"].flatten().tolist() == [1.0, 9.0, 3.0, 2.0]
    assert torch.allclose(out["log_var"], torch.tensor([0.1, 0.0, 0.3, 0.2])) and torch.equal(out["other"], old["other"])


def test_resnet_backbone_trains_and_reloads(tmp_path, monkeypatch):
    """model.arch resnet3d: a from-scratch ResNet under the multitask heads (slot readout), saved and rebuilt for
    evaluation (fewshot.load_model -> embed3d.load_brain3d)."""
    from brain.multitask.fewshot import load_model
    from brain.train import engine

    monkeypatch.setenv("BRAIN_PERSIST_SYNTHETIC", "1")
    engine.run(["multitask", "--synthetic", "--max-steps", "2", "--log-every", "1", "--out-dir", str(tmp_path),
                "--set", "model.arch=resnet3d", "--set", "model.task_tokens.enabled=false",
                "--set", "train.eval_every=100", "--set", "train.ckpt_every=2"])
    model, _, cfg = load_model(tmp_path / "last.pt", torch.device("cpu"))
    assert type(model.backbone).__name__ == "ResNetEncoder" and cfg["model"]["arch"] == "resnet3d"


def test_resnet_matches_vit_small_size():
    from brain.models.resnet3d import ResNetEncoder
    from brain.models.vit3d import BrainEncoder

    vit = BrainEncoder(size="small", proj_dim=16, proj_hidden=32)
    core = sum(p.numel() for n, p in vit.named_parameters() if not n.startswith(("projector", "token_head", "predictor")))
    res = ResNetEncoder()
    assert abs(res.param_count() / core - 1) < 0.02 and res.embed_dim == vit.embed_dim
