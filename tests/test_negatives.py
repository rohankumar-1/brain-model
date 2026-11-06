"""True negatives for lesion segmentation (labels v11): empty-mask loss, the label cap, Dice / false-positive scoring."""
import numpy as np
import pandas as pd
import torch

from brain.multitask.data import TYPE_ID
from brain.multitask.labels import empty_cap
from brain.multitask.losses import negative_seg_loss, seg_loss, slot_losses
from brain.multitask.test_eval import score


def _maps(fill: float, shape=(2, 1, 4, 8, 8)) -> torch.Tensor:
    return torch.full(shape, fill)


def test_negative_loss_rewards_silence_on_empty_scans():
    empty = torch.zeros(2, 1, 4, 8, 8)
    quiet, blob = _maps(-8.0), _maps(-8.0)
    blob[:, :, 1, 2:4, 2:4] = 8.0                       # a small false-positive blob
    lq, lb = negative_seg_loss(quiet, empty), negative_seg_loss(blob, empty)
    assert (lq < 0.01).all() and (lb > lq + 1.0).all()  # the scan term makes one blob cost far more than its voxels


def test_negative_loss_keeps_dice_on_lesions():
    tgt = torch.zeros(1, 1, 4, 8, 8)
    tgt[:, :, 1:3, 2:6, 2:6] = 1.0
    good = torch.where(tgt > 0, 8.0, -8.0)
    miss = _maps(-8.0, tgt.shape)
    assert negative_seg_loss(good, tgt) < 0.05
    # missing the lesion pays Dice and the scan term on top of seg_loss's BCE
    assert negative_seg_loss(miss, tgt) > seg_loss(miss, tgt)


def test_slot_losses_routes_negative_tasks():
    task = torch.tensor([[0], [1]])
    seg = torch.zeros(2, 1, 8, 16, 16)
    out = {"seg": torch.full((2, 1, 4, 8, 8), 3.0)}      # both slots predict foreground everywhere on empty targets
    pair = torch.tensor([[0, 0], [1, 0]])
    tt = torch.full((2,), TYPE_ID["segmentation"])
    y = torch.full((2, 1), float("nan"))
    base, _ = slot_losses(out, task, y, tt, seg, pair)
    neg, _ = slot_losses(out, task, y, tt, seg, pair, negatives=torch.tensor([True, False]))
    assert neg[0, 0] != base[0, 0] and torch.isclose(neg[1, 0], base[1, 0])


def test_empty_cap_is_stable_and_bounded():
    f = pd.DataFrame({"volume_id": [f"v{i}" for i in range(50)], "value": np.nan, "ref": "@empty",
                      "split": ["train"] * 40 + ["test"] * 10})
    a = empty_cap(f, 7, {"per_positive": 2.0, "eval_n": 3})
    b = empty_cap(f.sample(frac=1.0, random_state=1), 7, {"per_positive": 2.0, "eval_n": 3})
    assert (a.split == "train").sum() == 14 and (a.split == "test").sum() == 3
    assert sorted(a.volume_id) == sorted(b.volume_id)   # the same scans whatever the row order


def test_segmentation_score_ignores_empty_targets():
    y = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    p = np.array([0.6, 0.8, 0.0, 1.0, 0.7])
    assert np.isclose(score("segmentation", y, p), 0.7)
