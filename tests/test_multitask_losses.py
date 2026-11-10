"""Multitask losses and scoring: ignored voxels, true negatives (labels v11: empty-mask loss, slot routing, Dice /
false-positive scoring), uncertainty balancing."""
import numpy as np
import torch

from brain.multitask.data import TYPE_ID
from brain.multitask.evaluate import dice
from brain.multitask.losses import Balancer, negative_seg_loss, seg_loss, slot_losses
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


def test_segmentation_score_ignores_empty_targets():
    y = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    p = np.array([0.6, 0.8, 0.0, 1.0, 0.7])
    assert np.isclose(score("segmentation", y, p), 0.7)


def test_one_channel_target_is_unchanged_and_ignored_voxels_do_not_count():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 1, 4, 4, 4, generator=g)
    t = (torch.rand(2, 1, 4, 4, 4, generator=g) > 0.7).float()
    ones = torch.ones_like(t)
    assert torch.allclose(seg_loss(logits, t), seg_loss(logits, torch.cat([t, ones], 1)))
    # a confidently wrong prediction inside an ignored region costs nothing
    bad = logits.clone()
    ign = torch.zeros_like(t, dtype=torch.bool)
    ign[:, :, :2] = True
    bad[ign] = 20.0
    w = (~ign).float()
    t2 = torch.where(ign, torch.zeros_like(t), t)
    clean = torch.where(ign, torch.full_like(logits, -20.0), logits)
    assert torch.allclose(seg_loss(bad, torch.cat([t2, w], 1)), seg_loss(clean, torch.cat([t2, w], 1)), atol=1e-4)
    assert (dice(bad.masked_fill(w < 0.5, -1.0), t2) == dice(clean, t2)).all()


def test_uncertainty_weights_are_clamped_and_the_data_term_stays_positive():
    bal = Balancer(3, clamp=1.0)
    with torch.no_grad():
        bal.log_var.copy_(torch.tensor([-5.0, 0.0, 5.0]))
    w = bal.weights()
    assert float(w.max()) <= torch.e + 1e-4 and float(w.min()) >= 1 / torch.e - 1e-4
    loss = torch.tensor([[0.05, 0.05, 0.05]])
    total, _ = bal(loss, torch.ones_like(loss, dtype=torch.bool), torch.tensor([[0, 1, 2]]))
    assert float(bal.data_term) > 0 and float(total) < float(bal.data_term)    # the +s part can push the total down


def test_balancer_asymmetric_clamp_bounds_the_task_weights():
    """``uncertainty_clamp: [lo, hi]``: s within [-lo, hi], weights exp(-s) within [e^-hi, e^lo]; a scalar is symmetric."""
    import math

    import torch

    from brain.multitask.losses import Balancer

    b = Balancer(3, clamp=[1.0, 2.0])
    with torch.no_grad():
        b.log_var.copy_(torch.tensor([-50.0, 0.0, 50.0]))
    w = b.weights()
    assert math.isclose(float(w[0]), math.e, rel_tol=1e-4) and float(w[1]) == 1.0
    assert math.isclose(float(w[2]), math.exp(-2.0), rel_tol=1e-4)
    s = Balancer(1, clamp=1.0)
    with torch.no_grad():
        s.log_var.fill_(50.0)
    assert math.isclose(float(s.weights()[0]), math.exp(-1.0), rel_tol=1e-4)
