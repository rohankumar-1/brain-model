"""Segmentation targets with an ignore region (``path|ids|ignored ids``): left out of the loss and the Dice."""
import torch

from brain.multitask.evaluate import dice
from brain.multitask.losses import seg_loss


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
