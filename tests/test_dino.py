"""CPU tests for the BrainDINO-style DINO + iBOT pieces."""

import torch

from models.vit import VisionTransformer
from braindino.encoder import BrainDINO
from braindino.heads import DINOHead, koleo_loss, softmax_center, update_center
from braindino.loss import dino_loss, ibot_loss, sample_masks


def test_dino_head_l2_then_prototypes():
    h = DINOHead(16, 32, hidden=32, bottleneck=8)
    y = h(torch.randn(4, 16))
    assert y.shape == (4, 32) and torch.isfinite(y).all()


def test_identical_student_teacher_gives_finite_dino_loss():
    logits = torch.randn(6, 20)
    center = torch.zeros(20)
    loss = dino_loss([logits, logits], [logits, logits], center, 0.1, 0.04)
    assert torch.isfinite(loss) and float(loss) > 0


def test_ibot_loss_only_uses_masked_tokens():
    s = torch.randn(2, 8, 16)
    t = s.clone()
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[0, 3] = True
    empty = ibot_loss(s, t, torch.zeros_like(mask), torch.zeros(16), 0.1, 0.04)
    one = ibot_loss(s, t, mask, torch.zeros(16), 0.1, 0.04)
    assert float(empty) == 0.0 and torch.isfinite(one)


def test_koleo_is_finite_on_normalized_features():
    x = torch.randn(16, 8)
    assert torch.isfinite(koleo_loss(x))


def test_center_ema_moves_toward_batch_mean():
    center = torch.zeros(4)
    logits = torch.ones(3, 4)
    update_center(center, logits, momentum=0.5)
    assert torch.allclose(center, torch.full((4,), 0.5))


def test_masks_respect_apply_probability_range():
    m = sample_masks(32, 64, torch.device("cpu"), p_apply=1.0, ratio=(0.2, 0.2))
    frac = m.float().mean().item()
    assert 0.05 < frac < 0.4
    z = sample_masks(16, 32, torch.device("cpu"), p_apply=0.0)
    assert int(z.sum()) == 0


def test_backbone_state_loads_into_a_plain_vit():
    vit = VisionTransformer(size="tiny", img_size=32, patch_size=16, proj_dim=16, proj_hidden=32, registers=2)
    net = BrainDINO(vit, proto_dim=32, hidden=32, bottleneck=8)
    net.backbone.projector.requires_grad_(False)
    x = torch.randn(2, 1, 32, 32)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, 0] = True
    cls, patches = net.tokens(x, mask)
    assert cls.shape == (2, vit.embed_dim) and patches.shape == (2, 4, vit.embed_dim)
    dest = VisionTransformer(size="tiny", img_size=32, patch_size=16, proj_dim=16, proj_hidden=32, registers=2)
    dest.load_state_dict(net.backbone_state())
    assert "mask_token" not in net.backbone_state()


def test_one_forward_trains_dino_ibot_and_skips_the_lejepa_projector():
    vit = VisionTransformer(size="tiny", img_size=32, patch_size=16, proj_dim=16, proj_hidden=32, registers=2)
    net = BrainDINO(vit, proto_dim=32, hidden=32, bottleneck=8)
    net.backbone.projector.requires_grad_(False)
    gv = torch.randn(2, 3, 1, 32, 32)
    lv = torch.randn(2, 3, 1, 32, 32)
    masks = torch.zeros(2, 3, 4, dtype=torch.bool)
    masks[0, 0, 1] = True
    t_dino = torch.randn(2, 3, 32)
    t_ibot = torch.randn(2, 3, 4, 32)
    center = torch.zeros(32)
    l_d, l_i, l_k, cls = net(gv, lv, masks, t_dino, t_ibot, center, center, 0.1, 0.04)
    (l_d + l_i + l_k).backward()
    assert cls.shape == (3, vit.embed_dim)
    assert net.dino_head.weight_v.grad is not None and net.ibot_head.weight_v.grad is not None
    assert net.mask_token.grad is not None
    assert net.backbone.projector.net[0].weight.grad is None


def test_softmax_center_is_a_distribution():
    p = softmax_center(torch.randn(5, 7), torch.zeros(7), 0.04)
    assert torch.allclose(p.sum(-1), torch.ones(5), atol=1e-5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
