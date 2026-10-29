"""Three role-window input channels (windowing.ROLES): aligned channels for CT and MRI, a widened one-channel checkpoint
that computes what it did, the native-unit augmentation path, a pretraining step, and single-pass evaluation."""

from __future__ import annotations

import torch

from brain.common.windowing import CANONICAL, ROLES, asinh_window, role_windows
from brain.models.vit3d import BrainEncoder, widen_patch_embed

PATCH = (8, 8, 4)
SHAPE_ZYX = (16, 32, 32)


def _encoder(in_chans: int) -> BrainEncoder:
    torch.manual_seed(0)
    return BrainEncoder(size="test", patch_xyz=PATCH, in_chans=in_chans, n_cls=1, qk_norm=True, proj_dim=16,
                        proj_hidden=32, checkpoint_blocks=False).eval()


def test_role_channel_one_is_the_canonical_window_for_both_modalities():
    x = torch.randn(2, 1, 4, 5, 6) * 50 + 30
    mod = torch.tensor([0, 1])
    y = role_windows(x, mod)
    assert y.shape == (2, 3, 4, 5, 6)
    for i, m in enumerate(("ct", "mri")):
        assert ROLES[m][0] == CANONICAL[m]
        assert torch.allclose(y[i, 0], asinh_window(x[i, 0], *CANONICAL[m]))
        assert torch.allclose(y[i, 2], asinh_window(x[i, 0], *ROLES[m][2]))
    jit = role_windows(x, mod, jitter=0.1)
    assert jit.shape == y.shape and not torch.allclose(jit, y)


def test_widened_checkpoint_computes_what_the_one_channel_encoder_did():
    one, three = _encoder(1), _encoder(3)
    r = three.load_state_dict(widen_patch_embed(one.state_dict(), 3))
    assert not r.missing_keys and not r.unexpected_keys
    view = torch.randn(1, 1, *SHAPE_ZYX[::-1])                              # (B, C, X, Y, Z)
    n = (SHAPE_ZYX[2] // PATCH[0]) * (SHAPE_ZYX[1] // PATCH[1]) * (SHAPE_ZYX[0] // PATCH[2])
    coords = torch.randn(1, n, 3) * 20
    pmm = torch.tensor([[8.0, 8.0, 4.0]])
    with torch.no_grad():
        a = one(view, coords, pmm)["tokens"]
        b = three(torch.cat([view, torch.randn_like(view), torch.randn_like(view)], 1), coords, pmm)["tokens"]
    assert torch.allclose(a, b, atol=1e-5)                                  # the new channels start at zero


def test_roles_appearance_is_native_units_then_three_channels():
    from brain.data.augment import appearance

    x = torch.randn(4, 1, *SHAPE_ZYX) * 40 + 40
    mod = torch.tensor([0, 0, 1, 1])
    aug = {"window": {"mode": "roles", "jitter": 0.05}, "bias": {"p": 1.0, "amp": 0.2},
           "noise": {"p": 1.0, "kind": "rician", "snr": [20.0, 50.0], "sigma_hu": [2.0, 10.0]},
           "curve": {"p": 1.0}, "blur": {"p": 1.0, "sigma_vox": [0.5, 1.0]}}
    y = appearance(x, mod, aug)
    assert y.shape == (4, 3, *SHAPE_ZYX) and torch.isfinite(y).all()
    plain = appearance(x, mod, {"window": {"mode": "roles"}})
    assert torch.allclose(plain, role_windows(x, mod))                      # no augmentation: just the role windows


def test_the_contrast_curve_never_touches_ct():
    """CT density is physical: in both appearance paths the curve (here always on, always inverting) re-renders only
    the MRI rows."""
    from brain.data.augment import appearance

    x = torch.randn(4, 1, *SHAPE_ZYX) * 40 + 40
    mod = torch.tensor([0, 0, 1, 1])
    curve = {"p": 1.0, "knots": 5, "strength": 0.6, "flip": 1.0}
    one, plain = (appearance(x, mod, {"window": {"mode": "fixed"}, **extra}) for extra in ({"curve": curve}, {}))
    assert torch.equal(one[:2], plain[:2]) and not torch.allclose(one[2:], plain[2:])
    roles, full = appearance(x, mod, {"window": {"mode": "roles"}, "curve": curve}), role_windows(x, mod)
    assert torch.equal(roles[:2], full[:2]) and not torch.allclose(roles[2:], full[2:])


def test_one_channel_control_gets_one_role_window_per_row():
    from brain.data.augment import appearance

    x = torch.randn(12, 1, *SHAPE_ZYX) * 40 + 40
    mod = torch.tensor([0] * 6 + [1] * 6)
    torch.manual_seed(0)
    y, full = appearance(x, mod, {"window": {"mode": "roles", "channels": 1}}), role_windows(x, mod)
    assert y.shape == (12, 1, *SHAPE_ZYX)
    picked = [next(k for k in range(3) if torch.allclose(y[i, 0], full[i, k])) for i in range(12)]
    assert len(set(picked)) > 1                                             # drawn per row, not one window for all


def test_pretraining_step_with_three_channels():
    from brain.data.batch3d import lejepa_batch
    from brain.models.viewnet3d import ViewNet, batch_meta
    from brain.objectives.lejepa3d import SIGReg, lejepa_loss
    from tests.test_lejepa3d import _batch, _cfg

    cfg = _cfg()
    cfg["aug"]["window"] = {"mode": "roles", "jitter": 0.05}
    torch.manual_seed(0)
    enc = BrainEncoder(size="test", patch_xyz=(8, 8, 4), in_chans=3, checkpoint_blocks=False, proj_dim=16,
                       proj_hidden=32, qk_norm=True, proj_norm="batch", dense_dim=16, dense_hidden=32)
    views = lejepa_batch(_batch(), cfg, torch.device("cpu"), gen=torch.Generator().manual_seed(0))
    assert views["global_view"].shape[2] == 3 and views["win_view"].shape[2] == 3
    out = ViewNet(enc, local_pooled=False, dense=True)(views)
    parts = lejepa_loss(out, batch_meta(views), SIGReg(n_proj=16, knots=9), lamb=0.02, win_weight=1.0,
                        dense_weight=1.0, tokens_per_view=3, gather_ddp=False)
    assert all(torch.isfinite(v) for v in parts.values())
    parts["loss"].backward()
    assert float(enc.patch_embed.proj.weight.grad[:, 1:].abs().sum()) > 0   # the new channels learn


def test_frozen_three_channel_encoder_is_one_pass():
    from brain.downstream.embed3d import FrozenEncoder3D

    enc = FrozenEncoder3D(_encoder(3), PATCH, SHAPE_ZYX[::-1], (1.0, 1.0, 1.0))
    assert enc.role_channels
    vol = torch.randn(*SHAPE_ZYX[::-1]) * 40 + 40                            # (X, Y, Z), HU-like
    seqs, xyz, fg = enc.embed_windows(vol, [(0.0, 1.0)], modality="ct")
    assert seqs.shape[0] == 1 and seqs.shape[-1] == enc.dim and torch.isfinite(seqs).all()
