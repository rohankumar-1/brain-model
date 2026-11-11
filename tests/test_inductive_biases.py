"""Depthwise MLP convolution and bilateral mirror mixing in the encoder (R18 inductive biases)."""
import torch
import torch.nn as nn

from brain.models.vit3d import BrainEncoder, MirrorMix, grid_conv, grid_layout, pack_kept


def _coords(nz, ny, nx, step=(16.0, 16.0, 8.0), origin=(0.0, 0.0, 0.0)):
    """(N, 3) x, y, z mm of token centres in z, y, x order; x spans [origin, origin + nx * step] symmetrically."""
    z, y, x = torch.meshgrid(torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij")
    return torch.stack([origin[0] + (x + 0.5) * step[0], origin[1] + (y + 0.5) * step[1],
                        origin[2] + (z + 0.5) * step[2]], -1).reshape(-1, 3).float()


def _enc(**kw):
    torch.manual_seed(0)
    return BrainEncoder(size="test", patch_xyz=(8, 8, 4), qk_norm=True, checkpoint_blocks=False, **kw).eval()


def _inputs(b=2, g=(2, 4, 4), patch=(8, 8, 4)):
    nz, ny, nx = g
    view = torch.randn(b, 1, nx * patch[0], ny * patch[1], nz * patch[2])
    c = _coords(nz, ny, nx, (float(patch[0]), float(patch[1]), float(patch[2])))
    size = torch.tensor([[float(v) for v in patch]] * b)
    keep = torch.ones(b, nz * ny * nx, dtype=torch.bool)
    keep[1, :3] = False
    return view, c.expand(b, -1, -1).contiguous(), size, keep


def test_partner_is_grid_mirror_and_respects_dropped_tokens():
    nz, ny, nx = 2, 3, 4
    c = _coords(nz, ny, nx).unsqueeze(0)
    mid = nx * 16.0 / 2
    keep = torch.ones(1, nz * ny * nx, dtype=torch.bool)
    keep[0, 0] = False                                    # (z0, y0, x0) dropped: its mirror (x3) has no partner
    _, _, pad, order = pack_kept(torch.zeros(1, nz * ny * nx, 1), c, keep)
    lay = grid_layout((nz, ny, nx), c, order, pad, mid)
    full = order[0].tolist()
    for j, o in enumerate(full):
        if not pad[0, j]:
            assert lay["partner"][0, j] == -1
            continue
        iz, iy, ix = o // (ny * nx), (o // nx) % ny, o % nx
        want = (iz * ny + iy) * nx + (nx - 1 - ix)
        p = int(lay["partner"][0, j])
        if want == 0:
            assert p == -1
        else:
            assert full[p] == want


def test_partner_on_anisotropic_offset_grid():
    """Native CT: thick slices, anisotropic tokens, grid not starting at 0 (only the mirror plane matters)."""
    nz, ny, nx = 3, 2, 6
    c = _coords(nz, ny, nx, step=(14.0, 14.0, 5.0), origin=(86.0, 10.0, 3.0)).unsqueeze(0)
    mid = 86.0 + nx * 14.0 / 2
    keep = torch.ones(1, nz * ny * nx, dtype=torch.bool)
    _, _, pad, order = pack_kept(torch.zeros(1, nz * ny * nx, 1), c, keep)
    lay = grid_layout((nz, ny, nx), c, order, pad, mid)
    for j, o in enumerate(order[0].tolist()):
        ix = o % nx
        assert order[0, lay["partner"][0, j]] == o - ix + (nx - 1 - ix)


def test_grid_conv_matches_dense_convolution():
    nz, ny, nx, ch = 2, 3, 4, 5
    conv = nn.Conv3d(ch, ch, 3, padding=1, groups=ch, bias=False)
    h = torch.randn(1, nz * ny * nx, ch)
    keep = torch.rand(1, nz * ny * nx) > 0.3
    c = _coords(nz, ny, nx).unsqueeze(0)
    packed, _, pad, order = pack_kept(h, c, keep)
    got = grid_conv(conv, packed, grid_layout((nz, ny, nx), c, order, pad))
    dense = conv((h * keep.unsqueeze(-1)).transpose(1, 2).reshape(1, ch, nz, ny, nx)).reshape(1, ch, -1).transpose(1, 2)
    want = dense.gather(1, order.unsqueeze(-1).expand(-1, -1, ch)) * pad.unsqueeze(-1)
    assert torch.allclose(got, want, atol=1e-5)


def test_mirror_mix_is_flip_equivariant():
    nz, ny, nx, d = 2, 2, 4, 8
    mix = MirrorMix(d)
    nn.init.normal_(mix.proj.weight)
    c = _coords(nz, ny, nx).unsqueeze(0)
    keep = torch.ones(1, nz * ny * nx, dtype=torch.bool)
    _, _, pad, order = pack_kept(torch.zeros(1, nz * ny * nx, 1), c, keep)
    partner = grid_layout((nz, ny, nx), c, order, pad, nx * 16.0 / 2)["partner"]
    x = torch.randn(1, nz * ny * nx, d)
    flip = lambda t: t.reshape(1, nz, ny, nx, -1).flip(3).reshape(1, nz * ny * nx, -1)   # noqa: E731
    assert torch.allclose(mix(flip(x), partner), flip(mix(x, partner)), atol=1e-5)


def test_biases_start_as_the_plain_encoder_and_train():
    view, c, size, keep = _inputs()
    plain = _enc()
    biased = _enc(dw_conv_blocks=1, mirror_blocks=(0, 1), mirror_mid_mm=16.0)
    missing, unexpected = biased.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(("mirror.", "blocks.0.dw")) for k in missing)
    with torch.no_grad():
        a = plain(view, c, size, keep=keep)["tokens"]
        b = biased(view, c, size, keep=keep)["tokens"]
    assert torch.allclose(a, b, atol=1e-5)
    biased.train()
    biased(view, c, size, keep=keep)["tokens"].pow(2).mean().backward()
    assert biased.blocks[0].dw.weight.grad.abs().sum() > 0
    assert biased.mirror["0"].proj.weight.grad.abs().sum() > 0
    assert biased.blocks[1].dw is None


def test_task_tokens_and_registers_pass_through():
    view, c, size, keep = _inputs()
    enc = _enc(dw_conv_blocks=2, mirror_blocks=(1,), mirror_mid_mm=16.0, n_registers=4)
    nn.init.normal_(enc.blocks[0].dw.weight)
    nn.init.normal_(enc.mirror["1"].proj.weight, std=0.02)
    tt = torch.randn(3, enc.embed_dim)
    out = enc(view, c, size, keep=keep, task_tokens=tt)
    assert out["task_tokens"].shape == (2, 3, enc.embed_dim) and out["registers"].shape[1] == 4
    assert torch.isfinite(out["tokens"]).all()
    # dropped tokens stay zero
    assert out["tokens"][1, :3].abs().sum() == 0


def test_resume_may_add_the_zero_init_biases_only():
    from brain.train.multitask import is_new_key

    assert is_new_key("backbone.blocks.0.dw.weight") and is_new_key("backbone.blocks.11.dw.weight")
    assert is_new_key("backbone.mirror.5.proj.weight") and is_new_key("backbone.mirror.2.norm.bias")
    assert not is_new_key("backbone.blocks.0.mlp.0.weight") and not is_new_key("backbone.prefix")
