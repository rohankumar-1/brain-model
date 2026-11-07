"""The 3D ViT encoder's numerics and inputs: fp32 RoPE angles, the fp32 residual stream, the metadata embedding."""

from __future__ import annotations

import torch

from brain.models.rope3d import RoPE3D


def test_fp32_rope_is_accurate_under_bf16_and_legacy_is_not():
    c = torch.rand(1, 500, 3) * torch.tensor([256.0, 256.0, 160.0])
    q = torch.randn(1, 2, 500, 64)
    ref, _ = RoPE3D(64).rotate(q, q, c)                    # fp32 everywhere
    fixed, _ = RoPE3D(64, fp32=True).rotate(q.bfloat16(), q.bfloat16(), c)
    legacy, _ = RoPE3D(64).rotate(q.bfloat16(), q.bfloat16(), c)
    err_fixed, err_legacy = (fixed.float() - ref).abs().max(), (legacy.float() - ref).abs().max()
    assert err_fixed < 0.1 < err_legacy


def test_fp32_residual_stream_under_autocast():
    from brain.models.vit3d import BrainEncoder

    seen = []
    for flag in (False, True):
        enc = BrainEncoder(size="test", patch_xyz=(8, 8, 4), qk_norm=True, n_cls=1, proj_dim=16, proj_hidden=32,
                           checkpoint_blocks=False, fp32_residual=flag).eval()
        h = enc.blocks[1].register_forward_pre_hook(lambda m, a: seen.append(a[0].dtype))
        from brain.multitask.explain import encode_backbone

        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            encode_backbone(enc, torch.randn(1, 1, 16, 32, 32), torch.ones(1, 4 * 4 * 4, dtype=torch.bool))
        h.remove()
    assert seen == [torch.bfloat16, torch.float32]


def test_metadata_embedding_starts_neutral_and_can_be_added_to_a_trained_encoder():
    from brain.models.vit3d import BrainEncoder
    from brain.multitask.explain import encode_backbone

    torch.manual_seed(0)
    enc = BrainEncoder(size="test", patch_xyz=(8, 8, 4), qk_norm=True, n_cls=1, proj_dim=16, proj_hidden=32,
                       checkpoint_blocks=False).eval()
    x, keep = torch.randn(2, 1, 16, 32, 32), torch.ones(2, 64, dtype=torch.bool)
    before = encode_backbone(enc, x, keep)["tokens"]
    enc.enable_meta(11)
    coords_out = enc(x.permute(0, 1, 4, 3, 2), *_coords(enc, x), keep=keep, meta=torch.tensor([3, 4]))["tokens"]
    assert torch.allclose(before, coords_out, atol=1e-5)      # zero-initialised: nothing changes
    with torch.no_grad():
        enc.meta_emb.weight[3] += 1.0
    after = enc(x.permute(0, 1, 4, 3, 2), *_coords(enc, x), keep=keep, meta=torch.tensor([3, 4]))["tokens"]
    assert not torch.allclose(after[0], before[0]) and torch.allclose(after[1], before[1], atol=1e-5)


def _coords(enc, x):
    from brain.data.coords import crop_bbox_mm, patch_size_mm, token_centers_mm

    size = (x.shape[4], x.shape[3], x.shape[2])
    bbox = crop_bbox_mm((0, 0, 0), size, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    c, p = token_centers_mm(bbox, size, enc.patch_xyz), patch_size_mm(bbox, size, enc.patch_xyz)
    return c.unsqueeze(0).expand(x.shape[0], -1, -1), p.unsqueeze(0).expand(x.shape[0], -1)
