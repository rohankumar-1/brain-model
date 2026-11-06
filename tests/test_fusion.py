"""Study-level input (R14-F): sequence fusion inside the encoder; a single scan keeps the unfused path exactly."""
import torch

from brain.models.vit3d import BrainEncoder


def _enc():
    torch.manual_seed(0)
    e = BrainEncoder(size="test", patch_xyz=(8, 8, 4), in_chans=1, qk_norm=True, n_cls=1, proj_dim=16, proj_hidden=32)
    e.enable_fusion(after_block=1)
    return e.eval()


def _inputs(rows):
    torch.manual_seed(1)
    x = torch.randn(rows, 1, 16, 16, 8)
    n = (16 // 8) * (16 // 8) * (8 // 4)
    coords = torch.rand(rows, n, 3) * 100
    pmm = torch.full((rows, 3), 8.0)
    return x, coords, pmm, torch.ones(rows, n, dtype=torch.bool)


def test_single_scan_study_is_unchanged_by_fusion():
    e = _enc()
    with torch.no_grad():                                     # make the residual non-trivial
        e.fusion.o.weight.normal_(std=0.1)
    x, c, p, keep = _inputs(3)                                # rows: anchor 0 (alone), anchor 1, partner of 1
    fuse = {"n_out": 2, "index": torch.tensor([[0, -1], [1, 2]]), "seq": torch.tensor([[2, 0], [3, 5]])}
    with torch.no_grad():
        fused = e(x, c, p, keep=keep, fuse=fuse)
        alone = e(x[:1], c[:1], p[:1], keep=keep[:1])
        solo1 = e(x[1:2], c[1:2], p[1:2], keep=keep[1:2])
    assert fused["tokens"].shape[0] == 2
    assert torch.allclose(fused["tokens"][0], alone["tokens"][0], atol=1e-5)    # no partner: exactly the plain path
    assert not torch.allclose(fused["tokens"][1], solo1["tokens"][0], atol=1e-3)  # a partner changes the anchor


def test_fusion_starts_as_the_mean_of_the_scans():
    e = _enc()
    t = torch.randn(1, 2, 5, e.embed_dim)
    out = e.fusion(t, torch.tensor([[True, True]]), torch.tensor([[2, 4]]))
    assert torch.allclose(out, t.mean(1), atol=1e-6)
