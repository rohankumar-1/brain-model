"""Two-stage convolution stem before the ViT token grid (model.conv_stem, patch_embed.ConvStemEmbed3D)."""
import torch

from brain.models.patch_embed import ConvStemEmbed3D, PatchEmbed3D
from brain.models.vit3d import BrainEncoder


def test_stem_keeps_the_token_grid_and_resizes():
    stem, flat = ConvStemEmbed3D(1, 48, (16, 16, 8), (8, 16)), PatchEmbed3D(1, 48, (16, 16, 8))
    x = torch.randn(2, 1, 64, 64, 32)                          # (B, C, X, Y, Z)
    (a, ga), (b, gb) = stem(x), flat(x)
    assert a.shape == b.shape and ga == gb == (4, 4, 4)
    half, gh = stem(x, (8, 8, 4))                               # the frozen suite's half patches
    assert gh == (8, 8, 8) and half.shape == (2, 512, 48)


def test_stem_trains_and_reloads(tmp_path, monkeypatch):
    """A from-scratch ViT with the stem under the multitask heads, saved and rebuilt for evaluation."""
    from brain.multitask.fewshot import load_model
    from brain.train import engine

    monkeypatch.setenv("BRAIN_PERSIST_SYNTHETIC", "1")
    engine.run(["multitask", "--synthetic", "--max-steps", "2", "--log-every", "1", "--out-dir", str(tmp_path),
                "--set", "model.conv_stem=[8,16]", "--set", "train.eval_every=100", "--set", "train.ckpt_every=2"])
    model, _, cfg = load_model(tmp_path / "last.pt", torch.device("cpu"))
    assert isinstance(model.backbone.patch_embed, ConvStemEmbed3D) and cfg["model"]["conv_stem"] == [8, 16]


def test_stem_adds_little_to_vit_small():
    core = lambda m: sum(p.numel() for n, p in m.named_parameters()  # noqa: E731
                         if not n.startswith(("projector", "token_head", "predictor")))
    plain = BrainEncoder(size="small", proj_dim=16, proj_hidden=32)
    hybrid = BrainEncoder(size="small", proj_dim=16, proj_hidden=32, conv_stem=(32, 64))
    assert 0 < core(hybrid) / core(plain) - 1 < 0.02


def test_shared_decoder_matches_slots_to_task_channels(tmp_path, monkeypatch):
    """model.seg.shared: one decode per volume, each slot reads its task's channel; trains and reloads."""
    from brain.multitask.fewshot import load_model
    from brain.multitask.model import SharedSegDecoder
    from brain.train import engine

    dec = SharedSegDecoder(24, 8, (8, 16, 16), (32, 16, 8, 4), in_chans=1, n_tasks=5)
    grid, img = torch.randn(2, 24, 2, 2, 2), torch.randn(2, 1, 16, 32, 32)
    rows, chan = torch.tensor([1, 0, 1]), torch.tensor([4, 2, 2])
    out = dec(grid, img, rows, torch.zeros(3, 8), chan=chan)
    assert out.shape == (3, 1, 8, 16, 16)
    # the two slots of volume 1 differ only through their channel's 1x1 weights
    alone = dec(grid[1:], img[1:], torch.tensor([0, 0]), torch.zeros(2, 8), chan=torch.tensor([4, 2]))
    assert torch.allclose(out[[0, 2]], alone, atol=1e-5)
    monkeypatch.setenv("BRAIN_PERSIST_SYNTHETIC", "1")
    engine.run(["multitask", "--synthetic", "--max-steps", "2", "--log-every", "1", "--out-dir", str(tmp_path),
                "--set", "model.seg.shared=true", "--set", "tasks.all_seg=true",
                "--set", "train.eval_every=100", "--set", "train.ckpt_every=2"])
    model, _, _ = load_model(tmp_path / "last.pt", torch.device("cpu"))
    assert isinstance(model.seg, SharedSegDecoder)


def test_shared_decoder_v2_per_task_heads():
    from brain.multitask.model import SharedSegDecoder

    dec = SharedSegDecoder(24, 8, (8, 16, 16), (32, 24, 16, 16), in_chans=1, n_tasks=5, head_hidden=4)
    grid, img = torch.randn(2, 24, 2, 2, 2), torch.randn(2, 1, 16, 32, 32)
    out = dec(grid, img, torch.tensor([1, 0, 1]), torch.zeros(3, 8), chan=torch.tensor([4, 2, 2]))
    alone = dec(grid[1:], img[1:], torch.tensor([0, 0]), torch.zeros(2, 8), chan=torch.tensor([4, 2]))
    assert out.shape == (3, 1, 8, 16, 16) and torch.allclose(out[[0, 2]], alone, atol=1e-5)
    out.sum().backward()
    assert dec.hid_w.grad[[2, 4]].abs().sum() > 0 and dec.hid_w.grad[[0, 1, 3]].abs().sum() == 0
