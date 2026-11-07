"""3D token segmentation: decoder shape, Dice bookkeeping, and the training path end to end."""

from __future__ import annotations

import numpy as np
import torch

from brain.downstream import seg3d as S
from brain.downstream.tasks import TaskBundle
from brain.common.grid import DEPTH

N_TOK = int(np.prod(S.GRID))  # tokens of a full volume

REGIONS = {"LES": (1,)}


def _cache(n=3, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, *S.VOX), np.uint8)
    for i in range(n):
        masks[i, 20 + 4 * i:60, 40:120, 50:130] = 1
    feats = rng.normal(size=(n, N_TOK, dim)).astype(np.float16)
    aug = rng.normal(size=(n, 2, N_TOK, dim)).astype(np.float16)  # two extra copies per volume
    return S.Cache([f"v{i}" for i in range(n)], feats, masks, aug, {i: i for i in range(n)})


class _Oracle(torch.nn.Module):
    """Returns the cached mask of whichever volume it was last asked about, as logits."""

    def __init__(self, cache, flip=False):
        super().__init__()
        self.cache, self.flip, self.k = cache, flip, -1

    def forward(self, tok, grid=None):
        outs = []
        for _ in range(tok.shape[0]):
            self.k += 1
            m = torch.from_numpy(self.cache.masks[self.k]).long()
            if self.flip:
                m = 1 - m
            outs.append(torch.nn.functional.one_hot(m, 2).permute(3, 0, 1, 2).float())
        return torch.stack(outs, 0)


class _Tiny(torch.nn.Module):
    """Linear token map, nearest-upsampled. Avoids 3D convs so login-CPU tests stay short."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, tok, grid=None):
        b, n, _ = tok.shape
        grid = tuple(grid or S.GRID)
        pz, py, px = S.PATCH[2], S.PATCH[1], S.PATCH[0]
        x = tok.mean(-1).view(b, 1, *grid)
        x = torch.nn.functional.interpolate(x, size=(grid[0] * pz, grid[1] * py, grid[2] * px), mode="nearest")
        return torch.cat([self.w - x, x], 1)


def _train(dec, cache, **kw):
    task = TaskBundle.__new__(TaskBundle)
    task.regions = REGIONS
    split = lambda s: np.array({"train": [0, 1], "val": [2], "test": []}[s])  # noqa: E731
    real = S.score_volumes
    S.score_volumes = lambda *a, **k: ({"dice_macro": 0.5}, None)  # noqa: ARG005
    try:
        return S.train_decoder(dec, cache, task, split, steps=kw.get("steps", 2), batch_size=1, lr=1e-2,
                               seed=0, eval_every=1, device=torch.device("cpu"),
                               patience=kw.get("patience", 8), min_delta=kw.get("min_delta", 0.001),
                               crop=kw.get("crop", (8, 8, 8)))
    finally:
        S.score_volumes = real


def test_decoder_maps_the_token_grid_to_voxel_logits():
    dec = S.TokenDecoder3D(16, 3, grid=(8, 8, 8), widths=(8, 8, 8, 8, 8), out_vox=(32, 64, 64))
    assert dec(torch.randn(1, 512, 16)).shape == (1, 3, 32, 64, 64)


def test_decoder_stages_follow_the_grid():
    assert S.PATCH == (16, 16, 8) and S.GRID == (DEPTH // 8, 16, 16)
    dec = S.TokenDecoder3D(16, 2, (16, 16, 16), widths=(8, 8, 8, 8, 8), out_vox=(64, 128, 128))
    assert len(dec.stages) == 3
    small = S.TokenDecoder3D(16, 2, (8, 8, 8), widths=(8, 8, 8, 8, 8), out_vox=(32, 32, 32))
    assert small(torch.randn(1, 512, 16)).shape == (1, 2, 32, 32, 32)


def test_decoder_accepts_a_cropped_token_grid():
    dec = S.TokenDecoder3D(16, 2, (8, 8, 8), widths=(8, 8, 8, 8, 8), out_vox=(32, 64, 64))
    out = dec(torch.randn(2, 4 ** 3, 16), grid=(4, 4, 4))
    assert out.shape == (2, 2, 16, 32, 32) and dec.n_down == 2


def test_sample_crop_prefers_foreground():
    mask = np.zeros(S.VOX, np.uint8)
    mask[8:16, 16:32, 16:32] = 1  # token (1, 1, 1)
    rng = np.random.default_rng(0)
    hits = [S.sample_crop(mask, (8, 8, 8), rng, p_fg=1.0) for _ in range(20)]
    assert all(o == (0, 0, 0) for o in hits)
    tok = torch.arange(N_TOK * 2, dtype=torch.float32).reshape(N_TOK, 2)
    t, m = S.crop_tokens(tok, torch.from_numpy(mask), (0, 1, 2), (4, 4, 4))
    assert t.shape == (64, 2) and m.shape == (32, 64, 64)
    assert int(m.sum()) == 0  # origin (0,1,2) misses the lesion at token (1,1,1)


def test_dice_is_scored_on_the_reference_slices_and_on_the_whole_volume():
    cache = _cache()
    m, df = S.score_volumes(_Oracle(cache), cache, np.array([0, 1, 2]), REGIONS, torch.device("cpu"))
    assert abs(m["dice_macro"] - 1) < 1e-6 and abs(m["dice_full_macro"] - 1) < 1e-6 and m["n_volumes"] == 3
    bad, _ = S.score_volumes(_Oracle(cache, flip=True), cache, np.array([0]), REGIONS, torch.device("cpu"))
    assert bad["dice_macro"] < 0.1
    cache.masks[1] = 0  # a volume without the region: an empty prediction scores 1, a painted one 0
    m, df = S.score_volumes(_Oracle(cache), cache, np.array([0, 1]), REGIONS, torch.device("cpu"))
    assert df["LES"].iloc[1] == 1.0 and m["n_positive"] == {"LES": 1}


def test_augmented_masks_move_with_their_volumes():
    vol = torch.zeros(256, 256, 128)  # (X, Y, Z)
    vol[100:120, 30:50, 20:30] = 1.0
    mask = (vol > 0).permute(2, 1, 0).contiguous()  # (Z, Y, X)
    for k in range(len(S.AUG)):
        v, m = S.aug_volume(vol, k), S.aug_mask(mask, k)
        assert torch.equal(m, (v > 0).permute(2, 1, 0)), k  # same object, same place
        assert int(m.sum()) == int(mask.sum())  # nothing leaves the grid for these shifts
    assert not torch.equal(S.aug_mask(mask, 1), mask) and not torch.equal(S.aug_mask(mask, 2), mask)
    assert float(S.shift_fill(torch.ones(4, 4, 4), (1, 0, -1), 7.0).sum()) == 3 * 4 * 3 + 7.0 * (64 - 36)


def test_training_runs_and_restores_the_best_validation_state():
    cache = _cache()
    dec = _Tiny()
    before = {k: v.clone() for k, v in dec.state_dict().items()}
    hist = _train(dec, cache)
    assert len(hist) == 2 and all(np.isfinite(h["train_loss"]) and "val_dice_macro" in h for h in hist)
    assert any(not torch.equal(before[k], v) for k, v in dec.state_dict().items())


def test_training_on_foreground_crops():
    hist = _train(_Tiny(), _cache(), crop=(8, 8, 8))
    assert len(hist) == 2 and all(np.isfinite(h["train_loss"]) for h in hist)


def test_non_finite_loss_skips_the_step():
    """2026-11-05 (AISD): a batch whose loss is not finite is named and skipped, not stepped on."""
    class _NaN(_Tiny):
        def forward(self, tok, grid=None):
            return super().forward(tok, grid) * float("nan")

    dec = _NaN()
    before = {k: v.clone() for k, v in dec.state_dict().items()}
    _train(dec, _cache())
    assert all(torch.equal(before[k], v) for k, v in dec.state_dict().items())


def test_cache_tokens_pick_the_encoded_copy():
    cache = _cache()
    assert cache.n_aug == 3
    assert np.array_equal(cache.tokens(1), cache.feats[1]) and np.array_equal(cache.tokens(1, 2), cache.aug[1, 1])


def test_training_stops_when_validation_stops_improving():
    hist = _train(_Tiny(), _cache(), steps=20, patience=1, min_delta=0.05)
    assert hist[-1]["step"] == 2 and len(hist) == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print("run", name, flush=True)
            fn()
            print("ok", name, flush=True)
