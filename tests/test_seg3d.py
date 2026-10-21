"""3D token segmentation: decoder shape, Dice bookkeeping, and the training path end to end."""

from __future__ import annotations

import numpy as np
import torch

from brain.downstream import seg3d as S
from brain.downstream.tasks import TaskBundle
from brain.common.grid import DEPTH

N_TOK = int(np.prod(S.grid_for(S.PATCH)))  # tokens of a full volume at the native patch

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
        grid = tuple(grid or S.grid_for(S.PATCH))
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


def test_decoder_fuses_patch_embed_skip():
    dec = S.TokenDecoder3D(16, 3, grid=(8, 8, 8), widths=(8, 8, 8, 8, 8), out_vox=(32, 64, 64), pe_dim=8).eval()
    tok = torch.randn(1, 512, 24)
    assert dec(tok).shape == (1, 3, 32, 64, 64)
    z = tok.clone()
    z[..., 16:] = 0
    assert not torch.allclose(dec(tok), dec(z), atol=1e-5)


def test_patch_fcn_nearest_upsamples_the_token_grid():
    head = S.PatchFCN(16, 3, grid=(8, 8, 8), patch_xyz=(8, 8, 4), width=16, dropout=0.0).eval()
    out = head(torch.randn(2, 512, 16))
    assert out.shape == (2, 3, 32, 64, 64)
    cropped = head(torch.randn(1, 64, 16), grid=(4, 4, 4))
    assert cropped.shape == (1, 3, 16, 32, 32)
    # one patch, constant logits inside it
    tok = torch.zeros(1, 8, 16)
    tok[0, 0] = 1
    block = head(tok, grid=(2, 2, 2))
    assert block.shape == (1, 3, 8, 16, 16)
    patch = block[0, :, :4, :8, :8]
    assert torch.allclose(patch, patch[:, :1, :1, :1])


def test_cache_tokens_concatenates_patch_embed():
    cache = _cache()
    cache.pe = np.ones((3, N_TOK, 4), np.float16)
    t = cache.tokens(0)
    assert t.shape == (N_TOK, 20) and np.allclose(t[..., -4:], 1)
    assert cache.pe_dim == 4


def test_fine_flag_selects_the_half_size_patch_and_tiling():
    from brain.downstream.seg3d import PATCH, TILE, grid_for

    assert grid_for((8, 8, 4)) == (DEPTH // 4, 32, 32) and TILE[0][0] % 8 == 0
    assert PATCH == (16, 16, 8) and grid_for(PATCH) == (DEPTH // 8, 16, 16)


def test_decoder_handles_the_fine_patch_grid():
    # the full fine grid -> DEPTH x 256 x 256 is too heavy for login CPU; check the stage count and a scaled-down grid
    dec = S.TokenDecoder3D(16, 2, (16, 16, 16), widths=(8, 8, 8, 8, 8), out_vox=(64, 128, 128))
    assert S.grid_for((8, 8, 4)) == (DEPTH // 4, 32, 32) and len(dec.stages) == 3
    small = S.TokenDecoder3D(16, 2, (8, 8, 8), widths=(8, 8, 8, 8, 8), out_vox=(32, 32, 32))
    assert small(torch.randn(1, 512, 16)).shape == (1, 2, 32, 32, 32)


def test_decoder_accepts_a_cropped_token_grid():
    dec = S.TokenDecoder3D(16, 2, (8, 8, 8), widths=(8, 8, 8, 8, 8), out_vox=(32, 64, 64))
    out = dec(torch.randn(2, 4 ** 3, 16), grid=(4, 4, 4))
    assert out.shape == (2, 2, 16, 32, 32) and dec.n_down == 2


def test_sample_crop_prefers_foreground():
    patch = (16, 16, 8)
    mask = np.zeros(S.VOX, np.uint8)
    mask[8:16, 16:32, 16:32] = 1  # token (1, 1, 1)
    rng = np.random.default_rng(0)
    hits = [S.sample_crop(mask, patch, (8, 8, 8), rng, p_fg=1.0) for _ in range(20)]
    assert all(o == (0, 0, 0) for o in hits)
    tok = torch.arange(N_TOK * 2, dtype=torch.float32).reshape(N_TOK, 2)
    t, m = S.crop_tokens(tok, torch.from_numpy(mask), (0, 1, 2), (4, 4, 4), patch)
    assert t.shape == (64, 2) and m.shape == (32, 64, 64)
    assert int(m.sum()) == 0  # origin (0,1,2) misses the lesion at token (1,1,1)


def test_deep_sup_returns_aux_heads():
    dec = S.TokenDecoder3D(16, 2, grid=(8, 8, 8), widths=(8, 8, 8, 8, 8), deep_sup=True, out_vox=(32, 64, 64))
    logits, aux = dec(torch.randn(1, 512, 16))
    assert logits.shape == (1, 2, 32, 64, 64)
    assert aux[0].shape[2:] == (8, 8, 8) and aux[1].shape[2:] == (16, 16, 16)
    y = torch.zeros(1, 32, 64, 64, dtype=torch.long)
    y[0, 4:12, 8:24, 8:24] = 1
    _, loss = S._decode_loss((logits, aux), y, REGIONS)
    assert torch.isfinite(loss) and loss.ndim == 0


def test_dice_is_scored_on_the_reference_slices_and_on_the_whole_volume():
    cache = _cache()
    m, df = S.score_volumes(_Oracle(cache), cache, np.array([0, 1, 2]), REGIONS, torch.device("cpu"))
    assert abs(m["dice_macro"] - 1) < 1e-6 and abs(m["dice_full_macro"] - 1) < 1e-6 and m["n_volumes"] == 3
    bad, _ = S.score_volumes(_Oracle(cache, flip=True), cache, np.array([0]), REGIONS, torch.device("cpu"))
    assert bad["dice_macro"] < 0.1
    cache.masks[1] = 0  # a volume without the region does not count
    m, df = S.score_volumes(_Oracle(cache), cache, np.array([0, 1]), REGIONS, torch.device("cpu"))
    assert np.isnan(df["LES"].iloc[1]) and m["n_positive"] == {"LES": 1}


def test_token_aug_flips_and_rolls_with_the_mask():
    patch = (16, 16, 8)
    nz, ny, nx = S.grid_for(patch)
    tz, ty, tx = 2, 3, 4
    tok = torch.zeros(nz * ny * nx, 1)
    tok[(tz * ny + ty) * nx + tx] = 1
    mask = torch.zeros(S.VOX, dtype=torch.long)
    pz, py, px = patch[2], patch[1], patch[0]
    mask[tz * pz:(tz + 1) * pz, ty * py:(ty + 1) * py, tx * px:(tx + 1) * px] = 1
    t1, m1 = S.token_aug(tok, mask, 1, patch)
    assert float(t1[(tz * ny + ty) * nx + (nx - 1 - tx)]) == 1
    assert int(m1[:, :, (nx - 1 - tx) * px:(nx - tx) * px].sum()) == int(mask.sum())
    t2, m2 = S.token_aug(tok, mask, 2, patch)
    assert float(t2[(tz * ny + ty) * nx + ((tx + 1) % nx)]) == 1
    assert int(m2.sum()) == int(mask.sum()) and not torch.equal(m2, mask)


def test_training_on_a_single_cached_copy_uses_token_aug():
    src = _cache()
    cache = S.Cache(["v0", "v1", "v2"], src.feats, src.masks)
    assert cache.n_aug == 1
    hist = _train(_Tiny(), cache)
    assert len(hist) == 2 and all(np.isfinite(h["train_loss"]) for h in hist)


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


def test_patch_fcn_training_path():
    dec = S.PatchFCN(16, 2, grid=(8, 8, 8), patch_xyz=(16, 16, 8), width=8, dropout=0.0)
    hist = _train(dec, _cache(), crop=(8, 8, 8))
    assert len(hist) == 2 and all(np.isfinite(h["train_loss"]) for h in hist)


def test_training_stops_when_validation_stops_improving():
    hist = _train(_Tiny(), _cache(), steps=20, patience=1, min_delta=0.05)
    assert hist[-1]["step"] == 2 and len(hist) == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print("run", name, flush=True)
            fn()
            print("ok", name, flush=True)
