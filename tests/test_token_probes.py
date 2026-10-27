"""Full-token probes: finite on empty volumes, and the shared training path learns a simple signal."""

import numpy as np
import torch

from brain.downstream.heads import AttnEnsemble
from brain.downstream.token_probes import AttnProbe, ConvProbe, GRID_ZYX, make_token_heads
from brain.common.grid import FINE_TOKEN_GRID, TOKEN_GRID

N_TOK = TOKEN_GRID[0] * TOKEN_GRID[1] * TOKEN_GRID[2]
N_FINE = FINE_TOKEN_GRID[0] * FINE_TOKEN_GRID[1] * FINE_TOKEN_GRID[2]


def _data(n, dim=16, seed=0):
    """Positives carry a small shift on channel 0 of every token."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, N_TOK, dim)).astype(np.float32)
    y = (np.arange(n) % 2).astype(np.float32)
    x[y == 1, :, 0] += 0.3
    g = np.stack(np.meshgrid(*[np.linspace(0, 1, s) for s in GRID_ZYX], indexing="ij"), -1).reshape(-1, 3)
    return x, y.reshape(-1, 1), np.ones((n, N_TOK), bool), np.broadcast_to(g, (n, N_TOK, 3)).astype(np.float32)


def _auc(y, s):
    y, s = np.asarray(y).ravel(), np.asarray(s).ravel()
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def test_default_token_heads_are_tok_conv_only():
    heads = make_token_heads(16, 1, False, 0, False)
    assert [h.name for h in heads] == ["tok_conv"]


def test_attn_ensemble_trains_a_large_grid_from_host_memory():
    """32^3-scale token counts must not be copied onto the GPU as one tensor."""
    n, t, d = 4, 8193, 4

    class _Pool(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(d, 1)

        def forward(self, x, mask, xyz):
            return self.lin((x * mask.unsqueeze(-1).to(x.dtype)).mean(1))

    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, t, d)).astype(np.float16)
    y = (np.arange(n) % 2).astype(np.float32).reshape(-1, 1)
    m = np.ones((n, t), bool)
    z = np.zeros((n, t, 3), np.float32)
    head = AttnEnsemble(d, 1, False, 0, n_seeds=1, epochs=2, patience=2, batch_size=2, half=True,
                        model_factory=_Pool)
    head.fit(x, y, m, x, y, m, z=z, z_val=z)
    scores = head.predict_scores(x, m, z=z)
    assert scores.shape == (n,) and np.isfinite(scores).all()


def test_conv_probe_accepts_the_fine_grid():
    x = torch.zeros(2, N_FINE, 4)
    m = torch.ones(2, N_FINE, dtype=torch.bool)
    z = torch.zeros(2, N_FINE, 3)
    out = ConvProbe(4, 1, grid_zyx=FINE_TOKEN_GRID, width=8)(x, m, z)
    assert out.shape == (2, 1) and bool(torch.isfinite(out).all())
    from brain.downstream.token_readout import token_grid_zyx
    assert token_grid_zyx(N_TOK) == TOKEN_GRID
    assert token_grid_zyx(N_FINE) == FINE_TOKEN_GRID and token_grid_zyx(4096) is None
    assert token_grid_zyx(100) is None


def test_probes_shapes_and_masked_rows():
    x, _, m, z = (torch.from_numpy(a) for a in _data(4))
    m[0] = False  # a volume with no foreground token must not produce NaN
    for net in (AttnProbe(16, 3), ConvProbe(16, 3)):
        out = net(x, m, z)
        assert out.shape == (4, 3) and bool(torch.isfinite(out).all())


def test_probes_learn_a_simple_signal_through_the_shared_training_path():
    xt, yt, mt, zt = _data(48, seed=0)
    xv, yv, mv, zv = _data(16, seed=1)
    xe, ye, me, ze = _data(32, seed=2)
    for name, build in (("attn", lambda: AttnProbe(16, 1, width=32, heads=2, queries=2)),
                        ("conv", lambda: ConvProbe(16, 1, width=16))):
        head = AttnEnsemble(16, 1, False, 0, n_seeds=1, epochs=40, patience=40, lr=3e-3, batch_size=8,
                            model_factory=build, half=True)
        head.fit(xt, yt, mt, xv, yv, mv, z=zt, z_val=zv)
        auc = _auc(ye, head.predict_scores(xe, me, z=ze))
        assert auc > 0.95, (name, auc)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            print("ok", k)


def test_cross_fitted_validation_scores_each_half_with_seeds_blind_to_it():
    """With ``val_half``, seed i early-stops on half i % 2, and a validation volume is scored only by the seeds that
    stopped on the other half; test scores still average every seed."""
    x, y, m, z = _data(24)
    xv, yv, mv, zv = _data(8, seed=1)
    half = np.array([0, 1] * 4)
    h = AttnEnsemble(16, 1, False, 0, n_seeds=3, epochs=2, patience=1, model_factory=lambda: ConvProbe(16, 1))
    h.fit(x, y, m, xv, yv, mv, z=z, z_val=zv, val_half=half)
    assert h.stop_half_ == [0, 1, 0]
    lg = h._all_logits(xv, mv, zv)                              # (3, 8, 1)
    want = torch.where(torch.as_tensor(half == 0)[:, None], lg[1], (lg[0] + lg[2]) / 2).sigmoid().numpy().ravel()
    assert np.allclose(h.predict_val(xv, mv, half, z=zv), want, atol=1e-6)
    assert np.allclose(h.predict_scores(xv, mv, z=zv), lg.mean(0).sigmoid().numpy().ravel(), atol=1e-6)
