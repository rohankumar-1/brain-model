"""Full-token probes: finite on empty volumes, and the shared training path learns a simple signal."""

import numpy as np
import torch

from downstream.heads import AttnEnsemble
from downstream.token_probes import AttnProbe, ConvProbe, GRID_ZYX, make_token_heads


def _data(n, dim=16, seed=0):
    """Positives carry a small shift on channel 0 of every token."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4096, dim)).astype(np.float32)
    y = (np.arange(n) % 2).astype(np.float32)
    x[y == 1, :, 0] += 0.3
    g = np.stack(np.meshgrid(*[np.linspace(0, 1, s) for s in GRID_ZYX], indexing="ij"), -1).reshape(-1, 3)
    return x, y.reshape(-1, 1), np.ones((n, 4096), bool), np.broadcast_to(g, (n, 4096, 3)).astype(np.float32)


def _auc(y, s):
    y, s = np.asarray(y).ravel(), np.asarray(s).ravel()
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def test_default_token_heads_are_tok_conv_only():
    heads = make_token_heads(16, 1, False, 0, False)
    assert [h.name for h in heads] == ["tok_conv"]


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
        head = AttnEnsemble(16, 0, 1, False, 0, n_seeds=1, epochs=40, patience=40, lr=3e-3, batch_size=8, spatial=True,
                            model_factory=build, half=True)
        head.fit(xt, yt, mt, xv, yv, mv, z=zt, z_val=zv)
        auc = _auc(ye, head.predict_scores(xe, me, z=ze))
        assert auc > 0.95, (name, auc)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            print("ok", k)
