"""BrainDINO-recipe MLP probe (brain.downstream.heads.MLPProbe) on a separable pooled embedding."""
import numpy as np

from brain.downstream.heads import MLPProbe
from brain.downstream.metrics import _safe_auc


def _data(n, d=32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1, d)).astype(np.float32)
    w = rng.normal(size=d)
    return x, x[:, 0] @ w, np.ones((n, 1), bool)


def test_binary_multilabel_and_regression_learn_and_cross_fit():
    x, s, m = _data(600)
    tr, va, te = slice(0, 400), slice(400, 500), slice(500, 600)
    half = np.arange(100) % 2
    for kind in ("binary", "multilabel", "regression"):
        if kind == "binary":
            y = (s > 0).astype(np.float32)
        elif kind == "multilabel":
            y = np.stack([s > 0, s > 1], 1).astype(np.float32)
        else:
            y = s.astype(np.float32)
        h = MLPProbe(2 if kind == "multilabel" else 1, kind == "multilabel", seed=0, regression=kind == "regression")
        h.fit(x[tr], y[tr], m[tr], x[va], y[va], m[va], val_half=half)
        p = h.predict_scores(x[te], m[te])
        pv = h.predict_val(x[va], m[va], half)
        assert pv.shape == p.shape[:1] + p.shape[1:] or pv.shape[0] == 100
        if kind == "regression":
            assert np.corrcoef(p, y[te])[0, 1] > 0.8, kind
        else:
            assert _safe_auc(y[te], p) > 0.85, kind
