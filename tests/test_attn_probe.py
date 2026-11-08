"""Attentive-probe readout (brain.downstream.heads.AttentiveProbe, head ``attn_probe``) on synthetic token arrays, and
its status as an extra readout: never pinned, never a fallback, never in a headline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from brain.downstream.heads import AttentiveProbe, make_attn_probe
from brain.downstream.metrics import ATTN_PROBE, EXTRA_READOUTS, _safe_auc

REPO = Path(__file__).resolve().parents[1]


def _tokens(n: int, t: int = 48, d: int = 16, seed: int = 0):
    """Tokens of n scans on a line of t positions: noise everywhere, a signal direction added to the tokens of one
    region when the label is 1; the last quarter of every scan is gated out (air) and carries a decoy that would give
    the label away if the mask were ignored."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, t, d)).astype(np.float32)
    y = (rng.random(n) < 0.5).astype(np.float32)
    sig = rng.normal(size=d).astype(np.float32)
    sig /= np.linalg.norm(sig)
    region = slice(t // 4, t // 4 + 6)
    x[:, region] += 2.5 * y[:, None, None] * sig
    mask = np.ones((n, t), dtype=bool)
    mask[:, 3 * t // 4:] = False
    x[:, 3 * t // 4:] += 5.0 * (1 - y)[:, None, None] * sig        # decoy in masked tokens
    z = np.stack(np.meshgrid(np.linspace(0, 1, t), [0.5], [0.5], indexing="ij"), -1).reshape(t, 3)
    return x, y, mask, np.broadcast_to(z, (n, t, 3)).astype(np.float32).copy()


def test_masked_tokens_do_not_change_the_output():
    torch.manual_seed(0)
    probe = AttentiveProbe(16, 3, n_queries=2, width=32, heads=4).eval()
    x, _, mask, z = (torch.from_numpy(a) for a in _tokens(4))
    mask_t = torch.from_numpy(np.asarray(mask))
    out, w = probe(x, mask_t, z, return_attn=True)
    assert out.shape == (4, 3) and w.shape == (4, 2, 48)
    assert torch.all(w[..., 36:] == 0) and torch.allclose(w.sum(-1), torch.ones(4, 2))
    x2 = x.clone()
    x2[:, 36:] = torch.randn_like(x2[:, 36:]) * 100
    assert torch.allclose(probe(x2, mask_t, z), out, atol=1e-5)
    none = torch.zeros_like(mask_t)                                  # a scan with no kept token still scores
    assert torch.isfinite(probe(x, none, z)).all()


def test_binary_probe_learns_and_validation_is_cross_fitted():
    x, y, m, z = _tokens(240, seed=1)
    tr, va, te = slice(0, 140), slice(140, 180), slice(180, 240)
    head = make_attn_probe(16, 1, multilabel=False, seed=0, regression=False, n_seeds=2)
    head.epochs, head.patience = 40, 10
    half = np.arange(40) % 2
    head.fit(x[tr], y[tr], m[tr], x[va], y[va], m[va], z=z[tr], z_val=z[va], val_half=half)
    assert head.name == ATTN_PROBE and head.stop_half_ == [0, 1]
    s = head.predict_scores(x[te], m[te], z=z[te])
    assert s.shape == (60,) and _safe_auc(y[te], s) > 0.9
    v = head.predict_val(x[va], m[va], half, z=z[va])
    assert v.shape == (40,) and np.isfinite(v).all()


def test_regression_scores_come_back_in_target_units_and_multilabel_shapes():
    x, y, m, z = _tokens(120, seed=2)
    target = 50.0 + 20.0 * y + np.random.default_rng(0).normal(size=len(y)).astype(np.float32)
    head = make_attn_probe(16, 1, multilabel=False, seed=0, regression=True, n_seeds=1)
    head.epochs, head.patience = 30, 10
    head.fit(x[:80], target[:80], m[:80], x[80:100], target[80:100], m[80:100], z=z[:80], z_val=z[80:100])
    s = head.predict_scores(x[100:], m[100:], z=z[100:])
    assert s.shape == (20,) and 40 < s.mean() < 80                   # z-scored on train, mapped back
    yl = np.stack([y, 1 - y, y], 1)
    ml = make_attn_probe(16, 3, multilabel=True, seed=0, regression=False, n_seeds=1)
    ml.epochs = 3
    ml.fit(x[:80], yl[:80], m[:80], x[80:100], yl[80:100], m[80:100], z=z[:80], z_val=z[80:100])
    assert ml.predict_scores(x[100:], m[100:], z=z[100:]).shape == (20, 3)


def test_extra_readouts_are_never_pinned_fallback_or_headline():
    from brain.eval.report import fixed_heads, mark_readouts

    assert ATTN_PROBE in EXTRA_READOUTS
    rows = []
    for mid in ("m1", "m2"):
        rows += [{"model_id": mid, "task": "t", "head": "linear", "val": 0.6},
                 {"model_id": mid, "task": "t", "head": ATTN_PROBE, "val": 0.9}]   # best on val, yet never chosen
    t = mark_readouts(pd.DataFrame(rows))
    assert list(t.extra) == [False, True, False, True] and not t[t.extra].encoder.any()
    assert fixed_heads(t) == {"t": "linear"}
    # a model lacking the pinned head falls back to its best encoder readout on val: never the extra readout
    from brain.bench.board import _pick

    only = mark_readouts(pd.DataFrame([{"model_id": "m3", "task": "t", "head": ATTN_PROBE, "val": 0.9},
                                       {"model_id": "m3", "task": "t", "head": "mlp", "val": 0.5}]))
    assert _pick(only, "m3", "t", "tok_conv")["head"] == "mlp"
    assert _pick(only[only["head"] == ATTN_PROBE], "m3", "t", "tok_conv") is None
    # the pinned readouts do not list it
    pinned = yaml.safe_load((REPO / "configs" / "bench" / "readouts_v3.yaml").read_text())["readouts"]
    assert ATTN_PROBE not in set(pinned.values())


def test_run_suite_mode_needs_a_token_sequence():
    from brain.downstream.run_suite import attn_mode

    emb = {"mus": np.zeros((2, 8, 4)), "z": np.zeros((2, 8, 3))}
    assert attn_mode("off", emb) is None and attn_mode("on", emb) == "on" and attn_mode("only", emb) == "only"
    flat = {"mus": np.zeros((2, 1, 4))}
    assert attn_mode("on", flat) is None
    with pytest.raises(SystemExit):
        attn_mode("only", flat)
