"""Joint SSL + multitask training: curriculum, loss normalisation, encoder options. python tests/run.py tests/test_joint.py"""

from __future__ import annotations

import numpy as np
import torch

from brain.data.gate import token_keep
from brain.models.rope3d import RoPE3D
from brain.train.joint import schedule

J = {"warmup_frac": 0.25, "task_ramp_frac": 0.15, "taper_end_frac": 0.6, "ssl_floor": 0.5}


def test_schedule_is_ssl_only_then_ramps_tasks_and_floors_ssl():
    assert schedule(0, 1000, J) == (1.0, 0.0)
    assert schedule(249, 1000, J) == (1.0, 0.0)
    w_ssl, w_task = schedule(320, 1000, J)
    assert 0.5 < w_ssl < 1.0 and 0.0 < w_task < 1.0
    assert schedule(400, 1000, J)[1] == 1.0
    assert schedule(600, 1000, J) == (0.5, 1.0)
    assert schedule(999, 1000, J) == (0.5, 1.0)
    ssl_only = {**J, "warmup_frac": 1.0}                      # the SSL-only control never asks tasks
    assert all(schedule(s, 1000, ssl_only) == (1.0, 0.0) for s in (0, 500, 999))


def test_schedule_is_monotone():
    w = np.array([schedule(s, 1000, J) for s in range(1000)])
    assert (np.diff(w[:, 0]) <= 1e-9).all() and (np.diff(w[:, 1]) >= -1e-9).all()


def test_rope_min_period_removes_aliasing_and_default_is_unchanged():
    def periods(r):
        f = torch.cat([getattr(r, f"freq_{a}") for a in "xyz"])
        return 2 * np.pi / f.numpy()

    assert periods(RoPE3D(64)).min() < 7.0                    # the original 6.3 mm
    assert periods(RoPE3D(64, min_period_mm=32.0)).min() >= 31.99
    q = torch.randn(1, 2, 5, 64)
    c = torch.rand(1, 5, 3) * 100
    a, _ = RoPE3D(64).rotate(q, q, c)
    b, _ = RoPE3D(64, min_period_mm=None).rotate(q, q, c)
    assert torch.equal(a, b)


def test_gate_max_mode_keeps_rims_center_mode_is_unchanged():
    v = torch.full((1, 1, 32, 32, 16), -6.0)
    v[0, 0, 0:2, :, :] = 1.0                                  # tissue only at the patch edge, not at its centre
    m = torch.tensor([1])
    assert int(token_keep(v, m, (16, 16, 8)).sum()) == 0
    assert int(token_keep(v, m, (16, 16, 8), mode="max").sum()) == 4
    r = torch.randn(1, 1, 32, 32, 16)
    assert torch.equal(token_keep(r, m, (16, 16, 8)), token_keep(r, m, (16, 16, 8), mode="center"))
