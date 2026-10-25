"""Run evaluation: the skill scale and the report's headline arithmetic. python tests/run.py tests/test_eval_report.py"""

from __future__ import annotations

import numpy as np
import pandas as pd

from brain.eval.report import core_tasks, headline
from brain.eval.skill import scale, skill, summarize


def test_skill_is_zero_at_chance_and_one_when_perfect():
    assert skill("auc", 0.5) == 0.0 and skill("auc_ovr_macro", 1.0) == 1.0
    assert skill("bacc", 0.25, n_classes=4) == 0.0 and skill("bacc", 1.0, n_classes=4) == 1.0
    assert skill("r2", -0.4) == 0.0 and skill("r2", 0.7) == 0.7
    assert skill("dice_macro", 0.3) == 0.3
    assert np.isnan(skill("r2", float("nan")))
    assert scale("auc") == 2.0 and abs(scale("bacc", 3) - 1.5) < 1e-12


def test_summarize_averages_group_means_so_big_groups_do_not_dominate():
    s = summarize([("anatomy", 0.9)] * 20 + [("hemorrhage", 0.1), ("hemorrhage", float("nan"))])
    assert abs(s["group_anatomy"] - 0.9) < 1e-12 and s["group_hemorrhage"] == 0.1
    assert abs(s["score"] - 0.5) < 1e-12


def test_headline_weights_groups_equally_and_widens_with_task_noise():
    rows = pd.DataFrame({"group": ["a", "a", "b"], "task": ["x", "y", "z"], "run": [0.6, 0.8, 0.4],
                         "ref": [0.5, 0.7, 0.4], "se": [0.02, 0.02, 0.04]})
    h = headline(rows)
    assert abs(h["run"] - 0.55) < 1e-12 and abs(h["delta"] - 0.05) < 1e-12 and h["n_tasks"] == 3
    se = np.sqrt(2 * 0.25 ** 2 * 0.02 ** 2 + 0.5 ** 2 * 0.04 ** 2)
    assert abs((h["hi"] - h["lo"]) / 2 - 1.96 * se) < 1e-9


def test_core_transfer_set_has_no_duplicates():
    ts = core_tasks(with_seg=True)
    assert len(ts) == len(set(ts)) and "cq500" in ts
