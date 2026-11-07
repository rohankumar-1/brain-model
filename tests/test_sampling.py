"""Family- and subject-balanced sampling weights (brain.data.volumes.family_weights, subject_balance)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from brain.data.volumes import family_weights, subject_balance


def _mass(fams: pd.Series, w: np.ndarray) -> pd.Series:
    m = pd.Series(w, index=fams.index).groupby(fams).sum()
    return m / m.sum()


def test_ungrouped_matches_n_to_the_alpha():
    fams = pd.Series(["a"] * 100 + ["b"] * 25)
    m = _mass(fams, family_weights(fams, 0.5))
    assert np.isclose(m["a"] / m["b"], (100 / 25) ** 0.5)


def test_group_shares_one_mass_and_splits_by_n_alpha():
    fams = pd.Series(["g1"] * 100 + ["g2"] * 25 + ["c"] * 125)
    m = _mass(fams, family_weights(fams, 0.5, {"g": ["g1", "g2"]}))
    assert np.isclose(m["g1"] + m["g2"], m["c"])                 # group of 125 volumes == family of 125
    assert np.isclose(m["g1"] / m["g2"], (100 / 25) ** 0.5)      # split inside the group by n ** alpha


def test_subject_counts_once_and_family_mass_is_kept():
    fam = pd.Series(["a"] * 5 + ["b"] * 2)
    sub = pd.Series(["s1", "s1", "s1", "s1", "s2", "t1", "t2"])
    w = np.array([0.2] * 5 + [0.5] * 2)
    out = subject_balance(w, fam, sub, 1.0)
    assert np.isclose(out[:4].sum(), out[4]) and np.isclose(out[:5].sum(), 1.0) and np.isclose(out[5:].sum(), 1.0)
    assert np.array_equal(subject_balance(w, fam, sub, 0.0), w)
