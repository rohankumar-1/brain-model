"""Family-balanced sampling weights (brain.data.volumes.family_weights)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from brain.data.volumes import family_weights


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
