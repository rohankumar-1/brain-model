"""data.subject_beta (R15): a subject's volumes share one subject's mass; family masses are unchanged."""
import numpy as np
import pandas as pd

from brain.data.volumes import subject_balance


def test_subject_counts_once_and_family_mass_is_kept():
    fam = pd.Series(["a"] * 5 + ["b"] * 2)
    sub = pd.Series(["s1", "s1", "s1", "s1", "s2", "t1", "t2"])
    w = np.array([0.2] * 5 + [0.5] * 2)
    out = subject_balance(w, fam, sub, 1.0)
    assert np.isclose(out[:4].sum(), out[4]) and np.isclose(out[:5].sum(), 1.0) and np.isclose(out[5:].sum(), 1.0)
    assert np.array_equal(subject_balance(w, fam, sub, 0.0), w)
