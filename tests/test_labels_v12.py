"""Labels v12 (R15): post-contrast-only sources, Hounsfield-range CT masks."""
import numpy as np
import pandas as pd

from brain.multitask.labels import refine, src_hu


def _vols():
    return pd.DataFrame({"volume_id": ["a", "b", "c"], "contrast_std": ["t1c", "T1", "flair"],
                         "modality": ["mri", "mri", "mri"], "dataset": ["x", "x", "x"], "case_id": ["1", "2", "3"]})


def test_contrasts_keeps_only_listed_sequences():
    f = pd.DataFrame({"volume_id": ["a", "b", "c"], "value": np.nan, "ref": "m|3"})
    out = refine(f, {"contrasts": ["t1c"]}, _vols())
    assert list(out.volume_id) == ["a"]
    assert len(refine(f, {"contrasts": ["T1", "FLAIR"]}, _vols())) == 2      # case-insensitive


def test_hu_source_writes_range_refs():
    v = pd.DataFrame({"volume_id": ["ct1", "ct2"], "modality": ["ct", "ct"], "brain_ok": [True, False],
                      "brain_method": ["synthstrip", "synthstrip"], "reg_ok": [True, True]})
    out = src_hu(v, None, {"range": [300, 3000]})
    assert set(out["ref"]) == {"@hu:300:3000"} and len(out) == 2
