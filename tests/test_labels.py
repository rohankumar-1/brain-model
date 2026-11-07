"""Multitask label building (brain.multitask.labels): source refinement (bins, thresholds, modality, dataset and
contrast filters), Hounsfield-range CT masks, the empty-mask cap for true negatives."""
import numpy as np
import pandas as pd

from brain.multitask.labels import empty_cap, refine, src_hu


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


def test_empty_cap_is_stable_and_bounded():
    f = pd.DataFrame({"volume_id": [f"v{i}" for i in range(50)], "value": np.nan, "ref": "@empty",
                      "split": ["train"] * 40 + ["test"] * 10})
    a = empty_cap(f, 7, {"per_positive": 2.0, "eval_n": 3})
    b = empty_cap(f.sample(frac=1.0, random_state=1), 7, {"per_positive": 2.0, "eval_n": 3})
    assert (a.split == "train").sum() == 14 and (a.split == "test").sum() == 3
    assert sorted(a.volume_id) == sorted(b.volume_id)   # the same scans whatever the row order


def test_label_refinement_bins_thresholds_modality_and_dataset_exclusion():
    v = pd.DataFrame({"volume_id": ["a", "b", "c", "d"], "modality": ["mri", "mri", "ct", "ct"],
                      "dataset": ["oasis1", "oasis2", "rsna_ich", "rsna_ica_cta"]})
    f = pd.DataFrame({"volume_id": ["a", "b", "c", "d"], "value": [14.0, 25.0, 30.0, 27.0], "ref": ""})
    assert refine(f, {"bins": [21, 27]}, v)["value"].tolist() == [0.0, 1.0, 2.0, 2.0]
    assert refine(f, {"threshold": 26}, v)["value"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert refine(f, {"modality": "mri"}, v)["volume_id"].tolist() == ["a", "b"]
    assert refine(f, {"exclude_datasets": ["*_cta"]}, v)["volume_id"].tolist() == ["a", "b", "c"]
