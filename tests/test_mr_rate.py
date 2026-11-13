"""MR-RATE: series choice (post-contrast rule), study labels onto existing tasks (soft negatives), index keys and
official splits. Synthetic frames only (no MR-RATE records in tests)."""
import numpy as np
import pandas as pd
import pytest

from brain.multitask.registry import Task
from brain.prep import index as ix
from brain.prep import mr_rate as M


def _series(study, sid, mod, desc, proc="MR BEYIN (KRANIAL)", acq="3D", plane="AXIAL", spacing="(1.0, 1.0, 1.0)",
            shape="(256, 256, 176)", center=False):
    return dict(study_uid=study, series_id=sid, classified_modality=mod, SeriesDescription=desc, ProtocolName=desc,
                PerformedProcedureStepDescription=proc, RequestedProcedureDescription=proc, StudyDescription="",
                is_derived="False", is_localizer="False", is_subtraction="False", is_center_modality=str(center),
                MRAcquisitionType=acq, acquisition_plane=plane, array_spacing_mm=spacing, array_shape=shape)


def test_post_contrast_rule_and_preference():
    con = "MR, BEYIN KONTRASTLI(ILAÇLI)"
    m = pd.DataFrame([
        # non-contrast study: the 3D T1 wins over the 2D one; FLAIR / T2 kept; spine series dropped
        _series("A", "t1-2d", "T1w", "T1_SE_TRA", acq="2D", spacing="(0.5, 0.5, 5.0)", shape="(512, 512, 24)"),
        _series("A", "t1-3d", "T1w", "T1_MPRAGE_SAG_P2_ISO"),
        _series("A", "fl", "FLAIR", "3D_BRAIN_VIEW_FLAIR"),
        _series("A", "t2", "T2w", "T2_TSE_TRA", acq="2D", spacing="(0.4, 0.4, 5.0)", shape="(512, 512, 24)"),
        _series("A", "sp", "T2w", "T2_TSE_SAG_C-SPINE", acq="2D", spacing="(0.4, 0.4, 3.0)", shape="(512, 512, 16)"),
        # contrast study with a pre/post pair: unmarked twin is pre, marked is post
        _series("B", "pre", "T1w", "T1_SE_TRA", proc=con, acq="2D", spacing="(0.5, 0.5, 5.0)", shape="(512, 512, 24)"),
        _series("B", "post", "T1w", "T1_SE_TRA+C", proc=con, acq="2D", spacing="(0.5, 0.5, 5.0)", shape="(512, 512, 24)"),
        # contrast study, single unmarked T1: ambiguous, skipped
        _series("C", "amb", "T1w", "ST1W_3D_TFE", proc=con),
        _series("C", "fl", "FLAIR", "3D_BRAIN_VIEW_FLAIR", proc=con),
        # knee study: not head
        _series("D", "k", "T1w", "T1_SE_COR", proc="SOL DIZ MR"),
    ])
    p = M.pick_series(m).set_index(["study_uid", "contrast"])["series_id"].to_dict()
    assert p[("A", "t1w")] == "t1-3d" and p[("A", "flair")] == "fl" and p[("A", "t2w")] == "t2"
    assert ("A", "t1c") not in p
    assert p[("B", "t1w")] == "pre" and p[("B", "t1c")] == "post"
    assert ("C", "t1w") not in p and ("C", "t1c") not in p and p[("C", "flair")] == "fl"
    assert not any(k[0] == "D" for k in p)
    assert "sp" not in set(p.values())


@pytest.fixture
def labels(monkeypatch):
    lab = pd.DataFrame({
        "study_uid": ["s1", "s2", "s3", "s4", "s5"],
        "Cerebral hemorrhage": [1, 0, 0, 0, np.nan],
        "Cerebral infarction": [0, 1, 0, 0, np.nan],
        "Glioma": [0, 0, 1, 1, np.nan], "Intracranial meningioma": [0, 0, 0, 1, np.nan],
        "nv:subdural_hematoma": [0, 0, 0, 0, 1],
        "nv:low_grade_glioma": [0, 0, 1, 0, 0], "nv:high_grade_glioma": [0, 0, 0, 1, 0],
    })
    from brain.multitask import labels as L
    monkeypatch.setattr(L, "_mr_rate_labels", lambda: lab)
    vols = pd.DataFrame({"case_id": [f"s{i}-{c}" for i in range(1, 6) for c in ("t1w", "flair")]})
    vols["volume_id"] = "mr_rate_x:" + vols.case_id
    vols["dataset_family"] = "mr_rate"
    return L, vols


def _val(f):
    return dict(zip(f.volume_id.str.split(":").str[1].str.rsplit("-", n=1).str[0], f.value))


def test_binary_soft_negatives_and_unknown(labels):
    L, v = labels
    t = Task("hemorrhage_any", "binary", ["?"], [])
    f = L.src_mr_rate(v, t, {"positive": ["Cerebral hemorrhage", "nv:subdural_hematoma"]})
    assert len(f) == 10                            # every series of every study (s5 only in the NeuroVFM table)
    d = _val(f)
    assert d["s1"] == 1.0 and d["s5"] == 1.0 and d["s2"] == L.MR_RATE_NEGATIVE
    t = Task("mri_infarct", "binary", ["?"], [])
    f = L.src_mr_rate(v, t, {"positive": ["Cerebral hemorrhage"], "unknown": ["Cerebral infarction"]})
    assert "s2" not in _val(f)                     # acute / unspecified infarct: unknown for the old-infarct task
    assert "s5" not in _val(f)                     # not in the 37-category table: no label, not a negative


def test_explicit_negative_and_multiclass(labels):
    L, v = labels
    t = Task("glioma_lower_grade", "binary", ["?"], [])
    d = _val(L.src_mr_rate(v, t, {"positive": ["nv:low_grade_glioma"], "negative": ["nv:high_grade_glioma"]}))
    assert d == {"s3": 1.0, "s4": 0.0}
    t = Task("tumor_type", "multiclass", ["?"], [], classes=["glioma", "meningioma"])
    d = _val(L.src_mr_rate(v, t, {"classes": {"Glioma": "glioma", "Intracranial meningioma": "meningioma"}}))
    assert d == {"s3": 0}                          # s4 names two tumour types: dropped


def test_index_family_key_and_official_split(tmp_path, monkeypatch):
    (tmp_path / "labels").mkdir()
    pd.DataFrame({"study_uid": ["s1", "s2"], "patient_uid": ["p1", "p1"], "split": ["test", "test"]}).to_csv(
        tmp_path / "labels" / "mr_rate_studies.csv", index=False)
    ix.load_mr_rate_studies(tmp_path)
    assert ix.dataset_family("mr_rate_t1w") == "mr_rate" and ix.dataset_family("mr_rate_flair") == "mr_rate"
    assert ix.contrast_of("mr_rate_t1c", "s1-t1c") == "t1c"
    assert ix.dedup_key("mr_rate_t1w", "s1-t1w") == ix.dedup_key("mr_rate_flair", "s2-flair") == "mr_rate:p1"
    vols = pd.DataFrame({"dataset_family": ["mr_rate", "mr_rate", "ixi"], "case_id": ["s1-t1w", "s2-flair", "x"],
                         "split": ["train", "val", "train"]})
    out = ix.pin_official_splits(vols)
    assert out.split.tolist() == ["test", "test", "train"]
    monkeypatch.setattr(ix, "_MR_RATE", {})
    with pytest.raises(SystemExit):
        ix.pin_official_splits(vols)


def test_soft_binary_labels_score_as_negatives():
    from brain.multitask.test_eval import score

    y = np.array([1.0, 0.05, 0.05, 1.0, 0.05])
    p = np.array([0.9, 0.1, 0.2, 0.8, 0.3])
    assert score("binary", y, p) == pytest.approx(1.0)
