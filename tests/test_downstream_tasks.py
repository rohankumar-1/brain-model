"""Every frozen and seg task name loads; 3D readout heads stay linear. python -m tests.test_downstream_tasks"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from brain.downstream.run_suite import readout_head_sets
from brain.downstream.tasks import (
    CLASSIFICATION_TASKS,
    CQ500_LABELS,
    DEFAULT_VOLUMES,
    FROZEN_TASKS,
    ICH_REGIONS,
    REGRESSION_TASKS,
    RSNA_LABELS,
    SEGMENTATION_TASKS,
    TASK_NAMES,
    TaskBundle,
    load_msd_task,
    load_rsna_task,
    load_task,
    msd_join_key,
)

# dataset column values each named task reads from volumes.csv
TASK_DATASETS = {
    "cq500": {"cq500", "seg_cq500"},
    "rsna": {"rsna_ich"},
    "rsna_ich": {"rsna_ich"},
    "oasis1_cdr": {"oasis1"},
    "oasis2_cdr": {"oasis2"},
    "oasis2_conversion": {"oasis2"},
    "abide_asd": {"abide1"},
    "msd": set(),  # joined on volume_id from a labels csv, not a dataset name
    "oasis1_age": {"oasis1"},
    "ixi_age": {"ixi_t1"},
    "pixar_age": {"pixar"},
    "piop1_age": {"piop1"},
    "piop2_age": {"piop2"},
    "id1000_age": {"id1000"},
    "seg_cq500": {"seg_cq500"},
    "brats2023_gli": {"brats2023_gli_t1n"},
    "brats2023_men": {"brats2023_men_t1n"},
    "brats2024_gli": {"brats2024_gli_t1n"},
    "brats2023_ped": {"brats2023_ped_t1n"},
    "atlas": {"atlas"},
    "isles22": {"isles22_dwi"},
    "wmh": {"wmh_flair"},
    "isles24": {"isles24_ncct"},
    "aisd": {"aisd"},
    "instance": {"instance"},
}

TASK_TYPES = {
    **{n: "multilabel" for n in ("cq500", "rsna", "rsna_ich")},
    **{n: "binary" for n in ("oasis1_cdr", "oasis2_cdr", "oasis2_conversion", "abide_asd", "msd")},
    **{n: "regression" for n in REGRESSION_TASKS},
    **{n: "segmentation" for n in SEGMENTATION_TASKS},
}

SEG_MASK = {
    "seg_cq500": ("mask_ich", "/tmp/s.ICH.zip"),
    "brats2023_gli": ("mask_seg", "/tmp/g.seg.zip"),
    "brats2023_men": ("mask_seg", "/tmp/m.seg.zip"),
    "brats2024_gli": ("mask_seg", "/tmp/g4.seg.zip"),
    "brats2023_ped": ("mask_seg", "/tmp/p.seg.zip"),
    "atlas": ("mask_lesion", "/tmp/a.lesion.zip"),
    "isles22": ("mask_lesion", "/tmp/i.lesion.zip"),
    "wmh": ("mask_lesion", "/tmp/w.lesion.zip"),
    "isles24": ("mask_lesion", "/tmp/i24.lesion.zip"),
    "aisd": ("mask_lesion", "/tmp/aisd.lesion.zip"),
    "instance": ("mask_ich", "/tmp/inst.ICH.zip"),
}


def test_task_catalog_is_complete():
    assert set(TASK_DATASETS) == set(TASK_NAMES)
    assert set(TASK_TYPES) == set(TASK_NAMES)
    src = inspect.getsource(load_task)
    for name in TASK_NAMES:
        assert name in src, name


def _row(dataset: str, vid: str, **extra) -> dict:
    case = vid.split(":", 1)[1] if ":" in vid else vid
    out = {
        "volume_id": vid,
        "dataset": dataset,
        "split": extra.pop("split", "train"),
        "image_path": extra.pop("image_path", f"/tmp/{dataset}.zip"),
        "modality": extra.pop("modality", "mri"),
        "case_id": extra.pop("case_id", case),
    }
    out.update(extra)
    return out


def _volumes() -> pd.DataFrame:
    rows = [
        _row("cq500", "cq500:CT-1", modality="ct"),
        _row("seg_cq500", "seg_cq500:CT-1", modality="ct", mask_ich="/tmp/s.ICH.zip"),
        _row("rsna_ich", "ct:RSNA-1", modality="ct"),
        _row("oasis1", "oasis1:OAS1_0001_MR1"),
        _row("oasis2", "oasis2:OAS2_0001_MR1"),
        _row("oasis2", "oasis2:OAS2_0002_MR1"),
        _row("abide1", "abide1:SITE_x", case_id="SITE_x"),
        _row("ixi_t1", "ixi_t1:IXI012", case_id="IXI012"),
        _row("pixar", "pixar:sub-01", case_id="sub-01"),
        _row("piop1", "piop1:sub-01", case_id="sub-01"),
        _row("piop2", "piop2:sub-01", case_id="sub-01"),
        _row("id1000", "id1000:sub-01", case_id="sub-01"),
        _row("msd_braintumor", "msd:1"),
    ]
    for name, (col, path) in SEG_MASK.items():
        ds = next(iter(TASK_DATASETS[name]))
        rows.append(_row(ds, f"{ds}:v0", modality="ct" if name in ("seg_cq500", "isles24", "aisd", "instance") else "mri", **{col: path}))
    return pd.DataFrame(rows)


def _write_clinical(root: Path) -> Path:
    (root / "cq500_meta").mkdir(parents=True)
    reads = {"name": ["CT-1"]}
    for lab in CQ500_LABELS:
        reads[f"R1:{lab}"] = [1]
        reads[f"R2:{lab}"] = [1]
        reads[f"R3:{lab}"] = [0]
    pd.DataFrame(reads).to_csv(root / "cq500_meta" / "reads.csv", index=False)

    o1 = root / "oasis" / "oasis1"
    o1.mkdir(parents=True)
    pd.DataFrame({"ID": ["OAS1_0001_MR1"], "Age": [70], "CDR": [0.5]}).to_csv(
        o1 / "oasis_cross-sectional.csv", index=False
    )
    o2 = root / "oasis" / "oasis2"
    o2.mkdir(parents=True)
    pd.DataFrame({
        "MRI ID": ["OAS2_0001_MR1", "OAS2_0001_MR2", "OAS2_0002_MR1"],
        "Subject ID": ["OAS2_0001", "OAS2_0001", "OAS2_0002"],
        "Visit": [1, 2, 1],
        "CDR": [0.0, 1.0, 1.0],
    }).to_csv(o2 / "oasis_longitudinal.csv", index=False)

    a1 = root / "abide1"
    a1.mkdir()
    pd.DataFrame({"FILE_ID": ["SITE_x"], "DX_GROUP": [1], "SITE_ID": ["SITE"]}).to_csv(
        a1 / "Phenotypic_V1_0b.csv", index=False
    )
    for ds in ("pixar", "piop1", "piop2", "id1000"):
        d = root / ds
        d.mkdir()
        pd.DataFrame({"participant_id": ["sub-01"], "age": [25.0]}).to_csv(
            d / "participants.tsv", sep="\t", index=False
        )
    msd = root / "msd_volume_labels.csv"
    pd.DataFrame({"volume_id": ["msd:1"], "enhancing_high": [1]}).to_csv(msd, index=False)
    return msd


def _rsna_labels() -> pd.DataFrame:
    return pd.DataFrame([{"volume_id": "ct:RSNA-1", **{k: 1 for k in RSNA_LABELS}}])


def test_load_task_covers_every_name():
    vols = _volumes()
    xls = pd.DataFrame({"IXI_ID": [12], "AGE": [41.0]})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        msd = _write_clinical(root)
        with (
            patch("brain.downstream.tasks.DATASETS", root),
            patch("brain.downstream.tasks.build_rsna_volume_labels", return_value=_rsna_labels()),
            patch("pandas.read_excel", return_value=xls),
        ):
            for name in TASK_NAMES:
                task = load_task(name, vols, msd_labels=msd)
                assert task.name == name, name
                assert task.task_type == TASK_TYPES[name], (name, task.task_type)
                assert len(task.volume_ids) >= 1, name
                assert set(task.splits) == set(task.volume_ids)


def test_msd_joins_store_ids_and_builds_enhancing_high():
    assert msd_join_key("msd_braintumor:BRATS_001") == "BRATS_001"
    assert msd_join_key("mri:MSD-BRATS_001") == "BRATS_001"
    vols = pd.DataFrame([{
        "volume_id": "msd_braintumor:BRATS_001",
        "dataset": "msd_braintumor",
        "split": "train",
        "image_path": "/tmp/a.zip",
        "modality": "mri",
        "case_id": "BRATS_001",
    }])
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "volume_labels.csv"
        pd.DataFrame({
            "name": ["BRATS_001", "BRATS_002"],
            "enhancing_vox": [30000, 10],
        }).to_csv(src, index=False)
        task = load_msd_task(vols, labels_csv=src)
    assert task.name == "msd" and task.task_type == "binary"
    assert task.volume_ids == ["msd_braintumor:BRATS_001"]
    assert int(task.y[0]) == 1


def test_rsna_aliases_share_labels_and_differ_by_name():
    vols = _volumes()
    labels = _rsna_labels()
    a = load_rsna_task(vols, labels=labels, name="rsna")
    b = load_rsna_task(vols, labels=labels, name="rsna_ich")
    assert a.task_type == b.task_type == "multilabel"
    assert a.label_names == b.label_names == list(RSNA_LABELS)
    assert a.name == "rsna" and b.name == "rsna_ich"


def test_instance_is_in_the_seg_suite():
    assert "instance" in SEGMENTATION_TASKS
    assert "instance" not in FROZEN_TASKS
    assert ICH_REGIONS["ICH"] == (1, 2)


def _fake_views():
    n, d = 4, 8
    pack = {
        "train": {"mus": np.zeros((n, 3, d), np.float32), "mask": np.ones((n, 3), bool),
                  "y": np.zeros(n), "volume_ids": [f"v{i}" for i in range(n)]},
        "val": {"mus": np.zeros((2, 3, d), np.float32), "mask": np.ones((2, 3), bool),
                "y": np.zeros(2), "volume_ids": ["a", "b"]},
        "test": {"mus": np.zeros((2, 3, d), np.float32), "mask": np.ones((2, 3), bool),
                 "y": np.zeros(2), "volume_ids": ["c", "d"]},
    }
    return {"plane": pack, "grid": pack, "global": pack}


def test_readout_heads_are_linear_probes():
    views = _fake_views()
    clf = TaskBundle("t", "binary", ["v0"], {"v0": "train"}, np.array([1]), ["y"])
    names = [h.name for _, _, heads in readout_head_sets(clf, views, 0) for h in heads]
    assert names == ["plane_logreg", "grid_logreg", "pca_global_logreg"]
    for h in (h for _, _, hs in readout_head_sets(clf, views, 0) for h in hs):
        assert "xgb" not in h.name and "attn" not in h.name

    reg = TaskBundle("t", "regression", ["v0"], {"v0": "train"}, np.array([1.0]), ["Age"])
    names = [h.name for _, _, heads in readout_head_sets(reg, views, 0) for h in heads]
    assert names == ["plane_ridge", "grid_ridge", "pca_global_ridge"]


def test_store_has_every_frozen_and_seg_family():
    """Skip when the zarr index is not on this machine; otherwise every catalog family must exist."""
    if not DEFAULT_VOLUMES.exists():
        return
    ds = set(pd.read_csv(DEFAULT_VOLUMES, usecols=["dataset"], low_memory=False)["dataset"].astype(str))
    missing = []
    for name in TASK_NAMES:
        need = TASK_DATASETS[name]
        if not need:
            continue
        if ds.isdisjoint(need):
            missing.append(f"{name} ({sorted(need)})")
    assert not missing, "volumes.csv is missing task families:\n  " + "\n  ".join(missing)


def test_frozen_suite_is_clf_plus_age():
    assert FROZEN_TASKS == CLASSIFICATION_TASKS + REGRESSION_TASKS
    assert not set(SEGMENTATION_TASKS) & set(FROZEN_TASKS)


def test_pi_job_defaults_to_the_full_frozen_suite():
    text = (Path(__file__).resolve().parents[1] / "jobs" / "downstream_vit_pi.sbatch").read_text()
    assert "TASK_LIST=(oasis1_age oasis1_cdr cq500)" not in text
    for name in FROZEN_TASKS:
        assert name in text, name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
