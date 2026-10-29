"""``data.ssl_exclude``: datasets a pretraining round leaves out of the SSL pool, without touching the index."""

from __future__ import annotations

import pandas as pd

from brain.data.volumes import excluded
from brain.downstream.records import pretrain_overlap, ssl_exclude


def _index() -> pd.DataFrame:
    return pd.DataFrame({
        "volume_id": ["g1", "g2", "g3", "m1", "x1"],
        "dataset": ["brats2023_gli_t1c", "brats2023_gli_t2f", "brats2024_gli_t1c", "brats2023_men_t1c", "ixi_t1"],
        "dataset_family": ["brats_gli", "brats_gli", "brats_gli", "brats_men", "ixi"],
        "split": ["train", "train", "train", "train", "test"],
        "pretrain": [True] * 5, "ssl_keep": [True] * 5,
    })


def test_patterns_match_dataset_or_family():
    df = _index()
    assert excluded(df, ["brats2023_gli_*"]).tolist() == [True, True, False, False, False]
    assert excluded(df, ["brats_gli"]).tolist() == [True, True, True, False, False]
    assert not excluded(df, []).any()


def test_overlap_drops_excluded_datasets(tmp_path):
    df = _index()
    ids = ["g1", "g2", "g3", "m1"]
    assert pretrain_overlap(ids, df) == 1.0
    assert pretrain_overlap(ids, df, ["brats2023_gli_*"]) == 0.5
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text("data:\n  ssl_exclude: [brats2023_gli_*]\n")
    assert ssl_exclude(run / "best.pt") == ["brats2023_gli_*"]
    assert ssl_exclude(tmp_path / "none" / "best.pt") == []


def test_a_held_out_scan_is_seen_through_its_pretraining_twin():
    df = pd.concat([_index().assign(dedup_key=["brats:gli:1", "brats:gli:1", "brats:gli:3", "men:1", "ixi:1"]),
                    pd.DataFrame({"volume_id": ["u1", "u2"], "dataset": ["ucsf_pdgm_t1c"] * 2,
                                  "dataset_family": ["ucsf_pdgm"] * 2, "split": ["train", "train"],
                                  "pretrain": [False] * 2, "ssl_keep": [True] * 2,
                                  "dedup_key": ["brats:gli:1", "ucsf_pdgm:UCSF-PDGM-0002"]})])
    assert pretrain_overlap(["u1", "u2"], df) == 0.5
    assert pretrain_overlap(["u1", "u2"], df, ["brats2023_gli_*"]) == 0.0
