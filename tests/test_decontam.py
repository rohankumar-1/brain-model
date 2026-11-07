"""Exclusion-list rules (brain.prep.decontam) on synthetic QC tables."""

from __future__ import annotations

import pandas as pd


def _pairs(tmp_path):
    d = pd.DataFrame([
        # a, b, dataset_a, dataset_b, held_out_a, held_out_b, split_a, split_b, r
        ("atlas:1", "ixi_t1:1", "atlas", "ixi_t1", True, False, "test", "train", 0.97),
        ("atlas:2", "atlas:3", "atlas", "atlas", True, True, "train", "test", 0.95),
        ("cq500:1", "aisd:1", "cq500", "aisd", True, True, "test", "test", 0.96),
        ("ixi_t1:2", "oasis1:2", "ixi_t1", "oasis1", False, False, "train", "val", 0.99),
        ("adni_t1w:I1", "fomo_x:1", "adni_t1w", "fomo_x", True, False, "test", "train", 0.98),
        ("ixi_t1:9", "oasis1:9", "ixi_t1", "oasis1", False, False, "train", "train", 0.5),
    ], columns=["a", "b", "dataset_a", "dataset_b", "held_out_a", "held_out_b", "split_a", "split_b", "r"])
    d["verified"] = d["r"] > 0.9
    d.to_csv(tmp_path / "duplicate_verified.csv", index=False)


def test_bench_exclude_drops_evaluation_copies_and_splits_off_dua_rows(tmp_path):
    from brain.prep.decontam import bench_exclude, write_bench_exclude

    _pairs(tmp_path)
    out = bench_exclude(tmp_path)
    assert dict(zip(out.volume_id, out.reason)) == {
        "atlas:1": "copy of a training scan", "atlas:3": "second copy within its evaluation cohort",
        "adni_t1w:I1": "copy of a training scan"}
    write_bench_exclude(out, tmp_path / "exclude.csv")
    public, private = pd.read_csv(tmp_path / "exclude.csv"), pd.read_csv(tmp_path / "exclude.private.csv")
    assert set(public.volume_id) == {"atlas:1", "atlas:3"} and len(private) == 1
    assert (tmp_path / "exclude.private.csv").stat().st_mode & 0o077 == 0


def test_train_exclude_drops_trainable_copies_swaps_and_bad_ct(tmp_path):
    from brain.prep.decontam import train_exclude

    _pairs(tmp_path)
    pd.DataFrame({"volume_id": ["fomo_a:1", "fomo_infant_t1w:1", "atlas:7"],
                  "dataset": ["fomo_a_t1w", "fomo_infant_t1w", "atlas"], "label": ["t1", "t1", "t1"],
                  "seen": ["t2-like", "t2-like", "t2-like"]}).to_csv(tmp_path / "contrast_mismatch.csv", index=False)
    pd.DataFrame({"volume_id": ["rsna:1", "cq500:5"], "dataset": ["rsna_ich", "cq500"], "brain_off": [True, True],
                  "air_off": [False, False]}).to_csv(tmp_path / "ct_calibration_flags.csv", index=False)
    pd.DataFrame({"volume_id": ["rsna:1", "cq500:5", "pedi:1"], "held_out": [False, True, False],
                  "dataset": ["rsna_ich", "cq500", "pedi_cranial_ct"], "reg_method": ["affine", "affine", "rigid"]}
                 ).to_csv(tmp_path / "volumes.csv", index=False)
    (tmp_path / "cfg.yaml").write_text("data:\n  exclude_datasets: [atlas, cq500]\n")
    out = train_exclude(tmp_path, tmp_path / "cfg.yaml", tmp_path / "volumes.csv")
    got = dict(zip(out.volume_id, out.reason))
    assert got["ixi_t1:1"] == "copy of a benchmark scan" and got["fomo_x:1"] == "copy of a benchmark scan"
    assert got["ixi_t1:2"] == "twin of a val / test scan" and "oasis1:2" not in got
    assert got["fomo_a:1"].startswith("contrast swap") and "fomo_infant_t1w:1" not in got and "atlas:7" not in got
    assert got["rsna:1"] == "CT Hounsfield scale off" and "cq500:5" not in got
    assert got["pedi:1"] == "infant CT not scaled to the template" and "ixi_t1:9" not in got
