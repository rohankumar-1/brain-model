"""Decontamination lists from the pixel-verified duplicate pairs and the data-QC flags (``brain.prep.data_qc``).

``bench-exclude`` (benchmark v3, 2026-11-05) -> ``configs/bench/exclude_v3.csv``, read by
``brain.downstream.tasks.load_task``. From ``duplicate_verified.csv`` (correlation > 0.9 against a random-pair null
that never exceeded 0.76):
  * one copy evaluation-only (index ``held_out``), the other a training / pre-training scan: the evaluation copy goes;
  * both copies in one evaluation cohort (e.g. ATLAS, MSD): the second goes, so one scan cannot sit in two folds;
  * both evaluation-only but in different cohorts: kept (no training exposure, and separate tasks);
  * both training scans: not a benchmark question (training deduplication handles them).
Rows that name an ADNI / PPMI scan (data-use agreements) go to the git-ignored, owner-only
``exclude_v3.private.csv`` beside it.

``train-exclude`` (R15, 2026-11-06) -> ``configs/multitask/train_exclude_r15.csv``, read by ``data.exclude_volumes``
(multitask task rows and the SSL pool alike):
  * one copy evaluation-only or in a cohort the run removes from training (the config's ``data.exclude_datasets``,
    the benchmark cohorts), the other trainable: the trainable copy goes ("copy of a benchmark scan");
  * both trainable but in different splits: the train-split copy goes (a test / val scan's twin is not trained on);
    val vs test: the val copy goes. Pairs inside the train split are left to the index's own dedup;
  * T1 <-> T2 swaps (``contrast_mismatch.csv``, trainable scans): labelled T1 / T1c with T2-like tissue ordering or
    T2 / PD with T1-like ordering; infant cohorts kept (their contrast inverts with myelination). FLAIR vs T1 is not
    decided (the GM / WM order is too close in FLAIR), so those flags stay;
  * CT whose Hounsfield scale is off (``ct_calibration_flags.csv``: brain median outside 15-50 HU or air not near
    -1000): mask in air, or rescaled;
  * Pedi-Cranial-CT infants whose adult-template affine failed (rigid fallback, scale 1): the brain sits small and
    unscaled in the grid, wrong for every size / atlas target (labels v12).

    python -m brain.prep.decontam bench-exclude [--qc-dir DIR] [--out configs/bench/exclude_v3.csv]
    python -m brain.prep.decontam train-exclude [--qc-dir DIR] [--config configs/joint/r15.yaml] [--volumes CSV]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from brain.common.paths import PROCESSED_ROOT, REPO_ROOT
from brain.prep.data_qc import QC_DIR

BENCH_OUT = REPO_ROOT / "configs" / "bench" / "exclude_v3.csv"
TRAIN_OUT = REPO_ROOT / "configs" / "multitask" / "train_exclude_r15.csv"
TRAIN_CONFIG = REPO_ROOT / "configs" / "joint" / "r15.yaml"
PRIVATE = "adni|ppmi"                         # cohorts under a data-use agreement: their ids never go into git
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def verified_pairs(qc_dir: Path) -> pd.DataFrame:
    d = pd.read_csv(qc_dir / "duplicate_verified.csv")
    return d[d["verified"].astype(bool)]


def bench_exclude(qc_dir: Path) -> pd.DataFrame:
    d = verified_pairs(qc_dir)
    a_ho, b_ho = _is_true(d["held_out_a"]), _is_true(d["held_out_b"])
    rows = []
    for r, ah, bh in zip(d.itertuples(), a_ho, b_ho):
        if ah and not bh:
            rows.append({"volume_id": r.a, "reason": "copy of a training scan", "other": r.b, "r": r.r})
        elif bh and not ah:
            rows.append({"volume_id": r.b, "reason": "copy of a training scan", "other": r.a, "r": r.r})
        elif ah and bh and r.dataset_a == r.dataset_b:
            drop, keep = sorted([r.a, r.b], reverse=True)
            rows.append({"volume_id": drop, "reason": "second copy within its evaluation cohort", "other": keep,
                         "r": r.r})
    return pd.DataFrame(rows).drop_duplicates("volume_id")


def write_bench_exclude(out: pd.DataFrame, path: Path) -> None:
    """The public list at ``path`` and the DUA rows at ``<stem>.private.csv`` (mode 600)."""
    priv = out["volume_id"].str.lower().str.contains(PRIVATE) | \
        out["other"].astype(str).str.lower().str.contains(PRIVATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    out[~priv].to_csv(path, index=False)
    private = path.with_name(path.stem + ".private.csv")
    out[priv].to_csv(private, index=False)
    private.chmod(0o600)
    ds = out["volume_id"].str.split(":").str[0]
    print(f"{len(out)} evaluation volumes excluded -> {path} ({int(priv.sum())} DUA rows -> {private.name})")
    print(out.assign(dataset=ds).groupby(["dataset", "reason"]).size().to_string())


def train_exclude(qc_dir: Path, config: Path, volumes: Path) -> pd.DataFrame:
    gone = tuple(yaml.safe_load(open(config))["data"]["exclude_datasets"])
    d = verified_pairs(qc_dir)
    bench = lambda ds, h: h | ds.astype(str).str.startswith(gone)       # noqa: E731
    a_b, b_b = bench(d["dataset_a"], _is_true(d["held_out_a"])), bench(d["dataset_b"], _is_true(d["held_out_b"]))
    rows = []
    for r, ab, bb in zip(d.itertuples(), a_b, b_b):
        if ab and not bb:
            rows.append((r.b, "copy of a benchmark scan", r.a, r.r))
        elif bb and not ab:
            rows.append((r.a, "copy of a benchmark scan", r.b, r.r))
        elif not ab and not bb and r.split_a != r.split_b:
            first_lower = SPLIT_ORDER.get(r.split_a, 0) < SPLIT_ORDER.get(r.split_b, 0)
            drop, keep = (r.a, r.b) if first_lower else (r.b, r.a)
            rows.append((drop, "twin of a val / test scan", keep, r.r))
    c = pd.read_csv(qc_dir / "contrast_mismatch.csv")
    swap = ((c.label.isin(["t1", "t1c"]) & (c.seen == "t2-like")) | (c.label.isin(["t2", "pd"]) & (c.seen == "t1-like"))) \
        & ~c.dataset.str.startswith(gone + ("fomo_infant",))
    rows += [(v, f"contrast swap: labelled {lab}, tissue order {seen}", "", None)
             for v, lab, seen in zip(c.volume_id[swap], c.label[swap], c.seen[swap])]
    ct = pd.read_csv(qc_dir / "ct_calibration_flags.csv")
    vol = pd.read_csv(volumes, low_memory=False, usecols=["volume_id", "held_out", "dataset", "reg_method"]) \
        .drop_duplicates("volume_id")
    held = vol.set_index("volume_id")["held_out"]
    bad = ct[(ct.brain_off | ct.air_off) & ~ct.dataset.str.startswith(gone)]
    bad = bad[~_is_true(held.reindex(bad.volume_id).fillna(True)).to_numpy()]
    rows += [(v, "CT Hounsfield scale off", "", None) for v in bad.volume_id]
    rig = vol[(vol.dataset == "pedi_cranial_ct") & (vol.reg_method.astype(str) == "rigid")]
    rows += [(v, "infant CT not scaled to the template", "", None) for v in rig.volume_id]
    return pd.DataFrame(rows, columns=["volume_id", "reason", "other", "r"]).drop_duplicates("volume_id")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("bench-exclude", help="benchmark exclusion list (+ its private DUA twin)")
    p.add_argument("--qc-dir", type=Path, default=QC_DIR)
    p.add_argument("--out", type=Path, default=BENCH_OUT)
    p = sub.add_parser("train-exclude", help="training exclusion list")
    p.add_argument("--qc-dir", type=Path, default=QC_DIR)
    p.add_argument("--config", type=Path, default=TRAIN_CONFIG, help="run config with data.exclude_datasets")
    p.add_argument("--volumes", type=Path, default=PROCESSED_ROOT / "volumes.csv")
    p.add_argument("--out", type=Path, default=TRAIN_OUT)
    a = ap.parse_args()
    if a.what == "bench-exclude":
        write_bench_exclude(bench_exclude(a.qc_dir), a.out)
        return
    out = train_exclude(a.qc_dir, a.config, a.volumes)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"{len(out)} training volumes excluded -> {a.out}")
    print(out.groupby(out.reason.str.split(":").str[0]).size().to_string())


if __name__ == "__main__":
    main()
