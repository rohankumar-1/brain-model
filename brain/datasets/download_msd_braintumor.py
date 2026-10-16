#!/usr/bin/env python3
"""Download Medical Decathlon Task01 BrainTumour (labeled multimodal MRI).

Source: https://msd-for-monai.s3.amazonaws.com/Task01_BrainTumour.tar (~7.6 GB)
Writes: $BRAIN_DATASETS/msd_braintumor/
Labels: tumor subregion segmentation in labelsTr/ — also writes a volume-level
summary CSV (enhancing / edema / core presence + voxel counts) for frozen-embedding eval.
"""

from __future__ import annotations

import argparse
import csv
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import DATASETS

DEFAULT_OUT = DATASETS / "msd_braintumor"
URL = "https://msd-for-monai.s3.amazonaws.com/Task01_BrainTumour.tar"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000_000:
        print(f"[skip] {dest} already present ({dest.stat().st_size / 1e9:.1f} GB)")
        return
    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def extract(tar_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "Task01_BrainTumour" / "dataset.json"
    if marker.exists():
        print(f"[skip] already extracted at {marker.parent}")
        return marker.parent
    print(f"[extract] {tar_path}")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(out_dir)
    if not marker.exists():
        raise SystemExit(f"Expected {marker} after extract")
    return marker.parent


def build_label_csv(task_dir: Path) -> Path:
    """Summarize segmentation labels into volume-level binary + count targets."""
    import nibabel as nib

    label_dir = task_dir / "labelsTr"
    rows = []
    for path in sorted(label_dir.glob("*.nii*")):
        if path.name.startswith("._"):
            continue
        seg = np.asanyarray(nib.load(str(path)).dataobj)
        # MSD BrainTumour: 1=edema, 2=non-enhancing, 3=enhancing
        counts = {
            "edema_vox": int((seg == 1).sum()),
            "non_enhancing_vox": int((seg == 2).sum()),
            "enhancing_vox": int((seg == 3).sum()),
        }
        rows.append(
            {
                "name": path.name.replace(".nii.gz", "").replace(".nii", ""),
                "path": str(path.relative_to(task_dir)),
                "has_edema": int(counts["edema_vox"] > 0),
                "has_non_enhancing": int(counts["non_enhancing_vox"] > 0),
                "has_enhancing": int(counts["enhancing_vox"] > 0),
                **counts,
            }
        )
    if not rows:
        raise SystemExit(f"No label NIfTIs found under {label_dir}")
    out = task_dir / "volume_labels.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {out} ({len(rows)} volumes)")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-labels", action="store_true")
    return p.parse_args()


def export_msd_labels(msd_root: Path, out_csv: Path) -> None:
    """Copy volume_labels.csv with volume_id = msd_braintumor:<name> for frozen eval."""
    src = Path(msd_root) / "volume_labels.csv"
    if not src.exists():
        src = Path(msd_root).parent / "volume_labels.csv"
    if not src.exists():
        print(f"[msd] no {src}; run python -m brain.datasets.download_msd_braintumor --skip-download")
        return
    df = pd.read_csv(src)
    df["volume_id"] = "msd_braintumor:" + df["name"].astype(str)
    q = df["enhancing_vox"].quantile([1 / 3, 2 / 3]).to_list()
    df["enhancing_tertile"] = pd.cut(
        df["enhancing_vox"],
        bins=[-0.5, q[0], q[1], float(df["enhancing_vox"].max()) + 1],
        labels=[0, 1, 2],
    ).astype(int)
    df["enhancing_high"] = (df["enhancing_vox"] >= q[1]).astype(int)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(
        f"[msd] wrote labels {out_csv} (n={len(df)}, "
        f"enhancing_high rate={df.enhancing_high.mean():.2f})"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = args.out_dir / "Task01_BrainTumour.tar"
    if not args.skip_download:
        download(URL, tar_path)
    task_dir = extract(tar_path, args.out_dir)
    if not args.skip_labels:
        build_label_csv(task_dir)
    print(f"[done] MSD BrainTumour at {task_dir}")


if __name__ == "__main__":
    main()
