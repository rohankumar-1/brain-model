#!/usr/bin/env python3
"""Download additional CQ500 head CTs (beyond Seg-CQ500) and convert to NIfTI.

Source: Qure.ai S3 zips listed in Jason-Adam/cv-medical.
Writes: $BRAIN_DATASETS/cq500/<CQ500-CT-N>/CT.nii.gz
Skips any case ID already present under Seg-CQ500 (no overlap / leakage).
Also keeps prediction_probabilities.csv for full-cohort outcome labels.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATASETS = Path("/home/roku/orcd/pool/datasets")
DEFAULT_OUT = DATASETS / "cq500"
DEFAULT_EXISTING = DATASETS / "seg_cq500" / "data" / "volumes"
DEFAULT_META = DATASETS / "cq500_meta"
S3_PREFIX = "https://s3.ap-south-1.amazonaws.com/qure.headct.study/"
URL_LIST = (
    "https://raw.githubusercontent.com/Jason-Adam/cv-medical/master/docs/cq500_files.txt"
)


def existing_case_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for p in paths:
        if not p.is_dir():
            continue
        for d in p.iterdir():
            if d.is_dir() and d.name.startswith("CQ500"):
                ids.add(d.name)
    return ids


def fetch_url_list() -> list[str]:
    import urllib.request

    with urllib.request.urlopen(URL_LIST, timeout=60) as r:
        text = r.read().decode("utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip().startswith("http")]


def case_id_from_url(url: str) -> str:
    name = Path(url).name  # CQ500-CT-0.zip
    return name.replace(".zip", "")


def pick_dicom_series(extract_dir: Path) -> Path | None:
    """Prefer the series directory with the most .dcm files (usually thin slices)."""
    best_dir = None
    best_n = 0
    for d in extract_dir.rglob("*"):
        if not d.is_dir():
            continue
        n = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in {".dcm", ""})
        # also count files without extension that look like DICOM
        if n == 0:
            n = sum(
                1
                for f in d.iterdir()
                if f.is_file() and f.suffix == "" and f.stat().st_size > 10_000
            )
        if n > best_n:
            best_n = n
            best_dir = d
    if best_dir is None or best_n < 8:
        return None
    return best_dir


def convert_series_to_nifti(series_dir: Path, out_nii: Path) -> None:
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(series_dir))
    if not ids:
        # try parent levels
        for parent in [series_dir] + list(series_dir.parents)[:3]:
            ids = reader.GetGDCMSeriesIDs(str(parent))
            if ids:
                series_dir = parent
                break
    if not ids:
        raise RuntimeError(f"No DICOM series in {series_dir}")

    # pick series with most files
    best_id, best_files = None, []
    for sid in ids:
        files = reader.GetGDCMSeriesFileNames(str(series_dir), sid)
        if len(files) > len(best_files):
            best_id, best_files = sid, files
    reader.SetFileNames(best_files)
    img = reader.Execute()
    out_nii.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_nii))


def download_zip(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return
    tmp = dest.with_suffix(dest.suffix + ".partial")
    subprocess.check_call(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "-A",
            "Mozilla/5.0",
            url,
            "-o",
            str(tmp),
        ]
    )
    tmp.rename(dest)


def process_one(url: str, out_root: Path, work_root: Path) -> tuple[str, bool, str]:
    cid = case_id_from_url(url)
    out_nii = out_root / cid / "CT.nii.gz"
    if out_nii.exists():
        return cid, True, "exists"
    # Per-case work dir avoids collisions under parallel download
    case_work = work_root / cid
    zpath = case_work / f"{cid}.zip"
    extract_dir = case_work / "extract"
    try:
        case_work.mkdir(parents=True, exist_ok=True)
        download_zip(url, zpath)
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(extract_dir)
        series = pick_dicom_series(extract_dir)
        if series is None:
            series = extract_dir
        convert_series_to_nifti(series, out_nii)
        ok = out_nii.exists()
        return cid, ok, "ok" if ok else "missing_output"
    except Exception as e:
        if out_nii.exists():
            out_nii.unlink()
        return cid, False, str(e)
    finally:
        if case_work.exists():
            shutil.rmtree(case_work, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--existing-dir", type=Path, default=DEFAULT_EXISTING)
    p.add_argument("--work-dir", type=Path, default=Path("/home/roku/orcd/scratch/brain-model/work/cq500_download"))
    p.add_argument("--max-extra", type=int, default=150, help="Max new CT volumes to add")
    p.add_argument("--workers", type=int, default=6, help="Parallel download+convert workers")
    p.add_argument("--meta-dir", type=Path, default=DEFAULT_META)
    return p.parse_args()


def main() -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.meta_dir.mkdir(parents=True, exist_ok=True)

    preds = args.meta_dir / "prediction_probabilities.csv"
    if not preds.exists():
        subprocess.check_call(
            [
                "curl",
                "-fsSL",
                "-o",
                str(preds),
                "https://huggingface.co/datasets/neurosentry/cq500/resolve/main/prediction_probabilities.csv",
            ]
        )

    have = existing_case_ids([args.existing_dir, args.out_dir])
    print(f"[cq500] already have {len(have)} case folders")
    urls = fetch_url_list()
    todo = [u for u in urls if case_id_from_url(u) not in have][: args.max_extra]
    print(
        f"[cq500] downloading/converting {len(todo)} new studies "
        f"with {args.workers} workers -> {args.out_dir}"
    )

    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, u, args.out_dir, args.work_dir): u for u in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="cq500-extra"):
            cid, success, msg = fut.result()
            if success:
                ok += 1
                have.add(cid)
            else:
                print(f"[warn] {cid}: {msg}")
    print(f"[ok] newly available: {ok}; total CT folders now ~{len(have)}")


if __name__ == "__main__":
    main()
