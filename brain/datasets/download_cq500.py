"""CQ500 head CTs as original DICOM (qure.ai), one series per study, onto pool.

The CQ500 we prepped until 2026-10-26 came as third-party pre-converted NIfTIs (one per study, series unknown): some
are mirrored (the bleed side disagrees with unanimous reads in 20 of 34 lateralised bleeds, both ways), and they mix
reconstruction kernels and contrast series. Each study zip (~165 MB, several series: plain thin / plain 5 mm soft
tissue, contrast series, sometimes bone) is downloaded to node-local disk, every series' header is read, and the series
``choose`` picks is written to ``<dest>/CQ500-CT-<i>.zip`` (stored, so prep reads members by offset like RSNA);
``<dest>/series.csv`` lists every series and the choice. Resumable: studies with a zip are skipped.

    python -m brain.datasets.download_cq500 --dest $BRAIN_DATASETS/cq500_dicom --workers 6
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

URL = "https://s3.ap-south-1.amazonaws.com/qure.headct.study/CQ500-CT-{}.zip"
N_STUDIES = 491
SOFT_NAMES = ("SOFT", "STANDARD", "STND", "BRAIN", "UB", "FC2")
SHARP_NAMES = ("BONE", "LUNG", "SHARP", "EDGE", "DETAIL", "FC3", "FC8")
CONTRAST = ("CC", "CONTRAST", "CTA", "ANGIO", "POST", "+C", "C+", "ENHANC")       # "4cc sec 150cc": an injection


def _kernel_class(kernel: str) -> str:
    """soft / bone / other from the reconstruction kernel name. Letter + number kernels (Siemens B30f / H31s / J40,
    Philips-style codes): below 60 is soft tissue, 60 and above sharp."""
    import re

    k = str(kernel or "").upper().replace(" ", "")
    if not k:
        return "soft"
    if k.startswith(SHARP_NAMES):
        return "bone"
    if k.startswith(SOFT_NAMES):
        return "soft"
    m = re.match(r"^[A-Z]{1,2}(\d{2})", k)
    if m:
        return "bone" if int(m.group(1)) >= 60 else "soft"
    return "other"


def choose(series: pd.DataFrame) -> str | None:
    """The plain (non-contrast) soft-tissue series, thinnest first (ties: most slices); None if there is none."""
    desc = series["description"].astype(str).str.upper()
    plain = ~desc.apply(lambda d: any(c in d.replace(" ", "") for c in CONTRAST))
    ok = series[plain & (series["kernel_class"] == "soft") & (series["n"] >= 20)]
    if ok.empty:
        return None
    return ok.sort_values(["thickness", "n"], ascending=[True, False]).iloc[0]["folder"]


def headers(zf: zipfile.ZipFile) -> pd.DataFrame:
    import pydicom

    files: dict[str, list[str]] = {}
    for n in zf.namelist():
        if not n.endswith("/"):
            files.setdefault(n.rsplit("/", 1)[0], []).append(n)
    rows = []
    for folder, names in files.items():
        ds = pydicom.dcmread(io.BytesIO(zf.read(names[0])), stop_before_pixels=True)
        rows.append({"folder": folder, "n": len(names), "description": str(ds.get("SeriesDescription", "")),
                     "kernel": str(ds.get("ConvolutionKernel", "")), "thickness": float(ds.get("SliceThickness", 0) or 0),
                     "patient_id": str(ds.get("PatientID", ""))})
    out = pd.DataFrame(rows)
    out["kernel_class"] = out["kernel"].map(_kernel_class)
    return out


def fetch(i: int, dest: Path, tmp: Path) -> list[dict]:
    out = dest / f"CQ500-CT-{i}.zip"
    if out.exists():
        return []
    src = tmp / f"CQ500-CT-{i}.zip"
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(URL.format(i), src)
            break
        except Exception:                                    # noqa: BLE001  transient network errors
            if attempt == 3:
                return [{"study": f"CQ500-CT-{i}", "error": "download failed"}]
            time.sleep(10 * (attempt + 1))
    try:
        with zipfile.ZipFile(src) as zf:
            s = headers(zf)
            pick = choose(s)
            if pick is not None:
                part = out.with_name(out.name + ".partial")
                with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as w:
                    for n in zf.namelist():
                        if n.rsplit("/", 1)[0] == pick and not n.endswith("/"):
                            w.writestr(n.rsplit("/", 1)[1], zf.read(n))
                os.replace(part, out)
    finally:
        src.unlink(missing_ok=True)
    return [{**r, "study": f"CQ500-CT-{i}", "chosen": r["folder"] == pick} for r in s.to_dict("records")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--studies", default=f"0-{N_STUDIES - 1}", help="range a-b (inclusive)")
    a = ap.parse_args()
    lo, hi = map(int, a.studies.split("-"))
    a.dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as t, ThreadPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(lambda i: fetch(i, a.dest, Path(t)), range(lo, hi + 1)) for r in rs]
    if rows:
        f = a.dest / "series.csv"
        old = pd.read_csv(f) if f.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(f, index=False)
    got = len(list(a.dest.glob("CQ500-CT-*.zip")))
    print(f"[cq500] {got} studies on {a.dest}; this run listed {len(rows)} series", flush=True)
    if rows:
        d = pd.DataFrame(rows)
        print(d[d.get("chosen", False) == True][["description", "kernel", "thickness"]].value_counts().head(10).to_string())  # noqa: E712
        none = sorted(set(d["study"]) - set(d.loc[d.get("chosen", False) == True, "study"]))  # noqa: E712
        print(f"[cq500] studies without a plain soft-tissue series: {len(none)} {none[:10]}", flush=True)


if __name__ == "__main__":
    main()
