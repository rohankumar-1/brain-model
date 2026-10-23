"""RSNA studies straight from the big zip: byte-range reads by stored offset, then a DICOM series read.

Opening ``zipfile.ZipFile`` per study would re-parse an 874k-entry central directory each time, so the
offsets of every member are recorded once (``build_offsets``) and each study reads only its own bytes.
"""

from __future__ import annotations

import json
import os
import sys
import struct
from concurrent.futures import ThreadPoolExecutor
import tempfile
import zipfile
import zlib
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from brain.prep import geometry

Member = tuple[str, int, int, int]  # (name, local header offset, compressed size, compress type)


def build_offsets(zip_path: Path, study_index: Path, out: Path) -> dict[str, list[Member]]:
    """Attach byte offsets to the study -> member-names index built by the original ingest (cached)."""
    if out.exists():
        return json.loads(out.read_text())
    names = json.loads(study_index.read_text())
    with zipfile.ZipFile(zip_path) as zf:
        info = {i.filename: (i.header_offset, i.compress_size, i.compress_type) for i in zf.infolist()}
    index = {uid: [[n, *info[n]] for n in members if n in info] for uid, members in names.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index))
    return index


HEADER_SLACK = 1024  # local header (30 B) + file name + extra field fit well inside this
IO_THREADS = 16  # a study's members are scattered over ~150 GB: storage latency, not bandwidth, is the cost


def _read_member(fd: int, offset: int, csize: int, ctype: int) -> bytes:
    """One ``pread`` covers the local header and payload; a second is issued only for oversized headers."""
    buf = os.pread(fd, csize + HEADER_SLACK, offset)
    if buf[:4] != b"PK\x03\x04":
        raise ValueError(f"bad local header at {offset}")
    n, m = struct.unpack("<HH", buf[26:30])
    start = 30 + n + m
    data = buf[start:start + csize]
    if len(data) < csize:
        data = os.pread(fd, csize, offset + start)
    return data if ctype == zipfile.ZIP_STORED else zlib.decompress(data, -15)


def read_study(zip_path: str, members, tmp_root: str | None = None) -> tuple[sitk.Image, dict]:
    """Extract a study's DICOMs to a temp dir, read its largest series -> (RAS float32 image, QC info)."""
    fd = os.open(zip_path, os.O_RDONLY)
    try:
        with ThreadPoolExecutor(IO_THREADS) as ex:
            blobs = list(ex.map(lambda m: _read_member(fd, *m[1:]), members))
    finally:
        os.close(fd)
    with tempfile.TemporaryDirectory(dir=tmp_root) as d:
        for k, b in enumerate(blobs):
            (Path(d) / f"{k:04d}.dcm").write_bytes(b)
        return read_series_dir(d)


def read_series_dir(folder: str | Path) -> tuple[sitk.Image, dict]:
    """Largest series in a folder of DICOM files -> (RAS float32 image, QC info). ``BRAIN_DICOM_READER=dcm2niix``
    assembles slices by position (dcm2niix: duplicate / mixed slices, gantry-tilt de-shear, even spacing); the
    default SimpleITK series reader assumes even, untilted slices (docs/prep.md, "RSNA DICOM assembly")."""
    if os.environ.get("BRAIN_DICOM_READER", "sitk") == "dcm2niix":
        # dcm2niix refuses some series (ImageType DERIVED / SECONDARY CTAs, malformed overlay tags): read those with
        # the SimpleITK series reader instead
        try:
            return read_series_dcm2niix(folder)
        except RuntimeError as e:
            if "wrote nothing" not in str(e):
                raise
            img, info = read_series_dir_sitk(folder)
            return img, {**info, "dcm2niix_variant": "sitk_fallback"}
    return read_series_dir_sitk(folder)


def read_series_dir_sitk(folder: str | Path) -> tuple[sitk.Image, dict]:
    """SimpleITK / GDCM series reader (the default ``read_series_dir``)."""
    folder = str(folder)
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    series = reader.GetGDCMSeriesIDs(folder) or [None]
    loose = sorted(str(p) for p in Path(folder).glob("*.dcm")) + sorted(str(p) for p in Path(folder).glob("*.DCM"))
    files = max((reader.GetGDCMSeriesFileNames(folder, s) if s else loose for s in series), key=len)
    if len(files) < 8:
        raise RuntimeError(f"too few DICOM slices ({len(files)}) in {folder}")
    reader.SetFileNames(files)
    img = reader.Execute()
    info = {"n_series": len(series), "n_slices": len(files), **_slice_gaps(reader, img)}
    return geometry.standardize(img), info


def _slice_gaps(reader: sitk.ImageSeriesReader, img: sitk.Image) -> dict:
    """Largest deviation (mm) of consecutive slice spacing from the median: missing or irregular slices."""
    try:
        normal = np.array(img.GetDirection()).reshape(3, 3)[:, 2]
        pos = np.sort([np.dot([float(v) for v in reader.GetMetaData(i, "0020|0032").split("\\")], normal)
                       for i in range(img.GetSize()[2])])
        gaps = np.diff(pos)
        return {"slice_gap_dev_mm": float(np.abs(gaps - np.median(gaps)).max())}
    except Exception:
        return {"slice_gap_dev_mm": float("nan")}


# dcm2niix output variants, best first: tilt-corrected and equidistant, tilt-corrected, equidistant, as acquired
_DCM2NIIX_PREF = ("_Tilt_Eq_1", "_Tilt_1", "_Eq_1", "")


def read_series_dcm2niix(folder: str | Path) -> tuple[sitk.Image, dict]:
    """dcm2niix (``-i y``: no derived / localizer images) on a folder; the largest series, preferring its
    tilt-corrected / equidistant reconstruction when dcm2niix writes one."""
    import shutil
    import subprocess

    exe = os.environ.get("BRAIN_DCM2NIIX") or shutil.which("dcm2niix") or str(Path(sys.executable).parent / "dcm2niix")
    with tempfile.TemporaryDirectory(dir=str(folder)) as out:
        r = subprocess.run([exe, "-b", "n", "-z", "n", "-i", "y", "-f", "s%s", "-o", out, str(folder)],
                           capture_output=True, text=True, timeout=600)
        files = sorted(Path(out).glob("*.nii"))
        if not files:
            raise RuntimeError(f"dcm2niix wrote nothing for {folder}: {r.stdout[-300:]} {r.stderr[-300:]}")
        def base(f):
            n = f.stem
            for suf in _DCM2NIIX_PREF[:-1]:
                n = n.replace(suf, "")
            return n
        series = {}
        for f in files:
            series.setdefault(base(f), []).append(f)
        def n_vox(fs):
            return max(np.prod(sitk.ReadImage(str(f)).GetSize()) for f in fs)
        group = max(series.values(), key=n_vox)
        pick = next(f for suf in _DCM2NIIX_PREF for f in group if f.stem == base(f) + suf)
        img = sitk.ReadImage(str(pick))
        info = {"n_series": len(series), "n_slices": int(img.GetSize()[2]), "dcm2niix_variant": pick.stem[len(base(pick)):] or "plain",
                "dcm2niix_warn": int(r.stdout.count("Warning"))}
        return geometry.standardize(img), info
