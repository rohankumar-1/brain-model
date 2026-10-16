"""SinoCT file index: reconstructed DICOMs only, no RSNA-ICH overlap.

The Redivis dataset is a file index (``data``, ~357k files) plus ``labels``
(patient_id, normal/abnormal; 9,779 scans). We never download sinograms.
SinoCT re-anonymized UIDs, so RSNA overlap is dropped by pixel fingerprint
(``--overlap``), not by ``ID_*`` path matching.

    python -m brain.datasets.download_sinoct --sample 100     # 100 reconstructed DICOMs + size check
    python -m brain.datasets.download_sinoct --blocklist-only  # RSNA ids only (no Redivis)
    python -m brain.datasets.download_sinoct --overlap         # pixel-fingerprint RSNA zip, probe SinoCT, drop hits
    python -m brain.datasets.download_sinoct --overlap --pull  # then download keep reconstructed DICOMs only

Redivis must be at *data* access, not metadata. Set ``REDIVIS_API_TOKEN`` after
requesting the dataset at https://stanford.redivis.com/datasets/eznj-8hywfc0mq
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from brain.prep.sources import DATASETS, SCRATCH

OUT = DATASETS / "sinoct"
RSNA_INDEX = SCRATCH / "work" / "rsna_ingest" / "study_index.json"
DATASET = "sinoct:eznj:v1_0"
DATA_TABLE = "data:9wbv"
LABELS_TABLE = "labels:xe79"

# Kaggle RSNA ICH ids: study ID_ + 12 hex, slice ID_ + 8–12 hex.
_ID = re.compile(r"\bID[_-]?([0-9a-fA-F]{8,12})\b")
_SINO = re.compile(r"(sino|sinogram|catsim|\.npy$|\.npz$)", re.I)
_DICOM = re.compile(r"(\.dcm$|\.dicom$|\.ima$|(^|/)dicom)", re.I)


def normalize_id(token: str) -> str:
    s = str(token).strip().replace("-", "_")
    s = s.removeprefix("RSNA_").removeprefix("RSNA-").removeprefix("ct:")
    m = _ID.search(s) or re.fullmatch(r"ID[_-]?([0-9a-fA-F]{8,12})", s, re.I)
    if m:
        return "ID_" + (m.group(1) if m.lastindex else m.group(0).split("_")[-1]).lower()
    hex_only = re.fullmatch(r"[0-9a-fA-F]{8,12}", s)
    return ("ID_" + s.lower()) if hex_only else s.lower()


def ids_in(text: str) -> set[str]:
    return {"ID_" + h.lower() for h in _ID.findall(str(text or ""))}


def rsna_blocklist(index_path: Path = RSNA_INDEX) -> set[str]:
    """Study ids and slice SOP stems from the local RSNA zip index."""
    index = json.loads(Path(index_path).read_text())
    out = {normalize_id(uid) for uid in index}
    for members in index.values():
        for m in members:
            stem = Path(str(m)).stem
            out.add(normalize_id(stem))
            out |= ids_in(stem)
    out.discard("")
    return out


def is_sinogram(file_name: str) -> bool:
    return bool(_SINO.search(str(file_name or "")))


def is_dicom(file_name: str) -> bool:
    n = str(file_name or "")
    if is_sinogram(n):
        return False
    return bool(_DICOM.search(n)) or n.lower().endswith(".dcm")


def patient_from_name(file_name: str, known: set[str] | None = None) -> str:
    """First path component that looks like a patient id (or a known labels id)."""
    parts = Path(str(file_name).replace("\\", "/")).parts
    if known:
        for p in parts:
            if p in known or normalize_id(p) in known:
                return p
    skip = {"data", "dicom", "dicoms", "image", "images", "sinogram", "sinograms", "npy", "raw"}
    for p in parts[:-1]:
        if p.lower().strip(".") not in skip:
            return p
    return Path(str(file_name)).stem


def hits_rsna(file_name: str, patient_id: str, blocklist: set[str]) -> bool:
    tokens = {normalize_id(patient_id)} | ids_in(file_name) | ids_in(patient_id)
    tokens |= {normalize_id(p) for p in Path(str(file_name).replace("\\", "/")).parts}
    tokens.discard("")
    return bool(tokens & blocklist)


def classify(files: pd.DataFrame, labels: pd.DataFrame, blocklist: set[str]) -> pd.DataFrame:
    """Add kind/keep/drop_reason. One row per file in the Redivis index."""
    known = set(labels["patient_id"].astype(str)) if "patient_id" in labels.columns else set()
    out = files.copy()
    name = out["file_name"].astype(str)
    out["patient_id"] = [patient_from_name(n, known) for n in name]
    lab = labels.rename(columns={"label": "scan_label"}) if "label" in labels.columns else labels
    if "patient_id" in lab.columns:
        out = out.merge(lab, on="patient_id", how="left")
    out["kind"] = ["sinogram" if is_sinogram(n) else "dicom" if is_dicom(n) else "other" for n in name]
    rsna = [hits_rsna(n, p, blocklist) for n, p in zip(name, out["patient_id"].astype(str))]
    leaked = {p for p, hit in zip(out["patient_id"].astype(str), rsna) if hit}
    out["rsna_overlap"] = [bool(hit or p in leaked) for hit, p in zip(rsna, out["patient_id"].astype(str))]
    reason = []
    keep = []
    for k, leak in zip(out["kind"], rsna):
        if k == "sinogram":
            reason.append("sinogram")
            keep.append(False)
        elif leak:
            reason.append("rsna")
            keep.append(False)
        elif k != "dicom":
            reason.append("not_dicom")
            keep.append(False)
        else:
            reason.append("")
            keep.append(True)
    out["keep"] = keep
    out["drop_reason"] = reason
    return out


def redivis_root():
    import redivis
    return redivis.table(f"aimi.{DATASET}.{DATA_TABLE}").to_directory()


def iter_recon_files(root=None, limit: int | None = None):
    """Yield reconstructed-image DICOM File objects. Never walks ``sinogram/``."""
    root = root or redivis_root()
    n = 0
    for batch in root.list(mode="directories"):
        if not str(batch.name).startswith("batch_"):
            continue
        for series in batch.list(mode="directories"):
            kids = {c.name: c for c in series.list(mode="directories")}
            recon = kids.get("reconstructed_image")
            if recon is None:
                continue
            for f in recon.list(mode="files"):
                if is_sinogram(f.name) or not is_dicom(f.name):
                    continue
                yield f
                n += 1
                if limit is not None and n >= limit:
                    return


def census_series(root=None, n_series: int = 8) -> dict:
    """Slice counts and byte sizes for a few series (DICOMs vs sinograms, no download)."""
    root = root or redivis_root()
    rows = []
    for batch in root.list(mode="directories"):
        if not str(batch.name).startswith("batch_"):
            continue
        for series in batch.list(mode="directories"):
            kids = {c.name: c for c in series.list(mode="directories")}
            rec = {"series": f"{batch.name}/{series.name}", "n_dcm": 0, "bytes_dcm": 0,
                   "n_sino": 0, "bytes_sino": 0}
            if "reconstructed_image" in kids:
                for f in kids["reconstructed_image"].list(mode="files"):
                    rec["n_dcm"] += 1
                    rec["bytes_dcm"] += int(f.size or 0)
            if "sinogram" in kids:
                for f in kids["sinogram"].list(mode="files"):
                    rec["n_sino"] += 1
                    rec["bytes_sino"] += int(f.size or 0)
            rows.append(rec)
            if len(rows) >= n_series:
                df = pd.DataFrame(rows)
                return {
                    "n_series_sampled": len(df),
                    "slices_per_series": float(df.n_dcm.mean()),
                    "bytes_per_slice": float(df.bytes_dcm.sum() / max(df.n_dcm.sum(), 1)),
                    "bytes_dcm_per_series": float(df.bytes_dcm.mean()),
                    "bytes_sino_per_series": float(df.bytes_sino.mean()),
                    "rows": df,
                }
    return {}


def check_ct_dicom(path: Path) -> dict:
    import pydicom
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    mod = str(getattr(ds, "Modality", "") or "")
    sop = str(getattr(ds, "SOPClassUID", "") or "")
    rows, cols = int(getattr(ds, "Rows", 0) or 0), int(getattr(ds, "Columns", 0) or 0)
    ok = mod.upper() == "CT" and path.suffix.lower() == ".dcm"
    return {"path": str(path), "ok": ok, "Modality": mod, "SOPClassUID": sop,
            "Rows": rows, "Columns": cols, "StudyInstanceUID": str(getattr(ds, "StudyInstanceUID", "") or "")}


def sample_dicoms(n: int = 100, dest: Path | None = None, n_series_census: int = 8) -> None:
    """Download ``n`` reconstructed DICOMs onto pool, verify they are CT, estimate full DICOM size."""
    dest = Path(dest or (OUT / "sample"))
    dest.mkdir(parents=True, exist_ok=True)
    root = redivis_root()
    lab = root.get("labels.csv")
    lab_path = dest / "labels.csv"
    print(f"[sinoct] labels.csv {lab.size} B", flush=True)
    lab.download(str(lab_path), overwrite=True, progress=False)
    labels = pd.read_csv(lab_path)
    print(f"[sinoct] labels {len(labels)} rows cols={list(labels.columns)}", flush=True)
    print(labels.head(3).to_string(), flush=True)
    block = rsna_blocklist()
    leaked = [p for p in labels["patient_id"].astype(str) if hits_rsna("", p, block)]
    print(f"[sinoct] RSNA id overlap in labels: {len(leaked)} / {len(labels)}", flush=True)

    print(f"[sinoct] census {n_series_census} series (no download)", flush=True)
    stats = census_series(root, n_series_census)
    n_scan = len(labels)
    dcm_tb = stats["bytes_dcm_per_series"] * n_scan / 1e12
    sino_tb = stats["bytes_sino_per_series"] * n_scan / 1e12
    print(f"[sinoct] ~{stats['slices_per_series']:.1f} slices/series, "
          f"{stats['bytes_per_slice']/1e3:.0f} KB/slice", flush=True)
    print(f"[sinoct] estimated reconstructed DICOMs for {n_scan} scans: {dcm_tb*1e3:.0f} GB "
          f"(sinograms, not downloaded: {sino_tb*1e3:.0f} GB)", flush=True)
    stats["rows"].to_csv(dest / "series_census.csv", index=False)

    got = []
    for f in iter_recon_files(root, limit=n):
        rel = Path(f.path)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            f.download(str(out), overwrite=False, progress=False)
        got.append(out)
        if len(got) % 20 == 0:
            print(f"[sinoct] downloaded {len(got)}/{n}", flush=True)
    rows = [check_ct_dicom(p) for p in got]
    chk = pd.DataFrame(rows)
    chk.to_csv(dest / "dicom_check.csv", index=False)
    n_ok = int(chk.ok.sum()) if len(chk) else 0
    nbytes = sum(p.stat().st_size for p in got)
    print(f"[sinoct] sample {n_ok}/{len(got)} CT DICOMs, {nbytes/1e6:.1f} MB on disk -> {dest}", flush=True)
    if n_ok != len(got):
        bad = chk.loc[~chk.ok, ["path", "Modality", "SOPClassUID"]]
        print(bad.to_string(), flush=True)
        raise SystemExit("sample contained non-CT files")


def pull_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    import redivis
    ds = redivis.user("aimi").dataset(DATASET)
    access = (ds.get().properties or {}).get("accessLevel", "")
    if access == "metadata":
        raise SystemExit(
            "SinoCT Redivis access is metadata-only on this host. Request data access at "
            "https://stanford.redivis.com/datasets/eznj-8hywfc0mq and set REDIVIS_API_TOKEN."
        )
    print("[sinoct] downloading labels (9779 patients) and file index (~357k files)", flush=True)
    labels = ds.table(LABELS_TABLE).to_pandas_dataframe()
    files = ds.table(DATA_TABLE).to_pandas_dataframe()
    return files, labels


def write_blocklist(path: Path, blocklist: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(blocklist)) + "\n")
    print(f"[sinoct] RSNA blocklist {len(blocklist)} ids -> {path}", flush=True)


def write_lists(classified: pd.DataFrame, labels: pd.DataFrame, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    classified.to_parquet(dest / "file_index.parquet", index=False)
    labels.to_csv(dest / "labels.csv", index=False)
    keep = classified[classified.keep]
    drop = classified[~classified.keep]
    keep.to_csv(dest / "keep_files.csv", index=False)
    drop.to_csv(dest / "drop_files.csv", index=False)
    agg = dict(n_dicom=("file_id", "size"), bytes=("size", "sum"))
    if "scan_label" in keep.columns:
        agg["scan_label"] = ("scan_label", "first")
    patients = keep.groupby("patient_id", as_index=False).agg(**agg)
    leaked = classified.loc[classified.rsna_overlap, "patient_id"].drop_duplicates()
    patients.to_csv(dest / "keep_patients.csv", index=False)
    pd.DataFrame({"patient_id": leaked}).to_csv(dest / "drop_rsna_patients.csv", index=False)
    print(
        f"[sinoct] files={len(classified)} keep={int(classified.keep.sum())} "
        f"sinogram={(classified.kind == 'sinogram').sum()} rsna_files={int(classified.rsna_overlap.sum())} "
        f"keep_patients={len(patients)} rsna_patients={len(leaked)}",
        flush=True,
    )
    print(f"[sinoct] lists -> {dest}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--blocklist-only", action="store_true")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="download N reconstructed DICOMs onto pool and verify they are CT")
    ap.add_argument("--overlap", action="store_true",
                    help="fingerprint RSNA pixels, probe SinoCT middles, write keep/drop lists")
    ap.add_argument("--fingerprint-rsna", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="download and hash middle reconstructed slices (sinograms skipped)")
    ap.add_argument("--match", action="store_true")
    ap.add_argument("--pull", action="store_true",
                    help="download reconstructed DICOMs for keep_series.txt (no sinograms, no RSNA drops)")
    ap.add_argument("--local-only", action="store_true",
                    help="hash DICOMs already on pool; do not call Redivis")
    ap.add_argument("--slices-per-series", type=int, default=1)
    ap.add_argument("--hamming", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap RSNA studies / SinoCT series (smoke)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    ap.add_argument("--rsna-index", type=Path, default=RSNA_INDEX)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    blocklist = rsna_blocklist(a.rsna_index)
    write_blocklist(a.out / "rsna_blocklist.txt", blocklist)
    if a.blocklist_only:
        return
    if a.sample:
        return sample_dicoms(a.sample, a.out / "sample")
    overlap = a.overlap or a.fingerprint_rsna or a.probe or a.match or a.pull
    if overlap:
        from brain.datasets.sinoct_overlap import fingerprint_rsna, match_fingerprints, probe_series, pull_keep, run_overlap
        lim = a.limit or None
        if a.overlap:
            run_overlap(dest=a.out, workers=a.workers, n_slices=a.slices_per_series,
                        limit=lim, hamming=a.hamming, local_only=a.local_only, pull=a.pull)
            return
        if a.fingerprint_rsna:
            fingerprint_rsna(dest=a.out, workers=a.workers, limit=lim)
        if a.probe:
            probe_series(dest=a.out, n_slices=a.slices_per_series, workers=min(8, a.workers),
                         limit=lim, local_only=a.local_only)
        if a.match:
            match_fingerprints(dest=a.out, hamming=a.hamming)
        if a.pull:
            pull_keep(dest=a.out, workers=min(8, a.workers))
        return
    files, labels = pull_tables()
    files.to_parquet(a.out / "redivis_files.parquet", index=False)
    labels.to_csv(a.out / "redivis_labels.csv", index=False)
    write_lists(classify(files, labels, blocklist), labels, a.out)


if __name__ == "__main__":
    main()
