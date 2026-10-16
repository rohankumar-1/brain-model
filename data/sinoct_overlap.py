"""RSNA ↔ SinoCT overlap by pixel content (IDs were re-anonymized and never match).

Each RSNA zip member is decoded to HU, center-cropped, air-padded, then hashed.
SinoCT series are probed with the middle reconstructed slice (sinograms never
touched). A series is dropped on an exact or 32×32 HU-thumbnail hit. Average-hash
distance is recorded only as a diagnostic (head CTs share the same silhouette).

    python -m data.download_sinoct --overlap
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from data.prep.sources import DATASETS, SCRATCH

RSNA_ZIP = DATASETS / "rsna" / "rsna-ihd-dataset.zip"
RSNA_OFFSETS = SCRATCH / "work" / "rsna_ingest" / "member_offsets.json"
OUT = DATASETS / "sinoct"
AIR = -1024
CROP = 384
HAMMING = 8
_IMAGE_N = re.compile(r"(\d+)")


def hu_from_dataset(ds) -> np.ndarray:
    arr = np.asarray(ds.pixel_array)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    hu = arr.astype(np.float32) * np.float32(slope) + np.float32(intercept)
    return np.clip(np.rint(hu), -32768, 32767).astype(np.int16)


def canon_plane(hu: np.ndarray, crop: int = CROP, air_below: int = -500) -> np.ndarray:
    """Clip HU, flatten FOV padding to air, center-crop so pad rings cannot change the hash."""
    x = np.clip(np.asarray(hu), AIR, 3071).astype(np.int16, copy=True)
    if x.ndim > 2:
        x = x.reshape(x.shape[-2], x.shape[-1])
    x[x < air_below] = AIR
    h, w = x.shape
    side = min(int(crop), h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return np.ascontiguousarray(x[y0:y0 + side, x0:x0 + side])


def block_mean(plane: np.ndarray, size: int) -> np.ndarray:
    h, w = plane.shape
    rh, rw = h // size, w // size
    if rh == 0 or rw == 0:
        return plane.astype(np.float32)
    a = plane[: rh * size, : rw * size]
    return a.reshape(size, rh, size, rw).mean(axis=(1, 3))


def exact_hash(plane: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(plane).tobytes()).hexdigest()


def coarse_hash(plane: np.ndarray, size: int = 32) -> str:
    q = np.rint(block_mean(plane, size)).astype(np.int16)
    return hashlib.sha1(np.ascontiguousarray(q).tobytes()).hexdigest()


def ahash64(plane: np.ndarray, size: int = 8) -> int:
    small = block_mean(plane, size)
    bits = small >= small.mean()
    val = 0
    for b in np.asarray(bits).ravel():
        val = (val << 1) | int(b)
    return val


def fingerprints_from_hu(hu: np.ndarray) -> dict:
    plane = canon_plane(hu)
    return {"exact": exact_hash(plane), "coarse": coarse_hash(plane), "ahash": f"{ahash64(plane):016x}"}


def fingerprints_from_bytes(blob: bytes) -> dict | None:
    import pydicom
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = pydicom.dcmread(io.BytesIO(blob), force=True)
    try:
        hu = hu_from_dataset(ds)
    except Exception:
        return None
    fp = fingerprints_from_hu(hu)
    ps = getattr(ds, "PixelSpacing", None)
    fp.update({
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "cols": int(getattr(ds, "Columns", 0) or 0),
        "pixel_spacing": float(ps[0]) if ps is not None else 0.0,
    })
    return fp


def fingerprints_from_file(path: Path) -> dict | None:
    return fingerprints_from_bytes(Path(path).read_bytes())


def _fp_member(fd: int, study_id: str, member) -> dict | None:
    from data.prep.dicom import _read_member
    name, offset, csize, ctype = member[0], int(member[1]), int(member[2]), int(member[3])
    try:
        blob = _read_member(fd, offset, csize, ctype)
        fp = fingerprints_from_bytes(blob)
    except Exception:
        return None
    if fp is None:
        return None
    fp["study_id"] = study_id
    fp["member"] = Path(str(name)).name
    return fp


def _write_part(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(path)


def load_rsna_fingerprints(dest: Path) -> pd.DataFrame:
    dest = Path(dest)
    parts = sorted((dest / "rsna_fp").glob("part_*.parquet"))
    merged = dest / "rsna_fingerprints.parquet"
    if parts:
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        dest.mkdir(parents=True, exist_ok=True)
        tmp = merged.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        tmp.replace(merged)
        return df
    if merged.exists() and merged.stat().st_size > 0:
        return pd.read_parquet(merged)
    return pd.DataFrame()


def fingerprint_rsna(
    zip_path: Path = RSNA_ZIP,
    offsets_path: Path = RSNA_OFFSETS,
    dest: Path = OUT,
    workers: int = 16,
    limit: int | None = None,
    checkpoint_slices: int = 8000,
) -> Path:
    """Hash every RSNA zip member in stored-offset order. Resumes from ``rsna_fp/part_*.parquet``."""
    print(f"[overlap] loading offsets {offsets_path}", flush=True)
    offsets = json.loads(Path(offsets_path).read_text())
    uids = sorted(offsets)
    if limit:
        uids = uids[:limit]
    part_dir = Path(dest) / "rsna_fp"
    part_dir.mkdir(parents=True, exist_ok=True)
    done_members: set[str] = set()
    n_done_slices = 0
    for p in part_dir.glob("part_*.parquet"):
        col = pd.read_parquet(p, columns=["member"])
        done_members |= set(col["member"].astype(str))
        n_done_slices += len(col)
    items = []
    for uid in uids:
        for m in offsets[uid]:
            if Path(str(m[0])).name not in done_members:
                items.append((uid, m))
    items.sort(key=lambda t: int(t[1][1]))
    print(f"[overlap] RSNA studies {len(uids)} members={len(items)} already={n_done_slices} "
          f"workers={workers} (zip-offset order)", flush=True)
    if not items:
        load_rsna_fingerprints(Path(dest))
        return Path(dest) / "rsna_fingerprints.parquet"

    fd = os.open(str(zip_path), os.O_RDONLY)
    n_part = max((int(p.stem.split("_")[-1]) for p in part_dir.glob("part_*.parquet")), default=-1) + 1
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for start in range(0, len(items), checkpoint_slices):
                batch = items[start:start + checkpoint_slices]
                got = list(ex.map(lambda t: _fp_member(fd, t[0], t[1]), batch))
                rows = [r for r in got if r]
                _write_part(rows, part_dir / f"part_{n_part:04d}.parquet")
                n_done_slices += len(rows)
                dt = max(time.time() - t0, 1e-6)
                print(f"[overlap] RSNA checkpoint {min(start + len(batch), len(items))}/{len(items)} "
                      f"part_{n_part:04d} +{len(rows)} skip={len(got) - len(rows)} "
                      f"total_slices={n_done_slices} {n_done_slices / dt:.0f} slices/s", flush=True)
                n_part += 1
    finally:
        os.close(fd)
    merged = load_rsna_fingerprints(Path(dest))
    print(f"[overlap] RSNA fingerprints {len(merged)} slices -> {dest / 'rsna_fingerprints.parquet'}", flush=True)
    return Path(dest) / "rsna_fingerprints.parquet"


def series_from_path(path: Path) -> str:
    for part in Path(path).parts:
        if part.startswith("series_"):
            return part
    return Path(path).parent.parent.name


def batch_from_path(path: Path) -> str:
    for part in Path(path).parts:
        if part.startswith("batch_"):
            return part
    return ""


def _slice_key(name: str) -> int:
    m = _IMAGE_N.search(Path(name).stem)
    return int(m.group(1)) if m else 0


def pick_probe_files(files, n_slices: int = 1):
    """Middle file, or quartile files when ``n_slices`` > 1."""
    ranked = sorted(files, key=lambda f: _slice_key(getattr(f, "name", str(f))))
    if not ranked:
        return []
    if n_slices <= 1:
        return [ranked[len(ranked) // 2]]
    n = min(n_slices, len(ranked))
    return [ranked[round(i * (len(ranked) - 1) / max(n - 1, 1))] for i in range(n)]


def hash_local_dicoms(roots: list[Path], dest: Path) -> pd.DataFrame:
    paths, seen = [], set()
    for root in roots:
        if root is None or not Path(root).exists():
            continue
        for p in sorted(Path(root).rglob("*.dcm")):
            r = p.resolve()
            if r in seen:
                continue
            seen.add(r)
            paths.append(p)
    rows = []
    for i, p in enumerate(paths, 1):
        fp = fingerprints_from_file(p)
        if fp is None:
            continue
        fp.update({"series": series_from_path(p), "batch": batch_from_path(p),
                   "path": str(p), "file": p.name, "source": "local"})
        rows.append(fp)
        if i % 50 == 0:
            print(f"[overlap] hashed local {i}/{len(paths)}", flush=True)
    df = pd.DataFrame(rows)
    if len(df):
        dest.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest / "probe_local.csv", index=False)
    print(f"[overlap] local DICOMs hashed {len(df)} / {len(paths)}", flush=True)
    return df


def _download_one(rec: dict, dest: Path) -> dict | None:
    out = dest / rec["rel"]
    out.parent.mkdir(parents=True, exist_ok=True)
    f = rec.get("file")
    if f is None:
        f = rec["root"].get(rec["rel"])
    if not out.exists() or out.stat().st_size < 1024:
        f.download(str(out), overwrite=True, progress=False)
    fp = fingerprints_from_file(out)
    if fp is None:
        return None
    fp.update({"series": rec["series"], "batch": rec["batch"], "path": str(out),
               "file": out.name, "n_series_files": rec["n_files"], "source": "probe"})
    return fp


def _list_inventory(dest: Path, n_slices: int, limit: int | None) -> pd.DataFrame:
    from data.download_sinoct import is_dicom, is_sinogram, redivis_root

    inv_path = dest / "series_inventory.csv"
    if inv_path.exists() and inv_path.stat().st_size > 0:
        inv = pd.read_csv(inv_path)
        print(f"[overlap] inventory {len(inv)} probe files from {inv_path}", flush=True)
        return inv

    print("[overlap] listing SinoCT reconstructed series (no sinograms)", flush=True)
    root = redivis_root()
    rows, recon_rows, series_seen = [], [], 0
    stop = False
    for batch in root.list(mode="directories"):
        if stop or not str(batch.name).startswith("batch_"):
            continue
        for series in batch.list(mode="directories"):
            if limit is not None and series_seen >= limit:
                stop = True
                break
            kids = {c.name: c for c in series.list(mode="directories")}
            recon = kids.get("reconstructed_image")
            if recon is None:
                continue
            files = [f for f in recon.list(mode="files") if is_dicom(f.name) and not is_sinogram(f.name)]
            for f in files:
                recon_rows.append({
                    "batch": batch.name, "series": series.name, "file": f.name,
                    "size": int(getattr(f, "size", 0) or 0),
                    "rel": str(Path(batch.name) / series.name / "reconstructed_image" / f.name),
                })
            for f in pick_probe_files(files, n_slices):
                rows.append({
                    "batch": batch.name, "series": series.name, "file": f.name,
                    "n_files": len(files),
                    "rel": str(Path(batch.name) / series.name / "reconstructed_image" / f.name),
                })
            series_seen += 1
            if series_seen % 200 == 0:
                print(f"[overlap] listed {series_seen} series", flush=True)
    inv = pd.DataFrame(rows)
    dest.mkdir(parents=True, exist_ok=True)
    inv.to_csv(inv_path, index=False)
    pd.DataFrame(recon_rows).to_csv(dest / "recon_files.csv", index=False)
    print(f"[overlap] inventory {len(inv)} probe files / {series_seen} series "
          f"recon={len(recon_rows)} -> {inv_path}", flush=True)
    return inv


def probe_series(
    dest: Path = OUT,
    n_slices: int = 1,
    workers: int = 4,
    limit: int | None = None,
    local_only: bool = False,
) -> Path:
    """Download ``n_slices`` reconstructed DICOMs per SinoCT series and hash them."""
    dest = Path(dest)
    probe_dir = dest / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    out_csv = dest / "probe_slices.csv"
    prev = pd.read_csv(out_csv) if out_csv.exists() and out_csv.stat().st_size > 0 else pd.DataFrame()
    have = set(prev["series"].astype(str)) if len(prev) else set()
    have_paths = set(prev["path"].astype(str)) if len(prev) and "path" in prev.columns else set()
    local_roots = [dest / "sample", probe_dir, OUT / "sample", OUT / "probe"]
    extra = hash_local_dicoms([p for p in local_roots if p.exists()], dest)
    if len(extra) and have_paths:
        extra = extra[~extra["path"].astype(str).isin(have_paths)]
    if len(extra):
        prev = pd.concat([prev, extra], ignore_index=True) if len(prev) else extra
        have |= set(prev["series"].astype(str))
    if local_only:
        prev.to_csv(out_csv, index=False)
        return out_csv

    from data.download_sinoct import redivis_root

    inv = _list_inventory(dest, n_slices, limit)
    if limit is not None:
        keep = list(dict.fromkeys(inv["series"].astype(str)))[:limit]
        inv = inv[inv["series"].astype(str).isin(keep)]
    pending = inv[~inv["series"].astype(str).isin(have)]
    print(f"[overlap] downloading {len(pending)} probe DICOMs -> {probe_dir} (have={len(have)})", flush=True)
    root = redivis_root()
    jobs = []
    for rec in pending.to_dict("records"):
        jobs.append({
            "batch": rec["batch"], "series": rec["series"], "n_files": int(rec["n_files"]),
            "rel": rec["rel"], "file": None, "root": root,
        })
    rows = list(prev.to_dict("records")) if len(prev) else []
    workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_download_one, j, probe_dir) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec:
                rows.append(rec)
            if k % 50 == 0 or k == len(futs):
                print(f"[overlap] probe {k}/{len(futs)} hashed={len(rows)}", flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    n_series = pd.DataFrame(rows)["series"].nunique() if rows else 0
    print(f"[overlap] probe slices={len(rows)} series={n_series} -> {out_csv}", flush=True)
    return out_csv


def _index_hashes(fp: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], np.ndarray, np.ndarray]:
    exact, coarse = {}, {}
    for h, s in zip(fp["exact"].astype(str), fp["study_id"].astype(str)):
        exact.setdefault(h, s)
    for h, s in zip(fp["coarse"].astype(str), fp["study_id"].astype(str)):
        coarse.setdefault(h, s)
    ahash = np.fromiter((int(x, 16) for x in fp["ahash"].astype(str)), dtype=np.uint64, count=len(fp))
    studies = fp["study_id"].astype(str).to_numpy()
    return exact, coarse, ahash, studies


def score_probe(probe: pd.DataFrame, fp: pd.DataFrame, hamming: int = HAMMING) -> pd.DataFrame:
    """One row per SinoCT series: exact/coarse hit and nearest RSNA average-hash."""
    if probe is None or len(probe) == 0:
        return pd.DataFrame(columns=["series", "batch", "n_probe", "exact_study",
                                    "coarse_study", "ahash_study", "min_hamming", "drop", "reason"])
    exact, coarse, gallery, gal_studies = _index_hashes(fp)
    rows = []
    for series, g in probe.groupby("series", sort=True):
        exact_hit = ""
        coarse_hit = ""
        min_d = 64
        near = ""
        for rec in g.to_dict("records"):
            e, c, a = rec["exact"], rec["coarse"], rec["ahash"]
            exact_hit = exact_hit or exact.get(str(e), "")
            coarse_hit = coarse_hit or coarse.get(str(c), "")
            d = np.bitwise_count(gallery ^ np.uint64(int(str(a), 16)))
            i = int(d.argmin())
            if int(d[i]) < min_d:
                min_d = int(d[i])
                near = str(gal_studies[i])
        drop = bool(exact_hit) or bool(coarse_hit)
        why = "exact" if exact_hit else "coarse" if coarse_hit else ""
        rows.append({
            "series": series,
            "batch": g["batch"].iloc[0] if "batch" in g.columns else "",
            "n_probe": len(g),
            "exact_study": exact_hit,
            "coarse_study": coarse_hit,
            "ahash_study": near,
            "min_hamming": min_d,
            "drop": drop,
            "reason": why,
        })
    return pd.DataFrame(rows)


def match_fingerprints(
    dest: Path = OUT,
    hamming: int = HAMMING,
    labels_path: Path | None = None,
) -> pd.DataFrame:
    dest = Path(dest)
    fp = load_rsna_fingerprints(dest)
    probe_path = dest / "probe_slices.csv"
    if not len(fp):
        raise SystemExit("no RSNA fingerprints; run --fingerprint-rsna first")
    if not probe_path.exists():
        raise SystemExit("no probe_slices.csv; run --probe first")
    probe = pd.read_csv(probe_path) if probe_path.stat().st_size else pd.DataFrame()
    if len(probe) == 0:
        print("[overlap] probe_slices.csv is empty", flush=True)
        return score_probe(probe, fp, hamming=hamming)
    print(f"[overlap] matching {probe['series'].nunique()} SinoCT series against "
          f"{len(fp)} RSNA slices (hamming<={hamming})", flush=True)
    scores = score_probe(probe, fp, hamming=hamming)
    lab = Path(labels_path or (dest / "sample" / "labels.csv"))
    if lab.exists():
        labels = pd.read_csv(lab)
        if "patient_id" in labels.columns:
            scores = scores.merge(labels.rename(columns={"patient_id": "series"}), on="series", how="left")
    scores.to_csv(dest / "overlap_scores.csv", index=False)
    dropped = scores.loc[scores["drop"], "series"].astype(str)
    keep = scores.loc[~scores["drop"], "series"].astype(str)
    (dest / "drop_rsna_series.txt").write_text("\n".join(sorted(dropped)) + ("\n" if len(dropped) else ""))
    (dest / "keep_series.txt").write_text("\n".join(sorted(keep)) + ("\n" if len(keep) else ""))
    print(
        f"[overlap] series={len(scores)} drop={int(scores['drop'].sum())} "
        f"exact={(scores.reason == 'exact').sum()} coarse={(scores.reason == 'coarse').sum()} "
        f"keep={int((~scores['drop']).sum())}",
        flush=True,
    )
    if len(scores):
        qs = scores["min_hamming"].quantile([0.0, 0.01, 0.05, 0.1, 0.5, 0.9]).to_dict()
        n_close = int((scores["min_hamming"] <= hamming).sum())
        print(f"[overlap] min_hamming quantiles {qs}; ahash<={hamming} (not dropped) {n_close}", flush=True)
    print(f"[overlap] lists -> {dest}/drop_rsna_series.txt {dest}/keep_series.txt", flush=True)
    return scores


def read_series_list(path: Path) -> set[str]:
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return set()
    return {ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()}


def select_pull_files(recon: pd.DataFrame, keep: set[str], drop: set[str]) -> pd.DataFrame:
    """Reconstructed DICOMs for keep series only. Drop list always wins."""
    s = recon["series"].astype(str)
    out = recon[s.isin(keep) & ~s.isin(drop)]
    return out.copy()


def _save_file(rec: dict, dest: Path) -> tuple[str, int]:
    out = dest / rec["rel"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1024:
        return str(out), out.stat().st_size
    rec["file"].download(str(out), overwrite=True, progress=False)
    return str(out), out.stat().st_size


def pull_keep(
    dest: Path = OUT,
    workers: int = 8,
    recon_dir: Path | None = None,
) -> Path:
    """Download reconstructed DICOMs for keep_series.txt. Never sinograms; never drop_rsna_series."""
    dest = Path(dest)
    keep = read_series_list(dest / "keep_series.txt")
    drop = read_series_list(dest / "drop_rsna_series.txt")
    if not keep:
        raise SystemExit("no keep_series.txt; run --overlap first so RSNA hits are filtered")
    recon_path = dest / "recon_files.csv"
    if recon_path.exists() and recon_path.stat().st_size > 0:
        recon = pd.read_csv(recon_path)
    else:
        inv = _list_inventory(dest, n_slices=1, limit=None)
        recon_path = dest / "recon_files.csv"
        if recon_path.exists():
            recon = pd.read_csv(recon_path)
        else:
            recon = inv.rename(columns={"n_files": "size"})
            recon["size"] = 0
    files = select_pull_files(recon, keep, drop)
    skipped = recon["series"].astype(str).isin(drop).sum()
    nbytes = int(files["size"].sum()) if "size" in files.columns else 0
    out_root = Path(recon_dir or (dest / "reconstructed"))
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[sinoct] pull keep={len(keep)} drop={len(drop)} files={len(files)} "
          f"~{nbytes / 1e9:.1f} GB (listed) skip_rsna_files={int(skipped)} -> {out_root}", flush=True)

    from data.download_sinoct import redivis_root
    root = redivis_root()
    jobs = []
    for rec in files.to_dict("records"):
        jobs.append({"rel": rec["rel"], "file": root.get(rec["rel"])})
    n_ok, n_bytes = 0, 0
    workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_save_file, j, out_root) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            _, sz = fut.result()
            n_ok += 1
            n_bytes += sz
            if k % 200 == 0 or k == len(futs):
                print(f"[sinoct] pull {k}/{len(futs)} {n_bytes / 1e9:.1f} GB", flush=True)
    print(f"[sinoct] pulled {n_ok} DICOMs, {n_bytes / 1e9:.1f} GB -> {out_root}", flush=True)
    return out_root


def run_overlap(
    dest: Path = OUT,
    workers: int = 16,
    n_slices: int = 1,
    limit: int | None = None,
    hamming: int = HAMMING,
    local_only: bool = False,
    skip_fingerprint: bool = False,
    pull: bool = False,
) -> pd.DataFrame:
    if not skip_fingerprint:
        fingerprint_rsna(dest=dest, workers=workers, limit=limit)
    probe_series(dest=dest, n_slices=n_slices, workers=min(8, workers), limit=limit, local_only=local_only)
    scores = match_fingerprints(dest=dest, hamming=hamming)
    if pull:
        pull_keep(dest=dest, workers=min(8, workers))
    return scores
