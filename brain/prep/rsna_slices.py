"""RSNA ICH slice labels on the stored grid (2026-11-01): the competition labels every axial slice with its
haemorrhage subtypes (``stage_2_train.csv``, ``ID_<slice>_<subtype>``); the study-level tasks keep only their OR. Here
each stored study keeps its slices' positions and labels, so training can ask *where* along the scan the bleed is
(``brain.multitask.losses.slab_mil_loss``).

Per slice: ``s = n . ImagePositionPatient`` with ``n`` the slice normal (row x column direction of
ImageOrientationPatient), its position along the acquisition axis, so gantry tilt needs no special case. Per study:
the map from stored voxel index (z, y, x) to that axis, ``s = g . (z, y, x) + c``, composed of the stored grid
(origin, direction, spacing) and the registration prep resampled with (``native.stored_transform``, stored physical
-> source physical; both LPS). A stored voxel takes the labels of the nearest slice when it lies within half the
slice spacing of it, and is unlabelled otherwise.

Headers come from the pool zip by byte offset (``work/rsna_ingest/member_offsets.json``): one ``pread`` of the first
few KB of each member, inflated only as far as the header needs. Writes ``labels/rsna_slices/rsna_slices.parquet``
(volume_id, g_z, g_y, g_x, c, spacing, s: sorted positions, bits: per slice bit k = RSNA_LABELS[k], 255 = no label)
and a QC table beside it (``n . native_origin`` must sit on an end slice).

    python -m brain.prep.rsna_slices [--limit 50] [--threads 16]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import DATASETS, PROCESSED_ROOT, SCRATCH
from brain.downstream.tasks import RSNA_LABELS
from brain.prep import native, store

ZIP = DATASETS / "rsna" / "rsna-ihd-dataset.zip"
OFFSETS = SCRATCH / "work" / "rsna_ingest" / "member_offsets.json"
STORE = PROCESSED_ROOT
OUT = STORE / "labels" / "rsna_slices"
LABEL_CSV = "rsna-intracranial-hemorrhage-detection/stage_2_train.csv"
HEAD = 16384          # compressed bytes read per member: the header inflates from well inside this
NO_LABEL = 255


def _header(fd: int, offset: int, csize: int, ctype: int):
    """The DICOM header of one zip member (pixel data skipped)."""
    import pydicom

    buf = os.pread(fd, min(csize, HEAD) + 1024, offset)
    if buf[:4] != b"PK\x03\x04":
        raise ValueError(f"bad local header at {offset}")
    n, m = struct.unpack("<HH", buf[26:30])
    data = buf[30 + n + m:30 + n + m + min(csize, HEAD)]
    raw = data if ctype == zipfile.ZIP_STORED else zlib.decompressobj(-15).decompress(data)
    try:
        return pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True, force=True)
    except Exception:  # noqa: BLE001  header longer than the read: take the whole member
        full = os.pread(fd, csize + 1024, offset)[30 + n + m:30 + n + m + csize]
        full = full if ctype == zipfile.ZIP_STORED else zlib.decompress(full, -15)
        return pydicom.dcmread(io.BytesIO(full), stop_before_pixels=True, force=True)


def slice_labels() -> dict[str, int]:
    """Slice id (``ID_xxx``) -> label bits (bit k = ``RSNA_LABELS[k]``)."""
    bit = {k: i for i, k in enumerate(RSNA_LABELS)}
    out: dict[str, int] = {}
    with zipfile.ZipFile(ZIP) as zf, zf.open(LABEL_CSV) as src:
        src.readline()
        for raw in src:
            sid, lab = raw.decode().strip().split(",")
            slice_id, sub = sid.rsplit("_", 1)
            if sub in bit:
                out[slice_id] = out.get(slice_id, 0) | (int(lab) << bit[sub])
    return out


def index_map(attrs: dict, normal: np.ndarray) -> tuple[np.ndarray, float]:
    """(g, c) with ``s = g . (z, y, x) + c``: stored voxel index -> position along ``normal`` in source space."""
    tf = native.stored_transform(attrs)
    m = np.asarray(tf.GetMatrix(), float).reshape(3, 3)
    t = np.asarray(tf.GetTranslation(), float)
    # sitk AffineTransform: y = M (x - centre) + centre + t; prep's transforms keep the centre at 0
    ctr = np.asarray(tf.GetCenter(), float)
    o = np.asarray(attrs["grid_origin"], float)
    d = np.asarray(attrs["grid_direction"], float).reshape(3, 3)
    sp = np.asarray(attrs["grid_spacing"], float)
    a = m @ d @ np.diag(sp)                                  # (x, y, z) index -> source physical
    b = m @ (o - ctr) + ctr + t
    g_xyz = normal @ a
    return g_xyz[::-1].copy(), float(normal @ b)             # (z, y, x) order


def one(fd: int, vid: str, image_path: str, members: list, labels: dict[str, int], pool) -> tuple[dict, dict]:
    heads = list(pool.map(lambda mm: (mm[0], _header(fd, *mm[1:])), members))
    rows = []
    for name, ds in heads:
        if "ImagePositionPatient" not in ds or "ImageOrientationPatient" not in ds:
            continue
        iop = np.asarray(ds.ImageOrientationPatient, float)
        n = np.cross(iop[:3], iop[3:])
        rows.append((Path(name).stem, np.asarray(ds.ImagePositionPatient, float), n))
    if not rows:
        return {}, {"volume_id": vid, "ok": False, "why": "no geometry"}
    normal = np.median(np.stack([r[2] for r in rows]), 0)
    normal /= np.linalg.norm(normal)
    s = np.array([normal @ r[1] for r in rows])
    bits = np.array([labels.get(r[0], NO_LABEL) for r in rows], np.int32)
    order = np.argsort(s)
    s, bits = s[order], bits[order]
    keep = np.r_[True, np.diff(s) > 0.05]                    # repeated positions: first slice wins
    s, bits = s[keep], bits[keep]
    attrs = store.read_attrs(STORE / image_path, "image")
    g, c = index_map(attrs, normal)
    spacing = float(np.median(np.diff(s))) if len(s) > 1 else 5.0
    o_s = float(normal @ np.asarray(attrs["native_origin"], float))
    end = min(abs(o_s - s[0]), abs(o_s - s[-1]))
    nz = int(attrs["grid_size"][2])
    corners = np.array([g @ np.array([z, y, x]) + c for z in (0, nz - 1) for y in (0, 255) for x in (0, 255)])
    rec = {"volume_id": vid, "g_z": g[0], "g_y": g[1], "g_x": g[2], "c": c, "spacing": spacing,
           "s": s.astype(np.float32).tolist(), "bits": bits.astype(np.int16).tolist()}
    qc = {"volume_id": vid, "ok": True, "n_slices": len(s), "spacing": spacing, "origin_end_mm": end,
          "labelled": float(np.mean(bits != NO_LABEL)), "any_pos": int(((bits != NO_LABEL) & (bits & 1 > 0)).sum()),
          "s_lo": s[0], "s_hi": s[-1], "grid_s_lo": corners.min(), "grid_s_hi": corners.max()}
    return rec, qc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    v = pd.read_csv(STORE / "volumes.csv", low_memory=False, usecols=["volume_id", "dataset", "case_id", "image_path"])
    v = v[v["dataset"] == "rsna_ich"].drop_duplicates("volume_id")
    offsets = json.loads(OFFSETS.read_text())
    labels = slice_labels()
    print(f"[rsna-slices] {len(v)} stored studies, {len(labels)} labelled slices", flush=True)
    v = v.assign(key=v["case_id"].astype(str).str.removeprefix("RSNA-"))
    v = v[v["key"].isin(offsets)]
    if a.limit:
        v = v.head(a.limit)
    recs, qcs = [], []
    fd = os.open(ZIP, os.O_RDONLY)
    try:
        with ThreadPoolExecutor(a.threads) as pool:
            for i, r in enumerate(v.itertuples()):
                members = [m for m in offsets[r.key] if m[0].lower().endswith(".dcm")]
                try:
                    rec, qc = one(fd, r.volume_id, r.image_path, members, labels, pool)
                except Exception as e:  # noqa: BLE001
                    rec, qc = {}, {"volume_id": r.volume_id, "ok": False, "why": str(e)[:120]}
                if rec:
                    recs.append(rec)
                qcs.append(qc)
                if (i + 1) % 500 == 0:
                    print(f"[rsna-slices] {i + 1}/{len(v)}", flush=True)
    finally:
        os.close(fd)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{a.limit}" if a.limit else ""
    pd.DataFrame(recs).to_parquet(OUT / f"rsna_slices{suffix}.parquet", index=False)
    q = pd.DataFrame(qcs)
    q.to_csv(OUT / f"rsna_slices_qc{suffix}.csv", index=False)
    ok = q[q["ok"]]
    print(f"[rsna-slices] wrote {len(recs)} studies; failed {int((~q['ok']).sum())}")
    if len(ok):
        print(f"  labelled studies {(ok.labelled > 0).sum()}  median slices {ok.n_slices.median():.0f}  "
              f"spacing {ok.spacing.median():.2f} mm")
        print(f"  native origin on an end slice (<0.5 mm): {(ok.origin_end_mm < 0.5).mean():.3f}  "
              f"median {ok.origin_end_mm.median():.3f} mm")
        inside = (ok.s_lo >= ok.grid_s_lo - 1) & (ok.s_hi <= ok.grid_s_hi + 1)
        print(f"  slice range inside the stored grid: {inside.mean():.3f}")


if __name__ == "__main__":
    main()
