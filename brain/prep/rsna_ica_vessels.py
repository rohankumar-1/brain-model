"""Vessel and aneurysm-location masks for the stored RSNA-ICA series (2026-10-30), from the competition zip, without
re-prepping: each native-space label goes through the stored volume's own registration (``native.stored_transform``,
the map prep resampled the image with) onto its 1 mm grid.

* ``<case>.cow.zip``       the organisers' 13-class vessel segmentation (``segmentations/<uid>_cowseg.nii``, 178
                           series), resampled per label (linear on the label's indicator, then argmax; a voxel is
                           vessel when its winning share is >= ``COW_MIN``) so 2-3 mm vessels survive 0.5 -> 1 mm.
* ``<case>.aneurysm.zip``  a 4 mm-radius ball per localizer point (``train_localizers.csv``: pixel x, y on one DICOM
                           instance, ``f`` = frame of a multi-frame file), valued by location group (``ANEURYSM_GROUP``,
                           sides merged as in the registry's aneurysm tasks); an empty mask for every series with no
                           aneurysm. Positive series without a localizer get no mask.

Writes runs/diagnostics/rsna_ica_vessels_qc.csv (image intensity inside the vessels / at each point vs the brain).

    python -m brain.prep.rsna_ica_vessels --workers 12 [--limit 5]
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from brain.prep import geometry, native, store
from brain.prep.dicom import _read_member
from brain.prep.sources import DATASETS, SCRATCH

STORE = SCRATCH / "data" / "processed" / "slices_zarr"
ZIP = DATASETS / "rsna_ica" / "rsna-intracranial-aneurysm-detection.zip"
WORK = SCRATCH / "work" / "rsna_ica"
QC_OUT = Path(__file__).resolve().parents[2] / "runs" / "diagnostics" / "rsna_ica_vessels_qc.csv"
COW_MIN = 0.35
BALL_MM = 4.0
ANEURYSM_GROUP = {   # train.csv / localizer location -> registry aneurysm task (1-based, sides merged)
    "Left Infraclinoid Internal Carotid Artery": 1, "Right Infraclinoid Internal Carotid Artery": 1,
    "Left Supraclinoid Internal Carotid Artery": 2, "Right Supraclinoid Internal Carotid Artery": 2,
    "Left Middle Cerebral Artery": 3, "Right Middle Cerebral Artery": 3,
    "Anterior Communicating Artery": 4,
    "Left Anterior Cerebral Artery": 5, "Right Anterior Cerebral Artery": 5,
    "Left Posterior Communicating Artery": 6, "Right Posterior Communicating Artery": 6,
    "Basilar Tip": 7, "Other Posterior Circulation": 8,
}


def members() -> tuple[dict, dict]:
    """Zip offsets of the segmentations and of every series' DICOMs (cached under work/rsna_ica)."""
    seg_path = WORK / "seg_members.json"
    if not seg_path.exists():
        with zipfile.ZipFile(ZIP) as zf:
            seg = {i.filename: [i.filename, i.header_offset, i.compress_size, i.compress_type]
                   for i in zf.infolist() if i.filename.startswith("segmentations/")}
        seg_path.write_text(json.dumps(seg))
    return json.loads(seg_path.read_text()), json.loads((WORK / "series_members.json").read_text())


def _grid(attrs: dict) -> sitk.Image:
    ref = sitk.Image([int(v) for v in attrs["grid_size"]], sitk.sitkFloat32)
    ref.SetOrigin(tuple(attrs["grid_origin"]))
    ref.SetSpacing(tuple(attrs["grid_spacing"]))
    ref.SetDirection(tuple(attrs["grid_direction"]))
    return ref


def _summary(img: np.ndarray, brain: np.ndarray | None, sel: np.ndarray) -> dict:
    out = {"img_in": float(img[sel].mean()) if sel.any() else np.nan}
    if brain is not None and brain.any():
        out["img_brain"] = float(img[brain > 0].mean())
        out["in_brain"] = float((brain[sel] > 0).mean()) if sel.any() else np.nan
    return out


def cow(job: tuple) -> dict:
    uid, rel, image_path, member = job
    fd = os.open(ZIP, os.O_RDONLY)
    try:
        raw = _read_member(fd, *member[1:])
    finally:
        os.close(fd)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        p = Path(d) / "cowseg.nii"
        p.write_bytes(raw)
        lab = geometry.read(p)
    a = {k: store.squeeze_channel(x) for k, x in store.read_many(STORE / image_path, ("image", "brain")).items()}
    attrs = store.read_attrs(STORE / image_path, "image")
    ref, tf = _grid(attrs), native.stored_transform(attrs)
    arr = sitk.GetArrayFromImage(lab).round().astype(np.int16)
    ids = [int(i) for i in np.unique(arr) if i > 0]
    share = np.zeros((len(ids) + 1, *ref.GetSize()[::-1]), np.float32)
    for k, i in enumerate(ids, 1):
        ind = sitk.GetImageFromArray((arr == i).astype(np.float32))
        ind.CopyInformation(lab)
        share[k] = sitk.GetArrayFromImage(sitk.Resample(ind, ref, tf, sitk.sitkLinear, 0.0, sitk.sitkFloat32))
    win = share.argmax(0)
    out = np.where(share.max(0) >= COW_MIN, np.asarray([0, *ids], np.uint8)[win], 0).astype(np.uint8)
    store.write(store.path_for(STORE, rel, uid, "cow"), "mask", out)
    img = a["image"]
    return {"case_id": uid, "kind": "cow", "ok": True, "n_vox": int((out > 0).sum()), "labels": len(ids),
            **_summary(img, a.get("brain"), out > 0)}


def _point_lps(dcm: bytes, x: float, y: float, frame: int | None) -> np.ndarray:
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(dcm), stop_before_pixels=True, force=True)
    if frame is not None and "PerFrameFunctionalGroupsSequence" in ds:
        fg = ds.PerFrameFunctionalGroupsSequence[int(frame)]
        sh = ds.SharedFunctionalGroupsSequence[0]
        ipp = fg.PlanePositionSequence[0].ImagePositionPatient
        iop = (fg if "PlaneOrientationSequence" in fg else sh).PlaneOrientationSequence[0].ImageOrientationPatient
        ps = (fg if "PixelMeasuresSequence" in fg else sh).PixelMeasuresSequence[0].PixelSpacing
    else:
        ipp, iop, ps = ds.ImagePositionPatient, ds.ImageOrientationPatient, ds.PixelSpacing
    ipp, iop, ps = np.asarray(ipp, float), np.asarray(iop, float), np.asarray(ps, float)
    # DICOM: x = column index along the row direction (iop[:3], spacing ps[1]); y = row index along iop[3:] (ps[0])
    return ipp + x * ps[1] * iop[:3] + y * ps[0] * iop[3:]


def aneurysm(job: tuple) -> list[dict]:
    uid, rel, image_path, points, sop_members = job
    attrs = store.read_attrs(STORE / image_path, "image")
    a = {k: store.squeeze_channel(x) for k, x in store.read_many(STORE / image_path, ("image", "brain")).items()}
    img = a["image"]
    size = [int(v) for v in attrs["grid_size"]]
    out = np.zeros(size[::-1], np.uint8)
    rows = []
    if points:
        inv = np.linalg.inv(np.asarray(attrs["reg_affine"], float).reshape(4, 4))
        org, sp = np.asarray(attrs["grid_origin"], float), np.asarray(attrs["grid_spacing"], float)
        d = np.asarray(attrs["grid_direction"], float).reshape(3, 3)
        zz, yy, xx = np.ogrid[:size[2], :size[1], :size[0]]
        fd = os.open(ZIP, os.O_RDONLY)
        try:
            for sop, x, y, f, group in points:
                m = sop_members.get(sop)
                if m is None:
                    rows.append({"case_id": uid, "kind": "point", "group": group, "ok": False, "why": "no_dicom"})
                    continue
                p = _point_lps(_read_member(fd, *m[1:]), x, y, f)
                q = inv[:3, :3] @ p + inv[:3, 3]                     # source physical -> stored-grid physical
                ijk = (d.T @ (q - org)) / sp                         # stored index (x, y, z)
                inside = bool(np.all(ijk >= 0) and np.all(ijk <= np.asarray(size) - 1))
                if inside:
                    ball = ((xx - ijk[0]) * sp[0]) ** 2 + ((yy - ijk[1]) * sp[1]) ** 2 + ((zz - ijk[2]) * sp[2]) ** 2
                    out[(ball <= BALL_MM ** 2) & (out == 0)] = group
                    c = tuple(int(round(v)) for v in ijk[::-1])
                    br = a.get("brain")
                    rows.append({"case_id": uid, "kind": "point", "group": group, "ok": True, "frame": f,
                                 "img_in": float(img[c]), "img_brain": float(img[br > 0].mean()) if br is not None else np.nan,
                                 "in_brain": float(br[c] > 0) if br is not None else np.nan})
                else:
                    rows.append({"case_id": uid, "kind": "point", "group": group, "ok": False, "why": "outside_grid"})
        finally:
            os.close(fd)
    store.write(store.path_for(STORE, rel, uid, "aneurysm"), "mask", out)
    return rows or [{"case_id": uid, "kind": "negative", "ok": True}]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N series of each kind (a pilot)")
    a = ap.parse_args()

    seg, series = members()
    v = pd.read_csv(STORE / "volumes.csv", low_memory=False)
    v = v[v["dataset_family"] == "rsna_ica"].astype({"case_id": str}).drop_duplicates("case_id").set_index("case_id")
    cow_jobs = [(uid, v.at[uid, "folder"], v.at[uid, "image_path"], m)
                for name, m in sorted(seg.items()) if name.endswith("_cowseg.nii")
                for uid in [name.split("/")[1].removesuffix("_cowseg.nii")] if uid in v.index]

    with zipfile.ZipFile(ZIP) as zf:
        meta = pd.read_csv(zf.open("train.csv"))
        loc = pd.read_csv(zf.open("train_localizers.csv"))
    pos = set(meta.loc[meta["Aneurysm Present"] == 1, "SeriesInstanceUID"].astype(str))
    pts: dict[str, list] = {}
    for r in loc.itertuples():
        c = ast.literal_eval(r.coordinates)
        pts.setdefault(str(r.SeriesInstanceUID), []).append(
            (str(r.SOPInstanceUID), float(c["x"]), float(c["y"]), c.get("f"), ANEURYSM_GROUP[r.location]))
    an_jobs = []
    for uid in v.index:
        if uid in pos and uid not in pts:
            continue
        by_sop = {Path(m[0]).stem: m for m in series.get(uid, [])}
        need = {p[0] for p in pts.get(uid, [])}
        an_jobs.append((uid, v.at[uid, "folder"], v.at[uid, "image_path"], pts.get(uid, []),
                        {k: m for k, m in by_sop.items() if k in need}))
    if a.limit:
        cow_jobs, an_jobs = cow_jobs[:a.limit], [j for j in an_jobs if j[3]][:a.limit]
    print(f"[vessels] cow {len(cow_jobs)}, aneurysm {len(an_jobs)} "
          f"({sum(bool(j[3]) for j in an_jobs)} with points)", flush=True)

    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        rows += list(ex.map(cow, cow_jobs, chunksize=1))
        for r in ex.map(aneurysm, an_jobs, chunksize=8):
            rows += r
    q = pd.DataFrame(rows)
    QC_OUT.parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(QC_OUT, index=False)
    q["dataset"] = q["case_id"].map(v["dataset"].astype(str))
    print(q.groupby(["kind", "dataset"]).agg(n=("case_id", "size"), ok=("ok", "mean"), img_in=("img_in", "median"),
                                              img_brain=("img_brain", "median"), in_brain=("in_brain", "mean"))
          .round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
