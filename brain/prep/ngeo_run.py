"""Build the native-geometry ("ngeo") CT copy (``brain.prep.ngeo``) for indexed CT volumes, and its index.

    # one shard of the CT volumes (rows of STORE/volumes.csv), resumable; raw data staged to node-local disk
    python -m brain.prep.ngeo_run build --root STORE --datasets rsna_ich --shard 3/32 --workers 11
    # a pilot into another root (the real store is only read)
    python -m brain.prep.ngeo_run build --root STORE --out-root runs/diagnostics/ngeo/pilot --cases pilot.csv
    # volumes_ngeo.csv from the manifests
    python -m brain.prep.ngeo_run index --root STORE
    # size / time estimate from finished manifests
    python -m brain.prep.ngeo_run estimate --root STORE

Outputs per case (next to the 1 mm image zip, ``path_for(root, folder, case_id, "ngeo")``): ``<case>.ngeo.zip`` and
``<case>.<mask>.ngeo.zip``; one manifest row per case in ``<out-root>/ngeo_manifests/``. A case is skipped when a
manifest marks it ok and its zip exists (``--force`` redoes it). Before every case the driver checks the space left:
``os.statvfs`` of the output filesystem and, because statvfs on the scratch NFS mount reports the whole filesystem and
not the user's quota, the quota cache (``~/orcd/.quota``) minus every ngeo byte the manifests record since that cache
was written (``space_left_gb``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import DATASETS, SCRATCH
from brain.prep import ngeo, store

STORE_ROOT = SCRATCH / "data" / "processed" / "slices_zarr"
CT_DATASETS = ("rsna_ich", "rsna_ica_cta", "hemseg", "cq500", "aisd", "synthrad_ct", "bhsd", "isles24_ncct",
               "isles24_cta", "synthrad_cbct", "instance", "pedi_cranial_ct", "seg_cq500")
SITK_FRAME = {"rsna_ich", "hemseg"}                  # the 1 mm store read these with SimpleITK's series reader
NATIVE_MASKS = {"bhsd": ("ICH",), "hemseg": ("ICH",), "instance": ("ICH",), "seg_cq500": ("ICH",),
                "aisd": ("lesion",), "isles24_ncct": ("lesion", "lvo", "cow")}
# template-grid-only label zips next to the image (nearest through the per-slice map); checked by name, per dataset
TEMPLATE_MASKS = {"rsna_ich": ("bleedpl",), "rsna_ica_cta": ("cow", "aneurysm"), "*": ("synthseg",)}
# the real 5 mm series of thin CQ500 studies (python -m brain.datasets.download_cq500 --thick); NGEO_CQ500_THICK overrides
CQ500_THICK = Path(os.environ.get("NGEO_CQ500_THICK", DATASETS / "cq500_dicom_thick"))
MIN_FREE_GB = float(os.environ.get("NGEO_MIN_FREE_GB", 20))
EST_MB_PER_CASE = 12.0                                # space reserved per case in flight
MANIFESTS = "ngeo_manifests"


# ------------------------------------------------------------------------------------------------------- space

def _quota_row(path: Path, quota: Path = Path.home() / "orcd" / ".quota") -> tuple[float, float, float] | None:
    """(used GB, limit GB, cache time) of the quota-cache row for ``path``'s filesystem, or None."""
    try:
        txt = quota.read_text()
    except OSError:
        return None
    p = str(Path(path).resolve())
    key = "SCRATCH" if "/scratch" in p else "POOL" if "/pool" in p else "HOME" if p.startswith(str(Path.home())) else None
    if key is None:
        return None
    used = limit = None
    for line in txt.splitlines():
        cols = [c.strip() for c in line.split("|")]
        if cols and cols[0] == key and len(cols) >= 3:
            used, limit = float(cols[1]), float(cols[2])
    stamp = time.time()
    for line in txt.splitlines():
        if line.startswith("Last updated:"):
            parts = line.split(":", 1)[1].split()
            try:   # "Sat Nov 07 21:49:07 EST 2026": drop the zone word
                stamp = datetime.strptime(" ".join(parts[:4] + parts[-1:]), "%a %b %d %H:%M:%S %Y").timestamp()
            except ValueError:
                stamp = time.time() - 3600
    return (used, limit, stamp) if used is not None else None


def _written_since(out_root: Path, since: float) -> float:
    """GB of ngeo zips the manifests under ``out_root`` record as written after ``since`` (epoch seconds)."""
    total = 0
    for m in sorted((Path(out_root) / MANIFESTS).glob("*.jsonl")):
        try:
            for line in m.read_text().splitlines():
                r = json.loads(line)
                if r.get("status") == "ok" and r.get("written_ts", 0) >= since:
                    total += int(r.get("bytes", 0))
        except (OSError, ValueError):
            continue
    return total / 1e9


def space_left_gb(out_root: Path) -> float:
    """Free space for new output: the smaller of statvfs and (quota limit - cached usage - ngeo bytes since)."""
    s = os.statvfs(out_root)
    free = s.f_bavail * s.f_frsize / 1e9
    q = _quota_row(out_root)
    if q is not None:
        used, limit, stamp = q
        free = min(free, limit - used - _written_since(out_root, stamp))
    return free


# ----------------------------------------------------------------------------------------------- case sources

def _read_members(zip_path: str, members) -> list[bytes]:
    from concurrent.futures import ThreadPoolExecutor

    from brain.prep import dicom

    fd = os.open(zip_path, os.O_RDONLY)
    try:
        with ThreadPoolExecutor(16) as ex:
            return list(ex.map(lambda m: dicom._read_member(fd, *m[1:]), members))
    finally:
        os.close(fd)


def _zip_all(zip_path: Path) -> list[bytes]:
    with zipfile.ZipFile(zip_path) as zf:
        return [zf.read(i) for i in zf.infolist() if not i.filename.endswith("/")]


def resolve(rows: pd.DataFrame, local: Path) -> dict[str, dict]:
    """volume row key (dataset:case_id) -> source spec for the worker: {"kind": "dicom", "zip", "members"} |
    {"kind": "dicom_zip", "zip"} | {"kind": "nifti", "path"}, plus native "masks" {name: path}."""
    from brain.prep import dicom, sources

    want = {(d, c) for d, c in zip(rows["dataset"], rows["case_id"])}
    dsets = {d for d, _ in want}
    spec: dict[str, dict] = {}

    def add_cases(cases):
        for c in cases:
            if (c.dataset, c.case_id) in want:
                s = {"masks": {n: str(p) for n, p in c.masks if n != "brain"}}
                if c.members:
                    s |= {"kind": "dicom", "zip": c.zip_path, "members": [list(m) for m in c.members]}
                else:
                    s |= {"kind": "nifti", "path": str(c.path)}
                spec[f"{c.dataset}:{c.case_id}"] = s

    if "rsna_ich" in dsets:
        work = SCRATCH / "work" / "rsna_ingest"
        rz = DATASETS / "rsna" / "rsna-ihd-dataset.zip"
        index = dicom.build_offsets(rz, work / "study_index.json", work / "member_offsets.json")
        for d, c in want:
            if d == "rsna_ich":
                uid = c.removeprefix("RSNA-")
                if uid in index:
                    spec[f"{d}:{c}"] = {"kind": "dicom", "zip": str(rz), "members": index[uid], "masks": {}}
    if "hemseg" in dsets:
        add_cases(sources.hemseg(local / "hemseg" / "extracted", DATASETS / "rsna" / "rsna-ihd-dataset.zip"))
    if "rsna_ica_cta" in dsets:
        idx = json.loads((SCRATCH / "work" / "rsna_ica" / "series_members.json").read_text())
        rz = DATASETS / "rsna_ica" / "rsna-intracranial-aneurysm-detection.zip"
        for d, c in want:
            if d == "rsna_ica_cta" and c in idx:
                spec[f"{d}:{c}"] = {"kind": "dicom", "zip": str(rz), "members": sorted(idx[c]), "masks": {}}
    if "cq500" in dsets:
        segs = {c.case_id: c for c in sources.seg_cq500(local / "seg_cq500" / "volumes")} \
            if (local / "seg_cq500" / "volumes").is_dir() else {}
        for d, c in want:
            if d == "cq500":
                thick = CQ500_THICK / f"{c}.zip"
                sc = segs.get(c)
                seg = {"path": str(sc.path), "ICH": str(dict(sc.masks)["ICH"])} \
                    if sc is not None and "ICH" in dict(sc.masks) else None
                spec[f"{d}:{c}"] = {"kind": "dicom_zip", "zip": str(DATASETS / "cq500_dicom" / f"{c}.zip"),
                                    "thick_zip": str(thick) if thick.exists() else None, "masks": {},
                                    "seg_cq500": seg}
    adapters = {"bhsd": lambda: sources.bhsd(local / "bhsd" / "extracted"),
                "seg_cq500": lambda: sources.seg_cq500(local / "seg_cq500" / "volumes"),
                "aisd": lambda: sources.aisd(local / "aisd" / "extracted"),
                "instance": lambda: sources.instance(local / "instance2022" / "extracted"),
                "isles24": lambda: sources.isles24(local / "isles24" / "train"),
                "synthrad": lambda: sources.synthrad(local / "synthrad2023"),
                "pedi_cranial_ct": lambda: sources.pedi_cranial_ct(local / "pedi_cranial_ct" / "nifti")}
    for name, fn in adapters.items():
        if any(d.startswith(name) for d in dsets):
            add_cases(fn())
    return spec


def stage(datasets: set[str], local: Path, case_ids: dict[str, set[str]] | None = None) -> None:
    """Raw data of the NIfTI-sourced datasets onto node-local disk (never scratch). DICOM datasets read from pool."""
    from brain.prep import rawpack

    local.mkdir(parents=True, exist_ok=True)
    pool = DATASETS

    def has(*names):
        return any(d.startswith(n) for d in datasets for n in names)

    def done(name):
        return (local / f".staged_{name}").exists()

    def mark(name):
        (local / f".staged_{name}").touch()

    for folder, names in (("seg_cq500", ("seg_cq500", "cq500")), ("aisd", ("aisd",)), ("instance2022", ("instance",)),
                          ("isles24", ("isles24",)), ("pedi_cranial_ct", ("pedi_cranial_ct",))):
        if has(*names) and not done(folder):
            rawpack.stage(pool / folder, local)
            mark(folder)
    if has("bhsd") and not done("bhsd"):
        with zipfile.ZipFile(pool / "bhsd" / "label_192.zip") as zf:
            zf.extractall(local / "bhsd" / "extracted")
        mark("bhsd")
    if has("hemseg") and not done("hemseg"):
        with zipfile.ZipFile(pool / "hemseg" / "HemSeg-500.zip") as zf:
            zf.extractall(local / "hemseg" / "extracted")
        shutil.copy(pool / "hemseg" / "annotion_file_info_500.jsonl", local / "hemseg")
        mark("hemseg")
    if has("synthrad") and not done("synthrad"):
        ids = set().union(*[case_ids.get(d, set()) for d in datasets if d.startswith("synthrad")]) if case_ids else None
        for t in ("Task1", "Task2"):
            with zipfile.ZipFile(pool / "synthrad2023" / f"{t}.zip") as zf:
                names = [n for n in zf.namelist() if n.startswith(f"{t}/brain/") and n.endswith(".nii.gz")
                         and (ids is None or n.split("/")[2] in ids)]
                zf.extractall(local / "synthrad2023", names)
        mark("synthrad")
    print(f"[ngeo] staged {sorted(datasets)} -> {local}", flush=True)


# --------------------------------------------------------------------------------------------------- one case

def _template_masks(store_root: Path, image_path: str, dataset: str) -> dict[str, np.ndarray]:
    out = {}
    for name in TEMPLATE_MASKS.get(dataset, ()) + TEMPLATE_MASKS["*"]:
        p = store_root / (image_path[:-4] + f".{name}.zip")
        if p.exists():
            out[name] = store.squeeze_channel(store.read(p, "mask"))
    return out


def read_case(dataset: str, spec: dict, attrs: dict, brain: np.ndarray | None):
    """(SliceStack with ``q`` set, native label stacks, extra attrs) for one case."""
    import SimpleITK as sitk

    from brain.prep import geometry, intensity

    extra: dict = {}
    if spec["kind"] in ("dicom", "dicom_zip"):
        series_src = "dicom"
        if spec["kind"] == "dicom":
            blobs = _read_members(spec["zip"], spec["members"])
            source = f"{spec['zip']}::{len(spec['members'])} members"
        else:
            zp = spec.get("thick_zip") or spec["zip"]
            blobs = _zip_all(Path(zp))
            source = zp
            series_src = "cq500_thick" if spec.get("thick_zip") else "dicom"
        st, main, info = ngeo.dicom_geometry(blobs)
        extra |= info | {"source": source, "series": main[0].series, "series_kind": series_src}
        if dataset in SITK_FRAME and os.environ.get("NGEO_IGNORE_STORE_FRAME") == "1":    # diagnostic only
            extra["store_frame"] = "ignored"
        elif dataset in SITK_FRAME:
            q, finfo = ngeo.sitk_store_origins(st.p, st.a, st.b, attrs)
            st.q = q
            extra |= finfo if q is not None else finfo | {"store_frame_ok": False}
        else:
            extra["store_frame"] = "true"            # dcm2niix read (true scanner frame)
        # decode only slices near the head range (head-and-neck CTA: skip the neck and chest)
        lo, hi, _, _ = ngeo.head_range_and_center(st, attrs, brain)
        t = st.p @ st.normal
        keep = np.flatnonzero((t >= lo - 2 * ngeo.SLAB_MM) & (t <= hi + 2 * ngeo.SLAB_MM))
        if len(keep) < 2:
            raise RuntimeError("no slices inside the head range")
        st, dinfo = ngeo.decode_stack(st.take(keep), blobs)
        extra |= dinfo
        masks = {}
        for name, p in spec.get("masks", {}).items():
            masks[name] = ngeo.mask_stack(geometry.read(Path(p)), st)
        return st, masks, extra
    img = geometry.read(Path(spec["path"]))
    img, off = intensity.fix_ct_offset(img)
    img = intensity.clamp_ct(img)
    st = ngeo.stack_from_sitk(img)
    extra |= {"source": spec["path"], "series": "", "series_kind": "nifti", "store_frame": "true",
              "offset_repaired": bool(off)}
    masks = {}
    for name, p in spec.get("masks", {}).items():
        m = sitk.Cast(geometry.read(Path(p)), sitk.sitkUInt8)
        masks[name] = ngeo.mask_stack(m, st)
    return st, masks, extra


def run_case(row: dict, spec: dict, store_root: str, out_root: str) -> dict:
    """Build and write one case; returns its manifest row (status ok / error)."""
    t0 = time.time()
    base = {"dataset": row["dataset"], "case_id": row["case_id"], "volume_id": row["volume_id"],
            "folder": row["folder"], "image_path": row["image_path"]}
    try:
        sr, orr = Path(store_root), Path(out_root)
        img_zip = sr / row["image_path"]
        arrays = store.read_many(img_zip, ("brain", "image"))
        attrs = store.read_attrs(img_zip, "image")
        brain = arrays.get("brain")
        st, nmasks, extra = read_case(row["dataset"], spec, attrs, brain)
        t1 = time.time()
        tmasks = _template_masks(sr, row["image_path"], row["dataset"])
        img1 = arrays.get("image")
        res = ngeo.build(st, attrs, brain, nmasks, tmasks, extra | {"volume_id": row["volume_id"]},
                         None if img1 is None else store.squeeze_channel(img1))
        t2 = time.time()
        stem = row["image_path"][:-4]
        out = orr / f"{stem}.ngeo.zip"
        arrs = {"image": (res.image[:, None], res.attrs)}
        if res.brain is not None:
            arrs["brain"] = (res.brain, {"source": "1mm_brain", "brain_method": attrs.get("brain_method")})
        store.write_arrays(out, arrs)
        nbytes = out.stat().st_size
        mpaths = {}
        for name, m in res.masks.items():
            mp = orr / f"{stem}.{name}.ngeo.zip"
            src = "native" if name in nmasks else "template"
            store.write(mp, "mask", m, {"kind": "mask_ngeo", "mask_source": src, "ngeo_rev": ngeo.NGEO_REV})
            nbytes += mp.stat().st_size
            mpaths[name] = str(mp.relative_to(orr))
        if spec.get("seg_cq500"):
            m, cinfo = cross_map_seg_cq500(spec["seg_cq500"], res)
            res.attrs.update(cinfo)
            if m is not None:
                mp = orr / f"{stem}.ICH.ngeo.zip"
                store.write(mp, "mask", m, {"kind": "mask_ngeo", "mask_source": "seg_cq500_physical",
                                            "ngeo_rev": ngeo.NGEO_REV, **cinfo})
                nbytes += mp.stat().st_size
                mpaths["ICH"] = str(mp.relative_to(orr))
        a = res.attrs
        flat = {k: v for k, v in a.items() if isinstance(v, (int, float, str, bool))}
        return base | flat | {
            "status": "ok", "path": str(out.relative_to(orr)), "S": int(res.image.shape[0]),
            "masks": mpaths, **{f"asm_{k}": v for k, v in a["assembly"].items()},
            "bytes": int(nbytes), "t_read": round(t1 - t0, 2), "t_build": round(t2 - t1, 2),
            "t_write": round(time.time() - t2, 2), "written_ts": time.time(), "host": os.uname().nodename}
    except Exception as e:                                         # noqa: BLE001  logged, retried on the next run
        return base | {"status": "error", "error": f"{type(e).__name__}: {e}"[:400],
                       "trace": traceback.format_exc()[-1500:], "written_ts": time.time()}


SEG_CQ500_MIN_CORR = 0.8


def cross_map_seg_cq500(seg: dict, res) -> tuple[np.ndarray | None, dict]:
    """Seg-CQ500's ICH mask (third-party NIfTI of the same study) onto the cq500 ngeo grid through physical space,
    accepted only when its CT matches the ngeo image there (the NIfTI release is mirrored for some studies)."""
    import SimpleITK as sitk

    from brain.prep import geometry

    ct = geometry.read(Path(seg["path"]))
    nsub = max(1, int(round(res.plan.D / 0.6))) if res.plan.D > 1.5 else 1
    x = ngeo.sample_physical(ct, res.grid, order=1, nsub=nsub, cval=ngeo.FLOOR_HU)
    a, b = res.image.astype(np.float32), x
    head = (a > ngeo.AIR_HU) | (b > ngeo.AIR_HU)
    corr = float(np.corrcoef(np.clip(a[head], -50, 150), np.clip(b[head], -50, 150))[0, 1]) if head.sum() > 100 else 0.0
    info = {"seg_cq500_corr": corr}
    if not corr >= SEG_CQ500_MIN_CORR:
        return None, info
    m = sitk.Cast(geometry.read(Path(seg["ICH"])), sitk.sitkUInt8)
    return ngeo.sample_physical(m, res.grid, order=0, nsub=nsub), info


# ---------------------------------------------------------------------------------------------------- driver

def select_rows(store_root: Path, datasets: list[str], shard: tuple[int, int], cases: Path | None) -> pd.DataFrame:
    cols = ["volume_id", "case_id", "dataset", "folder", "image_path", "modality"]
    v = pd.read_csv(store_root / "volumes.csv", usecols=cols, low_memory=False)
    v = v[(v["modality"] == "ct") & v["dataset"].isin(datasets)]
    if cases is not None:
        want = pd.read_csv(cases)
        v = v.merge(want[["dataset", "case_id"]].drop_duplicates(), on=["dataset", "case_id"])
    v = v.drop_duplicates(["dataset", "case_id"]).sort_values(["dataset", "case_id"]).reset_index(drop=True)
    i, n = shard
    return v.iloc[i::n].reset_index(drop=True)


def done_keys(out_root: Path) -> set[str]:
    keys = set()
    for m in sorted((out_root / MANIFESTS).glob("*.jsonl")):
        for line in m.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "ok" and r.get("ngeo_rev") == ngeo.NGEO_REV and (out_root / r["path"]).exists():
                keys.add(f"{r['dataset']}:{r['case_id']}")
    return keys


def cmd_build(a) -> None:
    store_root, out_root = Path(a.root), Path(a.out_root or a.root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / MANIFESTS).mkdir(exist_ok=True)
    i, n = (int(x) for x in a.shard.split("/"))
    rows = select_rows(store_root, [d for d in a.datasets.split(",") if d], (i, n), a.cases)
    if not a.force:
        done = done_keys(out_root)
        rows = rows[[f"{d}:{c}" not in done for d, c in zip(rows["dataset"], rows["case_id"])]]
    print(f"[ngeo] shard {i}/{n}: {len(rows)} cases to do (out {out_root})", flush=True)
    if rows.empty:
        return
    local = Path(a.local or os.environ.get("NGEO_LOCAL") or f"/tmp/{os.environ.get('USER', 'u')}_ngeo_"
                 f"{os.environ.get('SLURM_JOB_ID', 'local')}")
    ids = {d: set(g["case_id"]) for d, g in rows.groupby("dataset")}
    stage(set(rows["dataset"]), local, ids)
    spec = resolve(rows, local)
    manifest = out_root / MANIFESTS / f"ngeo_{a.tag}_{i}of{n}.jsonl"
    todo = [r for r in rows.to_dict("records")]
    missing = [r for r in todo if f"{r['dataset']}:{r['case_id']}" not in spec]
    with manifest.open("a") as mf:
        for r in missing:
            mf.write(json.dumps({**{k: r[k] for k in ("dataset", "case_id", "volume_id")}, "status": "error",
                                 "error": "no raw source found", "written_ts": time.time()}) + "\n")
        todo = [r for r in todo if f"{r['dataset']}:{r['case_id']}" in spec]
        n_ok = n_err = 0
        t_start = time.time()
        with ProcessPoolExecutor(max(1, a.workers)) as ex:
            running = set()
            it = iter(todo)
            stop = False
            while True:
                while not stop and len(running) < 2 * a.workers:
                    r = next(it, None)
                    if r is None:
                        break
                    left = space_left_gb(out_root)
                    if left < MIN_FREE_GB + EST_MB_PER_CASE * (len(running) + 1) / 1000:
                        print(f"[ngeo] only {left:.1f} GB left (min {MIN_FREE_GB}): stopping", flush=True)
                        stop = True
                        break
                    running.add(ex.submit(run_case, r, spec[f"{r['dataset']}:{r['case_id']}"], str(store_root),
                                          str(out_root)))
                if not running:
                    break
                fin, running = wait(running, return_when=FIRST_COMPLETED)
                for f in fin:
                    row = f.result()
                    mf.write(json.dumps(_jsonable(row)) + "\n")
                    mf.flush()
                    if row["status"] == "ok":
                        n_ok += 1
                    else:
                        n_err += 1
                        print(f"[ngeo] error {row['dataset']}:{row['case_id']}: {row['error']}", flush=True)
                    if (n_ok + n_err) % 50 == 0:
                        print(f"[ngeo] {n_ok} ok / {n_err} errors, {time.time() - t_start:.0f} s", flush=True)
    print(f"[ngeo] done: {n_ok} ok, {n_err} errors, {len(missing)} without source, {time.time() - t_start:.0f} s",
          flush=True)
    if stop:
        sys.exit(3)


def _jsonable(r: dict) -> dict:
    return {k: store.plain(v) if not isinstance(v, dict) else {kk: store.plain(vv) for kk, vv in v.items()}
            for k, v in r.items()}


# ----------------------------------------------------------------------------------------------------- index

QC_RULES = (("brain_off", lambda r: r.get("qc_brain_tissue", 1.0) < 0.8),
            ("frame_mismatch", lambda r: r.get("qc_corr_1mm", 1.0) < 0.8),
            ("tpl_nonlinear", lambda r: r.get("tpl_resid_mm", 0.0) > 3.0),
            ("gaps", lambda r: r.get("asm_n_gaps", 0) > 0),
            ("fov_cut", lambda r: r.get("qc_edge_tissue", 0.0) > 0.02),
            ("frame_assumed", lambda r: r.get("store_frame_ok") is False),
            ("short", lambda r: r.get("S", 99) < 12))


def load_manifests(out_root: Path) -> pd.DataFrame:
    rows = []
    for m in sorted((Path(out_root) / MANIFESTS).glob("*.jsonl")):
        for line in m.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return pd.DataFrame(rows)


def build_index(out_root: Path) -> pd.DataFrame:
    """volumes_ngeo.csv rows: the newest ok row per (dataset, case); the CQ500 / seg_cq500 volume_id collision keeps
    the cq500 copy."""
    m = load_manifests(out_root)
    if m.empty:
        return m
    m = m[m["status"] == "ok"].sort_values("written_ts").drop_duplicates(["dataset", "case_id"], keep="last")
    recs = m.to_dict("records")
    for r in recs:
        r["qc_flags"] = ",".join(name for name, rule in QC_RULES if rule(r))
        for name, p in (r.get("masks") or {}).items():
            r[f"mask_{name.lower()}_ngeo"] = p
    out = pd.DataFrame(recs)
    cq = set(out.loc[out["dataset"] == "cq500", "volume_id"])
    out = out[~((out["dataset"] == "seg_cq500") & out["volume_id"].isin(cq))]
    keep = ["volume_id", "dataset", "case_id", "path", "S", "slice_mm", "thick_mm", "synth_thick", "n_synth",
            "series_kind", "asm_n_dup_removed", "asm_n_gaps", "asm_uneven", "asm_tilt_deg", "asm_deshear",
            "store_frame", "tpl_resid_mm", "qc_brain_tissue", "qc_brain_air", "qc_corr_1mm", "qc_edge_tissue", "qc_flags",
            "bytes"]
    keep += sorted(c for c in out.columns if c.startswith("mask_") and c.endswith("_ngeo"))
    return out[[c for c in keep if c in out.columns]].sort_values(["dataset", "case_id"]).reset_index(drop=True)


def cmd_index(a) -> None:
    out_root = Path(a.out_root or a.root)
    idx = build_index(out_root)
    dst = out_root / "volumes_ngeo.csv"
    tmp = dst.with_suffix(".csv.partial")
    idx.to_csv(tmp, index=False)
    os.replace(tmp, dst)
    print(f"[ngeo] {dst}: {len(idx)} volumes; flags: "
          f"{idx['qc_flags'].replace('', 'none').str.split(',').explode().value_counts().to_dict()}")


def cmd_estimate(a) -> None:
    """Per-dataset MB / case and seconds / case from the ok manifest rows, scaled to every indexed CT volume."""
    out_root, store_root = Path(a.out_root or a.root), Path(a.root)
    m = load_manifests(out_root)
    m = m[m["status"] == "ok"]
    v = pd.read_csv(store_root / "volumes.csv", usecols=["dataset", "modality"], low_memory=False)
    counts = v[v["modality"] == "ct"]["dataset"].value_counts()
    m["t"] = m["t_read"] + m["t_build"] + m["t_write"]
    per = m.groupby("dataset").agg(n=("bytes", "size"), mb=("bytes", lambda s: s.mean() / 1e6), s=("t", "mean"))
    fallback_mb, fallback_s = float(m["bytes"].mean() / 1e6), float(m["t"].mean())
    tot_gb = tot_h = 0.0
    for ds, n in counts.items():
        mb = per["mb"].get(ds, fallback_mb)
        s = per["s"].get(ds, fallback_s)
        tot_gb += n * mb / 1e3
        tot_h += n * s / 3600
        print(f"{ds:18s} {n:6d} vols  {mb:6.2f} MB  {s:5.1f} s  -> {n * mb / 1e3:7.1f} GB {n * s / 3600:6.1f} core-h"
              f"{'' if ds in per.index else '  (no pilot rows: mean used)'}")
    print(f"total {counts.sum()} CT volumes: {tot_gb:.1f} GB, {tot_h:.1f} core-hours")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "index", "estimate"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=str(STORE_ROOT), help="the 1 mm store (read)")
        p.add_argument("--out-root", default=None, help="where ngeo zips + manifests go (default: --root)")
        if name == "build":
            p.add_argument("--datasets", default=",".join(CT_DATASETS))
            p.add_argument("--shard", default="0/1")
            p.add_argument("--cases", type=Path, default=None, help="CSV with dataset, case_id to restrict to")
            p.add_argument("--workers", type=int, default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - 1))
            p.add_argument("--local", default=None, help="node-local staging dir (default /tmp/$USER_ngeo_$JOBID)")
            p.add_argument("--tag", default="v1")
            p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"build": cmd_build, "index": cmd_index, "estimate": cmd_estimate}[a.cmd](a)


if __name__ == "__main__":
    main()
