"""Stage datasets kept on pool as their original downloads into a prep-ready tree (node-local disk), plus label tables.

* ``ms3seg`` (figshare 30393475): RAR5 volumes -> ``ms3seg/extracted/MS_100_patient_nifti/<id>/``. The three binary
  masks (ventricles, normal WMH, MS lesions; float 0/1 or 0/255, on the axial FLAIR grid) become one label map
  ``<id>_seg.nii.gz``: 1 ventricles, 2 normal WMH, 3 MS lesion (lesion wins overlaps). ``participants.tsv`` holds
  age and sex from the FLAIR JSON sidecars.
* ``valdo`` (Zenodo 4687995): the three task tarballs -> ``valdo/extracted/Task{1,2,3}/sub-*/`` (macOS ``._``
  resource files skipped). Rater masks of a subject (or its single annotation) are merged by union into ``*_desc-union_<Label>.nii.gz``;
  ``labels.tsv`` holds per-subject EPVS counts (Task 1) and lesion counts.
* ``isles24`` (already prepped from its pool zip): only the clinical table. ``clinical.tsv`` joins each subject's
  ``demographic_baseline.xlsx`` (ses-01) and ``outcome.xlsx`` (ses-02): admission NIHSS / mRS, risk factors, timings,
  post-interventional TICI, 24 h / discharge NIHSS and mRS, and the 3-month mRS.
* ``atlas`` (already prepped): only ``metadata.tsv``, the per-session metadata CSVs (days post stroke, site).

Label tables are also written to ``--labels`` (``<store>/labels/sources/<dataset>/``) for the downstream tasks.

    python -m brain.prep.stage_new --datasets ms3seg,valdo --dest $TMPDIR/datasets --labels $STORE/labels/sources
    python -m brain.prep.stage_new --datasets isles24,atlas --dest $TMPDIR/datasets --labels $STORE/labels/sources
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from brain.common.paths import DATASETS
from brain.prep.multirar import extract

MS3SEG_MASKS = (("Vent", 1), ("nWMH", 2), ("abWMH", 3))     # later entries win overlaps
# VALDO label files: ``sub-x_space-T1_desc-Rater3_Lacunes``, ``sub-x_space-T2S_CMB`` (single annotation), ``..._PVSSeg``
VALDO_LABEL = re.compile(r"_space-([A-Za-z0-9]+)_(?:desc-Rater\d+_)?(Lacunes|CMB|PVSSeg)\.nii\.gz$")


def _binary(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    return np.asanyarray(img.dataobj) > 0.5, img


def stage_ms3seg(pool: Path, dest: Path) -> pd.DataFrame:
    out = dest / "ms3seg" / "extracted"
    extract(sorted(pool.glob("MS_100_patient_nifti.part*.rar")), out)
    extract([pool / "MS_100_patient_masks.rar"], out)
    nifti, masks = out / "MS_100_patient_nifti", out / "MS_100_patient_masks"
    rows = []
    for subject in sorted(p for p in nifti.iterdir() if p.is_dir()):
        sid = subject.name
        flair = nib.load(subject / f"{sid}_FLAIR.nii.gz")
        seg = np.zeros(flair.shape, np.uint8)
        for name, label in MS3SEG_MASKS:
            m, img = _binary(masks / f"{name}_Masks" / sid / f"{sid}_{name}_Mask.nii.gz")
            if m.shape != flair.shape or not np.allclose(img.affine, flair.affine, atol=1e-3):
                raise RuntimeError(f"ms3seg {sid}: {name} mask is not on the FLAIR grid")
            seg[m] = label
        nib.save(nib.Nifti1Image(seg, flair.affine, flair.header), subject / f"{sid}_seg.nii.gz")
        meta = json.loads((subject / f"{sid}_FLAIR.json").read_text())
        age = re.match(r"(\d+)", str(meta.get("PatientAge", "")))
        rows.append({"participant_id": sid, "age": float(age.group(1)) if age else np.nan,
                     "sex": meta.get("PatientSex"), "manufacturer": meta.get("Manufacturer"),
                     "field_strength": meta.get("MagneticFieldStrength"),
                     "lesion_ml": float((seg == 3).sum() * np.prod(flair.header.get_zooms()[:3]) / 1000)})
    shutil.rmtree(masks)
    return pd.DataFrame(rows)


def stage_valdo(pool: Path, dest: Path) -> pd.DataFrame:
    out = dest / "valdo" / "extracted"
    out.mkdir(parents=True, exist_ok=True)
    for tar in sorted(pool.glob("Task*_v2.tar.gz")):
        with tarfile.open(tar) as tf:
            members = [m for m in tf.getmembers() if not Path(m.name).name.startswith("._")
                       and Path(m.name).name != ".DS_Store"]
            tf.extractall(out, members=members, filter="data")
    rows = []
    for task_dir in sorted(out.glob("Task*")):
        task = task_dir.name.lower().replace("task", "t")
        for subject in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            row = {"task": task, "participant_id": subject.name}
            raters: dict[str, list[Path]] = {}
            for f in subject.glob("*.nii.gz"):
                m = VALDO_LABEL.search(f.name)
                if m:
                    raters.setdefault(m.group(2), []).append(f)
            for label, files in raters.items():
                masks = [_binary(f) for f in sorted(files)]
                union = np.any([m for m, _ in masks], axis=0).astype(np.uint8)
                ref = masks[0][1]
                space = VALDO_LABEL.search(files[0].name).group(1)
                nib.save(nib.Nifti1Image(union, ref.affine, ref.header),
                         subject / f"{subject.name}_space-{space}_desc-union_{label}.nii.gz")
                row[f"{label.lower()}_voxels"] = int(union.sum())
                row[f"{label.lower()}_raters"] = len(files)
            counts = sorted(subject.glob("*Count.csv"))       # one per rater for the double-rated subjects
            if counts:
                totals = [float(pd.to_numeric(pd.read_csv(c)["count"], errors="coerce").sum()) for c in counts]
                row["epvs_count"] = float(np.mean(totals))
                row["epvs_raters"] = len(totals)
            rows.append(row)
    return pd.DataFrame(rows)


def stage_isles24(pool: Path, dest: Path) -> pd.DataFrame:
    import io
    import zipfile

    z = zipfile.ZipFile(pool / "isles24.raw.zip")
    tables = {}
    for name in z.namelist():
        m = re.search(r"phenotype/(sub-stroke\d+)/ses-0\d/[^/]+_(demographic_baseline|outcome)\.xlsx$", name)
        if m:
            row = pd.read_excel(io.BytesIO(z.read(name))).iloc[0].to_dict()
            tables.setdefault(m.group(1), {}).update(row)
    return pd.DataFrame([{"participant_id": sid, **row} for sid, row in sorted(tables.items())])


def stage_atlas(pool: Path, dest: Path) -> pd.DataFrame:
    import io
    import zipfile

    z = zipfile.ZipFile(pool / "atlas.raw.zip")
    rows = [pd.read_csv(io.BytesIO(z.read(n))) for n in z.namelist() if n.endswith("_metadata.csv")]
    meta = pd.concat(rows, ignore_index=True)
    meta.insert(0, "participant_id", meta["SESSION_ID"].astype(str).str.split("_").str[0])
    meta["DAYS_POST_STROKE"] = pd.to_numeric(meta["DAYS_POST_STROKE"], errors="coerce")
    return meta


STAGERS = {"ms3seg": (stage_ms3seg, "participants.tsv"), "valdo": (stage_valdo, "labels.tsv"),
           "isles24": (stage_isles24, "clinical.tsv"), "atlas": (stage_atlas, "metadata.tsv")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", required=True, help=f"comma list of {', '.join(STAGERS)}")
    ap.add_argument("--pool", type=Path, default=DATASETS)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=None, help="also copy label tables here (<dataset>/<file>)")
    a = ap.parse_args()
    for name in a.datasets.split(","):
        fn, table = STAGERS[name]
        df = fn(a.pool / name, a.dest)
        targets = [a.dest / name / table] + ([a.labels / name / table] if a.labels else [])
        for t in targets:
            t.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(t, sep="\t", index=False)
        print(f"[stage] {name}: {len(df)} subjects -> {a.dest / name}", flush=True)


if __name__ == "__main__":
    main()
