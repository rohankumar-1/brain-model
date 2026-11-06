"""Pedi-Cranial-CT-Healthy (TCIA, doi:10.7937/MQ33-AJ18, CC BY 4.0): 100 healthy children (age 0-9.9 years, 10 per
yearly bin; Children's National Hospital, FDA regulatory science tool RST26PD01.01), one non-contrast head CT each.
TCIA ships MetaImage files (``<age bin>do/<NNNNN>_CT.mha``, int16 HU, slope 1 / intercept 0, identity direction;
the curators' DICOM conversion already resolved slice assembly and gantry tilt: matrices up to 512 x 710).

    python -m brain.prep.pedi_ct stage --src <download>/Pedi-Cranial-CT-Healthy --demographics <tsv> --out <dir>

``stage`` writes ``<out>/nifti/<NNNNN>_CT.nii.gz`` (same voxels and geometry; each one read back and compared) and
``<out>/participants.tsv`` in the FOMO participants form (dataset, participant_id, session_id, sex, age in years) so
an age / sex source can read it like the FOMO cohorts'. ``brain.prep.sources.pedi_cranial_ct`` reads the NIfTI
folder; case_id ``sub-<NNNNN>_ses-01``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

DATASET = "pedi_cranial_ct"
NAME = re.compile(r"^(\d{5})_CT\.mha$")


def case_id(num: str) -> str:
    return f"sub-{num}_ses-01"


def participants(demographics: Path) -> pd.DataFrame:
    d = pd.read_csv(demographics, sep="\t")
    days = pd.to_numeric(d["Age at Imaging (Days)"], errors="raise")
    return pd.DataFrame({
        "dataset": DATASET,
        "participant_id": [f"sub-{int(i):05d}" for i in d["Id"]],
        "session_id": "ses-01",
        "sex": d["Sex"].map({"Male": "M", "Female": "F"}),
        "age": (days / 365.25).round(4),
        "age_days": days.astype(int),
        "age_bin_days": d["Age Bin (Days)"].astype(str),
        "group": "Control",
    })


def convert(src: Path, dest: Path) -> dict:
    """One MHA -> NIfTI, checked voxel for voxel and on spacing / origin / direction after reading it back."""
    img = sitk.ReadImage(str(src))
    if img.GetPixelID() != sitk.sitkInt16 or img.GetDimension() != 3:
        raise ValueError(f"{src.name}: expected 3-D int16, got {img.GetPixelIDTypeAsString()} {img.GetDimension()}-D")
    tmp = dest.with_name(f".{dest.name}.partial.nii.gz")
    sitk.WriteImage(img, str(tmp), useCompression=True)
    back = sitk.ReadImage(str(tmp))
    a, b = sitk.GetArrayViewFromImage(img), sitk.GetArrayViewFromImage(back)
    geo_ok = all(np.allclose(f(img), f(back), atol=1e-4) for f in (
        lambda x: x.GetSpacing(), lambda x: x.GetOrigin(), lambda x: x.GetDirection()))
    if back.GetPixelID() != sitk.sitkInt16 or a.shape != b.shape or not np.array_equal(a, b) or not geo_ok:
        tmp.unlink()
        raise ValueError(f"{src.name}: NIfTI does not reproduce the MHA")
    tmp.replace(dest)
    return {"file": dest.name, "size": "x".join(map(str, img.GetSize())),
            "spacing": "x".join(f"{s:.3f}" for s in img.GetSpacing()), "min": int(a.min()), "max": int(a.max())}


def stage(src: Path, demographics: Path, out: Path) -> None:
    mhas = sorted(p for p in src.glob("*do/*.mha") if NAME.match(p.name))
    nums = [NAME.match(p.name).group(1) for p in mhas]
    if len(set(nums)) != len(nums):
        raise SystemExit("[pedi_ct] duplicate subject ids across age bins")
    part = participants(demographics)
    have = set(part["participant_id"])
    missing = [n for n in nums if f"sub-{n}" not in have]
    if missing:
        raise SystemExit(f"[pedi_ct] {len(missing)} images without demographics, e.g. {missing[:3]}")
    # the folder's age bin must hold the subject's age (catches an id mix-up between images and the table)
    bins = dict(zip(part["participant_id"], part["age_bin_days"]))
    wrong = [p.name for p, n in zip(mhas, nums) if p.parent.name.removesuffix("do") != bins[f"sub-{n}"]]
    if wrong:
        raise SystemExit(f"[pedi_ct] {len(wrong)} images in another age bin than the table's, e.g. {wrong[:3]}")
    nifti = out / "nifti"
    nifti.mkdir(parents=True, exist_ok=True)
    rows = []
    for p, n in zip(mhas, nums):
        rows.append(convert(p, nifti / f"{n}_CT.nii.gz") | {"participant_id": f"sub-{n}"})
        print(f"[pedi_ct] {rows[-1]}", flush=True)
    part = part[part["participant_id"].isin([f"sub-{n}" for n in nums])]
    part.to_csv(out / "participants.tsv", sep="\t", index=False)
    print(f"[pedi_ct] {len(rows)} NIfTI in {nifti}; participants.tsv {len(part)} rows "
          f"(sex {part['sex'].value_counts().to_dict()}, age {part['age'].min():.2f}-{part['age'].max():.2f} y)",
          flush=True)


def pack(folder: Path, dest: Path, workers: int) -> None:
    """``<folder>/nifti`` -> ``dest`` as JPEG-XL lossless (``brain.prep.rawpack``, so ``rawpack stage`` restores it),
    verified by name / CRC and by decoding one member against its source."""
    import json
    import zipfile

    import nibabel as nib

    from brain.prep import archive, rawpack

    files = sorted((folder / "nifti").glob("*.nii.gz"))
    rawpack.write_zip(folder, files, dest, workers)
    ok, msg = rawpack.verify_zip(folder, files, dest)
    print(f"[pedi_ct] verify {ok}: {msg}", flush=True)
    if not ok:
        raise SystemExit(1)
    rel = f"nifti/{files[0].name}"
    with zipfile.ZipFile(dest) as zf:
        if rel + rawpack.JLS in zf.namelist():
            back = archive.restore_array(archive.decode_volume(zf.read(rel + rawpack.JLS)),
                                         json.loads(zf.read(rel + rawpack.META)))
            src = np.asanyarray(nib.load(str(files[0])).dataobj.get_unscaled())
            same = back.shape == src.shape and np.array_equal(back, src)
            print(f"[pedi_ct] round trip {rel}: {same}", flush=True)
            if not same:
                raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["stage", "pack"])
    ap.add_argument("--src", type=Path, help="stage: the downloaded Pedi-Cranial-CT-Healthy folder")
    ap.add_argument("--demographics", type=Path, help="stage: TCIA's demographics tsv")
    ap.add_argument("--out", type=Path, required=True, help="the staged folder (nifti/, participants.tsv)")
    ap.add_argument("--zip", type=Path, help="pack: the archive to write")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.action == "stage":
        stage(a.src, a.demographics, a.out)
    else:
        pack(a.out, a.zip, a.workers)


if __name__ == "__main__":
    main()
