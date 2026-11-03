"""fastMRI+ annotated brain scans (2026-11-01): the coil-combined images ``brain.prep.fastmri_fetch`` kept, as images
with geometry, for ``brain.prep.run`` (source ``fastmri``).

The fastMRI .h5 files carry no slice positions, so the geometry is fixed by convention, checked against the template
on the pilot scans (affine registration, mutual information, 4 of 4 scans each):

* rows flipped (``image[:, ::-1, :]``, fastMRI+'s display convention, the frame its boxes are drawn in): row 0 is
  anterior (MI -0.72..-0.88 vs -0.56..-0.63 with anterior at the bottom);
* slice 0 is inferior (vs -0.42..-0.48 reversed);
* columns run from the patient's right to left (radiological; left-right is not identifiable from a near-symmetric
  brain, and no fastMRI+ label is lateralised).

So the flipped array is LPS with identity direction, pixel spacing = reconstruction field of view / matrix, slice
spacing = slice thickness (the release's own DICOM converter assumes no gap). A box ``(slice, x, y, w, h)`` covers
array voxels ``[slice, y:y+h, x:x+w]`` of the flipped image.

    python -m brain.prep.fastmri --dest /tmp/.../datasets/fastmri_brain/nifti
      -> <dest>/<contrast>/<file>.nii.gz, <store>/labels/fastmri_patients.csv, <store>/labels/fastmri_scans.csv
"""

from __future__ import annotations

import argparse
import io
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from brain.common.paths import DATASETS, SCRATCH

POOL = DATASETS / "fastmri_brain"
STORE = SCRATCH / "data" / "processed" / "slices_zarr"
# file name contrast -> store dataset
DATASETS_OF = {"AXFLAIR": "fastmri_flair", "AXT1": "fastmri_t1w", "AXT1POST": "fastmri_t1c", "AXT2": "fastmri_t2w"}


def contrast_of(name: str) -> str:
    """``file_brain_AXFLAIR_200_6002425`` -> ``AXFLAIR``."""
    return name.split("_")[2]


def _fov(header: str) -> tuple[tuple[float, float, float], tuple[int, int]]:
    m = re.search(r"<reconSpace>\s*<matrixSize>\s*<x>(\d+)</x>\s*<y>(\d+)</y>.*?<fieldOfView_mm>\s*<x>([\d.]+)</x>"
                  r"\s*<y>([\d.]+)</y>\s*<z>([\d.]+)</z>", header, re.S)
    if not m:
        return (0.6875, 0.6875, 5.0), (0, 0)
    mx, my, fx, fy, fz = int(m.group(1)), int(m.group(2)), *map(float, m.groups()[2:])
    return (fx / mx, fy / my, fz), (mx, my)


def to_image(blob: bytes) -> tuple[sitk.Image, dict]:
    """A kept scan (fastmri_fetch's .h5) -> LPS image of the flipped array, and its attributes."""
    import h5py

    with h5py.File(io.BytesIO(blob), "r") as f:
        img = f["image"][()][:, ::-1, :].astype(np.float32)
        header = f["ismrmrd_header"][()]
        header = header.decode() if isinstance(header, bytes) else str(header)
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    (sx, sy, sz), (mx, my) = _fov(header)
    h, w = img.shape[1:]
    # pixel size per array axis: reconSpace x / y may be listed in either order relative to rows / columns
    col = sx if w == mx else sy if w == my else sx
    row = sy if h == my else sx if h == mx else sy
    im = sitk.GetImageFromArray(np.ascontiguousarray(img))
    im.SetSpacing((float(col), float(row), float(sz)))
    im.SetDirection((1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0))
    field = re.search(r"<systemFieldStrength_T>([\d.]+)<", header)
    return im, {**attrs, "field_T": float(field.group(1)) if field else np.nan, "n_slices": img.shape[0],
                "rows": h, "cols": w, "spacing": f"{col:.4f}x{row:.4f}x{sz:g}"}


def stage(dest: Path, store: Path = STORE) -> pd.DataFrame:
    """Every kept scan in the pool zips -> ``<dest>/<contrast>/<file>.nii.gz``; patient and scan tables in the store."""
    os.umask(0o077)
    rows = []
    for zp in sorted(POOL.glob("fastmri_plus_*.zip")):
        try:
            z = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:            # a batch still being fetched (no central directory until it closes)
            print(f"[fastmri] {zp.name}: still being written, skipped", flush=True)
            continue
        with z:
            for n in z.namelist():
                name = Path(n).stem
                ds = DATASETS_OF.get(contrast_of(name))
                if ds is None:
                    continue
                out = dest / ds / f"{name}.nii.gz"
                im, info = to_image(z.read(n))
                out.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(im, str(out), useCompression=True)
                rows.append({"case_id": name, "dataset": ds, "zip": zp.name, **info})
    df = pd.DataFrame(rows)
    (store / "labels").mkdir(parents=True, exist_ok=True)
    df[["case_id", "patient_id"]].to_csv(store / "labels" / "fastmri_patients.csv", index=False)
    df.drop(columns=["patient_id"]).to_csv(store / "labels" / "fastmri_scans.csv", index=False)
    print(f"[fastmri] staged {len(df)} scans of {df['patient_id'].nunique()} patients: "
          f"{df['dataset'].value_counts().to_dict()}", flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", type=Path, required=True)
    stage(ap.parse_args().dest)


if __name__ == "__main__":
    main()
