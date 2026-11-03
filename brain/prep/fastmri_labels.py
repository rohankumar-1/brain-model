"""fastMRI+ brain labels (2026-11-02) for the multitask model, from the fastMRI+ annotations (``Annotations/brain.csv``:
one row per box or study-level label; ``brain_file_list.csv``: every reviewed file).

* ``labels/fastmri_findings.csv``  per scan, the nine scan-level findings (``FINDINGS``): 1 when the file carries one
  of the finding's labels, 0 when it was reviewed without them.
* ``labels/fastmri_boxes.parquet`` per scan, the boxes of the five focal findings with boxes (``BOX_FINDINGS``; craniotomy,
  artifacts, nonspecific lesions, normal variants, dural thickening, sinus opacification and enlarged ventricles are
  left out as not focal or not what the finding is) as native continuous-index cuboids (k slice, j row, i column of
  the flipped image ``brain.prep.fastmri`` stored, the frame the boxes are drawn in), and the affine from stored voxel
  index (z, y, x) to that native index, so training can map its tokens onto the boxes
  (``brain.multitask.data.token_mil_labels``).

    python -m brain.prep.fastmri_labels
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from brain.common.paths import SCRATCH
from brain.prep import native, store

STORE = SCRATCH / "data" / "processed" / "slices_zarr"
ANN = SCRATCH / "work" / "fastmri" / "fastmri-plus" / "Annotations"
# task id -> the fastMRI+ labels that make it positive
FINDINGS = {
    "mri_normal_for_age": ["Normal for age"],
    "mri_small_vessel_disease": ["Global label: Small vessel chronic white matter ischemic change"],
    "mri_wm_lesion": ["Nonspecific white matter lesion"],
    "mri_postop": ["Posttreatment change", "Resection cavity", "Craniotomy", "Craniectomy",
                   "Craniectomy with Cranioplasty"],
    "mri_mass": ["Mass", "Extra-axial mass"],
    "mri_edema": ["Edema"],
    "mri_enlarged_ventricles": ["Enlarged ventricles"],
    "mri_infarct": ["Lacunar infarct", "Encephalomalacia"],
    "mri_artifact": ["Possible artifact", "Motion artifact"],
}
# box channels (order = channel index after the RSNA slice channels): the boxes that mark the finding itself
BOX_FINDINGS = {
    "mri_wm_lesion": ["Nonspecific white matter lesion"],
    "mri_postop": ["Posttreatment change", "Resection cavity"],
    "mri_mass": ["Mass", "Extra-axial mass"],
    "mri_edema": ["Edema"],
    "mri_infarct": ["Lacunar infarct", "Encephalomalacia"],
}


def index_to_native(attrs: dict, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    """(3, 4) affine: stored voxel index (z, y, x, 1) -> native continuous index (k, j, i) of the flipped fastMRI array
    (LPS, identity direction, origin 0, ``spacing_xyz`` = column, row, slice spacing; brain.prep.fastmri.to_image)."""
    tf = native.stored_transform(attrs)
    m = np.asarray(tf.GetMatrix(), float).reshape(3, 3)
    t = np.asarray(tf.GetTranslation(), float)
    ctr = np.asarray(tf.GetCenter(), float)
    o = np.asarray(attrs["grid_origin"], float)
    d = np.asarray(attrs["grid_direction"], float).reshape(3, 3)
    sp = np.asarray(attrs["grid_spacing"], float)
    a = m @ d @ np.diag(sp)                      # stored (x, y, z) index -> source physical
    b = m @ (o - ctr) + ctr + t
    inv = np.diag(1.0 / np.asarray(spacing_xyz, float))   # source physical -> native (i, j, k)
    a_ijk, b_ijk = inv @ a, inv @ b
    # rows (k, j, i) from (i, j, k); columns (z, y, x) from (x, y, z)
    return np.c_[a_ijk[::-1, ::-1], b_ijk[::-1]]


def main() -> None:
    os.umask(0o077)
    ann = pd.read_csv(ANN / "brain.csv")
    files = [f.strip() for f in open(ANN / "brain_file_list.csv") if f.strip()]
    scans = pd.read_csv(STORE / "labels" / "fastmri_scans.csv").set_index("case_id")
    have = {f for f in files if store.path_for(STORE, "fastmri", f).exists()}
    by_file = ann.groupby("file")["label"].apply(set).to_dict()
    find = pd.DataFrame({"case_id": sorted(have)})
    for t, labs in FINDINGS.items():
        find[t] = [int(bool(by_file.get(f, set()) & set(labs))) for f in find["case_id"]]
    find.to_csv(STORE / "labels" / "fastmri_findings.csv", index=False)
    print("[fastmri-labels] findings (positives of", len(find), "scans):",
          {t: int(find[t].sum()) for t in FINDINGS}, flush=True)

    boxes = ann[ann["x"].notna() & ann["file"].isin(have)]
    rows, n_box = [], 0
    for f in sorted(have):
        attrs = store.read_attrs(store.path_for(STORE, "fastmri", f), "image")
        sx, sy, sz = (float(v) for v in scans.at[f, "spacing"].split("x"))
        a = index_to_native(attrs, (sx, sy, sz))
        bx = []
        for ch, (t, labs) in enumerate(BOX_FINDINGS.items()):
            for b in boxes[(boxes["file"] == f) & boxes["label"].isin(labs)].itertuples():
                # pixel centres at integer indices: rows y..y+h-1 span [y - 0.5, y + h - 0.5)
                bx.append([ch, b.slice - 0.5, b.slice + 0.5, b.y - 0.5, b.y + b.height - 0.5,
                           b.x - 0.5, b.x + b.width - 0.5])
        n_box += len(bx)
        rows.append({"case_id": f, "affine": a.ravel().tolist(),
                     "shape": [int(scans.at[f, "n_slices"]), int(scans.at[f, "rows"]), int(scans.at[f, "cols"])],
                     "boxes": [x for b in bx for x in b]})
    pd.DataFrame(rows).to_parquet(STORE / "labels" / "fastmri_boxes.parquet", index=False)
    print(f"[fastmri-labels] {n_box} focal boxes over {sum(1 for r in rows if r['boxes'])} scans", flush=True)


if __name__ == "__main__":
    main()
