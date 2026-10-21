"""Map stored (template-space) arrays back onto the original scan: brain masks, label masks, model predictions.

Every format-6 volume carries, in its ``image`` attributes:

* ``reg_affine``       4 x 4 row-major map from stored-grid physical points to source-scan physical points (ITK/LPS mm),
                       i.e. the fitted rigid/affine registration (identity when only centered);
* ``grid_*``           origin, spacing, direction (9, row-major) and size (x, y, z) of the stored 1 mm grid;
* ``native_*``         the same four for the source scan as prep read it (reoriented to RAS index order; its physical
                       coordinates are those of the original file, so overlays line up in any viewer);
* ``native_source``    the file it came from (``<zip>::<study>`` for scans read from a DICOM zip).

``to_native`` resamples a stored array onto the source grid (or, with ``like``, the original file's own grid and
header, which is what to hand to a physician) through the inverse map: nearest neighbour for masks
(labels keep their values), linear for images or probabilities. Volumes are exact up to that one interpolation, since
the affine's Jacobian is constant.

    python -m brain.prep.native --image STORE/brats2023_gli/BraTS-GLI-00000-000-t1c.zip --array brain --out brain.nii.gz
    python -m brain.prep.native --image STORE/cq500/CQ500-CT-1.zip --npy pred.npy --out pred_native.nii.gz
    python -m brain.prep.native --image STORE/wmh/case.zip --mask-zip STORE/wmh/case.lesion.zip --out lesion.nii.gz \
        --like /path/to/original/FLAIR.nii.gz          # exact original grid and header
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

KEYS = ("reg_affine", "grid_origin", "grid_spacing", "grid_direction", "grid_size",
        "native_origin", "native_spacing", "native_direction", "native_size")


def _affine4(tf: sitk.Transform | None) -> list[float]:
    if tf is None:
        return np.eye(4).ravel().tolist()
    from brain.prep.registration import linear_part

    a, t = linear_part(tf)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = a, t
    return m.ravel().tolist()


def geometry_attrs(img: sitk.Image, grid, tf: sitk.Transform | None, case) -> dict:
    """Everything ``to_native`` needs, as JSON-able lists (stored with the volume)."""
    src = str(case.path) if getattr(case, "path", None) else f"{case.zip_path}::{case.case_id}"
    return {
        "reg_affine": _affine4(tf),
        "grid_origin": list(map(float, grid.origin)), "grid_spacing": list(map(float, grid.spacing)),
        "grid_direction": list(map(float, grid.direction)), "grid_size": list(map(int, grid.size)),
        "native_origin": list(map(float, img.GetOrigin())), "native_spacing": list(map(float, img.GetSpacing())),
        "native_direction": list(map(float, img.GetDirection())), "native_size": list(map(int, img.GetSize())),
        "native_source": src,
    }


def has_native(attrs: dict) -> bool:
    return all(k in attrs and attrs[k] is not None for k in KEYS)


def _image(arr_zyx: np.ndarray, origin, spacing, direction) -> sitk.Image:
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr_zyx))
    img.SetOrigin(tuple(origin))
    img.SetSpacing(tuple(spacing))
    img.SetDirection(tuple(direction))
    return img


def to_native(arr_zyx: np.ndarray, attrs: dict, *, nearest: bool = True, default: float = 0.0,
              like: sitk.Image | None = None) -> sitk.Image:
    """``arr_zyx`` on the stored grid -> image on the source scan's grid (physical space of the original file).

    By default the output grid is the source as prep read it (RAS index order). Pass ``like`` (the original file,
    e.g. ``sitk.ReadImage(native_source)``) to land on that file's exact grid and index order."""
    if not has_native(attrs):
        missing = [k for k in KEYS if k not in attrs]
        raise ValueError(f"volume has no native-space record (missing {missing}); re-prep it with format-6 code that "
                         "stores the transform")
    arr = np.asarray(arr_zyx)
    if arr.ndim == 4:
        arr = arr[:, 0]
    want = tuple(attrs["grid_size"])[::-1]
    if arr.shape != want:
        raise ValueError(f"array {arr.shape} is not on the stored grid {want}")
    stored = _image(arr if nearest else arr.astype(np.float32), attrs["grid_origin"], attrs["grid_spacing"],
                    attrs["grid_direction"])
    m = np.asarray(attrs["reg_affine"], dtype=np.float64).reshape(4, 4)
    inv = np.linalg.inv(m)                                   # source physical -> stored-grid physical
    tf = sitk.AffineTransform(3)
    tf.SetMatrix(tuple(inv[:3, :3].ravel()))
    tf.SetTranslation(tuple(inv[:3, 3]))
    if like is not None:
        ref = like
    else:
        ref = sitk.Image([int(v) for v in attrs["native_size"]], sitk.sitkUInt8)
        ref.SetOrigin(tuple(attrs["native_origin"]))
        ref.SetSpacing(tuple(attrs["native_spacing"]))
        ref.SetDirection(tuple(attrs["native_direction"]))
    return sitk.Resample(stored, ref, tf, sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear, default,
                         stored.GetPixelID())


def volume_ml(mask_zyx: np.ndarray, attrs: dict, label=None) -> float:
    """True (source-space) volume in mL of a stored mask: voxel count x 1 mm^3 x |det| of the registration."""
    m = np.asarray(mask_zyx)
    n = int((m == label).sum() if label is not None else (m > 0).sum())
    det = abs(np.linalg.det(np.asarray(attrs["reg_affine"], dtype=np.float64).reshape(4, 4)[:3, :3]))
    vox = float(np.prod(attrs.get("grid_spacing", (1.0, 1.0, 1.0))))
    return n * vox * det / 1000.0


def main() -> None:
    from brain.prep import store

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True, help="stored image zip (holds the transform)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--array", help="an array inside the image zip (e.g. brain, image)")
    src.add_argument("--mask-zip", type=Path, help="a label-mask zip next to the image (<case>.<name>.zip)")
    src.add_argument("--npy", type=Path, help="a (Z, Y, X) array on the stored grid, e.g. a model prediction")
    ap.add_argument("--linear", action="store_true", help="linear interpolation (images, probabilities)")
    ap.add_argument("--like", type=Path, default=None,
                    help="the original scan file: write onto its exact grid / header (default: RAS-ordered source grid)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    attrs = store.read_attrs(a.image, "image")
    if a.array:
        arr = store.read(a.image, a.array)
    elif a.mask_zip:
        arr = store.read(a.mask_zip, "mask")
    else:
        arr = np.load(a.npy)
    out = to_native(arr, attrs, nearest=not a.linear, like=sitk.ReadImage(str(a.like)) if a.like else None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(a.out))
    print(f"[native] {a.out}  size={out.GetSize()} spacing={tuple(round(s, 3) for s in out.GetSpacing())} "
          f"source={attrs.get('native_source')}")


if __name__ == "__main__":
    main()
