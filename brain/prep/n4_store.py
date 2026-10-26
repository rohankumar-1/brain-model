"""Bias-field correction (N4) of stored MRI volumes whose prep skipped it.

Until 2026-10-25 ``pipeline.needs_n4`` corrected the raw structural MRI cohorts (FOMO, OASIS-2, ABIDE, IXI, ...) but
not BraTS, MSD, BrainMetShare, VALDO or MS3SEG, so the residual bias field differed by cohort (median log-field spread
0.030 vs 0.021, BraTS-PED 0.079; ``runs/diagnostics/n4_check.py``), and among tumour types only the metastases
(Yale, a FOMO cohort) were corrected. ``needs_n4`` now covers them for future preps; this pass corrects what is stored.

A stored MRI value is ``z = (v - raw_median) / raw_mad`` (``intensity.mri_native``), so ``z + raw_median / raw_mad =
v / raw_mad``: the raw intensities up to a scale, which a multiplicative bias model does not see. N4 is fitted on that
inside the head foreground (on a 4x shrunk copy, applied at full size), then the volume is re-standardised exactly as
prep does (robust z from the brain mask, clipped, background at the floor) and rewritten with ``n4 = "store"`` and the
new raw-unit median / MAD. The registration, the grid and every mask stay as they are. Resumable: volumes whose image
already carries an ``n4`` attribute are skipped.

    python -m brain.prep.n4_store --shard 0/4 --workers 11
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from brain.prep import store
from brain.prep.intensity import MRI_FLOOR, MRI_HI, MRI_LO

SHRINK = 4


def legacy_n4(dataset: str) -> bool:
    """The rule prep used before 2026-10-25 (these volumes were corrected natively)."""
    from brain.prep.pipeline import LEGACY_N4_DATASETS

    return dataset in LEGACY_N4_DATASETS or str(dataset).startswith("fomo_")


def targets(volumes: pd.DataFrame) -> pd.DataFrame:
    """Stored MRI that the current rule corrects and the old one did not (BrainMetShare is re-prepped instead)."""
    from brain.prep.pipeline import needs_n4

    ds = volumes["dataset"].astype(str)
    sel = volumes["modality"].astype(str).str.lower().eq("mri") & ds.map(needs_n4) & ~ds.map(legacy_n4) \
        & ~ds.str.startswith("brainmetshare")
    return volumes[sel]


def correct(z: np.ndarray, brain: np.ndarray | None, raw_median: float, raw_mad: float) -> tuple[np.ndarray, dict]:
    """Stored robust-z volume (Z, Y, X) -> N4-corrected robust-z volume and the new raw-unit stats."""
    import SimpleITK as sitk

    z = z.astype(np.float32)
    fg = z > MRI_LO - 0.5
    x = np.where(fg, z + raw_median / max(raw_mad, 1e-6), 0.0).astype(np.float32)
    x = np.where(fg, np.maximum(x, 1e-3), 0.0)                 # strictly positive inside the mask for the log model
    img, mask = sitk.GetImageFromArray(x), sitk.GetImageFromArray(fg.astype(np.uint8))
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations([50, 50, 50])
    n4.Execute(sitk.Shrink(img, [SHRINK] * 3), sitk.Shrink(mask, [SHRINK] * 3))
    field = sitk.GetArrayFromImage(n4.GetLogBiasFieldAsImage(img))
    y = x / np.exp(field)
    ref = (brain.astype(bool) & fg) if brain is not None and (brain.astype(bool) & fg).any() else fg
    vals = y[ref]
    med = float(np.median(vals))
    mad = float(1.4826 * np.median(np.abs(vals - med)))
    out = np.clip((y - med) / max(mad, 1e-6), MRI_LO, MRI_HI)
    out[~fg] = MRI_FLOOR
    stats = {"n4": "store", "raw_median": med * raw_mad, "raw_mad": mad * raw_mad,
             "n4_field_spread": float(np.std(field[ref]))}
    return out.astype(np.float16), stats


def process(path: str) -> tuple[str, str]:
    try:
        arrays, attrs = store.read_all(Path(path))
        a = attrs.get("image", {})
        if a.get("n4"):
            return path, "done"
        if a.get("raw_median") is None or a.get("raw_mad") is None:
            return path, "no-stats"
        image = arrays["image"]
        z = store.squeeze_channel(image)
        brain = store.squeeze_channel(arrays["brain"]) if "brain" in arrays else None
        new, stats = correct(z, brain, float(a["raw_median"]), float(a["raw_mad"]))
        arrays["image"] = new[:, None] if image.ndim == 4 else new
        attrs["image"] = {**a, **stats}
        store.write_arrays(Path(path), {n: (arrays[n], {k: v for k, v in attrs[n].items() if k != "format_version"})
                                        for n in arrays})
        return path, "ok"
    except Exception as e:                                     # noqa: BLE001  reported, retried on the next run
        return path, f"error: {e!r}"[:200]


def main() -> None:
    from brain.downstream.tasks import DEFAULT_VOLUMES

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N targets only (a check)")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    root = DEFAULT_VOLUMES.parent
    v = targets(pd.read_csv(DEFAULT_VOLUMES, low_memory=False)).sort_values("volume_id")
    paths = [str(root / p) for p in v["image_path"]][i::n]
    paths = [p for p in paths if os.path.exists(p)]
    if a.limit:
        paths = paths[: a.limit]
    print(f"[n4-store] shard {i}/{n}: {len(paths)} volumes ({v['dataset_family'].value_counts().to_dict()} overall)",
          flush=True)
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (p, status) in enumerate(ex.map(process, paths, chunksize=4), 1):
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if status.startswith("error") or k % 500 == 0:
                print(f"[n4-store] {k}/{len(paths)} {counts} {p if status.startswith('error') else ''} {status}",
                      flush=True)
    print(f"[n4-store] done {counts}", flush=True)
    if counts.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    main()
