"""Per-scan label tables derived from store masks, for downstream tasks (``labels/sources/<dataset>/mask_stats.tsv``).

* ``atlas``: lesion volume and its left / right split (side of the subject's own brain-mask centroid).
* ``aisd``: voxels (ml) per infarct code: 1 remote, 2 clear acute, 3 blurred acute, 4 invisible, 5 infarct.
* ``isles24``: large-vessel occlusion side (LVO-mask centroid vs the brain centroid) and the Circle of Willis label
  (TopCoW codes) the occlusion overlaps most within 3 mm.

Store volumes are on the 1 mm RAS grid as (Z, Y, X): x increasing is the patient's right.

    python -m brain.downstream.mask_labels --datasets atlas,aisd,isles24 --workers 16
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from brain.downstream.tasks import DEFAULT_VOLUMES, LABEL_SOURCES
from brain.prep import store

ROOT = DEFAULT_VOLUMES.parent
MASK_COL = {"atlas": ("atlas", "mask_lesion"), "aisd": ("aisd", "mask_lesion"), "isles24": ("isles24_ncct", "mask_lvo")}


def _midline_x(image_path: str) -> float:
    brain = np.asarray(store.read(ROOT / image_path, "brain")) > 0
    return float(np.nonzero(brain.any(axis=(0, 1)))[0].mean()) if brain.any() else (brain.shape[2] - 1) / 2


def _sides(mask: np.ndarray, mid: float) -> tuple[float, float]:
    x = np.nonzero(mask)[2]
    return float((x < mid).sum() / 1000.0), float((x >= mid).sum() / 1000.0)


def _row(args: tuple[str, str, dict]) -> dict:
    dataset, image_path, r = args
    out = {"volume_id": r["volume_id"], "case_id": r["case_id"]}
    mid = _midline_x(image_path)
    if dataset == "atlas":
        m = store.read_image(ROOT / r["mask_lesion"], "mask") > 0
        out["lesion_ml"] = float(m.sum() / 1000.0)
        out["left_ml"], out["right_ml"] = _sides(m, mid)
    elif dataset == "aisd":
        m = store.read_image(ROOT / r["mask_lesion"], "mask")
        for code in range(1, 6):
            out[f"code{code}_ml"] = float((m == code).sum() / 1000.0)
    else:
        from scipy.ndimage import binary_dilation

        lvo = store.read_image(ROOT / r["mask_lvo"], "mask") > 0
        cow = store.read_image(ROOT / r["mask_cow"], "mask")
        out["lvo_ml"] = float(lvo.sum() / 1000.0)
        out["left_ml"], out["right_ml"] = _sides(lvo, mid)
        near = cow[binary_dilation(lvo, iterations=3)]
        near = near[near > 0]
        out["lvo_vessel"] = int(np.bincount(near).argmax()) if near.size else 0
    return out


def build(dataset: str, workers: int) -> pd.DataFrame:
    name, col = MASK_COL[dataset]
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
    v = v[(v["dataset"] == name) & v[col].notna() & (v[col].astype(str) != "")]
    jobs = [(dataset, r.image_path, r._asdict()) for r in v.itertuples(index=False)]
    with ProcessPoolExecutor(workers) as ex:
        return pd.DataFrame(list(ex.map(_row, jobs, chunksize=4)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="atlas,aisd,isles24")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for d in a.datasets.split(","):
        df = build(d, a.workers)
        out = LABEL_SOURCES / d / "mask_stats.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, sep="\t", index=False)
        print(f"[mask_labels] {d}: {len(df)} scans -> {out}", flush=True)


if __name__ == "__main__":
    main()
