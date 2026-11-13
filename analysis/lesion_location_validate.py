"""Lesion location from the scan's own anatomy (2026-11-12): features per BHSD subtype mask, to choose and validate
a compartment rule (the template-prior rule failed: EDH / SDH 87% "hemispheric", IVH 19% intraventricular).

Per scan and BHSD subtype id (1 EDH, 2 IPH, 3 IVH, 4 SAH, 5 SDH): share of the subtype's voxels
* near the inner skull (bone >= BONE_HU), at 3 / 6 / 10 mm;
* in components touching the skull band (<= 3 mm);
* in the scan's own ventricles (CSF density inside the dilated ventricle prior, closed so blood-filled ventricle
  is included) and in the plain ventricle prior;
* in the deep grey and posterior fossa priors.
Aggregates only, no case ids, written to <out>/features.csv (rows keyed by a running index).

    python analysis/lesion_location_validate.py --out ~/mask_labels_work/location --workers 8
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.downstream.tasks import DEFAULT_VOLUMES  # noqa: E402
from brain.multitask import lesion_measures as LM  # noqa: E402
from brain.prep import store  # noqa: E402

ROOT = DEFAULT_VOLUMES.parent
PRIORS: dict = {}
SUBTYPES = {1: "EDH", 2: "IPH", 3: "IVH", 4: "SAH", 5: "SDH"}


def _init(prior_path: str) -> None:
    PRIORS.update(LM.compartment_priors(Path(prior_path)))


def job(args) -> list[dict]:
    i, image_path, ref = args
    a = store.read_many(ROOT / image_path, ("image", "brain"))
    hu = store.squeeze_channel(a["image"]).astype(np.float32)
    brain = store.squeeze_channel(a["brain"]) > 0
    m = np.asarray(store.read_image(ROOT / ref, "mask")).astype(int)
    anat = LM.ct_anatomy(hu, brain, PRIORS)
    rows = []
    for sid, name in SUBTYPES.items():
        les = m == sid
        if les.sum() < 50:
            continue
        f = LM.location_features(les, anat, PRIORS)
        rows.append({"i": i, "subtype": name, "ml": les.sum() / 1000.0, **f,
                     "rule": LM.LOCATION_CLASSES[LM.location_class_ct(les, anat, PRIORS)]})
    alls = m > 0
    f = LM.location_features(alls, anat, PRIORS)
    present = "+".join(SUBTYPES[s] for s in SUBTYPES if (m == s).sum() >= 50)
    rows.append({"i": i, "subtype": "ALL:" + present, "ml": alls.sum() / 1000.0, **f,
                 "rule": LM.LOCATION_CLASSES[LM.location_class_ct(alls, anat, PRIORS)]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--priors", type=Path, default=Path.home() / "mask_labels_work/validate/ixi_synthseg_compartments.npz")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
    b = v[(v["dataset_family"] == "bhsd") & v["mask_ich"].notna()]
    if a.limit:
        b = b.head(a.limit)
    jobs = [(k, r.image_path, r.mask_ich) for k, r in enumerate(b.itertuples())]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(a.priors),)) as ex:
        rows = [r for rs in ex.map(job, jobs, chunksize=2) for r in rs]
    f = pd.DataFrame(rows)
    f.to_csv(a.out / "features.csv", index=False)
    sub = f[~f.subtype.str.startswith("ALL")]
    cols = [c for c in f.columns if c.startswith("share_")]
    print(sub.groupby("subtype")[cols].median().round(2).to_string())
    print()
    print(pd.crosstab(sub.subtype, sub.rule).to_string())


if __name__ == "__main__":
    main()
