"""How far CT SynthSeg is from MR SynthSeg on the same head (2026-10-30): SynthRAD task-1 patients have a CT and a T1
MR on one stored grid (the CT reuses the MR's registration), so the MR segmentation is a reference for the CT one.
Per structure group of the multitask anatomy tasks (``labels.SS_GROUPS``): Dice of CT vs MR mask, and the CT / MR
volume ratio. The baseline any CT label source (TotalSegmentator) has to beat.

    python runs/diagnostics/synthrad_ct_synthseg.py      -> runs/diagnostics/synthrad_ct_synthseg.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/roku/brain-model")
from brain.multitask.labels import SS_GROUPS  # noqa: E402
from brain.prep import store  # noqa: E402
from brain.prep.synthseg import LABELS  # noqa: E402

S = Path("/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr")
OUT = Path(__file__).with_suffix(".csv")


def main() -> None:
    v = pd.read_csv(S / "volumes.csv", low_memory=False)
    v = v[v["dataset"].isin(["synthrad_t1w", "synthrad_ct"]) & v["case_id"].astype(str).str.startswith("1B")]
    pairs = v.pivot_table(index="case_id", columns="dataset", values=["folder"], aggfunc="first").dropna()
    rows = []
    for cid, r in pairs.iterrows():
        segs = {}
        for ds in ("synthrad_t1w", "synthrad_ct"):
            p = store.path_for(S, str(r[("folder", ds)]), str(cid), "synthseg")
            if not p.exists():
                break
            segs[ds] = LABELS[np.asarray(store.read(p, "mask"))]          # index -> FreeSurfer label id
        if len(segs) < 2:
            continue
        mr, ct = segs["synthrad_t1w"], segs["synthrad_ct"]
        for g, ids in SS_GROUPS.items():
            a, b = np.isin(mr, list(ids)), np.isin(ct, list(ids))
            if not a.any():
                continue
            rows.append({"case_id": cid, "group": g, "dice": 2 * (a & b).sum() / (a.sum() + b.sum()),
                         "vol_ratio_ct_mr": b.sum() / a.sum()})
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    s = d.groupby("group").agg(n=("dice", "size"), dice_median=("dice", "median"), dice_p10=("dice", lambda x: x.quantile(0.1)),
                               vol_ratio_median=("vol_ratio_ct_mr", "median"))
    print(s.sort_values("dice_median").round(3).to_string())


if __name__ == "__main__":
    main()
