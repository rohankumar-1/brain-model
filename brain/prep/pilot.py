"""Pilot: run the v4 pipeline on a small dataset; save volumes, QC figures and a summary table.

    python -m brain.prep.pilot --dataset cq500 --n 0 --workers 8     # n=0 means all volumes
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import REPO_ROOT as ROOT, SCRATCH
from brain.prep import pipeline, qc, sources, store



def _run(args):
    case, out_root, fig_dir, z_anchor, opts = args
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    t0 = time.time()
    try:
        res = pipeline.process(case, z_anchor=z_anchor, keep_debug=True, opts=opts)
        store.write_case(out_root, case, res)
        png = Path(fig_dir) / f"{case.dataset}__{case.case_id}.png"
        qc.overview(png, case, res)
        return {"dataset": case.dataset, "case_id": case.case_id, "png": str(png), "seconds": time.time() - t0,
                **{k: v for k, v in res.info.items()}, **_mask_truth(res)}
    except Exception:
        return {"dataset": case.dataset, "case_id": case.case_id, "error": traceback.format_exc(limit=3)}


def _mask_truth(res) -> dict:
    """Distance (mm) from the crop center to the ground-truth brain mask centroid (Seg-CQ500 only)."""
    m = res.masks.get("brain")
    if m is None or not m.any():
        return {}
    area = m.sum((1, 2))
    zs = np.flatnonzero(area >= 0.6 * area.max())
    cy = np.median([np.nonzero(m[z])[0].mean() for z in zs])
    cx = np.median([np.nonzero(m[z])[1].mean() for z in zs])
    return {"brain_mask_offset_mm": float(np.hypot(cx - 127.5, cy - 127.5))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cq500", "oasis1", "oasis2", "brats", "abide", "ixi", "isles", "atlas", "pixar", "piop1", "piop2",
                                        "id1000", "wmh", "isles24"],
                    required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z-anchor", choices=["auto", "volume", "brain"], default="auto")
    ap.add_argument("--filter", default="", help="keep cases whose dataset tag contains this substring")
    pipeline.add_cli_options(ap)
    a = ap.parse_args()
    opts = pipeline.options_from_args(a)

    if a.dataset == "cq500":
        cases = sources.cq500(sources.DATASETS / "seg_cq500" / "volumes", sources.DATASETS / "cq500")
    elif a.dataset == "brats":
        cases = sources.brats()
    elif a.dataset in ("abide", "ixi", "isles", "atlas", "pixar", "piop1", "piop2", "id1000", "wmh", "isles24"):
        cases = {"abide": sources.abide1, "ixi": sources.ixi, "isles": sources.isles22, "atlas": sources.atlas,
                 "pixar": sources.pixar, "piop1": sources.piop1, "piop2": sources.piop2, "id1000": sources.id1000,
                 "wmh": sources.wmh, "isles24": sources.isles24}[a.dataset]()
        if a.dataset in ("isles", "wmh", "isles24"):  # keep whole subjects; draw the lesion mask on the other sequences to check alignment
            sub = (lambda c: c.case_id.rsplit("_", 1)[0]) if a.dataset == "isles" else (lambda c: c.case_id)
            per = 3 if a.dataset == "isles" else 2
            keep = sorted({sub(c) for c in cases})
            if a.n:
                step = max(len(keep) // max(a.n // per, 1), 1)  # spread over sites rather than the first few
                keep = keep[::step][: max(a.n // per, 1)]
            cases = [c for c in cases if sub(c) in keep]
            a.n = 0
            lesion = {sub(c): c.masks for c in cases if c.masks}
            cases = [dataclasses.replace(c, masks=lesion.get(sub(c), ())) for c in cases]
    else:
        cases = [c for c in sources.oasis() if c.dataset == a.dataset]
    cases = [c for c in cases if a.filter in c.dataset]
    if a.n:
        random.Random(a.seed).shuffle(cases)
        cases = cases[: a.n]
    tag = a.dataset + (a.filter if a.filter.startswith("_") else "") + ("_zbrain" if a.z_anchor == "brain" else "")
    out_root = SCRATCH / "data" / "processed" / "slices_zarr_v4_pilot" / ("zbrain" if a.z_anchor == "brain" else "zvolume")
    fig_dir = ROOT / "runs" / "prep_pilot" / tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pilot] {len(cases)} cases -> {out_root}, figures -> {fig_dir}", flush=True)

    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(_run, [(c, out_root, fig_dir, a.z_anchor, opts) for c in cases]), 1):
            rows.append(r)
            if i % 10 == 0 or "error" in r:
                print(f"[pilot] {i}/{len(cases)} {r.get('case_id')} {'ERROR ' + r['error'] if 'error' in r else ''}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(fig_dir / "summary.csv", index=False)
    print(df.drop(columns=["png", "error"], errors="ignore").describe(include="all").T.to_string(), flush=True)


if __name__ == "__main__":
    main()
