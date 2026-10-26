"""Cross-dataset duplicate scans: the same acquisition redistributed under another dataset (e.g. INDI / FCP releases
inside several FOMO folders). Run after prep + index for newly added families.

1. Shortlist from the index alone: within each bucket of identical source grid (``src_n*``, ``src_d*``), each new
   volume's nearest other-subject volumes by image fingerprint (brain fraction, brain/head ratio; each in units of
   its tolerance); only pairs within ``MAX_DIST`` tolerance units are kept. Before 2026-10-25 the shortlist also used
   the registration result (scale, rotation, final MI) and bucketed by contrast: copies registered by the other
   backend (whose MI is on another scale) or labelled with another contrast string (MSD "" vs BraTS "t1n") were
   never compared.
2. Confirm by image: Pearson correlation of the two registered volumes (2x downsampled, inside the union of their
   brain masks). The same scan gives ~1.0; different people stay far below.

Writes ``labels/dedup/candidates.csv`` (shortlist with correlations) and ``labels/pool_drop/duplicates.csv``
(``volume_id`` = the new copy, dropped from pretraining by the index; ``duplicate_of`` = the copy kept).

    python -m brain.prep.dedup --new-families fomo_small --workers 16
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from brain.downstream.tasks import DEFAULT_VOLUMES
from brain.prep import store

ROOT = DEFAULT_VOLUMES.parent
OUT = ROOT / "labels" / "dedup"
DROP = ROOT / "labels" / "pool_drop" / "duplicates.csv"
GRID = ["src_nx", "src_ny", "src_nz", "src_dx", "src_dy", "src_dz"]
FEAT = {"brain_frac": 0.01, "brain_head_ratio": 0.01}        # image-derived: the same for a copy whatever registered it
MAX_DIST = 1.0
DUP_CORR = 0.98


def new_families(spec: str) -> list[str]:
    if spec == "fomo_small":
        from brain.prep.fomo import PT_SHORT, SMALL_FOLDERS

        return ["fomo_" + PT_SHORT[pt] for pt in SMALL_FOLDERS]
    return spec.split(",")


def shortlist(v: pd.DataFrame, fams: list[str]) -> pd.DataFrame:
    v = v[v["modality"].astype(str).str.lower().eq("mri")].dropna(subset=GRID).copy()
    for c in GRID[3:]:
        v[c] = v[c].astype(float).round(3)
    for c, tol in FEAT.items():
        v["f_" + c] = pd.to_numeric(v[c], errors="coerce").fillna(0.0) / tol
    feats = ["f_" + c for c in FEAT]
    out = []
    for _, b in v.groupby(GRID):
        new = b[b["dataset_family"].isin(fams)]
        if new.empty or len(b) < 2:
            continue
        tree = cKDTree(b[feats].to_numpy())
        k = min(len(b), 8)
        dist, idx = tree.query(new[feats].to_numpy(), k=k)
        dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
        for r, (ds, ix) in enumerate(zip(dist, idx)):
            me = new.iloc[r]
            for d, j in zip(ds, ix):
                o = b.iloc[j]
                if d > MAX_DIST:
                    break
                if o["volume_id"] == me["volume_id"] or o["dedup_key"] == me["dedup_key"]:
                    continue
                if o["dataset_family"] in fams and o["volume_id"] > me["volume_id"]:
                    continue                                   # new-vs-new pairs once
                out.append({"volume_id": me["volume_id"], "dataset_family": me["dataset_family"],
                            "image_path": me["image_path"], "volume_id_o": o["volume_id"],
                            "dataset_family_o": o["dataset_family"], "image_path_o": o["image_path"], "dist": d})
    return pd.DataFrame(out)


def _load(path: str) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(store.read_image(ROOT / path), np.float32)[::2, ::2, ::2]
    brain = np.asarray(store.read(ROOT / path, "brain"))[::2, ::2, ::2] > 0
    return img, brain


def _corr(args: tuple[str, str]) -> float:
    (a, ma), (b, mb) = _load(args[0]), _load(args[1])
    m = ma | mb
    if m.sum() < 1000:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-families", default="fomo_small", help="'fomo_small' or a comma list of dataset_family")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=3000, help="refuse to read more pairs than this (I/O guard)")
    a = ap.parse_args()
    fams = new_families(a.new_families)
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
    pairs = shortlist(v, fams)
    print(f"[dedup] {int(v['dataset_family'].isin(fams).sum())} volumes in {len(fams)} new families; "
          f"{len(pairs)} shortlisted pairs (fingerprint distance <= {MAX_DIST})", flush=True)
    if len(pairs) > a.max_pairs:
        raise SystemExit(f"[dedup] {len(pairs)} pairs > --max-pairs {a.max_pairs}: tighten the shortlist first")
    OUT.mkdir(parents=True, exist_ok=True)
    DROP.parent.mkdir(parents=True, exist_ok=True)
    pairs["corr"] = np.nan
    if len(pairs):
        with ProcessPoolExecutor(a.workers) as ex:
            pairs["corr"] = list(ex.map(_corr, zip(pairs["image_path"], pairs["image_path_o"]), chunksize=2))
    pairs.to_csv(OUT / "candidates.csv", index=False)
    dup = pairs[pairs["corr"] >= DUP_CORR].sort_values("corr", ascending=False).drop_duplicates("volume_id")
    dup = dup.rename(columns={"volume_id_o": "duplicate_of"})
    dup = dup.assign(reason="duplicate")[["volume_id", "reason", "duplicate_of", "dataset_family", "dataset_family_o",
                                          "corr"]]
    if DROP.exists():                      # earlier runs' duplicates (other families) stay dropped
        old = pd.read_csv(DROP)
        dup = pd.concat([old, dup], ignore_index=True).drop_duplicates("volume_id", keep="last")
    dup.to_csv(DROP, index=False)
    print(f"[dedup] duplicates (corr >= {DUP_CORR}): {len(dup)}", flush=True)
    if len(dup):
        print(dup.groupby(["dataset_family", "dataset_family_o"]).size().to_string(), flush=True)
    if len(pairs):
        print(pairs["corr"].describe().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
