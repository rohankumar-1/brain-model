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

Content mode (``--content``, the dedup step of every prep round, 2026-10-25): each stored volume gets a fingerprint
(the registered image inside its brain mask, z-scored, 8x block-averaged: 20 x 32 x 32 values, L2-normalised;
cached per family in ``labels/dedup/fingerprints/``), every query volume is compared with every stored volume of the
same modality and another subject (cosine, on the GPU when there is one), pairs above ``FP_SIM`` are confirmed by
the 2x-downsampled correlation (>= ``DUP_CORR``), and of each confirmed pair the copy that ``keep_rank`` ranks lower
is listed as a duplicate and its store files are deleted at once (prep never writes a listed volume again). This
found the BraTS-GoAT re-issues of GLI / MEN / PED cases and the MSD copies of BraTS 2017 cases, which the index-
based shortlist above could not (other contrast string, other registration backend).

    python -m brain.prep.dedup --content --query all --workers 11          # whole store
    python -m brain.prep.dedup --content --query fomo_new_family --workers 11
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


FP_SIM = 0.95
FP_DIR = OUT / "fingerprints"


def fingerprint(path: str) -> np.ndarray | None:
    """Content fingerprint of a stored volume (see the module docstring), float16, or None (no usable brain mask)."""
    try:
        arrays = store.read_many(ROOT / path, ("image", "brain"))
    except Exception:                                    # noqa: BLE001
        return None
    if "brain" not in arrays:
        return None
    img = store.squeeze_channel(arrays["image"]).astype(np.float32)
    m = store.squeeze_channel(arrays["brain"]) > 0
    if m.sum() < 5000:
        return None
    x = np.where(m, img, 0.0)
    x[m] = (x[m] - x[m].mean()) / (x[m].std() + 1e-6)
    z, y, w = (s // 8 * 8 for s in x.shape)
    v = x[:z, :y, :w].reshape(z // 8, 8, y // 8, 8, w // 8, 8).mean((1, 3, 5)).reshape(-1)
    return (v / (np.linalg.norm(v) + 1e-6)).astype(np.float16)


def update_fingerprints(v: pd.DataFrame, workers: int) -> dict[str, np.ndarray]:
    """volume_id -> fingerprint for every stored volume in ``v``, reading only those not cached yet."""
    from concurrent.futures import ProcessPoolExecutor

    FP_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    for fam, g in v.groupby("dataset_family"):
        f = FP_DIR / f"{fam}.npz"
        have: dict[str, np.ndarray] = {}
        if f.exists():
            z = np.load(f)
            have = dict(zip(z["volume_ids"].astype(str), z["fp"]))
        want = dict(zip(g["volume_id"].astype(str), g["image_path"].astype(str)))
        todo = [k for k in want if k not in have]
        if todo:
            with ProcessPoolExecutor(workers) as ex:
                fps = list(ex.map(fingerprint, [want[k] for k in todo], chunksize=8))
            have.update({k: fp for k, fp in zip(todo, fps) if fp is not None})
            keep = {k: have[k] for k in want if k in have}
            np.savez(f, volume_ids=np.asarray(list(keep)), fp=np.stack(list(keep.values())) if keep else np.zeros((0, 1)))
            print(f"[dedup] fingerprints {fam}: {len(todo)} new, {len(keep)} total", flush=True)
        out.update({k: have[k] for k in want if k in have})
    return out


def keep_rank(r) -> tuple:
    """Which copy of a duplicate pair stays (higher wins): an original over a FOMO redistribution, not GoAT, the one
    with label masks, a pretraining volume over a held-out evaluation copy (the evaluation set must not contain
    training images), then the index's SSL priority and the id (deterministic)."""
    fam = str(r["dataset_family"])
    return (not fam.startswith("fomo_"), "goat" not in fam, bool(r.get("has_mask", False)),
            not bool(r.get("held_out", False)), int(r.get("priority", 0) or 0), str(r["volume_id"]))


def content_duplicates(v: pd.DataFrame, query: list[str] | None, workers: int, chunk: int = 2048) -> pd.DataFrame:
    """Confirmed duplicate pairs (volume_id dropped, duplicate_of kept) among stored volumes; ``query`` limits the
    volumes looked up to these families (None: all)."""
    import torch
    from concurrent.futures import ProcessPoolExecutor

    fps = update_fingerprints(v, workers)
    v = v[v["volume_id"].astype(str).isin(fps)].reset_index(drop=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = []
    for mod, g in v.groupby(v["modality"].astype(str).str.lower()):
        ids = g["volume_id"].astype(str).to_numpy()
        keys = g["dedup_key"].astype(str).to_numpy()
        X = torch.from_numpy(np.stack([fps[i] for i in ids])).to(dev)
        X = X if dev.type == "cuda" else X.float()
        q_idx = np.arange(len(g)) if query is None else np.flatnonzero(g["dataset_family"].isin(query).to_numpy())
        for s in range(0, len(q_idx), chunk):
            qi = q_idx[s:s + chunk]
            sim = (X[torch.from_numpy(qi).to(dev)] @ X.T).float()
            sim[torch.arange(len(qi), device=dev), torch.from_numpy(qi).to(dev)] = -1     # not itself
            top = sim.topk(min(4, sim.shape[1]), dim=1)
            for r, (vals, cols) in enumerate(zip(top.values.cpu().numpy(), top.indices.cpu().numpy())):
                a = qi[r]
                for val, b in zip(vals, cols):
                    if val >= FP_SIM and keys[a] != keys[b]:
                        pairs.append((ids[a], ids[b], float(val)))
        print(f"[dedup] {mod}: {len(g)} volumes, {len(q_idx)} queried, {len(pairs)} candidate pairs so far", flush=True)
    if not pairs:
        return pd.DataFrame(columns=["volume_id", "duplicate_of"])
    c = pd.DataFrame(pairs, columns=["a", "b", "fp_sim"])
    c["pair"] = [tuple(sorted(x)) for x in zip(c.a, c.b)]
    c = c.drop_duplicates("pair")
    path = dict(zip(v["volume_id"].astype(str), v["image_path"].astype(str)))
    with ProcessPoolExecutor(workers) as ex:
        c["corr"] = list(ex.map(_corr, [(path[a], path[b]) for a, b in zip(c.a, c.b)], chunksize=2))
    OUT.mkdir(parents=True, exist_ok=True)
    c.drop(columns="pair").to_csv(OUT / "content_candidates.csv", index=False)
    rows = v.set_index(v["volume_id"].astype(str))
    out = []
    for r in c[c["corr"] >= DUP_CORR].itertuples():
        ra, rb = rows.loc[r.a], rows.loc[r.b]
        drop, keep = (ra, rb) if keep_rank(ra) < keep_rank(rb) else (rb, ra)
        out.append({"volume_id": drop["volume_id"], "reason": "duplicate", "duplicate_of": keep["volume_id"],
                    "dataset_family": drop["dataset_family"], "dataset_family_o": keep["dataset_family"],
                    "corr": r.corr})
    d = pd.DataFrame(out).drop_duplicates("volume_id")
    kept = set(d["duplicate_of"])                        # a chain a~b~c: never drop a volume something else depends on
    return d[~d["volume_id"].isin(kept)]


def remove_files(v: pd.DataFrame, ids: set[str]) -> tuple[int, float]:
    """Delete the store files (image zip and label-mask zips) of ``ids``: (files, GiB)."""
    r = v[v["volume_id"].astype(str).isin(ids)]
    cols = ["image_path"] + [c for c in r.columns if c.startswith("mask_") and c != "mask_brain"]
    n, b = 0, 0
    for p in {str(x) for c in cols for x in r[c].dropna().astype(str) if x and x != "nan"}:
        f = ROOT / p
        if f.exists():
            b += f.stat().st_size
            f.unlink()
            n += 1
    return n, b / 2**30


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
    ap.add_argument("--content", action="store_true", help="content fingerprints over the store (module docstring)")
    ap.add_argument("--query", default="all", help="--content: 'all' or a comma list of dataset_family to look up")
    a = ap.parse_args()
    if a.content:
        v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
        v = v[[(ROOT / p).exists() for p in v["image_path"].astype(str)]]
        dup = content_duplicates(v, None if a.query == "all" else a.query.split(","), a.workers)
        print(f"[dedup] confirmed duplicates (corr >= {DUP_CORR}): {len(dup)}", flush=True)
        if len(dup):
            print(dup.groupby(["dataset_family", "dataset_family_o"]).size().to_string(), flush=True)
            old = pd.read_csv(DROP) if DROP.exists() else pd.DataFrame()
            pd.concat([old, dup], ignore_index=True).drop_duplicates("volume_id", keep="last").to_csv(DROP, index=False)
            n, gb = remove_files(v, set(dup["volume_id"].astype(str)))
            print(f"[dedup] listed in {DROP.name}; deleted {n} store files ({gb:.1f} GiB)", flush=True)
        return
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
