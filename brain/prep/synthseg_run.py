"""Run SynthSeg-robust 2.0 (``brain.prep.synthseg``) over store volumes, one GPU per shard.

    python -m brain.prep.synthseg_run --select --n 30000           # write labels/synthseg/select.csv (once)
    python -m brain.prep.synthseg_run --shard 0 --n-shards 8       # segment this shard's volumes (skips done ones)

Per volume: ``<folder>/<case_id>.synthseg.zip`` next to the image, array ``mask`` (Z, Y, X) uint8 holding the index
into ``LABELS`` (attrs: ``labels`` = FreeSurfer ids), and a row of structure volumes (ml, from the posteriors) in
``labels/synthseg/volumes_<shard>.csv``. Reading the next volumes overlaps with the GPU through a thread pool.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from brain.downstream.tasks import DEFAULT_VOLUMES
from brain.prep import store

ROOT = DEFAULT_VOLUMES.parent
OUT = ROOT / "labels" / "synthseg"


def seg_path(row) -> Path:
    return store.path_for(ROOT, str(row.folder), str(row.case_id), "synthseg")


def select(n: int, cap_frac: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """Train/val pretraining-pool volumes, family-balanced: every family keeps at most ``cap_frac * n`` volumes and
    the remainder is drawn uniformly from what is left (small families come in whole)."""
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
    v = v[v.pretrain.astype(bool) & ~v.held_out.astype(bool) & v.split.isin(["train", "val"])]
    cap = int(cap_frac * n)
    parts = [g.sample(min(len(g), cap), random_state=seed) for _, g in v.groupby("dataset_family")]
    pick = pd.concat(parts)
    if len(pick) > n:
        pick = pick.sample(n, random_state=seed)
    return pick[["volume_id", "folder", "case_id", "image_path", "modality", "dataset_family", "split"]]


def run_shard(shard: int, n_shards: int, batch: int = 1, amp: bool = False) -> None:
    """Three overlapped stages: loader threads read + normalise into pinned memory; the GPU segments batch i + 1
    while the CPU takes batch i's largest connected component; label zips are written by threads (checked)."""
    from brain.prep.synthseg import LABELS, NAMES, Segmenter

    sel = pd.read_csv(OUT / "select.csv")
    # done = has a volume row (written after its zip); a zip without a row (shard killed in between) is redone
    done_ids = set()
    for f in OUT.glob("volumes_*.csv"):
        done_ids |= set(pd.read_csv(f, usecols=["volume_id"])["volume_id"].astype(str))
    rows = [r for i, r in enumerate(sel.itertuples()) if i % n_shards == shard and str(r.volume_id) not in done_ids]
    for r in rows:                                       # half-written zips of killed workers
        for tmp in seg_path(r).parent.glob(seg_path(r).name + ".partial.*"):
            tmp.unlink(missing_ok=True)
    print(f"[synthseg] shard {shard}/{n_shards}: {len(rows)} to do", flush=True)
    seg = Segmenter("cuda", amp=amp)
    csv = OUT / f"volumes_{shard:03d}.csv"
    cols = ["volume_id"] + [NAMES[int(l)] for l in LABELS[1:]]
    chunks = [rows[i:i + batch] for i in range(0, len(rows), batch)]

    def load(chunk):
        vols = []
        for r in chunk:
            img = store.read(ROOT / str(r.image_path), "image")
            vols.append(np.transpose(np.asarray(img[:, 0] if img.ndim == 4 else img, np.float32), (2, 1, 0)))
        return seg.prepare(vols, [str(r.modality) for r in chunk])

    def save(chunk, out):
        recs = []
        for r, (lab, vol) in zip(chunk, out):
            idx = np.searchsorted(LABELS, lab).astype(np.uint8)
            store.write(seg_path(r), "mask", idx, {"labels": LABELS.tolist(), "model": "synthseg_robust_2.0"})
            recs.append([r.volume_id, *np.round(vol, 3)])
        return recs

    t0, done, writes = time.time(), 0, []
    with ThreadPoolExecutor(6) as io:
        loads = [io.submit(load, c) for c in chunks[:4]]
        pending = None                                        # (chunk, launched GPU work) of the previous batch
        for i, chunk in enumerate(chunks + [None]):
            launched = None
            if chunk is not None:
                x = loads[i].result()
                loads[i] = None                           # drop the future: it would keep the volume alive
                if i + 4 < len(chunks):
                    loads.append(io.submit(load, chunks[i + 4]))
                launched = seg.launch(x)                      # queued on the GPU; returns at once
            if pending is not None:
                prev_chunk, prev = pending
                writes.append(io.submit(save, prev_chunk, seg.finish(*prev)))   # CPU CC overlaps the forward above
                done += len(prev_chunk)
            pending = (chunk, launched) if chunk is not None else None
            while writes and (writes[0].done() or len(writes) > 8):
                pd.DataFrame(writes.pop(0).result(), columns=cols).to_csv(csv, mode="a", header=not csv.exists(),
                                                                          index=False)
            if done and done % 50 < batch:
                el = time.time() - t0
                print(f"[synthseg] {done}/{len(rows)} {el / done:.2f}s/vol eta {(len(rows) - done) * el / done / 3600:.1f}h",
                      flush=True)
        for w in writes:
            pd.DataFrame(w.result(), columns=cols).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(f"[synthseg] shard {shard} done: {done} volumes in {(time.time() - t0) / 3600:.2f}h", flush=True)


def forget(families: list[str]) -> None:
    """Drop the selected volumes of ``families`` from every ``volumes_*.csv`` (their zips get overwritten on redo)."""
    sel = pd.read_csv(OUT / "select.csv")
    ids = set(sel.loc[sel["dataset_family"].isin(families), "volume_id"].astype(str))
    for f in sorted(OUT.glob("volumes_*.csv")):
        d = pd.read_csv(f)
        keep = ~d["volume_id"].astype(str).isin(ids)
        if (~keep).any():
            d[keep].to_csv(f, index=False)
            print(f"[synthseg] {f.name}: {int((~keep).sum())} rows of {families} forgotten", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--redo-families", default="",
                    help="comma list of dataset_family whose volumes were re-prepped (new geometry): their rows are "
                         "removed from volumes_*.csv first, so this run segments them again")
    a = ap.parse_args()
    if a.redo_families:
        forget(a.redo_families.split(","))
    if a.select:
        sel = select(a.n)
        OUT.mkdir(parents=True, exist_ok=True)
        sel.to_csv(OUT / "select.csv", index=False)
        print(sel.groupby("dataset_family").size().to_string(), f"\ntotal {len(sel)}")
    if a.shard is not None:
        run_shard(a.shard, a.n_shards, a.batch, a.amp)


if __name__ == "__main__":
    main()
