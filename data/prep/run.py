"""Preprocess datasets into the store: sharded, resumable, flagged volumes excluded and listed.

    python -m data.prep.run --root STORE --datasets cq500,oasis,brats,rsna --shard 0/8 --workers 16
    python -m data.prep.run --root STORE --summary      # per-dataset timing and status counts
    python -m data.prep.run --root STORE --purge --yes  # delete store files this pipeline did not write
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import pandas as pd

from data.prep import pipeline, sources, store

STAGES = ("read", "n4", "center", "resample", "intensity")


def _plain(v):
    v = v.item() if hasattr(v, "item") else v
    return None if isinstance(v, float) and not math.isfinite(v) else v


def read_manifests(root: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_json(f, lines=True) for f in sorted(root.glob("manifest_*.jsonl"))], ignore_index=True)
    return df.drop_duplicates(["dataset", "case_id"], keep="last", ignore_index=True)  # a reprocessed case counts once


def discover(names: list[str]) -> list[sources.Case]:
    d = sources.DATASETS
    found = {"cq500": lambda: sources.cq500(d / "seg_cq500" / "volumes", d / "cq500"),
             "oasis": sources.oasis, "brats": sources.brats, "rsna": sources.rsna,
             "abide": sources.abide1, "ixi": sources.ixi, "isles": sources.isles22, "atlas": sources.atlas,
             "pixar": sources.pixar, "piop1": sources.piop1, "piop2": sources.piop2, "id1000": sources.id1000,
             "wmh": sources.wmh, "isles24": sources.isles24, "instance": sources.instance,
             "aisd": sources.aisd, "brainmetshare": sources.brainmetshare, "fomo": sources.fomo,
             "sinoct": sources.sinoct, "msd": sources.msd}
    return sorted((c for n in names for c in found[n]()), key=lambda c: (c.dataset, c.case_id))


def _remove(case, root: Path) -> None:
    image, masks = store.paths(root, case)
    for p in (image, *masks.values()):
        p.unlink(missing_ok=True)


def work(args) -> dict:
    """Process one case. Flagged or failed volumes are not stored and any old file is removed."""
    case, root, z_anchor = args
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    row = {"dataset": case.dataset, "case_id": case.case_id}
    t0 = time.perf_counter()
    try:
        res = pipeline.process(case, z_anchor=z_anchor)
        row |= res.info | {f"t_{k}": v for k, v in res.timing.items()}
        if res.info["qc_flag"]:
            _remove(case, root)
            row["status"] = "flagged"
        else:
            image, masks = store.paths(root, case)
            store.write(image, "image", res.vol[:, None], {**res.info, "kind": "ct_hu" if case.modality == "ct" else "mri_z"})
            for name, m in res.masks.items():
                store.write(masks[name], "mask", m)
            row["status"] = "ok"
    except Exception:
        _remove(case, root)
        row |= {"status": "error", "error": traceback.format_exc(limit=2)[-400:]}
    row["seconds"] = time.perf_counter() - t0
    return row


def summary(root: Path, names: list[str]) -> None:
    df = read_manifests(root)
    t = [f"t_{s}" for s in STAGES if f"t_{s}" in df]
    g = df.groupby("dataset").agg(n=("case_id", "size"), ok=("status", lambda s: (s == "ok").sum()),
                                  flagged=("status", lambda s: (s == "flagged").sum()),
                                  error=("status", lambda s: (s == "error").sum()),
                                  sec_mean=("seconds", "mean"), sec_p95=("seconds", lambda s: s.quantile(0.95)),
                                  **{c: (c, "mean") for c in t})
    print(g.round(2).to_string())
    total = pd.Series([c.folder for c in discover(names)]).value_counts()
    per = df.assign(folder=df.dataset.map(lambda d: next((d[: -len(c) - 1] for c in sources.BRATS_CONTRASTS
                                                          if d.endswith("_" + c)), d))).groupby("folder").seconds.mean()
    est = (total * per.reindex(total.index)).dropna()
    print(f"\nfull run estimate (single-core seconds summed): {est.sum() / 3600:.0f} core-hours")
    print((est / 3600).round(1).rename("core-hours").to_string())
    for w in (16, 32, 64, 128):
        print(f"  wall time on {w:3d} cores (ideal scaling): {est.sum() / 3600 / w:.1f} h")
    flagged = df[df.status != "ok"].reindex(columns=["dataset", "case_id", "status", "qc_reason", "error"])
    flagged.to_csv(root / "excluded.csv", index=False)
    print(f"excluded volumes listed in {root / 'excluded.csv'} ({len(flagged)})")


def purge(root: Path, names: list[str], yes: bool) -> None:
    keep = set()
    done = read_manifests(root)
    ok = set(done.loc[done.status == "ok", "case_id"])
    for c in discover(names):
        if c.case_id in ok:
            image, masks = store.paths(root, c)
            keep |= {image, *masks.values()}
    stale = [p for p in root.glob("*/*.zip*") if p not in keep]
    print(f"{len(keep)} files kept, {len(stale)} stale")
    if yes:
        for p in stale:
            p.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--datasets", default="cq500,oasis,brats,rsna")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit-per-dataset", type=int, default=0, help="first N cases of each dataset (benchmarks)")
    ap.add_argument("--z-anchor", choices=["auto", "volume", "brain"], default="auto")
    ap.add_argument("--match", default="", help="only cases whose dataset tag contains this substring")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest jsonl (default <root>/manifest_{shard}.jsonl)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--purge", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    names = a.datasets.split(",")
    if a.summary:
        return summary(a.root, names)
    if a.purge:
        return purge(a.root, names, a.yes)

    i, n = map(int, a.shard.split("/"))
    cases = [c for c in discover(names) if a.match in c.dataset]
    if a.limit_per_dataset:  # evenly spaced sample from each store folder (benchmarks)
        groups: dict[str, list] = {}
        for c in cases:
            groups.setdefault(c.folder, []).append(c)
        cases = [c for g in groups.values() for c in g[:: max(1, len(g) // a.limit_per_dataset)][: a.limit_per_dataset]]
    cases = cases[i::n]
    manifest = Path(a.manifest) if a.manifest else a.root / f"manifest_{i}of{n}.jsonl"
    if not manifest.is_absolute():
        manifest = a.root / manifest
    a.root.mkdir(parents=True, exist_ok=True)
    prev = pd.read_json(manifest, lines=True) if manifest.exists() and not a.force else None
    done = set(zip(prev.dataset, prev.case_id)) if prev is not None else set()  # ids repeat across BraTS cohorts
    todo = [c for c in cases if (c.dataset, c.case_id) not in done]
    if not a.force:
        keep, skipped = [], []
        for c in todo:
            if store.paths(a.root, c)[0].exists():
                skipped.append(c)
            else:
                keep.append(c)
        todo = keep
        if skipped:
            print(f"[run] skipping {len(skipped)} cases with existing zips", flush=True)
            with open(manifest, "a") as out:
                for c in skipped:
                    out.write(json.dumps({"dataset": c.dataset, "case_id": c.case_id,
                                          "status": "ok", "skipped_existing": True}) + "\n")
    print(f"[run] shard {i}/{n}: {len(cases)} cases, {len(done)} already done, {len(todo)} to do", flush=True)

    t0 = time.time()
    with mp.Pool(a.workers, maxtasksperchild=100) as pool, open(manifest, "a") as out:
        for k, row in enumerate(pool.imap_unordered(work, [(c, a.root, a.z_anchor) for c in todo]), 1):
            out.write(json.dumps({k: _plain(v) for k, v in row.items()}) + "\n")
            out.flush()
            if k % 50 == 0 or row["status"] == "error":
                print(f"[run] {k}/{len(todo)} {row['dataset']}/{row['case_id']} {row['status']} "
                      f"({(time.time() - t0) / k:.1f} s/case wall)", flush=True)


if __name__ == "__main__":
    main()
