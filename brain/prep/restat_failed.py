"""Re-standardise MRI volumes whose robust z-scores came from the whole head (2026-11-03 audit).

``pipeline`` takes the MRI z-score statistics from the brain mask only when that mask passed QC; otherwise from the
head foreground (``stats_from = head``), scalp and skull included, which puts the brain off the store's scale
(RSNA-ICA MRA: brain median 0.54, spread 0.36 instead of 0 / 1). Most of these masks failed only the template-Dice
check (partial-coverage slabs, infant brains against an adult template) and are fine, so this re-standardises on
the stored ``brain`` array (``intensity.mri_restat``, the same affine correction as ``remask --restat``, which skips
failed masks). Volumes whose mask extent is implausible (``brain_frac`` outside [0.02, 0.30]) are left alone.

The index is not changed (its sha1 pins benchmark results). Each rewritten zip records ``stats_from =
brain_restat`` and ``restat_m`` / ``restat_s`` (z_new = (z_old - m) / s, so it can be undone up to clip
saturation); every change is listed in ``<root>/labels/restat/restat_<shard>.jsonl``.

    python -m brain.prep.restat_failed --root STORE [--shard 0/4] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brain.prep import intensity, store
from brain.prep.remask import _squeeze

FRAC_RANGE = (0.02, 0.30)


def candidates(root: Path) -> pd.DataFrame:
    v = pd.read_csv(root / "volumes.csv", low_memory=False)
    mri = ~v["modality"].astype(str).str.lower().str.startswith("ct")
    ok = v["brain_ok"].astype(str).str.lower().isin(["true", "1", "1.0"])
    frac = pd.to_numeric(v["brain_frac"], errors="coerce")
    sel = mri & ~ok & frac.between(*FRAC_RANGE)
    return v.loc[sel, ["volume_id", "dataset", "image_path"]].drop_duplicates("image_path")


def restat_one(path: Path, dry_run: bool) -> dict:
    arrays, attrs = store.read_all(path)
    a = {k: v for k, v in attrs["image"].items() if k != "format_version"}
    if a.get("stats_from") != "head":
        return {"status": "skip", "stats_from": a.get("stats_from")}
    if "brain" not in arrays:
        return {"status": "no_mask"}
    vol, brain = _squeeze(arrays["image"]), _squeeze(arrays["brain"]) > 0
    res = intensity.mri_restat(vol, brain)
    if res is None:
        return {"status": "empty_mask"}
    z, m, s = res
    if abs(m) < 0.02 and abs(s - 1.0) < 0.02:          # already on scale (a stripped input: head = brain)
        return {"status": "unchanged", "m": round(m, 4), "s": round(s, 4)}
    if dry_run:
        return {"status": "would", "m": round(m, 4), "s": round(s, 4)}
    med, mad = a.get("raw_median"), a.get("raw_mad")
    a |= {"stats_from": "brain_restat", "restat_m": m, "restat_s": s}
    if med is not None and mad is not None:          # z = (raw - med) / mad, so z' = (raw - (med + mad m)) / (mad s)
        a |= {"raw_median": float(med) + float(mad) * m, "raw_mad": float(mad) * s}
    image = z[:, None] if arrays["image"].ndim == 4 else z
    out = {"image": (image, a)}
    out |= {n: (arr, {k: v for k, v in attrs[n].items() if k != "format_version"})
            for n, arr in arrays.items() if n != "image"}
    store.write_arrays(path, out)
    return {"status": "ok", "m": m, "s": s}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    rows = candidates(a.root).iloc[k::n]
    if a.limit:
        rows = rows.head(a.limit)
    log_dir = a.root / "labels" / "restat"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"restat_{k}{'_dry' if a.dry_run else ''}.jsonl"
    counts: dict[str, int] = {}
    with open(log, "a") as fh:
        for r in rows.itertuples():
            try:
                res = restat_one(a.root / str(r.image_path), a.dry_run)
            except Exception as e:                          # one unreadable zip must not stop the shard
                res = {"status": "error", "error": str(e)[:200]}
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            fh.write(json.dumps({"volume_id": r.volume_id, "dataset": r.dataset, **res}) + "\n")
    print(f"[restat] shard {k}/{n}: {len(rows)} candidates {counts} -> {log}", flush=True)


if __name__ == "__main__":
    main()
