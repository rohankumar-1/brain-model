"""Pre-compute the live targets (``labels.live_targets``: brain / intensity asymmetry, CT CSF fraction, brain HU,
calcification, skull HU) once per volume (2026-11-04). They depend only on the stored image and its brain mask, but were
computed in every dataloader item that asked one (0.35-0.85 s each, all measures for any one asked), the largest CPU cost
of a training item. With the table present the label build writes them as plain values (``labels.src_live``), so
the loader neither computes them nor reads the brain mask for them, and their standardisation uses every training row
instead of a 1,500-scan sample.

    python -m brain.prep.live_targets --shard 0/8      -> <store>/labels/live/live_<k>.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from brain.multitask.labels import LIVE, STORE, _live_one, _mask_ok

OUT = STORE / "labels" / "live"


def load_table() -> pd.DataFrame | None:
    """volume_id -> live measures, from every shard written so far; None before the first."""
    files = sorted(OUT.glob("live_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("volume_id").set_index("volume_id") \
        if files else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    v = pd.read_csv(STORE / "volumes.csv", low_memory=False).drop_duplicates("volume_id")
    v = v[_mask_ok(v)].iloc[k::n]                      # the volumes src_live labels (QC-passed SynthStrip mask)
    if a.limit:
        v = v.head(a.limit)
    rows = []
    for i, r in enumerate(v.itertuples()):
        rows.append({"volume_id": r.volume_id, **_live_one((str(r.image_path), str(r.modality)))})
        if (i + 1) % 2000 == 0:
            print(f"[live] shard {k}: {i + 1}/{len(v)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["volume_id", *LIVE])
    df.to_parquet(OUT / f"live_{k:02d}.parquet", index=False)
    print(f"[live] shard {k}/{n}: {len(df)} volumes -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
