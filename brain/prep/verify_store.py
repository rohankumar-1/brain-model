"""Final check of a built store: every indexed volume is current-format, complete and traceable to its source.

Per ``volumes.csv`` row: the image zip exists, is ``store.FORMAT_VERSION``, carries the registration transform and
geometry needed to map back to the source scan (``native.has_native``), has a ``brain`` array when ``mask_brain`` is
set, and every referenced label-mask zip exists. Also writes ``excluded.csv``: cases whose latest manifest row is not
``ok`` (flagged or errored), with the reason. Exit status 1 if any indexed volume fails a check.

    python -m brain.prep.verify_store --root STORE --workers 4
"""

from __future__ import annotations

import argparse
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from brain.prep import native, store
from brain.prep.index import _recency

MASK_COLS = ("mask_seg", "mask_ich", "mask_lesion", "mask_lvo", "mask_cow")


def _arrays(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as z:
        return {n.split("/", 1)[0] for n in z.namelist() if "/" in n}


def check_row(root: Path, row: dict) -> list[str]:
    img = root / row["image_path"]
    if not img.exists():
        return ["image missing"]
    problems = []
    try:
        attrs = store.read_attrs(img, "image")
        arrays = _arrays(img)
    except Exception as e:                                   # noqa: BLE001 - any unreadable zip is a failure
        return [f"unreadable: {type(e).__name__}"]
    if attrs.get("format_version") != store.FORMAT_VERSION:
        problems.append(f"format {attrs.get('format_version')}")
    if not native.has_native(attrs):
        problems.append("no transform")
    if str(row.get("mask_brain") or "") not in ("", "nan") and "brain" not in arrays:
        problems.append("brain array missing")
    for c in MASK_COLS:
        rel = str(row.get(c) or "")
        if rel not in ("", "nan") and not (root / rel).exists():
            problems.append(f"{c} missing")
    return problems


def excluded(root: Path, pattern: str) -> pd.DataFrame:
    """Cases whose latest manifest row (resume markers ignored) is flagged or errored."""
    last = {}
    for f in sorted(root.glob(pattern)):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r["dataset"], str(r["case_id"]))
            if r.get("skipped_existing") and key in last:
                continue
            if key in last and _recency(r) < _recency(last[key]):
                continue
            last[key] = r
    bad = [r for r in last.values() if r.get("status") != "ok"]
    return pd.DataFrame(bad).reindex(columns=["dataset", "case_id", "status", "qc_reason", "error"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifests", default=f"manifest_v{store.FORMAT_VERSION}_*.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    vols = pd.read_csv(a.root / "volumes.csv", low_memory=False)
    rows = vols.to_dict("records")
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda r: check_row(a.root, r), rows, chunksize=64))
    vols["problems"] = ["; ".join(p) for p in results]
    bad = vols[vols.problems != ""]
    print(f"[verify] {len(vols)} indexed volumes, {len(bad)} with problems")
    if len(bad):
        print(bad.groupby(["dataset", "problems"]).size().to_string())
        bad[["volume_id", "dataset", "image_path", "problems"]].to_csv(a.root / "verify_problems.csv", index=False)

    ex_df = excluded(a.root, a.manifests)
    ex_df.to_csv(a.root / "excluded.csv", index=False)
    print(f"[verify] excluded (latest manifest row not ok): {len(ex_df)}")
    if len(ex_df):
        print(ex_df.groupby(["dataset", "status"]).size().to_string())

    pre = vols[vols.pretrain.astype(str).str.lower().eq("true") & vols.ssl_keep.astype(str).str.lower().eq("true")]
    print(f"\n[verify] pretraining volumes: {len(pre)}  (CT {int((pre.modality == 'ct').sum())}, "
          f"MRI {int((pre.modality != 'ct').sum())})")
    print(vols.groupby("dataset").agg(n=("volume_id", "size"),
                                      brain_ok=("brain_ok", lambda s: s.astype(str).str.lower().eq("true").mean()))
          .round(3).to_string())
    raise SystemExit(1 if len(bad) else 0)


if __name__ == "__main__":
    main()
