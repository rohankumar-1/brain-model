"""Volume index with patient / timepoint / site columns preserved for later stages."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

def brats_patient(case_id: str) -> str:
    p = case_id
    for c in ("t1n", "t1c", "t2w", "t2f"):
        if p.endswith("-" + c):
            p = p[: -len(c) - 1]
            break
    parts = p.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else p


def oasis_subject(case_id: str) -> str:
    parts = case_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else case_id


_OASIS_TP = re.compile(r"_MR(\d+)$", re.I)
_BRATS_TP = re.compile(r"-(\d{3})(?:-|$)")


def timepoint_of(dataset: str, case_id: str) -> str:
    cid = str(case_id)
    m = _OASIS_TP.search(cid)
    if m and str(dataset).startswith("oasis"):
        return m.group(1)
    m = _BRATS_TP.search(cid)
    if m and str(dataset).startswith("brats"):
        return m.group(1)
    return ""


def patient_of(dataset: str, case_id: str, dedup_key: str) -> str:
    """Prefer the existing subject key; keep BraTS/OASIS parsers as a fallback."""
    if dedup_key:
        return str(dedup_key)
    if str(dataset).startswith("brats"):
        return brats_patient(case_id)
    if str(dataset).startswith("oasis"):
        return oasis_subject(case_id)
    return str(case_id)


def enrich(vols: pd.DataFrame) -> pd.DataFrame:
    out = vols.copy()
    if "patient_id" not in out.columns:
        out["patient_id"] = [
            patient_of(d, c, k) for d, c, k in out[["dataset", "case_id", "dedup_key"]].itertuples(index=False, name=None)
        ]
    if "timepoint" not in out.columns:
        out["timepoint"] = [timepoint_of(d, c) for d, c in out[["dataset", "case_id"]].itertuples(index=False, name=None)]
    if "site" not in out.columns:
        out["site"] = out["dataset_family"] if "dataset_family" in out.columns else out["dataset"]
    if "scanner" not in out.columns:
        out["scanner"] = ""
    if "sequence" not in out.columns:
        out["sequence"] = out["contrast"] if "contrast" in out.columns else ""
    return out


def build(root: Path, index_name: str = "volumes.csv") -> pd.DataFrame:
    vols = pd.read_csv(root / index_name)
    out = enrich(vols)
    dest = root / "brainmodel_index.parquet"
    out.to_parquet(dest, index=False)
    print(f"[index] wrote {dest} n={len(out)} patients={out['patient_id'].nunique()}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Write brainmodel_index.parquet with Stage-3 ID columns.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--index", default="volumes.csv")
    args = p.parse_args()
    build(args.root, args.index)


if __name__ == "__main__":
    main()
