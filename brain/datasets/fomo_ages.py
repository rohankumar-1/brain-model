"""Ages FOMO's participants.tsv leaves empty, from the original releases.

FOMO renamed subjects and sessions (``mapping.tsv``: ``old_path`` -> ``new_path``) and lists no age for some cohorts
(3,108 scans on 2026-10-26: NKI 1,449, M4Raw 624, Wayne 405, Beijing 180, Yale high-res 148, ...). Each fetcher maps a
cohort's (FOMO subject, session) to the original ids and reads the age from that release; the result goes to
``labels/sources/fomo260k/participants_extra.tsv`` (dataset, participant_id, session_id, age), which
``brain.multitask.labels`` uses where participants.tsv has no age.

Implemented: NKI-Rockland (per-subject ``sessions.tsv`` on the FCP-INDI bucket: one age per session).

    python -m brain.datasets.fomo_ages --cohorts nki --workers 8
"""

from __future__ import annotations

import argparse
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from brain.downstream.tasks import label_file

FCP = "https://fcp-indi.s3.amazonaws.com"
MAPPING = "/home/roku/orcd/pool/datasets/fomo260k/mapping.tsv"


def _get(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read().decode()
    except Exception:                                     # noqa: BLE001  a missing table: no age for that subject
        return None


def nki(mapping: pd.DataFrame, workers: int) -> pd.DataFrame:
    """PT027_NKI: ``old_path`` ``sub-A00075210/ses-FLU1/...`` -> that session's age in the subject's sessions.tsv."""
    m = mapping[mapping["dataset"] == "PT027_NKI"].copy()
    ids = m["old_path"].astype(str).str.extract(r"sub-(A\d+)/ses-([A-Z0-9]+)/")
    m["orig_sub"], m["orig_ses"] = ids[0], ids[1]
    pairs = m.dropna(subset=["orig_sub"]).drop_duplicates(["participant_id", "session_id"])

    def ages(sub: str) -> pd.DataFrame | None:
        t = _get(f"{FCP}/data/Projects/RocklandSample/RawDataBIDSLatest/sub-{sub}/sub-{sub}_sessions.tsv")
        return pd.read_csv(io.StringIO(t), sep="\t") if t else None

    subs = sorted(pairs["orig_sub"].unique())
    with ThreadPoolExecutor(workers) as ex:
        tables = dict(zip(subs, ex.map(ages, subs)))
    rows = []
    for r in pairs.itertuples():
        t = tables.get(r.orig_sub)
        if t is None or "age" not in t:
            continue
        hit = t[t["session"].astype(str) == r.orig_ses]
        age = pd.to_numeric(hit["age"], errors="coerce").dropna()
        if len(age):
            rows.append({"dataset": "PT027_NKI", "participant_id": r.participant_id, "session_id": r.session_id,
                         "age": float(age.iloc[0]), "source": "fcp-indi RocklandSample sessions.tsv"})
    print(f"[ages] nki: {len(pairs)} FOMO sessions, {len(subs)} subjects, {len(rows)} ages found", flush=True)
    return pd.DataFrame(rows)


COHORTS = {"nki": nki}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohorts", default=",".join(COHORTS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    mapping = pd.read_csv(MAPPING, sep="\t", low_memory=False)
    out_path = label_file("fomo260k", "participants.tsv").with_name("participants_extra.tsv")
    old = pd.read_csv(out_path, sep="\t") if out_path.exists() else pd.DataFrame()
    new = pd.concat([COHORTS[c](mapping, a.workers) for c in a.cohorts.split(",")], ignore_index=True)
    out = pd.concat([old, new], ignore_index=True).drop_duplicates(["dataset", "participant_id", "session_id"],
                                                                   keep="last")
    out.to_csv(out_path, sep="\t", index=False)
    print(f"[ages] {len(out)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
