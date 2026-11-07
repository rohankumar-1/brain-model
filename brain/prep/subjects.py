"""Subject identity tables the index reads for its ``dedup_key`` (so one person's scans share a split).

``rsna-patients``: RSNA ICH splits were by study, and 810 patients had studies in more than one split. One DICOM
header per study (``PatientID``, read by member byte offset from the pool zip) -> ``<root>/labels/rsna_patients.csv``
(``case_id, patient_id``), which ``index.load_rsna_patients`` reads; BHSD / HemSeg studies key by the same patient.

``glioma-twins`` (2026-10-29): BraTS copies of UCSF-PDGM / UPENN-GBM patients by image content. The generic duplicate
search (``brain.prep.dedup``) needs near-identical copies and found only 26 of UCSF's 261 known BraTS twins: BraTS
re-processed the scans. Here each query patient's baseline sequences are compared with every BraTS-family volume of
the same contrast (cached ``dedup`` fingerprints, cosine); a BraTS patient's score is the mean over the shared
contrasts of its best volume, and the best patient is a twin when it beats the runner-up by ``MARGIN``. The margin was
chosen on UCSF (precision 1.0 on the TCIA pairs whose BraTS copy is in the store) and applied to UPENN. Writes the
key tables ``index`` reads: ``ucsf_pdgm/meta/brats_twins_content.csv`` (UCSF patients without a TCIA BraTS 2021
mapping) and ``upenn_gbm/meta/brats_twins.csv`` (TCIA gives UPENN no mapping), plus the full score tables.

    python -m brain.prep.subjects rsna-patients [--root STORE] [--workers 12]
    python -m brain.prep.subjects glioma-twins [--root STORE] [--out-dir DIR]     (GPU when available)
"""

from __future__ import annotations

import argparse
import io
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import PROCESSED_ROOT

GLIOMA = ("ucsf_pdgm", "upenn_gbm")
BRATS = ("brats_gli", "brats_goat", "brats_men", "brats_ped", "msd_braintumor")
SEQ = ("t1n", "t1c", "t2w", "t2f")
MARGIN = 0.05


# ---- RSNA ICH patients ---------------------------------------------------------------------------------------------

def _patient(case) -> tuple[str, str]:
    import pydicom

    from brain.prep.dicom import _read_member

    fd = os.open(case.zip_path, os.O_RDONLY)
    try:
        raw = _read_member(fd, *(int(x) for x in case.members[0][1:4]))
        ds = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True, specific_tags=["PatientID"])
        return case.case_id, str(ds.get("PatientID", ""))
    except Exception as e:                               # noqa: BLE001
        return case.case_id, f"ERR:{e!r}"[:60]
    finally:
        os.close(fd)


def rsna_patients(root: Path, workers: int) -> None:
    from brain.prep import sources

    cases = [c for c in sources.rsna() if c.members]
    with ProcessPoolExecutor(workers) as ex:
        p = pd.DataFrame(list(ex.map(_patient, cases, chunksize=64)), columns=["case_id", "patient_id"])
    v = pd.read_csv(root / "volumes.csv", usecols=["case_id", "dataset", "split"], low_memory=False)
    v = v[v.dataset == "rsna_ich"].astype({"case_id": str})
    print(f"studies read {len(p)}; in the index {int(p.case_id.isin(v.case_id).sum())} of {len(v)} RSNA volumes",
          flush=True)
    ok = ~p.patient_id.str.startswith("ERR") & p.patient_id.ne("")
    out = v[["case_id"]].merge(p[ok], on="case_id", how="inner")
    out.to_csv(root / "labels" / "rsna_patients.csv", index=False)
    m = v.merge(out, on="case_id")
    per = m.groupby("patient_id")["split"].agg(lambda s: ",".join(sorted(set(s))))
    print(f"{len(out)} studies of {len(per)} patients -> {root / 'labels' / 'rsna_patients.csv'}; patients with "
          f"studies in more than one split: {int(per.str.contains(',').sum())}", flush=True)


# ---- UCSF-PDGM / UPENN-GBM BraTS twins -----------------------------------------------------------------------------

def _seq(ds: str, contrast_col: str) -> str:
    c = str(contrast_col).lower()
    if c in SEQ:
        return c
    tail = str(ds).rsplit("_", 1)[-1]
    return {"flair": "t2f", "t1": "t1n", "t1gd": "t1c", "t2": "t2w"}.get(tail, tail)


def _twin_inputs(root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    from brain.prep import dedup

    v = pd.read_csv(root / "volumes.csv", low_memory=False).drop_duplicates("volume_id")
    v = v[v.dataset_family.isin(BRATS + GLIOMA)].copy()
    v["seq"] = [_seq(d, c) for d, c in zip(v.dataset, v.get("contrast", pd.Series("", index=v.index)).fillna(""))]
    v = v[v.seq.isin(SEQ)]
    fps = {}
    for fam in v.dataset_family.unique():
        z = np.load(dedup.FP_DIR / f"{fam}.npz")
        fps.update(zip(z["volume_ids"].astype(str), z["fp"]))
    v = v[v.volume_id.astype(str).isin(fps)]
    q = v.dataset_family.isin(GLIOMA)
    base = ~v.case_id.astype(str).str.contains("_FU") & ~v.case_id.astype(str).str.contains("_21_")
    v["patient"] = np.where(q, v.dedup_key.str.split(":", n=1).str[1], v.dedup_key)
    return v[~q | base], fps


def twin_scores(v: pd.DataFrame, fps: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per query patient: the best-scoring BraTS patient (``brats_key``), its ``score`` and ``margin`` over the
    runner-up."""
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for cohort in GLIOMA:
        per_seq = []
        for s in SEQ:
            qa = v[(v.dataset_family == cohort) & (v.seq == s)]
            br = v[v.dataset_family.isin(BRATS) & (v.seq == s)]
            if qa.empty or br.empty:
                continue
            Q = torch.from_numpy(np.stack([fps[i] for i in qa.volume_id])).to(dev).float()
            B = torch.from_numpy(np.stack([fps[i] for i in br.volume_id])).to(dev).float()
            df = pd.DataFrame((Q @ B.T).cpu().numpy(), index=qa.patient.to_numpy(), columns=br.patient.to_numpy())
            df = df.T.groupby(level=0).max().T.groupby(level=0).max()     # best volume per patient pair
            per_seq.append(df.stack().rename(s))
        m = pd.concat(per_seq, axis=1)
        m["score"] = m.mean(axis=1)                    # over the contrasts both sides have
        m["n_seq"] = m[list(SEQ)].notna().sum(axis=1) if set(SEQ) <= set(m.columns) else m.notna().sum(axis=1) - 1
        m = m.reset_index().rename(columns={"level_0": "patient", "level_1": "brats_key"})
        m = m.sort_values(["patient", "score"], ascending=[True, False])
        top = m.groupby("patient").head(2)
        best = top.groupby("patient").head(1).set_index("patient")
        second = top.groupby("patient").nth(1).set_index("patient")["score"]
        best["margin"] = best["score"] - second.reindex(best.index).fillna(0)
        best["cohort"] = cohort
        rows.append(best.reset_index())
    return pd.concat(rows, ignore_index=True)


def glioma_twins(root: Path, out_dir: Path | None) -> None:
    from brain.prep import index

    v, fps = _twin_inputs(root)
    b = twin_scores(v, fps)
    u = b[b.cohort == "ucsf_pdgm"].copy()
    truth = {f"UCSF-PDGM-{k:04d}": f"brats:gli:{t}" for k, t in index.ucsf_pdgm_brats().items()}
    in_store = set(v.loc[v.dataset_family == "brats_gli", "patient"])
    u["truth"] = u.patient.map(truth)
    u["correct"] = u.brats_key == u.truth
    known = u.truth.isin(in_store)
    call = u.margin >= MARGIN
    print(f"[ucsf] patients {len(u)}; TCIA twin in the store {int(known.sum())}; at margin >= {MARGIN}: correct "
          f"{int((call & u.correct).sum())}, wrong twin {int((call & known & ~u.correct).sum())}, no TCIA twin "
          f"{int((call & u.truth.isna()).sum())}", flush=True)
    ucsf_path = out_dir / index.UCSF_PDGM_CONTENT_TWINS.name if out_dir else index.UCSF_PDGM_CONTENT_TWINS
    upenn_path = out_dir / index.UPENN_GBM_TWINS.name if out_dir else index.UPENN_GBM_TWINS
    for cohort, path in (("ucsf_pdgm", ucsf_path), ("upenn_gbm", upenn_path)):
        b[b.cohort == cohort].to_csv(path.with_name(f"glioma_twins_{cohort}.csv"), index=False)
    method = f"(brain.prep.subjects glioma-twins, margin >= {MARGIN})"
    uc = u[call & u.truth.isna()]
    pd.DataFrame({"ucsf_id": uc.patient, "brats_id": uc.brats_key, "score": uc.score, "margin": uc.margin,
                  "method": f"content, no TCIA mapping {method}"}).to_csv(ucsf_path, index=False)
    p = b[(b.cohort == "upenn_gbm") & (b.margin >= MARGIN)]
    pd.DataFrame({"upenn_id": p.patient, "brats_id": p.brats_key, "score": p.score, "margin": p.margin,
                  "method": f"content {method}"}).to_csv(upenn_path, index=False)
    print(f"[twins] UCSF content twins {len(uc)} -> {ucsf_path}; UPENN twins {len(p)} of "
          f"{int((b.cohort == 'upenn_gbm').sum())} -> {upenn_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("rsna-patients", help="labels/rsna_patients.csv from one DICOM header per RSNA study")
    p.add_argument("--root", type=Path, default=PROCESSED_ROOT)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 12)))
    p = sub.add_parser("glioma-twins", help="UCSF-PDGM / UPENN-GBM BraTS twin keys by image content")
    p.add_argument("--root", type=Path, default=PROCESSED_ROOT)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="write the twin tables here instead of the pool meta folders the index reads")
    a = ap.parse_args()
    if a.what == "rsna-patients":
        rsna_patients(a.root, a.workers)
    else:
        glioma_twins(a.root, a.out_dir)


if __name__ == "__main__":
    main()
