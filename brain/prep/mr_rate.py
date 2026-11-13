"""MR-RATE (Forithmus/MR-RATE on Hugging Face; gated, CC BY-NC-SA 4.0, academic use only, no redistribution): clinical
brain MRI studies of one Istanbul hospital network with radiology reports, study-level pathology labels (37 report
categories and 74 NeuroVFM diagnoses, both from an LLM pipeline) and patient-level splits.

We keep a label-targeted subset of head studies, one series per contrast (T1 pre-contrast, T1 post-contrast, T2,
FLAIR): see ``select``. The images are not archived on pool (Hugging Face keeps the originals; pool is ~92% full):
``fetch`` streams study zips onto local disk, keeps only the chosen series in a home holding folder in chunks, and
jobs/prep_mr_rate.sbatch preps each ready chunk into the store and deletes it.

    python -m brain.prep.mr_rate select                     # META/selection.csv + STORE/labels/mr_rate_studies.csv
    python -m brain.prep.mr_rate fetch --hold ~/brain_tmp/mr_rate --chunk-size 300 --max-ready 2
    python -m brain.prep.mr_rate status --hold ~/brain_tmp/mr_rate

Never print study / patient ids (data use agreement): logs report counts only.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import PROCESSED_ROOT

REPO_ID = "Forithmus/MR-RATE"
META = Path(os.environ.get("MR_RATE_META", "/home/roku/orcd/pool/datasets/mr_rate/meta"))
STUDIES_CSV = "mr_rate_studies.csv"          # STORE/labels: study -> patient, split (brain.prep.index reads it)
CONTRASTS = ("t1w", "t1c", "t2w", "flair")
DATASET_OF = {c: f"mr_rate_{c}" for c in CONTRASTS}

# body part from the (Turkish) procedure text: BEYIN = brain, KRANIAL = cranial, SERVIKAL = cervical, LOMBER = lumbar
HEAD = r"BEY[IİÝ]N|KRAN|KRN|BRAIN|HEAD|H[IİÝ]POF|PITU|ORB|SELLA|IAC|TEMPORAL"
SPINE = r"SERV[IİÝ]KAL|CERVIC|C[-_ ]SP|LOMBER|LUMBAR|L[-_ ]SP|TORAKAL|THORAC|T[-_ ]SP|VERTEB|SPIN|MYELO|SAKR|SACR|OMURGA"
OTHER = r"D[IİÝ]Z |KNEE|AYAK|ANKLE|OMUZ|SHOULDER|ABDOMEN|PELV|KALÇA|WRIST|BREAST|MEME|PROSTAT|KARACI|LIVER"
# post-contrast marker in a series description / protocol (KONTRAST = contrast, KM = contrast medium)
POST = r"\+C\b|\bC\+|\+GD|\bGD\b|GAD|POST|KONTRAST|CONTRAST|\bCE\b|DOTAREM|MULTIHANCE|\bKM\b|_C$"

# 37-category columns that are spine findings: not used (head studies only)
SPINE_CATS = ("Herniation of nucleus pulposus", "Spinal cord compression", "Foraminal Spinal Stenosis",
              "Spinal stenosis", "Hemangioma of vertebral column")
# label columns whose every training positive is kept (the first MRI haematoma labels; user 2026-11-12)
HEMORRHAGE_COLS = ("Cerebral hemorrhage", "Subdural intracranial hemorrhage", "nv:intracranial_hemorrhage",
                   "nv:subdural_hematoma", "nv:epidural_hematoma", "nv:intraparenchymal_hemorrhage",
                   "nv:intraventricular_hemorrhage", "nv:traumatic_subarachnoid_hemorrhage",
                   "nv:aneurysmal_subarachnoid_hemorrhage")
# NeuroVFM columns the registry maps onto existing tasks (configs/multitask/registry.yaml, source mr_rate)
NV_USED = ("intracranial_hemorrhage", "subdural_hematoma", "epidural_hematoma", "intraparenchymal_hemorrhage",
           "intraventricular_hemorrhage", "traumatic_subarachnoid_hemorrhage", "aneurysmal_subarachnoid_hemorrhage",
           "small_vessel_ischemic_disease", "chronic_ischemic_stroke", "lacunar_stroke", "acute_ischemic_stroke",
           "brain_tumor", "cerebral_edema", "ventriculomegaly", "obstructive_hydrocephalus", "hydrocephalus_ex_vacuo",
           "postsurgical_changes", "craniotomy_craniectomy", "tumor_resection_cavity", "high_grade_glioma",
           "low_grade_glioma", "brain_midline_shift", "brain_mass_effect", "multiple_sclerosis",
           "brain_metastasis", "meningioma", "glioma")


def up(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.upper()


def load_labels(meta: Path = META) -> pd.DataFrame:
    """Study-level labels: the 37 report categories (their names) and the NeuroVFM diagnoses (``nv:<name>``), 0/1,
    one row per study in either table (a study missing from one table has NaN there), plus split / patient."""
    a = pd.read_csv(meta / "pathology_labels" / "mrrate_labels.csv")
    b = pd.read_csv(meta / "pathology_labels_neurovfm" / "mrrate_neurovfm74_labels.csv")
    b = b.rename(columns={c: f"nv:{c}" for c in b.columns if c != "study_uid"})
    sp = pd.read_csv(meta / "splits.csv")[["study_uid", "patient_uid", "split"]]
    return sp.merge(a, on="study_uid", how="left").merge(b, on="study_uid", how="left")


def head_studies(m: pd.DataFrame) -> pd.Series:
    """study_uid -> True for head studies (head procedure text, not limb / body). Studies that combine head and
    spine stay; their spine series are dropped series by series."""
    txt = up(m.PerformedProcedureStepDescription) + " " + up(m.RequestedProcedureDescription) + " " + \
        up(m.StudyDescription)
    f = pd.DataFrame({"study_uid": m.study_uid, "head": txt.str.contains(HEAD, regex=True),
                      "other": txt.str.contains(OTHER, regex=True)}).groupby("study_uid").any()
    return f["head"] & ~f["other"]


def _spacing(s: pd.Series) -> np.ndarray:
    """array_spacing_mm text -> (n, 3) float (NaN when unparsable)."""
    out = np.full((len(s), 3), np.nan)
    for i, v in enumerate(s.astype(str)):
        nums = re.findall(r"[0-9]*\.?[0-9]+", v)
        if len(nums) >= 3:
            out[i] = [float(x) for x in nums[:3]]
    return out


def _shape(s: pd.Series) -> np.ndarray:
    out = np.zeros((len(s), 3))
    for i, v in enumerate(s.astype(str)):
        nums = re.findall(r"\d+", v)
        if len(nums) >= 3:
            out[i] = [int(x) for x in nums[:3]]
    return out


def candidate_series(m: pd.DataFrame) -> pd.DataFrame:
    """One row per usable head series with its contrast (t1w / t1c / t2w / flair) and a preference rank.

    Post-contrast T1 is not a metadata class: a T1 is ``t1c`` when its description / protocol carries a contrast
    marker. In a contrast study (procedure text KONTRAST) an unmarked T1 is pre-contrast only when a marked T1 of the
    same base protocol exists (the pre/post pair); otherwise it is ambiguous and skipped (many contrast protocols
    acquire only the post-contrast 3D T1, unmarked)."""
    flag = lambda c: m[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    ser_txt = up(m.SeriesDescription) + " " + up(m.ProtocolName)
    proc = up(m.PerformedProcedureStepDescription) + " " + up(m.RequestedProcedureDescription)
    c_study = proc.groupby(m.study_uid).transform(lambda s: s.str.contains("KONTRAST").any())
    ok = ~(flag("is_derived") | flag("is_localizer") | flag("is_subtraction"))
    ok &= ~ser_txt.str.contains(SPINE, regex=True)
    ok &= m.classified_modality.isin(["T1w", "T2w", "FLAIR"])
    sp, sh = _spacing(m.array_spacing_mm), _shape(m.array_shape)
    ok &= (np.nanmax(sp, 1) <= 7.0) & (sh.min(1) >= 16) & (np.sort(sh, 1)[:, 1] >= 128)
    marked = ser_txt.str.contains(POST, regex=True)
    base = up(m.SeriesDescription).str.replace(POST, "", regex=True).str.replace(r"[^A-Z0-9]", "", regex=True)
    key = m.study_uid.astype(str) + "|" + base
    has_marked_twin = key.isin(set(key[marked & (m.classified_modality == "T1w")]))
    t1 = m.classified_modality == "T1w"
    contrast = np.select(
        [t1 & marked, t1 & ~marked & (~c_study | has_marked_twin), t1, m.classified_modality == "T2w",
         m.classified_modality == "FLAIR"],
        ["t1c", "t1w", "", "t2w", "flair"], default="")
    c = m.assign(contrast=contrast, vox_mm3=np.prod(sp, 1), is3d=(m.MRAcquisitionType.astype(str) == "3D"))[
        ok & (pd.Series(contrast, index=m.index) != "")]
    c = c.assign(rank_axial=(c.acquisition_plane.astype(str) != "AXIAL").astype(int),
                 rank_center=(~c.is_center_modality.astype(str).str.lower().isin(["true", "1"])).astype(int),
                 sn=pd.to_numeric(c["SeriesNumber"], errors="coerce").fillna(1e9) if "SeriesNumber" in c else 0)
    return c.sort_values(["study_uid", "contrast", "is3d", "vox_mm3", "rank_axial", "rank_center", "sn"],
                         ascending=[True, True, False, True, True, True, True])


def pick_series(m: pd.DataFrame) -> pd.DataFrame:
    """The preferred series per (study, contrast): 3D first, then the finest voxels, axial, the centre modality."""
    c = candidate_series(m[m.study_uid.isin(set(head_studies(m).pipe(lambda s: s[s].index)))])
    return c.groupby(["study_uid", "contrast"], sort=False).head(1)[["study_uid", "series_id", "contrast", "is3d"]]


def greedy_cap(frame: pd.DataFrame, cols: list[str], cap: int, rng: np.random.Generator,
               chosen: set | None = None) -> set:
    """Studies such that every column has up to ``cap`` positives (all of them when it has fewer), rarest first."""
    chosen = set() if chosen is None else set(chosen)
    pos = frame[cols].fillna(0).sum()
    for col in pos.sort_values().index:
        ids = frame.study_uid[frame[col].fillna(0) == 1].to_numpy()
        have = sum(i in chosen for i in ids)
        rest = [i for i in ids if i not in chosen]
        chosen.update(rng.permutation(rest)[:max(0, cap - have)].tolist())
    return chosen


def select(meta: Path, cap: int, n_normal: int, n_val: int, n_test: int, seed: int) -> pd.DataFrame:
    m = pd.read_parquet(meta / "series_index.parquet")
    picks = pick_series(m)
    lab = load_labels(meta)
    lab = lab[lab.study_uid.isin(set(picks.study_uid))]
    cats = [c for c in pd.read_csv(meta / "pathology_labels" / "mrrate_labels.csv", nrows=0).columns
            if c != "study_uid" and c not in SPINE_CATS]
    cols = cats + [f"nv:{c}" for c in NV_USED]
    hem = list(HEMORRHAGE_COLS)
    rng = np.random.default_rng(seed)
    out = {}
    for split, cap_s, n_rand in (("train", cap, n_normal), ("val", max(20, cap * n_val // 10000), n_val),
                                 ("test", max(40, cap * n_test // 10000), n_test)):
        f = lab[lab.split == split]
        keep = set(f.study_uid[(f[hem].fillna(0) == 1).any(axis=1)])          # every haematoma positive
        keep = greedy_cap(f, cols, cap_s, rng, keep)
        none = f.study_uid[(f[cats].fillna(0).sum(axis=1) == 0) & ~f.study_uid.isin(keep)].to_numpy()
        rest = f.study_uid[~f.study_uid.isin(keep)].to_numpy()
        if split == "train":                                                     # report-normal studies
            keep.update(rng.permutation(none)[:n_rand].tolist())
        else:                                                                    # fill to n with random studies
            keep.update(rng.permutation(rest)[:max(0, n_rand - len(keep))].tolist())
        out[split] = keep
    studies = lab[lab.study_uid.isin(set().union(*out.values()))][["study_uid", "patient_uid", "split"]]
    sel = picks.merge(studies, on="study_uid")
    batch = pd.read_csv(meta / "splits.csv", usecols=["study_uid", "batch_id"])
    sel = sel.merge(batch, on="study_uid", how="left")
    sel["dataset"] = sel.contrast.map(DATASET_OF)
    sel["case_id"] = sel.study_uid.astype(str) + "-" + sel.contrast
    return sel.sort_values(["batch_id", "study_uid", "contrast"]).reset_index(drop=True)


def zip_path(study_uid: str, batch_id) -> str:
    b = str(batch_id)
    b = b if b.startswith("batch") else f"batch{int(float(b)):02d}"
    return f"mri/{b}/{study_uid}.zip"


# ------------------------------------------------------------------------------------------------ fetch (streaming)
def _chunks(sel: pd.DataFrame, size: int) -> list[list[str]]:
    studies = list(dict.fromkeys(sel.study_uid.astype(str)))
    return [studies[i:i + size] for i in range(0, len(studies), size)]


def _ready_pending(hold: Path) -> int:
    return sum(1 for d in hold.glob("chunk_*") if (d / "READY").exists() and not (d / "DONE").exists())


def _fetch_study(study: str, g: pd.DataFrame, d: Path, tmp: Path, k: int) -> tuple[int, int]:
    """One study: download its zip into its own folder under ``tmp``, extract the chosen series, delete the zip.
    Returns (studies ok, series / studies missing)."""
    from huggingface_hub import hf_hub_download

    dests = {r.series_id: d / "datasets" / "mr_rate" / "nifti" / r.dataset / f"{r.case_id}.nii.gz"
             for r in g.itertuples()}
    if all(p.exists() for p in dests.values()):
        return 1, 0
    rel = zip_path(study, g.batch_id.iloc[0])
    work = Path(tempfile.mkdtemp(prefix="s_", dir=tmp))
    try:
        for attempt in range(5):
            try:
                zp = Path(hf_hub_download(REPO_ID, rel, repo_type="dataset", local_dir=work))
                break
            except Exception as e:                          # transient network / rate limit
                print(f"[mr_rate] chunk {k}: download retry {attempt + 1} ({type(e).__name__})", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            return 0, 1
        missing = 0
        with zipfile.ZipFile(zp) as z:
            members = {n.rsplit("/", 1)[-1].removesuffix(".nii.gz"): n for n in z.namelist() if "/img/" in n}
            for sid, dest in dests.items():
                name = members.get(f"{study}_{sid}")
                if name is None:
                    missing += 1
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                part = dest.with_name(dest.name + ".part")
                with z.open(name) as src, open(part, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 22)
                part.rename(dest)
        return 1, missing
    finally:
        shutil.rmtree(work, ignore_errors=True)


def fetch(sel: pd.DataFrame, hold: Path, chunk_size: int, max_ready: int, tmp: Path | None = None,
          limit_chunks: int = 0, workers: int = 8) -> None:
    """Download study zips (``workers`` at a time) onto local disk, keep only the chosen series as
    ``<hold>/chunk_NNN/datasets/mr_rate/nifti/<dataset>/<case_id>.nii.gz`` and mark the chunk READY. Waits while
    ``max_ready`` chunks are prepped-pending (bounded home use). Resumable: READY / DONE chunks and studies whose
    series are all present are skipped."""
    from concurrent.futures import ThreadPoolExecutor

    hold.mkdir(parents=True, exist_ok=True)
    by_study = {s: g for s, g in sel.groupby(sel.study_uid.astype(str))}
    tmp = Path(tmp or tempfile.mkdtemp(prefix="mr_rate_dl_"))
    tmp.mkdir(parents=True, exist_ok=True)
    chunks = _chunks(sel, chunk_size)
    for k, chunk in enumerate(chunks[:limit_chunks] if limit_chunks else chunks):
        d = hold / f"chunk_{k:03d}"
        if (d / "READY").exists() or (d / "DONE").exists():
            continue
        while _ready_pending(hold) >= max_ready:
            time.sleep(60)
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda st: _fetch_study(st, by_study[st], d, tmp, k), chunk))
        n_ok, n_missing = sum(r[0] for r in res), sum(r[1] for r in res)
        (d / "READY").write_text(f"{n_ok} studies, {n_missing} missing series/studies, {time.time() - t0:.0f} s\n")
        print(f"[mr_rate] chunk {k}: {n_ok}/{len(chunk)} studies ready, {n_missing} missing, "
              f"{time.time() - t0:.0f} s", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    (hold / "FETCH_DONE").write_text("ok\n")       # a pilot (limit_chunks) also ends the prep loop


def status(hold: Path, sel: pd.DataFrame | None = None, chunk_size: int = 300) -> None:
    ch = sorted(hold.glob("chunk_*"))
    n = len(_chunks(sel, chunk_size)) if sel is not None else "?"
    print(f"chunks total {n}: present {len(ch)}, ready {sum((c / 'READY').exists() for c in ch)}, "
          f"done {sum((c / 'DONE').exists() for c in ch)}, fetch finished {(hold / 'FETCH_DONE').exists()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--meta", type=Path, default=META)
    s.add_argument("--store", type=Path, default=PROCESSED_ROOT)
    s.add_argument("--cap", type=int, default=600, help="training positives kept per label column")
    s.add_argument("--normal", type=int, default=2000, help="report-normal training studies")
    s.add_argument("--val", type=int, default=800)
    s.add_argument("--test", type=int, default=1500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("fetch")
    f.add_argument("--meta", type=Path, default=META)
    f.add_argument("--hold", type=Path, required=True)
    f.add_argument("--chunk-size", type=int, default=300)
    f.add_argument("--max-ready", type=int, default=2)
    f.add_argument("--tmp", type=Path, default=None)
    f.add_argument("--limit-chunks", type=int, default=0, help="pilot: only the first N chunks")
    f.add_argument("--workers", type=int, default=8, help="concurrent study downloads")
    st = sub.add_parser("status")
    st.add_argument("--meta", type=Path, default=META)
    st.add_argument("--hold", type=Path, required=True)
    st.add_argument("--chunk-size", type=int, default=300)
    a = ap.parse_args()
    os.umask(0o077)
    if a.cmd == "select":
        sel = select(a.meta, a.cap, a.normal, a.val, a.test, a.seed)
        st_ = sel.drop_duplicates("study_uid")
        print(f"[mr_rate] studies {len(st_)} {st_.split.value_counts().to_dict()}; series {len(sel)} "
              f"{sel.contrast.value_counts().to_dict()}; ~{len(sel) * 4.17 / 1000:.0f} GB in the store at 4.17 MB")
        if not a.dry_run:
            sel.to_csv(a.meta / "selection.csv", index=False)
            (a.store / "labels").mkdir(parents=True, exist_ok=True)
            st_[["study_uid", "patient_uid", "split"]].to_csv(a.store / "labels" / STUDIES_CSV, index=False)
            print(f"[mr_rate] wrote {a.meta / 'selection.csv'} and labels/{STUDIES_CSV}")
    elif a.cmd == "fetch":
        fetch(pd.read_csv(a.meta / "selection.csv"), a.hold, a.chunk_size, a.max_ready, a.tmp, a.limit_chunks, a.workers)
    else:
        status(a.hold, pd.read_csv(a.meta / "selection.csv") if (a.meta / "selection.csv").exists() else None,
               a.chunk_size)


if __name__ == "__main__":
    main()
