"""Build the index tables and leakage-safe splits from the processed store.

    python -m brain.prep.index --root STORE --manifests "manifest_v6_*.jsonl" [--inherit-splits OLD/volumes.csv]

Writes ``volumes.csv``, ``subjects.csv`` and ``labels/bhsd_studies.csv``. Splits are inherited from an existing
``volumes.csv`` (``--inherit-splits``, else the store's own); only subjects new to it get a fresh stratified split.
BHSD RSNA studies and held-out datasets never enter pretraining (``pretrain``). Subject-level ``dedup_key`` is the split
atom: OASIS-2 sessions, IXI T1/T2, BraTS year/timepoint, and co-registered sequences of one person always share a
split. AISD keeps its official 52-case test split.
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import DATASETS, PROCESSED_ROOT
from brain.data.volumes import META, meta_id
from brain.prep import store
from brain.prep.fomo import CONTRAST_SUFFIXES, load_holdout_families, strip_contrast
SSL_QC_DROP_DATASETS = ("rsna_ich",)  # read from DICOM series by brain.prep.dicom
# Whole datasets kept out of every training pool (SSL and multitask), stay in the store: HBN proton-density scans,
# many of them noise / artefact with no brain contrast (SynthSeg QC passed 61%, vs 99% for HBN T1w), 2026-10-21.
# SynthRAD cone-beam CTs (2026-10-30): artefact-heavy, not calibrated HU (CT density targets would be wrong); the CT
# and MR of those patients stay.
POOL_DROP_DATASETS = ("fomo_hbn_pd", "synthrad_cbct")
BHSD_ZIP = DATASETS / "bhsd" / "label_192.zip"
SEED, TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 42, 0.75, 0.10, 0.15
SPLIT_RANK = {"test": 0, "val": 1, "train": 2}  # lower wins when merging inherited splits

# Entire families stay out of the pretraining index (brain.downstream still gets train/val/test).
HELD_OUT = {"cq500", "isles22", "atlas", "wmh", "isles24", "instance", "aisd", "brainmetshare",
            "msd_braintumor", "ms3seg", "valdo", "ucsf_pdgm", "upenn_gbm", "adni", "ppmi"}
# Held-out families some of whose subjects are also in a pretraining family under another name (UCSF-PDGM patients in
# BraTS 2021; UPENN-GBM patients, the BraTS site-1 cases): they share the pretraining copy's dedup_key, so both copies
# get one split, and since 2026-10-29 the BraTS copies leave SSL and every training pool (``twin_drop`` in ``build``).
# Their own scans still never enter SSL.
SHARED_SUBJECTS = {"ucsf_pdgm", "upenn_gbm"}
# held out of SSL but a multitask training cohort (glioma option B, 2026-10-29): UPENN-GBM trains survival / MGMT, UCSF-PDGM
# stays the external glioma test
MULTITASK_FAMILIES = {"upenn_gbm"}


def all_held_out() -> set[str]:
    """Static eval families plus FOMO sources whose mapping.tsv overlaps them."""
    return set(HELD_OUT) | load_holdout_families()


def dataset_family(dataset: str) -> str:
    ds = strip_contrast(str(dataset))
    if ds in ("cq500", "seg_cq500"):
        return "cq500"
    if ds.startswith("brats"):
        if "_gli" in ds:
            return "brats_gli"
        if "men_rt" in ds:
            return "brats_men_rt"
        if "_men" in ds:
            return "brats_men"
        if "_ped" in ds:
            return "brats_ped"
        if "_met" in ds:
            return "brats_met"
        if "_goat" in ds:
            return "brats_goat"
        if "_ssa" in ds:
            return "brats_ssa"
        return ds
    if ds.startswith("ixi_"):
        return "ixi"
    if ds.startswith("isles22_"):
        return "isles22"
    if ds.startswith("wmh_"):
        return "wmh"
    if ds.startswith("isles24_"):
        return "isles24"
    if ds.startswith("brainmetshare"):
        return "brainmetshare"
    if ds.startswith("ms3seg"):
        return "ms3seg"
    if ds.startswith("valdo"):
        return "valdo"
    if ds.startswith("rsna_ica"):                # CTA / MRA / T2 / T1post series of one aneurysm dataset
        return "rsna_ica"
    if ds.startswith("synthrad"):                # SynthRAD2023 brain: one patient's CT + MR (task 1) or CT + CBCT (task 2)
        return "synthrad"
    return ds


def contrast_of(dataset: str, case_id: str) -> str:
    for c in CONTRAST_SUFFIXES:
        if dataset.endswith("_" + c) or case_id.endswith("-" + c):
            return c
    if dataset.endswith("_t1"):
        return "t1"
    if dataset.endswith("_t2"):
        return "t2"
    if dataset.endswith("_t2s"):
        return "t2s"
    if dataset.endswith(("_dwi", "_adc", "_flair", "_ncct", "_cta")):
        return dataset.rsplit("_", 1)[1]
    return ""


UCSF_PDGM_META = DATASETS / "ucsf_pdgm" / "meta" / "UCSF-PDGM-metadata_v5.csv"


@functools.cache
def ucsf_pdgm_brats() -> dict[int, str]:
    """UCSF-PDGM case number -> its BraTS 2021 copy as a BraTS-GLI patient (BraTS2021_NNNNN == BraTS-GLI-NNNNN; 262
    of these are in the store's brats2023_gli), from the TCIA metadata (IDs there have three digits, files four)."""
    if not UCSF_PDGM_META.exists():
        return {}
    m = pd.read_csv(UCSF_PDGM_META, usecols=["ID", "BraTS21 ID"]).dropna()
    m = m[~m["ID"].str.contains("_FU")]       # a follow-up row names its own BraTS case: the person keys by the baseline
    num = m["ID"].str.extract(r"UCSF-PDGM-(\d+)")[0].astype(int)
    return dict(zip(num, "BraTS-GLI-" + m["BraTS21 ID"].str.extract(r"BraTS2021_(\d{5})")[0]))


UCSF_PDGM_CONTENT_TWINS = DATASETS / "ucsf_pdgm" / "meta" / "brats_twins_content.csv"


@functools.cache
def ucsf_pdgm_content_twins() -> dict[str, str]:
    """UCSF-PDGM patient -> BraTS dedup_key for the patients TCIA maps to no BraTS 2021 case but whose BraTS copy
    was found by image content (runs/diagnostics/glioma_twins.py, 2026-10-29; mostly newer BraTS-GLI ids)."""
    if not UCSF_PDGM_CONTENT_TWINS.exists():
        return {}
    m = pd.read_csv(UCSF_PDGM_CONTENT_TWINS, usecols=["ucsf_id", "brats_id"]).dropna()
    return dict(zip(m["ucsf_id"], m["brats_id"]))


def ucsf_pdgm_key(case_id: str) -> str:
    """One person: the BraTS twin's key when the patient is also in BraTS (TCIA's BraTS 2021 mapping, else a content
    match), so both copies share a split, else the UCSF number. Follow-ups (``_FU###d``) and the contrast suffix are
    dropped."""
    num = int(case_id.split("_")[0].removeprefix("UCSF-PDGM-"))
    twin = ucsf_pdgm_brats().get(num)
    if twin:
        return f"brats:gli:{twin}"
    return ucsf_pdgm_content_twins().get(f"UCSF-PDGM-{num:04d}", f"ucsf_pdgm:UCSF-PDGM-{num:04d}")


UPENN_GBM_TWINS = DATASETS / "upenn_gbm" / "meta" / "brats_twins.csv"


@functools.cache
def ida_subjects(project: str) -> dict[str, str]:
    """IDA image ID -> subject for the ADNI / PPMI baseline picks (``sources.ida``): a subject's T1 / T2 / FLAIR share
    one dedup_key."""
    from brain.prep.sources import ida_pick

    pick = ida_pick(project)
    if pick.empty:
        return {}
    return dict(zip("I" + pick["Image ID"].astype(str).str.lstrip("I"), pick["Subject ID"].astype(str)))


@functools.cache
def upenn_gbm_brats() -> dict[str, str]:
    """UPENN-GBM patient -> its BraTS copy's dedup_key (``brats:gli:BraTS-GLI-NNNNN`` or ``brats:goat:...``). TCIA
    gives no mapping (UPenn is BraTS site 1, without local IDs), so ``brats_twins.csv`` comes from image content
    (runs/diagnostics/glioma_twins.py, 2026-10-29: 451 of 611 baseline patients, every GLI match a site-1 case)."""
    if not UPENN_GBM_TWINS.exists():
        return {}
    m = pd.read_csv(UPENN_GBM_TWINS, usecols=["upenn_id", "brats_id"]).dropna()
    return dict(zip(m["upenn_id"], m["brats_id"]))


def upenn_gbm_key(case_id: str) -> str:
    """One person: the BraTS twin's key when found, else the UPENN patient (``_11`` / ``_21`` timepoints and the
    contrast suffix dropped)."""
    pid = case_id.split("_")[0]
    twin = upenn_gbm_brats().get(pid)
    if not twin:
        return f"upenn_gbm:{pid}"
    return twin if twin.startswith("brats:") else f"brats:gli:{twin}"


BRATS_GLI_SITES = DATASETS / "brats" / "meta" / "BraTS2023_GLI_sites.csv"   # from BraTS2023_2017_GLI_Mapping.xlsx
# BraTS-GLI originating sites that are an external cohort's source: every case from them leaves the training pools.
# Site 1 is UPenn (UPENN-GBM): content matched 403 of its 511 BraTS patients to a UPENN-GBM patient, and an unmatched
# one may still be a UPENN-GBM scan too re-processed to match, so the whole site goes (2026-10-29). UCSF (site 18) is
# handled per patient: TCIA's mapping plus content matches.
EXTERNAL_BRATS_SITES = {1: "upenn_gbm"}


@functools.cache
def brats_gli_sites() -> dict[str, int]:
    """BraTS-GLI patient -> originating site number (BraTS 2023 GLI training mapping; newer ids are absent)."""
    if not BRATS_GLI_SITES.exists():
        return {}
    m = pd.read_csv(BRATS_GLI_SITES, usecols=["brats_id", "site"]).dropna()
    return dict(zip(m["brats_id"], m["site"].astype(int)))


def brats_patient(case_id: str) -> str:
    """BraTS-GLI-00005-000-t1n and BraTS-GLI-00005-100-t1n are one person."""
    p = case_id
    for c in CONTRAST_SUFFIXES:
        if p.endswith("-" + c):
            p = p[: -len(c) - 1]
            break
    parts = p.split("-")
    if len(parts) >= 4 and parts[1].upper() == "MEN" and parts[2].upper() == "RT":
        return "-".join(parts[:4])
    return "-".join(parts[:3]) if len(parts) >= 3 else p


def oasis_subject(case_id: str) -> str:
    """OAS1_0001_MR1 / OAS2_0001_MR2 → OAS1_0001 / OAS2_0001."""
    parts = case_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else case_id


def ixi_subject(case_id: str) -> str:
    return case_id.split("-", 1)[0]


def isles22_subject(case_id: str) -> str:
    return case_id.rsplit("_", 1)[0]  # sub-strokecase0002_dwi


def valdo_subject(case_id: str) -> str:
    """VALDO case ids are ``<task>_<subject>`` (``t3_sub-101``); the subject is shared across that task's sequences."""
    return case_id.split("_", 1)[1] if case_id[:1] == "t" and "_" in case_id else case_id


_RSNA_PATIENTS: dict[str, str] = {}


def load_rsna_patients(root: Path) -> None:
    """RSNA case_id -> DICOM PatientID from ``<root>/labels/rsna_patients.csv`` (read once from one header per study
    by ``runs/diagnostics/rsna_patients.py``); ``build`` calls this before computing dedup keys."""
    global _RSNA_PATIENTS
    f = root / "labels" / "rsna_patients.csv"
    _RSNA_PATIENTS = dict(zip(*[pd.read_csv(f)[c].astype(str) for c in ("case_id", "patient_id")])) if f.exists() else {}
    print(f"[rsna] {len(_RSNA_PATIENTS)} studies mapped to {len(set(_RSNA_PATIENTS.values()))} patients", flush=True)


def rsna_patients() -> dict[str, str]:
    return _RSNA_PATIENTS


def load_fastmri_patients(root: Path) -> None:
    """fastMRI file -> its anonymised patient id (``labels/fastmri_patients.csv``, written by ``brain.prep.fastmri``),
    so one patient's FLAIR / T1 / T1 post-contrast share a dedup_key and a split."""
    global _FASTMRI_PATIENTS
    f = root / "labels" / "fastmri_patients.csv"
    _FASTMRI_PATIENTS = dict(zip(*[pd.read_csv(f)[c].astype(str) for c in ("case_id", "patient_id")])) \
        if f.exists() else {}


_FASTMRI_PATIENTS: dict[str, str] = {}


def dedup_key(dataset: str, case_id: str) -> str:
    fam = dataset_family(dataset)
    if fam.startswith("brats"):
        return "brats:" + fam.removeprefix("brats_") + ":" + brats_patient(case_id)
    if fam == "cq500":
        return "cq500:" + case_id
    if fam == "ucsf_pdgm":
        return ucsf_pdgm_key(case_id)
    if fam == "upenn_gbm":
        return upenn_gbm_key(case_id)
    if fam in ("rsna_ich", "bhsd", "hemseg"):       # BHSD / HemSeg volumes are RSNA studies (same subject, split)
        pid = rsna_patients().get(case_id)          # by patient: 810 patients had studies in more than one split
        return "rsna:" + (f"patient:{pid}" if pid else case_id)
    if fam in ("oasis1", "oasis2"):
        return "oasis:" + oasis_subject(case_id)
    if fam in ("adni", "ppmi"):
        return f"{fam}:" + ida_subjects(fam).get(case_id, case_id)
    if fam == "fastmri":
        return "fastmri:" + _FASTMRI_PATIENTS.get(case_id, case_id)
    if fam == "ixi":
        return "ixi:" + ixi_subject(case_id)
    if fam == "isles22":
        return "isles22:" + isles22_subject(case_id)
    if fam == "isles24":
        return "isles24:" + case_id
    if fam == "wmh":
        return "wmh:" + case_id
    if fam == "atlas":
        return "atlas:" + case_id
    if fam == "ms3seg":
        return "ms3seg:" + case_id
    if fam == "valdo":
        return "valdo:" + valdo_subject(case_id)
    if fam.startswith("fomo_"):
        cid = str(case_id)
        for c in CONTRAST_SUFFIXES:
            if cid.endswith("_" + c):
                cid = cid[: -len(c) - 1]
                break
        parts = cid.split("_")
        sub = cid
        for i, p in enumerate(parts):
            if p.startswith("ses-"):
                sub = "_".join(parts[:i]) or cid
                break
        return f"{fam}:{sub}"
    return f"{fam}:{case_id}"


def volume_id(dataset: str, case_id: str, modality: str) -> str:
    """Keep v3 ids for the original tasks; prefix new datasets so PIOP1/PIOP2 cannot collide."""
    if dataset == "rsna_ich" or dataset in ("cq500", "seg_cq500"):
        return f"ct:{case_id}"
    if dataset in ("oasis1", "oasis2") or dataset.startswith("brats"):
        return f"mri:{case_id}"
    return f"{dataset}:{case_id}"


def ssl_priority(dataset: str, case_id: str) -> int:
    if dataset.startswith("brats") and "goat" in dataset:
        return 55
    if "brats2024_gli" in dataset:
        return 50
    if "brats2023_gli" in dataset:
        return 40
    if dataset == "seg_cq500":
        return 20
    if dataset == "oasis2" and case_id.endswith("_MR1"):
        return 10
    return 0


def bhsd_case_ids(zip_path: Path = BHSD_ZIP) -> set[str]:
    """RSNA study ids that BHSD labeled. Filenames are ID_<short>_ID_<rsna_hash>.nii.gz."""
    if not zip_path.exists():
        return set()
    ids = set()
    with zipfile.ZipFile(zip_path) as zf:
        for n in zf.namelist():
            if "/images/" in n and n.endswith(".nii.gz"):
                ids.add("RSNA-ID_" + n.rsplit("ID_", 1)[-1].removesuffix(".nii.gz"))
    return ids


def merge_split(names: set[str]) -> str:
    return min(names, key=lambda s: SPLIT_RANK.get(s, 9))


def stratified_subjects(keys: list[str], strata: dict[str, tuple], seed: int = SEED) -> dict[str, str]:
    rng = random.Random(seed)
    by: dict[tuple, list[str]] = defaultdict(list)
    for k in keys:
        by[strata[k]].append(k)
    out = {}
    for stratum, dks in sorted(by.items()):
        rng.shuffle(dks)
        n = len(dks)
        n_test, n_val = int(round(n * TEST_FRAC)), int(round(n * VAL_FRAC))
        if n >= 3:
            n_test = min(max(1, n_test), n - 2)
            n_val = min(max(1, n_val), n - n_test - 1)
        elif n == 2:
            n_test, n_val = 1, 0
        else:
            n_test = n_val = 0
        n_train = n - n_test - n_val
        i = 0
        for name, count in (("test", n_test), ("val", n_val), ("train", n_train)):
            for _ in range(count):
                out[dks[i]] = name
                i += 1
        print(f"[split] {stratum[0]}/{stratum[1]}: n={n} train={n_train} val={n_val} test={n_test}", flush=True)
    return out


# Per-volume prep measurements carried from the manifests into volumes.csv. The 3D loader uses the native-geometry
# columns for resolution-matching augmentation; the registration columns are for QC and for filtering analyses.
NATIVE_COLS = ("src_dx", "src_dy", "src_dz", "src_nx", "src_ny", "src_nz", "thick_mm", "thick_axis", "aniso",
               "reg_method", "reg_ok", "reg_rot_deg", "reg_scale", "reg_aniso", "reg_mi", "brain_method", "brain_frac",
               "brain_cover", "stripped_input", "brain_ok", "brain_dice_tpl", "reg_template", "brain_head_ratio", "reg_backend")


def _recency(r: dict) -> tuple:
    """Newer manifest rows win: by ``written_at``; rows from before timestamps prefer the one with a stored transform."""
    return (r.get("written_at", 0), r.get("reg_affine") is not None)


def load_ok_manifests(root: Path, pattern: str = "manifest_*.jsonl") -> pd.DataFrame:
    rows = {}
    for f in sorted(root.glob(pattern)):
        for line in open(f):
            r = json.loads(line)
            r["case_id"] = str(r["case_id"])
            key = (r["dataset"], r["case_id"])
            if r.get("skipped_existing") and key in rows:  # a resume marker carries no metadata; keep the full row
                continue
            # a rerun may re-shard cases into an earlier-sorted manifest: the most recently written full row wins
            if key in rows and _recency(r) < _recency(rows[key]):
                continue
            rows[key] = r
    ok = [r for r in rows.values() if r.get("status") == "ok"]
    return pd.DataFrame(ok)


def mask_rel(root: Path, folder: str, case_id: str, name: str) -> str:
    p = store.path_for(root, folder, case_id, name)
    return p.relative_to(root).as_posix() if p.exists() else ""


def folder_of(dataset: str) -> str:
    """Store folder of a dataset: the same rule ``sources.Case.folder`` uses when writing, so paths always resolve.

    A trailing contrast token is stripped (``brats2023_gli_t1n`` -> ``brats2023_gli``, ``fomo_hbn_t1w`` ->
    ``fomo_hbn``, ``wmh_flair`` -> ``wmh``); other sequences keep their own folder (``wmh_t1``, ``isles22_dwi``).
    """
    return strip_contrast(dataset)


def inherit_current_splits(vols: pd.DataFrame, old_vols: pd.DataFrame) -> pd.DataFrame:
    """Keep splits already in this store; only assign keys that are new (AISD/INSTANCE)."""
    old = (old_vols.dropna(subset=["dedup_key", "split"]).drop_duplicates("dedup_key")
           .set_index("dedup_key")["split"].astype(str).to_dict())
    strata = {k: (m, f) for k, m, f in vols[["dedup_key", "modality", "dataset_family"]].itertuples(index=False, name=None)}
    fresh = [k for k in vols["dedup_key"].unique() if k not in old]
    # a re-keyed subject (RSNA studies grouped by patient) keeps the majority old split of its volumes (ties: train,
    # then val), so only subjects whose volumes straddled splits move; truly new subjects get a stratified split
    old_vol = old_vols.dropna(subset=["split"]).set_index("volume_id")["split"].astype(str).to_dict() \
        if {"volume_id", "split"} <= set(old_vols.columns) else {}
    by_key = vols.groupby("dedup_key")["volume_id"].apply(list).to_dict() if "volume_id" in vols else {}
    rekeyed = {}
    for k in fresh:
        votes = [old_vol[v] for v in by_key.get(k, []) if v in old_vol]
        if votes:
            counts = {s: votes.count(s) for s in set(votes)}
            rekeyed[k] = max(counts, key=lambda s: (counts[s], {"train": 2, "val": 1}.get(s, 0)))
    new = [k for k in fresh if k not in rekeyed]
    assigned = stratified_subjects(new, strata) if new else {}
    assigned.update(old)
    assigned.update(rekeyed)
    vols = vols.copy()
    vols["split"] = vols["dedup_key"].map(assigned)
    moved = sum(1 for k in rekeyed for v in by_key[k] if old_vol.get(v) != rekeyed[k])
    print(f"[split] inherited current store subjects={len(old)} re-keyed={len(rekeyed)} ({moved} volumes changed "
          f"split) newly assigned={len(new)}", flush=True)
    return pin_official_splits(vols)


def pin_official_splits(vols: pd.DataFrame) -> pd.DataFrame:
    """AISD published a 52-case test set; do not let stratified assignment reshuffle it."""
    from brain.datasets.download_aisd import splits_for
    from brain.prep.sources import aisd_id

    m = vols["dataset_family"].astype(str).eq("aisd")
    if not m.any():
        return vols
    ids = [aisd_id(c) for c in vols.loc[m, "case_id"]]
    pinned = splits_for(ids)
    vols = vols.copy()
    vols.loc[m, "split"] = [pinned[aisd_id(c)] for c in vols.loc[m, "case_id"]]
    n = pd.Series(pinned).value_counts().to_dict()
    print(f"[split] aisd official {n}", flush=True)
    return vols


def assert_no_leak(vols: pd.DataFrame, ssl: pd.DataFrame, bhsd: set[str]) -> None:
    """Hard stop if a subject crosses SSL splits or a BHSD/held-out study is among the SSL rows (``ssl``: the
    train / val volumes of the pretraining pool)."""
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        sa, sb = set(ssl.loc[ssl.split == a, "dedup_key"]), set(ssl.loc[ssl.split == b, "dedup_key"])
        leak = sa & sb
        assert not leak, f"dedup_key in SSL {a} and {b}: {list(leak)[:5]}"
    ssl_keys = set(ssl["dedup_key"])
    held = set(vols.loc[vols.dataset_family.isin(all_held_out()), "dedup_key"])     # incl. the glioma twins' keys
    assert not (vols.dataset_family.isin(SHARED_SUBJECTS) & vols.pretrain).any(), "a shared-subject family in SSL"
    assert not (ssl_keys & held), f"held-out subjects in SSL index: {list(ssl_keys & held)[:5]}"
    bhsd_keys = {"rsna:" + i for i in bhsd}
    assert not (ssl_keys & bhsd_keys), f"BHSD studies in SSL index: {list(ssl_keys & bhsd_keys)[:5]}"
    in_ssl = vols[vols.pretrain & vols.ssl_keep]
    for dk, g in in_ssl.groupby("dedup_key"):
        assert g["split"].nunique() == 1, dk
    # multi-sequence families share a split
    for fam in ("ixi", "isles22", "wmh", "isles24", "brats_gli", "brats_men", "brats_goat", "oasis2", "ms3seg", "valdo"):
        sub = vols[vols.dataset_family == fam]
        bad = sub.groupby("dedup_key")["split"].nunique()
        assert (bad == 1).all(), f"{fam} split conflict: {list(bad[bad > 1].index)[:5]}"
    print("[leak] ok: no subject in two SSL splits; BHSD and held-out families absent from SSL", flush=True)


def build(root: Path, manifests: str = "manifest_*.jsonl", inherit_splits: Path | None = None,
          out_dir: Path | None = None) -> None:
    """``inherit_splits``: the volumes.csv whose subject splits are kept (default: ``root/volumes.csv``; with neither,
    every subject gets a fresh stratified split). ``out_dir``: write volumes.csv / subjects.csv / bhsd_studies.csv there
    instead of into the store (a candidate index built beside the live one, e.g. jobs/index_append.sbatch)."""
    bhsd = bhsd_case_ids()
    print(f"[bhsd] {len(bhsd)} RSNA studies with hemorrhage masks", flush=True)
    load_rsna_patients(root)
    load_fastmri_patients(root)
    man = load_ok_manifests(root, manifests)
    recs = []
    for r in man.itertuples(index=False):
        ds, cid, mod = r.dataset, str(r.case_id), r.modality
        fam = dataset_family(ds)
        folder = folder_of(ds)
        img = store.path_for(root, folder, cid)
        if not img.exists():
            continue
        held = fam in all_held_out()
        is_bhsd = cid in bhsd
        recs.append({
            "volume_id": volume_id(ds, cid, mod),
            "dedup_key": dedup_key(ds, cid),
            "subject_id": dedup_key(ds, cid),
            "case_id": cid,
            "modality": mod,
            "dataset": ds,
            "dataset_family": fam,
            "contrast": contrast_of(ds, cid),
            "contrast_std": META[meta_id(contrast_of(ds, cid), mod, ds)],     # harmonised (brain.data.volumes.META)
            "folder": folder,
            "image_path": img.relative_to(root).as_posix(),
            "priority": ssl_priority(ds, cid),
            "held_out": held,
            "bhsd": is_bhsd,
            "role": "heldout" if held or is_bhsd else "pretrain_eval",
            "mask_seg": mask_rel(root, folder, cid.removesuffix("-t1n") if ds.startswith("brats") else cid, "seg"),
            "mask_ich": mask_rel(root, folder, cid, "ICH"),
            # format 6: the brain mask is the ``brain`` array inside the image zip
            "mask_brain": img.relative_to(root).as_posix()
            if str(getattr(r, "brain_method", "none")) not in ("none", "nan", "None", "") else "",
            **{c: getattr(r, c, None) for c in NATIVE_COLS},
            "mask_lesion": mask_rel(root, folder, cid, "lesion"),
            "mask_lvo": mask_rel(root, folder, cid, "lvo"),
            "mask_cow": mask_rel(root, folder, cid, "cow"),
        })
    vols = pd.DataFrame(recs)
    old_path = Path(inherit_splits) if inherit_splits else root / "volumes.csv"
    if inherit_splits and not old_path.exists():
        raise SystemExit(f"--inherit-splits {old_path} does not exist")
    old_vols = pd.read_csv(old_path, low_memory=False) if old_path.exists() else \
        pd.DataFrame(columns=["dedup_key", "volume_id", "split"])
    vols = inherit_current_splits(vols, old_vols)
    # one SSL volume per (subject, contrast): 2024 GLI beats 2023; OASIS-2 MR1 beats later visits
    vols = vols.sort_values(["dedup_key", "contrast", "priority", "case_id"], ascending=[True, True, False, True])
    ssl_keep = ~vols.duplicated(["dedup_key", "contrast", "modality"], keep="first")
    vols["ssl_keep"] = ssl_keep.to_numpy()
    vols["pretrain"] = (~vols["held_out"]) & (~vols["bhsd"])
    # Volumes that fail brain-mask QC in datasets read from DICOM series are kept out of SSL (not out of downstream
    # tasks): there it mostly means a mis-assembled series (duplicate / mixed-thickness slices -> stripes) or an
    # uncorrected gantry-tilt shear that registration could not remove (see docs/prep.md, "RSNA DICOM assembly").
    failed_qc = vols.get("brain_ok", pd.Series(True, index=vols.index)).astype(str).str.lower().eq("false")
    vols["ssl_qc_drop"] = vols["dataset"].isin(SSL_QC_DROP_DATASETS) & failed_qc
    vols["pretrain"] &= ~vols["ssl_qc_drop"]
    print(f"[ssl] QC drop (DICOM-assembled, brain_ok=False): {int(vols.ssl_qc_drop.sum())} volumes", flush=True)
    vols["pool_drop"] = vols["dataset"].isin(POOL_DROP_DATASETS)
    print(f"[ssl] dataset drop {POOL_DROP_DATASETS}: {int(vols.pool_drop.sum())} volumes", flush=True)
    # per-volume drops (labels/pool_drop/*.csv: volume_id, reason): cross-dataset copies of a scan (brain.prep.dedup),
    # partial-coverage slabs and unscaled fallback registrations found by prep QC; they stay out of SSL and training
    for f in sorted((root / "labels" / "pool_drop").glob("*.csv")):
        ids = set(pd.read_csv(f)["volume_id"].astype(str))
        hit = vols["volume_id"].astype(str).isin(ids)
        vols["pool_drop"] |= hit
        print(f"[ssl] volume drop {f.name}: {int(hit.sum())} volumes", flush=True)
    vols["pretrain"] &= ~vols["pool_drop"]
    # The pretraining-family copies of the external glioma cohorts' patients (BraTS 2021 / 2024 cases that are UCSF-PDGM
    # or UPENN-GBM patients: they share the twin's dedup_key) leave SSL and every training pool (2026-10-29), so
    # neither cohort's evaluation has seen its patients through BraTS.
    twin_keys = set(vols.loc[vols["dataset_family"].isin(SHARED_SUBJECTS), "dedup_key"])
    vols["twin_drop"] = vols["dedup_key"].isin(twin_keys) & ~vols["dataset_family"].isin(SHARED_SUBJECTS)
    n_twin = int(vols.twin_drop.sum())
    sites = brats_gli_sites()
    gli = vols["dataset_family"].eq("brats_gli")
    site = vols.loc[gli, "dedup_key"].str.rsplit(":", n=1).str[1].map(sites)
    vols.loc[gli, "twin_drop"] |= site.isin(list(EXTERNAL_BRATS_SITES)).to_numpy()
    vols["pretrain"] &= ~vols["twin_drop"]
    print(f"[ssl] BraTS copies of {sorted(SHARED_SUBJECTS)} patients: {n_twin} volumes matched, "
          f"{int(vols.twin_drop.sum())} with the whole of sites {EXTERNAL_BRATS_SITES}", flush=True)
    # label masks only: mask_brain (the brain array inside the image zip) is not a downstream label
    vols["has_mask"] = vols.filter(like="mask_").drop(columns=["mask_brain"], errors="ignore").astype(str).ne("").any(axis=1)

    ssl_vols = vols[vols.pretrain & vols.ssl_keep].copy()
    print(f"[ssl] volumes eligible={len(ssl_vols)} / {len(vols)} (BHSD {int(vols.bhsd.sum())}, held-out {int(vols.held_out.sum())})", flush=True)

    # the leak check runs on the SSL volume rows (train / val)
    ssl_rows = ssl_vols.loc[ssl_vols["split"].isin(["train", "val"]), ["volume_id", "dedup_key", "split", "modality"]]
    vols["vol_idx"] = np.arange(len(vols))
    vols["status"] = "ok"
    assert_no_leak(vols, ssl_rows, bhsd)

    subj = (vols.groupby("dedup_key", as_index=False)
            .agg(split=("split", "first"), modality=("modality", "first"),
                 dataset_family=("dataset_family", "first"), role=("role", "first"),
                 pretrain=("pretrain", "max"), n_volumes=("volume_id", "nunique"),
                 bhsd=("bhsd", "max")))

    out = out_dir or root
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    pd.Series(sorted(bhsd), name="case_id").to_csv(out / "labels" / "bhsd_studies.csv", index=False)
    vols.to_csv(out / "volumes.csv", index=False)
    subj.to_csv(out / "subjects.csv", index=False)
    print("\n[volumes] by family × split:")
    print(vols.groupby(["dataset_family", "split"]).size().unstack(fill_value=0).to_string())
    print("\n[ssl volumes] by split × modality:")
    print(ssl_rows.groupby(["split", "modality"]).size().to_string())
    print(f"\n[ok] volumes={len(vols)} subjects={len(subj)} ssl_volumes={len(ssl_rows)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=PROCESSED_ROOT)
    ap.add_argument("--manifests", default="manifest_*.jsonl", help="manifest glob under --root (e.g. manifest_v6_*)")
    ap.add_argument("--inherit-splits", type=Path, default=None,
                    help="volumes.csv whose subject splits are kept (default: --root/volumes.csv)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write volumes.csv / subjects.csv here instead of --root (leave the live index untouched)")
    a = ap.parse_args()
    build(a.root, manifests=a.manifests, inherit_splits=a.inherit_splits, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
