"""Labeled brain.downstream tasks.

Classification and regression use the frozen-embedding heads (``brain.downstream.run_suite``). Segmentation
tasks point at the mask zip beside each volume; ``brain.downstream.seg3d`` trains a decoder on frozen 3D tokens
and scores per-volume Dice.

Splits are the ones stored in ``volumes.csv``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import replace, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import DATASETS, PROCESSED_ROOT, SCRATCH

DEFAULT_VOLUMES = PROCESSED_ROOT / "volumes.csv"

# CQ500 findings. EDH is omitted (about 1.6% positive).
CQ500_LABELS = [
    "ICH",
    "IPH",
    "IVH",
    "SDH",
    "SAH",
    "CalvarialFracture",
    "MassEffect",
    "MidlineShift",
]

RSNA_LABELS = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]

# BraTS integer labels after the shared-space nearest-neighbor resample.
GLI_REGIONS = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)}
# 2023 MEN masks use the same 1/2/3 codes. Label 1 alone is often empty;
# the meningioma is the whole foreground.
MEN_REGIONS = {"meningioma": (1, 2, 3)}
# 2024 post-treatment glioma. 4 is the resection cavity.
GLI2024_REGIONS = {"ET": (3,), "RC": (4,)}
# A few Seg-CQ500 masks store the hemorrhage as 2 rather than 1.
ICH_REGIONS = {"ICH": (1, 2)}
# ISLES'24 Circle of Willis (TopCoW codes): 1 BA, 2/3 R/L PCA, 4/6 R/L ICA, 5/7 R/L MCA, 8/9 R/L Pcom, 10 Acom,
# 11/12 R/L ACA, 13 third A2. Grouped by artery, sides merged.
COW_REGIONS = {"ICA": (4, 6), "MCA": (5, 7), "ACA": (10, 11, 12, 13), "PCA": (2, 3, 8, 9), "BA": (1,)}

CLASSIFICATION_TASKS = (
    "cq500",
    "rsna_ich",
    "oasis1_cdr",
    "oasis2_cdr",
    "oasis2_conversion",
    "abide_asd",
    "msd",
    "isles24_mrs",
    "isles24_mrs_discharge",
    "isles24_lvo_side",
    "cq500_bleed_side",
    "cq500_chronic_bleed",
    "cq500_edh",
    "cq500_other_fracture",
    "aisd_remote_infarct",
    "atlas_lesion_side",
    "instance_large_ich",
    "instance_ivh",
    "instance_infratentorial",
    "instance_side",
    "gli_survival_12m",
    "gli_lgg",
    "gli_idh",
    "ucsf_survival_12m",
    "ucsf_lgg",
    "ucsf_idh",
    "ucsf_mgmt",
    "upenn_survival_12m",
    "upenn_mgmt",
    "rsna_ica_cta_aneurysm",
    "rsna_ica_mr_aneurysm",
    "adni_dx3",
    "adni_cn_ad",
    "adni_cn_mci",
    "ppmi_pd",
)
REGRESSION_TASKS = (
    "oasis1_age",
    "ixi_age",
    "pixar_age",
    "piop1_age",
    "piop2_age",
    "id1000_age",
    "ms3seg_age",
    "valdo_epvs",
    "isles24_nihss",
    "isles24_nihss_discharge",
    "oasis1_mmse",
    "oasis2_mmse",
    "adni_mmse",
    "abide_ados",
    "atlas_lesion_volume",
    "atlas_days_post_stroke",
    "ms3seg_lesion_load",
)
SEGMENTATION_TASKS = (
    "seg_cq500",
    "brats2023_gli",
    "brats2023_men",
    "brats2024_gli",
    "brats2023_ped",
    "atlas",
    "isles22",
    "wmh",
    "isles24",
    "aisd",
    "instance",
    "ms3seg",
    "valdo_cmb",
    "valdo_lacunes",
    "isles24_cow",
)
CQ500_FINDING_TASKS = {"cq500_chronic_bleed": "ChronicBleed", "cq500_edh": "EDH", "cq500_other_fracture": "OtherFracture"}
# Frozen clf/reg suite (segmentation is ``seg3d`` / ``run_suite`` separately).
FROZEN_TASKS = CLASSIFICATION_TASKS + REGRESSION_TASKS
LESION_REGIONS = {"lesion": (1,)}
# AISD: remote=1, clear acute=2, blurred acute=3, invisible=4, infarct=5. Dice on {1,2,3,5}.
AISD_REGIONS = {"infarct": (1, 2, 3, 5)}
WMH_REGIONS = {"WMH": (1,)}
# MS3SEG staged label map (brain.prep.stage_new): 1 ventricles, 2 normal WMH, 3 MS lesion.
MS3SEG_REGIONS = {"MS_lesion": (3,), "normal_WMH": (2,), "ventricles": (1,)}
TASK_NAMES = CLASSIFICATION_TASKS + REGRESSION_TASKS + SEGMENTATION_TASKS


@dataclass
class TaskBundle:
    name: str
    task_type: str  # binary | multilabel | regression | segmentation
    volume_ids: list[str]
    splits: dict[str, str]
    y: np.ndarray
    label_names: list[str]
    modality_filter: str | None = None
    cohort: dict[str, str] = field(default_factory=dict)
    image_paths: dict[str, str] = field(default_factory=dict)
    mask_paths: dict[str, str] = field(default_factory=dict)
    regions: dict[str, tuple[int, ...]] = field(default_factory=dict)
    modalities: dict[str, str] = field(default_factory=dict)


def load_volumes(path: Path | None = None) -> pd.DataFrame:
    """The store index. An id listed twice keeps its last row: since the 2026-10-26 re-index, 48 CQ500 studies have
    both a ``cq500`` (DICOM) and a ``seg_cq500`` row under one id, and the ``cq500`` row comes last."""
    path = Path(path or DEFAULT_VOLUMES)
    if not path.exists():
        raise FileNotFoundError(f"Missing volume table {path}")
    return pd.read_csv(path, low_memory=False).drop_duplicates("volume_id", keep="last")


def _case_name(volume_id: str) -> str:
    vid = str(volume_id)
    return vid.split(":", 1)[1] if ":" in vid else vid


def _bundle_from_frame(
    name: str,
    task_type: str,
    frame: pd.DataFrame,
    y: np.ndarray,
    label_names: list[str],
    *,
    modality: str | None = None,
    cohort_col: str | None = None,
    mask_col: str | None = None,
    regions: dict[str, tuple[int, ...]] | None = None,
) -> TaskBundle:
    frame = frame.reset_index(drop=True)
    ids = frame["volume_id"].astype(str).tolist()
    mods = {}
    if "modality" in frame.columns:
        mods = dict(zip(ids, frame["modality"].astype(str)))
    elif modality:
        mods = {i: modality for i in ids}
    return TaskBundle(
        name=name,
        task_type=task_type,
        volume_ids=ids,
        splits=dict(zip(ids, frame["split"].astype(str))),
        y=np.asarray(y),
        label_names=list(label_names),
        modality_filter=modality,
        cohort=(
            dict(zip(ids, frame[cohort_col].astype(str)))
            if cohort_col and cohort_col in frame.columns
            else {}
        ),
        image_paths=dict(zip(ids, frame["image_path"].astype(str)))
        if "image_path" in frame.columns
        else {},
        mask_paths=dict(zip(ids, frame[mask_col].astype(str)))
        if mask_col and mask_col in frame.columns
        else {},
        regions=dict(regions or {}),
        modalities=mods,
    )


def _binarize(frame: pd.DataFrame, labels: list[str], threshold: float) -> np.ndarray:
    out = pd.DataFrame(index=frame.index)
    for col in labels:
        out[col] = (pd.to_numeric(frame[col], errors="coerce") >= threshold).astype(np.int64)
    return out.to_numpy(dtype=np.int64)


# Label tables copied from the raw dataset folders into the store (pool keeps only archives): read them there first.
LABEL_SOURCES = PROCESSED_ROOT / "labels" / "sources"
LABEL_FILES = (
    ("cq500_meta", "reads.csv"),
    ("oasis", "oasis1", "oasis_cross-sectional.csv"),
    ("oasis", "oasis2", "oasis_longitudinal.csv"),
    ("abide1", "Phenotypic_V1_0b.csv"),
    ("ixi", "IXI.xls"),
    ("pixar", "participants.tsv"),
    ("piop1", "participants.tsv"),
    ("piop2", "participants.tsv"),
    ("id1000", "participants.tsv"),
    ("msd_braintumor", "volume_labels.csv"),
    ("ms3seg", "participants.tsv"),        # written by brain.prep.stage_new
    ("isles24", "clinical.tsv"),           # written by brain.prep.stage_new
    ("atlas", "metadata.tsv"),             # written by brain.prep.stage_new
    ("valdo", "labels.tsv"),
    ("ucsf_pdgm", "meta", "UCSF-PDGM-metadata_v5.csv"),
    ("upenn_gbm", "meta", "UPENN-GBM_clinical_info_v2.1.csv"),
    ("rsna_ica", "train.csv"),             # copied out of the competition zip (sources.rsna_ica caches it too)
    ("fomo260k", "participants.tsv"),    # demographics / diagnosis for BrainQuery's tasks (brain.multitask)
)


def label_file(*parts: str) -> Path:
    """``labels/sources/<dataset>/...`` in the store when synced, else the file in the pool dataset folder."""
    local = LABEL_SOURCES.joinpath(*parts)
    return local if local.exists() else DATASETS.joinpath(*parts)


def sync_label_files(dest: Path = LABEL_SOURCES) -> list[Path]:
    """Copy every downstream label table from pool into the store (idempotent; run before packing pool)."""
    import shutil

    out = []
    for parts in LABEL_FILES:
        src, dst = DATASETS.joinpath(*parts), dest.joinpath(*parts)
        if not src.exists():
            continue
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        out.append(dst)
    return out


def cq500_majority(threshold: float = 0.5, findings: list[str] | None = None) -> pd.DataFrame:
    """Reader vote from Qure.ai reads.csv (0/1 per finding; default the multilabel task's ``CQ500_LABELS``): positive
    when at least ``threshold`` of the three readers marked it (0.5 = majority)."""
    reads = pd.read_csv(label_file("cq500_meta", "reads.csv"))
    out = pd.DataFrame({"name": reads["name"].astype(str)})
    for lab in findings or CQ500_LABELS:
        cols = [c for c in reads.columns if c.endswith(":" + lab)]
        if not cols:
            raise ValueError(f"CQ500 reads.csv has no reader columns for {lab}")
        out[lab] = (reads[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) >= threshold).astype(np.int64)
    return out


def _cq500_frame(volumes: pd.DataFrame, findings: list[str], threshold: float = 0.5) -> pd.DataFrame:
    """CQ500 and Seg-CQ500 volumes with 3-reader majority ``findings``, one volume per study: Seg-CQ500 re-releases
    some CQ500 studies, and a study present in both is kept once (the ``cq500`` copy) so it does not count twice in
    train or test."""
    ct = volumes[volumes["dataset"].isin(["cq500", "seg_cq500"])].copy()
    ct["name"] = ct["volume_id"].map(_case_name)
    ct = ct.sort_values("dataset").drop_duplicates("name", keep="first")   # "cq500" sorts before "seg_cq500"
    return ct.merge(cq500_majority(threshold, findings), on="name", how="inner")


def load_cq500_finding_task(volumes: pd.DataFrame, name: str) -> TaskBundle:
    """One CQ500 reader finding the main multilabel task leaves out (``CQ500_FINDING_TASKS``; rare: cross-validate)."""
    finding = CQ500_FINDING_TASKS[name]
    m = _cq500_frame(volumes, [finding])
    return _bundle_from_frame(name, "binary", m, m[finding].to_numpy(np.int64), [finding], modality="ct",
                              cohort_col="dataset")


def load_cq500_bleed_side_task(volumes: pd.DataFrame) -> TaskBundle:
    """Side of the bleed among unilateral hemorrhages (3-reader majority ICH, exactly one of left / right): 1 = left."""
    m = _cq500_frame(volumes, ["ICH", "BleedLocation-Left", "BleedLocation-Right"])
    m = m[(m["ICH"] == 1) & (m["BleedLocation-Left"] != m["BleedLocation-Right"])]
    return _bundle_from_frame("cq500_bleed_side", "binary", m, m["BleedLocation-Left"].to_numpy(np.int64), ["left"],
                              modality="ct", cohort_col="dataset")


def load_cq500_task(volumes: pd.DataFrame, threshold: float = 0.5) -> TaskBundle:
    """CQ500 and Seg-CQ500 together, multilabel ``CQ500_LABELS``; ``cohort`` is the dataset folder. ``threshold`` is
    the reader-agreement fraction for a positive label."""
    m = _cq500_frame(volumes, CQ500_LABELS, threshold)
    return _bundle_from_frame("cq500", "multilabel", m, m[CQ500_LABELS].to_numpy(np.int64), CQ500_LABELS,
                              modality="ct", cohort_col="dataset")


def _rsna_cache_path() -> Path:
    return SCRATCH / "work" / "rsna_ingest" / "volume_labels.csv"


def build_rsna_volume_labels(
    zip_path: Path | None = None,
    index_path: Path | None = None,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """One row per study: a subtype is positive if any slice in the study is."""
    cache_path = Path(cache_path or _rsna_cache_path())
    if cache_path.exists():
        return pd.read_csv(cache_path)

    import zipfile

    zip_path = Path(zip_path or (DATASETS / "rsna" / "rsna-ihd-dataset.zip"))
    index_path = Path(index_path or (SCRATCH / "work" / "rsna_ingest" / "study_index.json"))
    with index_path.open() as f:
        index = json.load(f)
    stem_to_uid: dict[str, str] = {}
    for uid, members in index.items():
        for member in members:
            stem_to_uid[Path(member).stem] = uid

    # Long CSV: ID is ``{slice}_{subtype}``, Label is 0/1.
    flags: dict[str, dict[str, int]] = {}
    csv_name = "rsna-intracranial-hemorrhage-detection/stage_2_train.csv"
    with zipfile.ZipFile(zip_path) as zf, zf.open(csv_name) as src:
        header = src.readline()
        if not header.startswith(b"ID,"):
            raise ValueError(f"Unexpected RSNA label header: {header[:40]!r}")
        for raw in src:
            line = raw.decode().strip()
            if not line:
                continue
            sid, bit = line.split(",", 1)
            slice_id, subtype = sid.rsplit("_", 1)
            if subtype not in RSNA_LABELS:
                continue
            uid = stem_to_uid.get(slice_id)
            if uid is None:
                continue
            row = flags.setdefault(uid, {k: 0 for k in RSNA_LABELS})
            if bit.strip() == "1":
                row[subtype] = 1

    records = []
    for uid, row in flags.items():
        safe = "RSNA-" + uid.replace(".", "_")[:80]
        records.append({"volume_id": f"ct:{safe}", **row})
    out = pd.DataFrame(records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


RSNA_MAX_TRAIN = 2000   # frozen heads need far fewer than 18k training studies; the token cache is ~11.8 MB per study
RSNA_MAX_EVAL = {"val": 500, "test": 1000}   # of ~5.4k val + test studies: test AUC intervals about +-0.02


def load_rsna_task(volumes: pd.DataFrame, labels: pd.DataFrame | None = None, name: str = "rsna_ich",
                   max_train: int | None = None, max_eval: int | None = None) -> TaskBundle:
    """RSNA study-level subtypes. The train split is subsampled to ``max_train`` studies (env ``BRAIN_RSNA_MAX_TRAIN``;
    0 = all) and validation / test to ``RSNA_MAX_EVAL`` (env ``BRAIN_RSNA_MAX_EVAL`` / ``max_eval`` for both),
    stratified on ``any``, with a fixed seed. Whole-split token embeddings (~5.4k val + test studies at 5120 tokens)
    ran out of host memory, and even 6k studies' cache (~70 GB per model) overfilled scratch when several models ran."""
    labels = labels if labels is not None else build_rsna_volume_labels()
    rsna = volumes[volumes["dataset"] == "rsna_ich"].copy()
    merged = rsna.merge(labels, on="volume_id", how="inner")
    caps = {"train": int(os.environ.get("BRAIN_RSNA_MAX_TRAIN", RSNA_MAX_TRAIN) if max_train is None else max_train)}
    for split in ("val", "test"):
        caps[split] = int(os.environ.get("BRAIN_RSNA_MAX_EVAL", RSNA_MAX_EVAL[split]) if max_eval is None else max_eval)
    for split, cap in caps.items():
        part = merged["split"] == split
        if cap and part.sum() > cap:
            keep = merged[part].groupby("any").sample(frac=cap / part.sum(), random_state=0).index
            merged = merged[~part | merged.index.isin(keep)].reset_index(drop=True)
    y = merged[RSNA_LABELS].to_numpy(dtype=np.int64)
    return _bundle_from_frame(name, "multilabel", merged, y, RSNA_LABELS, modality="ct")


def _oasis_sessions(volumes: pd.DataFrame, dataset: str, clinical: pd.DataFrame, id_col: str) -> pd.DataFrame:
    vol = volumes[volumes["dataset"] == dataset].copy()
    vol["session"] = vol["volume_id"].map(_case_name)
    clin = clinical.rename(columns={id_col: "session"})
    return vol.merge(clin, on="session", how="inner")


def load_oasis1_age_task(
    volumes: pd.DataFrame,
    clinical_csv: Path | None = None,
) -> TaskBundle:
    clinical_csv = clinical_csv or (label_file("oasis", "oasis1", "oasis_cross-sectional.csv"))
    clin = pd.read_csv(clinical_csv)
    merged = _oasis_sessions(volumes, "oasis1", clin, "ID")
    merged = merged.dropna(subset=["Age"])
    y = merged["Age"].to_numpy(dtype=np.float64)
    return _bundle_from_frame("oasis1_age", "regression", merged, y, ["Age"], modality="mri")


def load_oasis1_cdr_task(
    volumes: pd.DataFrame,
    clinical_csv: Path | None = None,
) -> TaskBundle:
    clinical_csv = clinical_csv or (label_file("oasis", "oasis1", "oasis_cross-sectional.csv"))
    clin = pd.read_csv(clinical_csv)
    merged = _oasis_sessions(volumes, "oasis1", clin, "ID")
    merged = merged.dropna(subset=["CDR"])
    y = (merged["CDR"].astype(float) > 0).to_numpy(dtype=np.int64)
    return _bundle_from_frame("oasis1_cdr", "binary", merged, y, ["CDR>0"], modality="mri")


def load_oasis2_cdr_task(
    volumes: pd.DataFrame,
    clinical_csv: Path | None = None,
) -> TaskBundle:
    clinical_csv = clinical_csv or (label_file("oasis", "oasis2", "oasis_longitudinal.csv"))
    clin = pd.read_csv(clinical_csv)
    merged = _oasis_sessions(volumes, "oasis2", clin, "MRI ID")
    merged = merged.dropna(subset=["CDR"])
    y = (merged["CDR"].astype(float) > 0).to_numpy(dtype=np.int64)
    return _bundle_from_frame("oasis2_cdr", "binary", merged, y, ["CDR>0"], modality="mri")


def load_oasis2_conversion_task(
    volumes: pd.DataFrame,
    clinical_csv: Path | None = None,
) -> TaskBundle:
    """Baseline session with CDR 0, labeled by whether a later visit has CDR > 0."""
    clinical_csv = clinical_csv or (label_file("oasis", "oasis2", "oasis_longitudinal.csv"))
    clin = pd.read_csv(clinical_csv)
    clin["CDR"] = pd.to_numeric(clin["CDR"], errors="coerce")
    base = clin[clin["Visit"] == 1].dropna(subset=["CDR"])
    base = base[base["CDR"] <= 0]
    later = clin[clin["Visit"] > 1]
    converted = set(
        later.loc[later["CDR"] > 0, "Subject ID"].astype(str)
    )
    base = base.copy()
    base["converted"] = base["Subject ID"].astype(str).isin(converted).astype(np.int64)
    merged = _oasis_sessions(volumes, "oasis2", base, "MRI ID")
    y = merged["converted"].to_numpy(dtype=np.int64)
    return _bundle_from_frame(
        "oasis2_conversion", "binary", merged, y, ["converted"], modality="mri"
    )


def load_oasis_mmse_task(volumes: pd.DataFrame, which: str) -> TaskBundle:
    """Mini-Mental State Examination (0-30) per OASIS-1 / OASIS-2 session."""
    path, col = ((("oasis", "oasis1", "oasis_cross-sectional.csv"), "ID") if which == "oasis1"
                 else (("oasis", "oasis2", "oasis_longitudinal.csv"), "MRI ID"))
    merged = _oasis_sessions(volumes, which, pd.read_csv(label_file(*path)), col)
    merged["MMSE"] = pd.to_numeric(merged["MMSE"], errors="coerce")
    merged = merged.dropna(subset=["MMSE"])
    return _bundle_from_frame(f"{which}_mmse", "regression", merged, merged["MMSE"].to_numpy(np.float64), ["MMSE"],
                              modality="mri")


def load_abide_ados_task(volumes: pd.DataFrame) -> TaskBundle:
    """ABIDE I autism severity: ADOS calibrated severity score (Gotham, 1-10; comparable across ADOS modules), ASD
    participants only. Missing values are -9999 in the phenotypic file."""
    pheno = pd.read_csv(label_file("abide1", "Phenotypic_V1_0b.csv"))
    pheno["case_id"] = pheno["FILE_ID"].astype(str)
    pheno["severity"] = pd.to_numeric(pheno["ADOS_GOTHAM_SEVERITY"], errors="coerce")
    pheno = pheno[(pheno["DX_GROUP"] == 1) & pheno["severity"].between(1, 10)]
    vol = volumes[volumes["dataset"] == "abide1"].copy()
    merged = vol.merge(pheno[["case_id", "severity", "SITE_ID"]], on="case_id", how="inner").rename(
        columns={"SITE_ID": "site"})
    return _bundle_from_frame("abide_ados", "regression", merged, merged["severity"].to_numpy(np.float64),
                              ["ADOS_severity"], modality="mri", cohort_col="site")


def load_abide_asd_task(
    volumes: pd.DataFrame,
    phenotypic: Path | None = None,
) -> TaskBundle:
    """ABIDE I: DX_GROUP 1 = ASD, 2 = control. Cohort is the scan site (FILE_ID prefix)."""
    phenotypic = phenotypic or (label_file("abide1", "Phenotypic_V1_0b.csv"))
    pheno = pd.read_csv(phenotypic)
    pheno["case_id"] = pheno["FILE_ID"].astype(str)
    vol = volumes[volumes["dataset"] == "abide1"].copy()
    merged = vol.merge(pheno[["case_id", "DX_GROUP", "SITE_ID"]], on="case_id", how="inner")
    merged = merged[pd.to_numeric(merged["DX_GROUP"], errors="coerce").isin([1, 2])]
    y = (pd.to_numeric(merged["DX_GROUP"]) == 1).astype(np.int64).to_numpy()
    merged = merged.rename(columns={"SITE_ID": "site"})
    return _bundle_from_frame(
        "abide_asd", "binary", merged, y, ["ASD"], modality="mri", cohort_col="site",
    )


def load_ixi_age_task(volumes: pd.DataFrame, xls: Path | None = None) -> TaskBundle:
    xls = xls or (label_file("ixi", "IXI.xls"))
    clin = pd.read_excel(xls, engine="xlrd").dropna(subset=["AGE"])
    # IXI.xls repeats some subjects: identical rows collapse to one; subjects with conflicting ages are dropped
    ages = clin.groupby("IXI_ID")["AGE"].agg(lambda a: a.round(2).nunique())
    clin = clin[clin["IXI_ID"].isin(ages[ages == 1].index)].drop_duplicates("IXI_ID")
    vol = volumes[volumes["dataset"] == "ixi_t1"].copy()
    vol["ixi_id"] = vol["case_id"].str.extract(r"IXI(\d+)", expand=False).astype(int)
    merged = vol.merge(clin, left_on="ixi_id", right_on="IXI_ID", how="inner")
    merged = merged.dropna(subset=["AGE"])
    y = merged["AGE"].to_numpy(dtype=np.float64)
    return _bundle_from_frame("ixi_age", "regression", merged, y, ["Age"], modality="mri")


def load_bids_age_task(
    volumes: pd.DataFrame,
    dataset: str,
    tsv: Path,
    name: str | None = None,
) -> TaskBundle:
    part = pd.read_csv(tsv, sep="\t")
    age_col = "Age" if "Age" in part.columns else "age"
    vol = volumes[volumes["dataset"] == dataset].copy()
    merged = vol.merge(part, left_on="case_id", right_on="participant_id", how="inner")
    age = pd.to_numeric(merged[age_col], errors="coerce")
    merged = merged.loc[age.notna()].copy()
    y = age.loc[merged.index].to_numpy(dtype=np.float64)
    return _bundle_from_frame(name or f"{dataset}_age", "regression", merged, y, ["Age"], modality="mri")


def _seg_frame(
    volumes: pd.DataFrame,
    dataset: str,
    mask_suffix: str,
    mask_col: str | None = None,
) -> pd.DataFrame:
    vol = volumes[volumes["dataset"] == dataset].copy()
    col = mask_col or (
        "mask_seg" if str(dataset).startswith("brats")
        else "mask_ich" if "cq500" in str(dataset)
        else "mask_lesion"
    )
    if col in vol.columns and (vol[col].fillna("").astype(str).str.len() > 0).any():
        vol = vol[vol[col].fillna("").astype(str).str.len() > 0].copy()
        vol["mask_path"] = vol[col].astype(str)
        return vol
    if dataset.startswith("brats"):
        vol["mask_path"] = vol["image_path"].map(
            lambda p: str(Path(p).with_name(Path(p).name.replace("-t1n.zip", ".seg.zip")))
        )
    else:
        vol["mask_path"] = vol["image_path"].map(
            lambda p: str(Path(p).with_name(Path(p).stem + mask_suffix))
        )
    vol = vol[vol["mask_path"].fillna("").astype(str).str.len() > 0]
    return vol


def _load_seg(
    name: str,
    volumes: pd.DataFrame,
    dataset: str,
    regions: dict[str, tuple[int, ...]],
    mask_suffix: str,
    *,
    mask_col: str | None = None,
    modality: str | None = None,
) -> TaskBundle:
    frame = _seg_frame(volumes, dataset, mask_suffix, mask_col=mask_col)
    y = np.zeros((len(frame),), dtype=np.int64)
    return _bundle_from_frame(
        name,
        "segmentation",
        frame,
        y,
        list(regions),
        modality=modality or ("mri" if dataset.startswith("brats") else "ct"),
        mask_col="mask_path",
        regions=regions,
    )


def load_seg_cq500_task(volumes: pd.DataFrame) -> TaskBundle:
    return _load_seg("seg_cq500", volumes, "seg_cq500", ICH_REGIONS, ".ICH.zip",
                     mask_col="mask_ich", modality="ct")


def load_brats2023_gli_task(volumes: pd.DataFrame) -> TaskBundle:
    return _load_seg("brats2023_gli", volumes, "brats2023_gli_t1n", GLI_REGIONS, ".seg.zip")


def load_brats2023_men_task(volumes: pd.DataFrame) -> TaskBundle:
    return _load_seg("brats2023_men", volumes, "brats2023_men_t1n", MEN_REGIONS, ".seg.zip")


def load_brats2024_gli_task(volumes: pd.DataFrame) -> TaskBundle:
    return _load_seg("brats2024_gli", volumes, "brats2024_gli_t1n", GLI2024_REGIONS, ".seg.zip")


def load_brats2023_ped_task(volumes: pd.DataFrame) -> TaskBundle:
    return _load_seg("brats2023_ped", volumes, "brats2023_ped_t1n", GLI_REGIONS, ".seg.zip")


def load_atlas_task(volumes: pd.DataFrame) -> TaskBundle:
    return _load_seg("atlas", volumes, "atlas", LESION_REGIONS, ".lesion.zip",
                     mask_col="mask_lesion", modality="mri")


def load_isles22_task(volumes: pd.DataFrame) -> TaskBundle:
    return _load_seg("isles22", volumes, "isles22_dwi", LESION_REGIONS, ".lesion.zip",
                     mask_col="mask_lesion", modality="mri")


def load_wmh_task(volumes: pd.DataFrame) -> TaskBundle:
    return _load_seg("wmh", volumes, "wmh_flair", WMH_REGIONS, ".lesion.zip",
                     mask_col="mask_lesion", modality="mri")


def load_isles24_task(volumes: pd.DataFrame) -> TaskBundle:
    return _load_seg("isles24", volumes, "isles24_ncct", LESION_REGIONS, ".lesion.zip",
                     mask_col="mask_lesion", modality="ct")


def _isles24_clinical(volumes: pd.DataFrame, column: str) -> pd.DataFrame:
    """ISLES'24 admission NCCT volumes joined to one clinical column (``clinical.tsv``, brain.prep.stage_new)."""
    clin = pd.read_csv(label_file("isles24", "clinical.tsv"), sep="\t")
    vol = volumes[volumes["dataset"] == "isles24_ncct"].copy()
    vol["participant_id"] = vol["case_id"].astype(str)
    merged = vol.merge(clin[["participant_id", column, "Center"]], on="participant_id", how="inner")
    merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged.dropna(subset=[column]).rename(columns={"Center": "center"})


def load_isles24_mrs_task(volumes: pd.DataFrame) -> TaskBundle:
    """ISLES'24 3-month functional outcome from the admission NCCT: poor outcome = mRS 3-6 (the usual dichotomy;
    every patient had thrombectomy, so treatment success also drives it)."""
    m = _isles24_clinical(volumes, "mRS 3 months")
    return _bundle_from_frame("isles24_mrs", "binary", m, (m["mRS 3 months"] >= 3).astype(np.int64).to_numpy(),
                              ["poor_outcome"], modality="ct", cohort_col="center")


def load_isles24_mrs_discharge_task(volumes: pd.DataFrame) -> TaskBundle:
    """ISLES'24 poor outcome at discharge (mRS 3-6) from the admission NCCT."""
    m = _isles24_clinical(volumes, "mRS discharge")
    return _bundle_from_frame("isles24_mrs_discharge", "binary", m,
                              (m["mRS discharge"] >= 3).astype(np.int64).to_numpy(), ["poor_outcome"],
                              modality="ct", cohort_col="center")


def load_isles24_nihss_discharge_task(volumes: pd.DataFrame) -> TaskBundle:
    """ISLES'24 stroke severity at discharge (NIHSS) from the admission NCCT."""
    m = _isles24_clinical(volumes, "NIHSS discharge")
    return _bundle_from_frame("isles24_nihss_discharge", "regression", m, m["NIHSS discharge"].to_numpy(np.float64),
                              ["NIHSS"], modality="ct", cohort_col="center")


def _isles24_cta(volumes: pd.DataFrame) -> pd.DataFrame:
    """ISLES'24 CTA volumes (registered onto the NCCT grid) with the NCCT case's vessel masks."""
    cta = volumes[volumes["dataset"] == "isles24_cta"].copy()
    ncct = volumes[volumes["dataset"] == "isles24_ncct"][["case_id", "mask_cow", "mask_lvo"]]
    return cta.drop(columns=["mask_cow", "mask_lvo"], errors="ignore").merge(ncct, on="case_id", how="inner")


def load_isles24_lvo_side_task(volumes: pd.DataFrame) -> TaskBundle:
    """ISLES'24 side of the large-vessel occlusion from the CTA: 1 = left (``mask_labels``: >= 90% of the LVO mask
    on one side of the brain-mask centroid; the rest are dropped)."""
    stats = pd.read_csv(label_file("isles24", "mask_stats.tsv"), sep="\t")
    stats["left_frac"] = stats["left_ml"] / (stats["left_ml"] + stats["right_ml"]).clip(lower=1e-6)
    stats = stats[(stats["lvo_ml"] > 0) & ((stats["left_frac"] >= 0.9) | (stats["left_frac"] <= 0.1))]
    m = _isles24_cta(volumes).merge(stats[["case_id", "left_frac"]], on="case_id", how="inner")
    return _bundle_from_frame("isles24_lvo_side", "binary", m, (m["left_frac"] >= 0.9).astype(np.int64).to_numpy(),
                              ["left"], modality="ct")


def load_isles24_cow_task(volumes: pd.DataFrame) -> TaskBundle:
    """ISLES'24 Circle of Willis arteries on the CTA (``COW_REGIONS``)."""
    m = _isles24_cta(volumes)
    m = m[m["mask_cow"].fillna("").astype(str).str.len() > 0].assign(mask_path=lambda d: d["mask_cow"].astype(str))
    return _bundle_from_frame("isles24_cow", "segmentation", m, np.zeros(len(m), np.int64), list(COW_REGIONS),
                              modality="ct", mask_col="mask_path", regions=COW_REGIONS)


def load_isles24_nihss_task(volumes: pd.DataFrame) -> TaskBundle:
    """ISLES'24 admission stroke severity (NIHSS, 0-42) from the admission NCCT."""
    m = _isles24_clinical(volumes, "NIHSS at admission")
    return _bundle_from_frame("isles24_nihss", "regression", m, m["NIHSS at admission"].to_numpy(np.float64),
                              ["NIHSS"], modality="ct", cohort_col="center")


def _mask_stats(volumes: pd.DataFrame, dataset: str, folder: str) -> pd.DataFrame:
    stats = pd.read_csv(label_file(folder, "mask_stats.tsv"), sep="\t")
    vol = volumes[volumes["dataset"] == dataset]
    return vol.merge(stats.drop(columns=["case_id"]), on="volume_id", how="inner")


def load_atlas_lesion_volume_task(volumes: pd.DataFrame) -> TaskBundle:
    """ATLAS chronic stroke lesion volume, log(1 + ml) (``mask_labels``)."""
    m = _mask_stats(volumes, "atlas", "atlas")
    return _bundle_from_frame("atlas_lesion_volume", "regression", m, np.log1p(m["lesion_ml"].to_numpy(np.float64)),
                              ["log_lesion_ml"], modality="mri")


# INSTANCE 2022 ICH (99 NCCT): the imaging items of the ICH score (Hemphill 2001) from the hematoma masks
# (``mask_labels``): volume >= 30 ml, ventricular extension, infratentorial origin; plus the side of unilateral bleeds.
INSTANCE_TASKS = {
    "instance_large_ich": "hematoma volume >= 30 ml",
    "instance_ivh": ">= 1 ml of the hematoma inside the ventricle prior (intraventricular extension)",
    "instance_infratentorial": "> 50% of the hematoma inside the posterior-fossa prior",
    "instance_side": "side of a unilateral hematoma (>= 90% on one side): 1 = left",
}


def load_instance_ich_task(volumes: pd.DataFrame, name: str) -> TaskBundle:
    """One of ``INSTANCE_TASKS`` (binary, CT)."""
    m = _mask_stats(volumes, "instance", "instance")
    m = m[m["ich_ml"] > 0].copy()
    if name == "instance_large_ich":
        y = m["ich_ml"] >= 30
    elif name == "instance_ivh":
        y = m["ventricle_ml"] >= 1
    elif name == "instance_infratentorial":
        y = m["posterior_fossa_ml"] > 0.5 * m["ich_ml"]
    else:
        left = m["left_ml"] / (m["left_ml"] + m["right_ml"]).clip(lower=1e-6)
        m = m[(left >= 0.9) | (left <= 0.1)].copy()
        y = left[m.index] >= 0.9
    return _bundle_from_frame(name, "binary", m, y.astype(np.int64).to_numpy(), [name.removeprefix("instance_")],
                              modality="ct")


def load_atlas_lesion_side_task(volumes: pd.DataFrame) -> TaskBundle:
    """ATLAS lesion side among unilateral lesions (>= 90% of the lesion on one side): 1 = left."""
    m = _mask_stats(volumes, "atlas", "atlas")
    m["left_frac"] = m["left_ml"] / m["lesion_ml"].clip(lower=1e-6)
    m = m[(m["lesion_ml"] > 0) & ((m["left_frac"] >= 0.9) | (m["left_frac"] <= 0.1))]
    return _bundle_from_frame("atlas_lesion_side", "binary", m, (m["left_frac"] >= 0.9).astype(np.int64).to_numpy(),
                              ["left"], modality="mri")


def load_atlas_days_post_stroke_task(volumes: pd.DataFrame) -> TaskBundle:
    """ATLAS time since the stroke, log(1 + days) (session metadata; staged by brain.prep.stage_new)."""
    meta = pd.read_csv(label_file("atlas", "metadata.tsv"), sep="\t").dropna(subset=["DAYS_POST_STROKE"])
    vol = volumes[volumes["dataset"] == "atlas"].copy()
    m = vol.merge(meta[["participant_id", "DAYS_POST_STROKE", "SITE"]].rename(columns={"participant_id": "case_id"}),
                  on="case_id", how="inner")
    return _bundle_from_frame("atlas_days_post_stroke", "regression", m,
                              np.log1p(m["DAYS_POST_STROKE"].clip(lower=0).to_numpy(np.float64)), ["log_days"],
                              modality="mri", cohort_col="SITE")


def load_aisd_remote_infarct_task(volumes: pd.DataFrame) -> TaskBundle:
    """AISD acute-stroke CT: is an old (remote) infarct present besides the acute one (mask code 1, >= 0.1 ml)?"""
    m = _mask_stats(volumes, "aisd", "aisd")
    return _bundle_from_frame("aisd_remote_infarct", "binary", m, (m["code1_ml"] >= 0.1).astype(np.int64).to_numpy(),
                              ["remote_infarct"], modality="ct")


def load_ms3seg_lesion_load_task(volumes: pd.DataFrame) -> TaskBundle:
    """MS3SEG MS lesion load, log(1 + ml) (from the staged masks)."""
    part = pd.read_csv(label_file("ms3seg", "participants.tsv"), sep="\t", dtype={"participant_id": str})
    vol = volumes[volumes["dataset"] == "ms3seg_flair"].copy()
    vol["participant_id"] = vol["case_id"].astype(str)
    m = vol.merge(part, on="participant_id", how="inner").dropna(subset=["lesion_ml"])
    return _bundle_from_frame("ms3seg_lesion_load", "regression", m, np.log1p(m["lesion_ml"].to_numpy(np.float64)),
                              ["log_lesion_ml"], modality="mri")


def _gli_tcga(volumes: pd.DataFrame) -> pd.DataFrame:
    """BraTS 2023 GLI post-contrast T1 of the TCGA cases, with TCGA clinical data (brain.datasets.tcga_glioma)."""
    clin = pd.read_csv(label_file("tcga_glioma", "clinical.tsv"), sep="\t")
    vol = volumes[volumes["dataset"] == "brats2023_gli_t1c"].copy()
    vol["brats_id"] = vol["case_id"].astype(str).str.removesuffix("-t1c")
    return vol.merge(clin, on="brats_id", how="inner").assign(cohort=lambda d: d["cancer_type"])


def load_gli_survival_task(volumes: pd.DataFrame) -> TaskBundle:
    """Glioma overall survival beyond 12 months (1) vs death within 12 months (0); patients censored alive before 12
    months are dropped."""
    m = _gli_tcga(volumes)
    m = m[(m["deceased"] == 1) | (m["os_months"] >= 12)].dropna(subset=["os_months"])
    return _bundle_from_frame("gli_survival_12m", "binary", m, (m["os_months"] >= 12).astype(np.int64).to_numpy(),
                              ["survived_12m"], modality="mri", cohort_col="cohort")


def load_gli_lgg_task(volumes: pd.DataFrame) -> TaskBundle:
    """Lower-grade glioma (TCGA-LGG, WHO grade 2-3) vs glioblastoma (TCGA-GBM)."""
    m = _gli_tcga(volumes)
    return _bundle_from_frame("gli_lgg", "binary", m, (m["cancer_type"] == "LGG").astype(np.int64).to_numpy(),
                              ["LGG"], modality="mri", cohort_col="cohort")


def load_gli_idh_task(volumes: pd.DataFrame) -> TaskBundle:
    """IDH mutation (TCGA PanCancer subtype; known for a subset)."""
    m = _gli_tcga(volumes).dropna(subset=["idh_mutant"])
    return _bundle_from_frame("gli_idh", "binary", m, m["idh_mutant"].astype(np.int64).to_numpy(), ["IDH_mutant"],
                              modality="mri", cohort_col="cohort")


def _ucsf_pdgm(volumes: pd.DataFrame) -> pd.DataFrame:
    """UCSF-PDGM pre-operative post-contrast T1 (the TCGA glioma tasks' sequence), one baseline scan per patient, with
    the TCIA clinical metadata. Follow-up scans (``_FU###d``, post-treatment) are left out. Metadata IDs have three
    digits for baselines (``UCSF-PDGM-004``), files four."""
    meta = pd.read_csv(label_file("ucsf_pdgm", "meta", "UCSF-PDGM-metadata_v5.csv"))
    meta = meta[~meta["ID"].str.contains("_FU")]
    meta["case_id"] = meta["ID"].str.extract(r"UCSF-PDGM-(\d+)")[0].astype(int).map("UCSF-PDGM-{:04d}_t1c".format)
    vol = volumes[volumes["dataset"] == "ucsf_pdgm_t1c"]
    return vol.merge(meta, on="case_id", how="inner")


def load_ucsf_survival_task(volumes: pd.DataFrame) -> TaskBundle:
    """UCSF-PDGM overall survival (days from diagnosis) beyond 12 months (1) vs death within 12 months (0); patients
    censored alive before 12 months are dropped."""
    m = _ucsf_pdgm(volumes).dropna(subset=["OS"])
    m = m[(m["1-dead 0-alive"] == 1) | (m["OS"] >= 365)]
    return _bundle_from_frame("ucsf_survival_12m", "binary", m, (m["OS"] >= 365).astype(np.int64).to_numpy(),
                              ["survived_12m"], modality="mri")


def load_ucsf_lgg_task(volumes: pd.DataFrame) -> TaskBundle:
    """UCSF-PDGM WHO CNS 2021 grade 2-3 (1) vs grade 4 (0)."""
    m = _ucsf_pdgm(volumes)
    return _bundle_from_frame("ucsf_lgg", "binary", m, (m["WHO CNS Grade"] < 4).astype(np.int64).to_numpy(),
                              ["grade_2_3"], modality="mri")


def load_ucsf_idh_task(volumes: pd.DataFrame) -> TaskBundle:
    """UCSF-PDGM IDH1/2 mutation (any variant, UCSF500 sequencing) vs wildtype."""
    m = _ucsf_pdgm(volumes).dropna(subset=["IDH"])
    return _bundle_from_frame("ucsf_idh", "binary", m, (m["IDH"] != "wildtype").astype(np.int64).to_numpy(),
                              ["IDH_mutant"], modality="mri")


def load_ucsf_mgmt_task(volumes: pd.DataFrame) -> TaskBundle:
    """MGMT promoter methylation (positive vs negative) in IDH-wildtype glioblastoma, the setting of the BraTS 2021 /
    RSNA-MICCAI MGMT challenge. Unknown and indeterminate are dropped; restricting to one diagnosis keeps the label from
    standing in for IDH (IDH-mutant tumours are mostly methylated)."""
    m = _ucsf_pdgm(volumes)
    m = m[(m["Final pathologic diagnosis (WHO 2021)"] == "Glioblastoma, IDH-wildtype")
          & m["MGMT status"].isin(["positive", "negative"])]
    return _bundle_from_frame("ucsf_mgmt", "binary", m, (m["MGMT status"] == "positive").astype(np.int64).to_numpy(),
                              ["MGMT_methylated"], modality="mri")


def _upenn_gbm(volumes: pd.DataFrame) -> pd.DataFrame:
    """UPENN-GBM pre-operative post-contrast T1 (``_11`` scans; ``_21`` are post-treatment follow-ups) with the TCIA
    clinical table. A multitask training cohort under glioma option B (``index.MULTITASK_FAMILIES``)."""
    meta = pd.read_csv(label_file("upenn_gbm", "meta", "UPENN-GBM_clinical_info_v2.1.csv"))
    meta = meta[meta["ID"].str.endswith("_11")]
    meta["case_id"] = meta["ID"] + "_t1c"
    vol = volumes[volumes["dataset"] == "upenn_gbm_t1c"]
    return vol.merge(meta, on="case_id", how="inner")


def load_upenn_survival_task(volumes: pd.DataFrame) -> TaskBundle:
    """UPENN-GBM survival from surgery beyond 12 months (1) vs death within 12 months (0); alive or lost to follow-up
    with less than 12 months of follow-up is dropped, as is an uncertain date of death."""
    m = _upenn_gbm(volumes)
    days = pd.to_numeric(m["Survival_from_surgery_days_UPDATED"], errors="coerce")
    follow = pd.to_numeric(m["Survival_Censor"], errors="coerce")
    dead = m["Survival_Status"] == "Deceased"
    ok = (dead & days.notna()) | (~dead & (m["Survival_Status"] != "Deceased - uncertain date of death") & (follow >= 365))
    m, y = m[ok], np.where(dead[ok], days[ok] >= 365, True)
    return _bundle_from_frame("upenn_survival_12m", "binary", m, y.astype(np.int64), ["survived_12m"], modality="mri")


def load_upenn_mgmt_task(volumes: pd.DataFrame) -> TaskBundle:
    """UPENN-GBM MGMT promoter methylated (1) vs unmethylated (0); indeterminate / not available dropped."""
    m = _upenn_gbm(volumes)
    m = m[m["MGMT"].isin(["Methylated", "Unmethylated"])]
    return _bundle_from_frame("upenn_mgmt", "binary", m, (m["MGMT"] == "Methylated").astype(np.int64).to_numpy(),
                              ["MGMT_methylated"], modality="mri")


# ADNI / PPMI (LONI IDA, held out of every training pool; DUA: tables stay in the pool, owner-only, never in the store)
ADNI_DX3 = ["CN", "MCI", "AD"]


def _adni(volumes: pd.DataFrame) -> pd.DataFrame:
    """ADNI baseline 3D T1 (one session per subject, ADNI1-4; ``adni/tables/adni_labels.csv``): dx3 is the ADNIMERGE2
    diagnosis nearest the scan (within 120 days, else the IDA research group; SMC -> CN, EMCI / LMCI -> MCI), MMSE the
    nearest exam. BrainDINO's ADNI cohort is longitudinal ADNI1 (sc..m36), so its numbers are not directly comparable.
    Subjects of ``adni_exclude.csv`` (IDA duplicates across subject IDs) are dropped."""
    lab = pd.read_csv(label_file("adni", "tables", "adni_labels.csv"))
    drop = set(pd.read_csv(label_file("adni", "tables", "adni_exclude.csv"))["subject"].astype(str))
    lab = lab[~lab["subject"].astype(str).isin(drop) & lab["t1_image_id"].notna()]
    lab["case_id"] = "I" + lab["t1_image_id"].astype(np.int64).astype(str)
    vol = volumes[volumes["dataset"] == "adni_t1w"]
    return vol.merge(lab.drop(columns=["subject"]), on="case_id", how="inner")


def load_adni_dx3_task(volumes: pd.DataFrame) -> TaskBundle:
    """ADNI CN / MCI / AD, one-hot (scored by macro one-vs-rest AUROC, BrainDINO's 3-class metric)."""
    m = _adni(volumes)
    m = m[m["dx3"].isin(ADNI_DX3)]
    y = (m["dx3"].to_numpy()[:, None] == np.asarray(ADNI_DX3)[None, :]).astype(np.int64)
    return _bundle_from_frame("adni_dx3", "multilabel", m, y, ADNI_DX3, modality="mri")


def load_adni_pair_task(volumes: pd.DataFrame, pos: str) -> TaskBundle:
    """ADNI ``pos`` (AD or MCI, 1) vs cognitively normal (0)."""
    m = _adni(volumes)
    m = m[m["dx3"].isin(["CN", pos])]
    return _bundle_from_frame(f"adni_cn_{pos.lower()}", "binary", m, (m["dx3"] == pos).astype(np.int64).to_numpy(),
                              [pos], modality="mri")


def load_adni_mmse_task(volumes: pd.DataFrame) -> TaskBundle:
    """ADNI MMSE (0-30) nearest the baseline scan."""
    m = _adni(volumes).dropna(subset=["mmse"])
    return _bundle_from_frame("adni_mmse", "regression", m, m["mmse"].astype(float).to_numpy(), ["mmse"], modality="mri")


def load_ppmi_pd_task(volumes: pd.DataFrame) -> TaskBundle:
    """PPMI Parkinson's disease (1) vs healthy control (0) from the baseline 3D T1, IDA research group
    (``ppmi/tables/ppmi_baseline_pick.csv``; BrainDINO: PD 1,158 / HC 283). Unlike BrainDINO, PPMI is in no training
    pool here."""
    pick = pd.read_csv(label_file("ppmi", "tables", "ppmi_baseline_pick.csv"))
    pick = pick[(pick["kind"] == "t1") & pick["Research Group"].isin(["PD", "Control"])]
    pick["case_id"] = "I" + pick["Image ID"].astype(np.int64).astype(str)
    m = volumes[volumes["dataset"] == "ppmi_t1w"].merge(pick[["case_id", "Research Group"]], on="case_id", how="inner")
    return _bundle_from_frame("ppmi_pd", "binary", m, (m["Research Group"] == "PD").astype(np.int64).to_numpy(),
                              ["PD"], modality="mri")


# RSNA Intracranial Aneurysm Detection location columns, left and right merged (BrainQuery's training flips scans,
# so a sided target would swap; brain.multitask.labels.src_rsna_ica reads these)
RSNA_ICA_LOCATIONS = {
    "ica_infraclinoid": ("Left Infraclinoid Internal Carotid Artery", "Right Infraclinoid Internal Carotid Artery"),
    "ica_supraclinoid": ("Left Supraclinoid Internal Carotid Artery", "Right Supraclinoid Internal Carotid Artery"),
    "mca": ("Left Middle Cerebral Artery", "Right Middle Cerebral Artery"),
    "acom": ("Anterior Communicating Artery",),
    "aca": ("Left Anterior Cerebral Artery", "Right Anterior Cerebral Artery"),
    "pcom": ("Left Posterior Communicating Artery", "Right Posterior Communicating Artery"),
    "basilar_tip": ("Basilar Tip",),
    "posterior_other": ("Other Posterior Circulation",),
}


def rsna_ica_labels(volumes: pd.DataFrame) -> pd.DataFrame:
    """RSNA-ICA series (CTA, TOF MRA, T2, post-contrast T1; one per SeriesInstanceUID, no patient IDs) with the
    competition's ``train.csv``: ``Aneurysm Present`` and the side-merged ``RSNA_ICA_LOCATIONS``. A pretraining and
    multitask training cohort (2026-10-29); its test split is the aneurysm evaluation."""
    meta = pd.read_csv(label_file("rsna_ica", "train.csv"))
    meta = meta.rename(columns={"SeriesInstanceUID": "case_id"})
    for loc, cols in RSNA_ICA_LOCATIONS.items():
        meta[loc] = meta[list(cols)].max(axis=1)
    vol = volumes[volumes["dataset"].astype(str).str.startswith("rsna_ica")]
    return vol.merge(meta, on="case_id", how="inner")


def load_rsna_ica_aneurysm_task(volumes: pd.DataFrame, which: str) -> TaskBundle:
    """Aneurysm present (any location) on CTA (``cta``) or on MR (``mr``: TOF MRA, T2, post-contrast T1)."""
    m = rsna_ica_labels(volumes)
    m = m[m["dataset"].eq("rsna_ica_cta") if which == "cta" else ~m["dataset"].eq("rsna_ica_cta")]
    return _bundle_from_frame(f"rsna_ica_{which}_aneurysm", "binary", m, m["Aneurysm Present"].astype(np.int64)
                              .to_numpy(), ["aneurysm"], modality="ct" if which == "cta" else "mri",
                              cohort_col=None if which == "cta" else "dataset")


def load_ms3seg_task(volumes: pd.DataFrame) -> TaskBundle:
    return _load_seg("ms3seg", volumes, "ms3seg_flair", MS3SEG_REGIONS, ".seg.zip", mask_col="mask_seg",
                     modality="mri")


def load_valdo_cmb_task(volumes: pd.DataFrame) -> TaskBundle:
    """Cerebral microbleeds on T2* (VALDO Task 2), rater union."""
    return _load_seg("valdo_cmb", volumes, "valdo_t2s", {"CMB": (1,)}, ".lesion.zip", mask_col="mask_lesion",
                     modality="mri")


def load_valdo_lacunes_task(volumes: pd.DataFrame) -> TaskBundle:
    """Lacunes on T1 (VALDO Task 3), rater union. Only Task-3 T1 volumes carry the mask."""
    return _load_seg("valdo_lacunes", volumes, "valdo_t1", {"lacune": (1,)}, ".lesion.zip", mask_col="mask_lesion",
                     modality="mri")


def load_ms3seg_age_task(volumes: pd.DataFrame) -> TaskBundle:
    part = pd.read_csv(label_file("ms3seg", "participants.tsv"), sep="\t", dtype={"participant_id": str})
    vol = volumes[volumes["dataset"] == "ms3seg_flair"].copy()
    vol["participant_id"] = vol["case_id"].astype(str)
    merged = vol.merge(part, on="participant_id", how="inner")
    merged = merged.dropna(subset=["age"])
    return _bundle_from_frame("ms3seg_age", "regression", merged, merged["age"].to_numpy(np.float64), ["Age"],
                              modality="mri")


def load_valdo_epvs_task(volumes: pd.DataFrame) -> TaskBundle:
    """Enlarged perivascular space burden (VALDO Task 1): log(1 + rated EPVS count over the annotated regions)."""
    lab = pd.read_csv(label_file("valdo", "labels.tsv"), sep="\t", dtype={"participant_id": str})
    lab = lab[(lab["task"] == "t1") & lab["epvs_count"].notna()].copy()
    lab["case_id"] = "t1_" + lab["participant_id"]
    vol = volumes[volumes["dataset"] == "valdo_t1"].copy()
    vol["case_id"] = vol["case_id"].astype(str)
    merged = vol.merge(lab[["case_id", "epvs_count"]], on="case_id", how="inner")
    y = np.log1p(merged["epvs_count"].to_numpy(np.float64))
    return _bundle_from_frame("valdo_epvs", "regression", merged, y, ["log1p_EPVS"], modality="mri")


def load_aisd_task(volumes: pd.DataFrame) -> TaskBundle:
    return _load_seg("aisd", volumes, "aisd", AISD_REGIONS, ".lesion.zip",
                     mask_col="mask_lesion", modality="ct")


def load_instance_task(volumes: pd.DataFrame) -> TaskBundle:
    return _load_seg("instance", volumes, "instance", ICH_REGIONS, ".ICH.zip",
                     mask_col="mask_ich", modality="ct")


def msd_join_key(vid: str) -> str:
    """BRATS_001 from msd_braintumor:BRATS_001, mri:MSD-BRATS_001, or msd:1."""
    s = str(vid)
    if ":" in s:
        s = s.split(":", 1)[1]
    for prefix in ("MSD-", "TS_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def _msd_with_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "volume_id" not in df.columns:
        if "name" not in df.columns:
            raise ValueError("MSD labels need volume_id or name")
        df["volume_id"] = "msd_braintumor:" + df["name"].astype(str)
    if "enhancing_high" not in df.columns:
        if "enhancing_vox" not in df.columns:
            raise ValueError("MSD labels need enhancing_high or enhancing_vox")
        vox = pd.to_numeric(df["enhancing_vox"], errors="coerce")
        q = vox.quantile([1 / 3, 2 / 3]).to_list()
        df["enhancing_tertile"] = pd.cut(
            vox, bins=[-0.5, q[0], q[1], float(vox.max()) + 1], labels=[0, 1, 2],
        ).astype(int)
        df["enhancing_high"] = (vox >= q[1]).astype(int)
    return df


def resolve_msd_labels(labels_csv: Path | None = None) -> pd.DataFrame:
    """Scratch store labels, then the pool Task01 summary CSV."""
    candidates = []
    if labels_csv is not None:
        candidates.append(Path(labels_csv))
    candidates.append(DEFAULT_VOLUMES.parent / "labels" / "msd_volume_labels.csv")
    pool = label_file("msd_braintumor", "volume_labels.csv")
    for path in candidates:
        if path.is_file():
            return _msd_with_targets(pd.read_csv(path))
    if pool.is_file():
        df = _msd_with_targets(pd.read_csv(pool))
        out = DEFAULT_VOLUMES.parent / "labels" / "msd_volume_labels.csv"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
        except OSError:
            pass
        return df
    tried = ", ".join(str(p) for p in candidates + [pool])
    raise FileNotFoundError(f"MSD labels not found. Looked in: {tried}")


def load_msd_task(
    manifest: pd.DataFrame,
    labels_csv: Path | None = None,
    target: str = "enhancing_high",
) -> TaskBundle:
    labels = resolve_msd_labels(labels_csv)
    labels = labels.copy()
    labels["_msd_key"] = labels["volume_id"].map(msd_join_key)
    frame = manifest.copy()
    frame["_msd_key"] = frame["volume_id"].astype(str).map(msd_join_key)
    keep = ["volume_id", "split", "image_path", "_msd_key"]
    keep = [c for c in keep if c in frame.columns]
    vols = (
        frame[keep]
        .drop_duplicates("volume_id")
        .merge(labels, on="_msd_key", how="inner", suffixes=("", "_lab"))
        .reset_index(drop=True)
    )
    if vols.empty:
        raise FileNotFoundError(
            "MSD labels matched 0 store volumes. Pack Task01 T1w with "
            "`python -m brain.prep.run --datasets msd` (held-out), then rebuild the index."
        )
    if target not in vols.columns:
        raise KeyError(f"MSD labels missing column {target}")
    y = vols[target].to_numpy(dtype=np.int64)
    return _bundle_from_frame("msd", "binary", vols, y, [target], modality="mri")


# Benchmark v3 decontamination (2026-11-05): evaluation scans that are the same scan as a training / pre-training scan,
# or a second copy of one inside their own cohort (it would sit in two cross-validation folds), verified at the pixel
# level (python -m brain.prep.data_qc verify -> python -m brain.prep.decontam bench-exclude). BRAIN_BENCH_EXCLUDE=none disables.
BENCH_EXCLUDE = Path(__file__).resolve().parents[2] / "configs" / "bench" / "exclude_v3.csv"


def bench_exclusions() -> set[str]:
    path = os.environ.get("BRAIN_BENCH_EXCLUDE", str(BENCH_EXCLUDE))
    if path == "none":
        return set()
    # rows of DUA cohorts (ADNI / PPMI ids may not be published) live in a git-ignored, owner-only twin file
    paths = [Path(path), Path(path).with_name(Path(path).stem + ".private.csv")]
    return set().union(*(set(pd.read_csv(q)["volume_id"].astype(str)) for q in paths if q.exists()))


def drop_volumes(task: TaskBundle, drop: set[str]) -> TaskBundle:
    """``task`` without the volumes in ``drop`` (ids, labels and every per-volume map together)."""
    keep = [i for i, v in enumerate(task.volume_ids) if str(v) not in drop]
    if len(keep) == len(task.volume_ids):
        return task
    ids = [task.volume_ids[i] for i in keep]
    kept = set(ids)
    sub = lambda d: {k: v for k, v in d.items() if k in kept}  # noqa: E731
    y = task.y[keep] if getattr(task.y, "shape", None) and len(task.y) == len(task.volume_ids) else task.y
    return replace(task, volume_ids=ids, y=y, splits=sub(task.splits), cohort=sub(task.cohort),
                   image_paths=sub(task.image_paths), mask_paths=sub(task.mask_paths), modalities=sub(task.modalities))


def load_task(name: str, volumes: pd.DataFrame, **kw) -> TaskBundle:
    """A downstream task, without the benchmark's decontamination exclusions (``bench_exclusions``)."""
    task = _load_task(name, volumes, **kw)
    drop = bench_exclusions()
    out = drop_volumes(task, drop) if drop else task
    if len(out.volume_ids) != len(task.volume_ids):
        print(f"[task] {name}: {len(task.volume_ids) - len(out.volume_ids)} volumes excluded (benchmark "
              f"decontamination, {BENCH_EXCLUDE.name})", flush=True)
    return out


# task name -> loader of its TaskBundle from the volume table (cq500 and msd take options: ``_load_task``)
LOADERS: dict[str, Callable[..., TaskBundle]] = {
    "cq500": load_cq500_task,
    "msd": load_msd_task,
    "rsna_ich": load_rsna_task,
    "oasis1_age": load_oasis1_age_task,
    "oasis1_cdr": load_oasis1_cdr_task,
    "oasis2_cdr": load_oasis2_cdr_task,
    "oasis2_conversion": load_oasis2_conversion_task,
    "abide_asd": load_abide_asd_task,
    "ixi_age": load_ixi_age_task,
    "pixar_age": lambda v: load_bids_age_task(v, "pixar", label_file("pixar", "participants.tsv")),
    "piop1_age": lambda v: load_bids_age_task(v, "piop1", label_file("piop1", "participants.tsv")),
    "piop2_age": lambda v: load_bids_age_task(v, "piop2", label_file("piop2", "participants.tsv")),
    "id1000_age": lambda v: load_bids_age_task(v, "id1000", label_file("id1000", "participants.tsv")),
    "seg_cq500": load_seg_cq500_task,
    "brats2023_gli": load_brats2023_gli_task,
    "brats2023_men": load_brats2023_men_task,
    "brats2024_gli": load_brats2024_gli_task,
    "brats2023_ped": load_brats2023_ped_task,
    "atlas": load_atlas_task,
    "isles22": load_isles22_task,
    "wmh": load_wmh_task,
    "isles24": load_isles24_task,
    "aisd": load_aisd_task,
    "instance": load_instance_task,
    "ms3seg": load_ms3seg_task,
    "valdo_cmb": load_valdo_cmb_task,
    "valdo_lacunes": load_valdo_lacunes_task,
    "ms3seg_age": load_ms3seg_age_task,
    "valdo_epvs": load_valdo_epvs_task,
    "isles24_mrs_discharge": load_isles24_mrs_discharge_task,
    "isles24_nihss_discharge": load_isles24_nihss_discharge_task,
    "isles24_lvo_side": load_isles24_lvo_side_task,
    "isles24_cow": load_isles24_cow_task,
    "cq500_bleed_side": load_cq500_bleed_side_task,
    "aisd_remote_infarct": load_aisd_remote_infarct_task,
    "abide_ados": load_abide_ados_task,
    "atlas_lesion_volume": load_atlas_lesion_volume_task,
    "atlas_lesion_side": load_atlas_lesion_side_task,
    "atlas_days_post_stroke": load_atlas_days_post_stroke_task,
    "ms3seg_lesion_load": load_ms3seg_lesion_load_task,
    "gli_survival_12m": load_gli_survival_task,
    "gli_lgg": load_gli_lgg_task,
    "gli_idh": load_gli_idh_task,
    "ucsf_survival_12m": load_ucsf_survival_task,
    "ucsf_lgg": load_ucsf_lgg_task,
    "ucsf_idh": load_ucsf_idh_task,
    "ucsf_mgmt": load_ucsf_mgmt_task,
    "upenn_survival_12m": load_upenn_survival_task,
    "upenn_mgmt": load_upenn_mgmt_task,
    "adni_dx3": load_adni_dx3_task,
    "adni_mmse": load_adni_mmse_task,
    "ppmi_pd": load_ppmi_pd_task,
    "isles24_mrs": load_isles24_mrs_task,
    "isles24_nihss": load_isles24_nihss_task,
    "cq500_chronic_bleed": lambda v: load_cq500_finding_task(v, "cq500_chronic_bleed"),
    "cq500_edh": lambda v: load_cq500_finding_task(v, "cq500_edh"),
    "cq500_other_fracture": lambda v: load_cq500_finding_task(v, "cq500_other_fracture"),
    "oasis1_mmse": lambda v: load_oasis_mmse_task(v, "oasis1"),
    "oasis2_mmse": lambda v: load_oasis_mmse_task(v, "oasis2"),
    "instance_large_ich": lambda v: load_instance_ich_task(v, "instance_large_ich"),
    "instance_ivh": lambda v: load_instance_ich_task(v, "instance_ivh"),
    "instance_infratentorial": lambda v: load_instance_ich_task(v, "instance_infratentorial"),
    "instance_side": lambda v: load_instance_ich_task(v, "instance_side"),
    "rsna_ica_cta_aneurysm": lambda v: load_rsna_ica_aneurysm_task(v, "cta"),
    "rsna_ica_mr_aneurysm": lambda v: load_rsna_ica_aneurysm_task(v, "mr"),
    "adni_cn_ad": lambda v: load_adni_pair_task(v, "AD"),
    "adni_cn_mci": lambda v: load_adni_pair_task(v, "MCI"),
}


def _load_task(name: str, volumes: pd.DataFrame, *, threshold: float = 0.5, msd_labels: Path | None = None,
               msd_target: str = "enhancing_high") -> TaskBundle:
    if name not in LOADERS:
        raise KeyError(f"Unknown task {name}. Choices: {', '.join(TASK_NAMES)}")
    opts = {"cq500": {"threshold": threshold}, "msd": {"labels_csv": msd_labels, "target": msd_target}}
    return LOADERS[name](volumes, **opts.get(name, {}))


def split_xy(task: TaskBundle, emb: dict) -> dict:
    """Align embeddings with labels; return train/val/test packs."""
    id_to_i = {vid: i for i, vid in enumerate(emb["volume_ids"])}
    rows = []
    for i, vid in enumerate(task.volume_ids):
        if vid not in id_to_i:
            continue
        rows.append((vid, task.splits[vid], i, id_to_i[vid]))
    if not rows:
        raise SystemExit(f"No overlap between task {task.name} labels and embeddings")

    def pack(split: str):
        chosen = [r for r in rows if r[1] == split]
        task_idx = [r[2] for r in chosen]
        emb_idx = [r[3] for r in chosen]
        ids = [r[0] for r in chosen]
        out = {
            "volume_ids": ids,
            "mus": emb["mus"][emb_idx],
            "mask": emb["mask"][emb_idx],
            "y": task.y[task_idx],
        }
        if "z" in emb:
            out["z"] = emb["z"][emb_idx]
        if task.cohort:
            out["cohort"] = [task.cohort[v] for v in ids]
        return out

    return {s: pack(s) for s in ("train", "val", "test")}


def task_summary(task: TaskBundle) -> str:
    lines = [
        f"{task.name}  type={task.task_type}  n={len(task.volume_ids)}  labels={task.label_names}"
    ]
    if task.regions:
        lines.append(f"  regions={ {k: list(v) for k, v in task.regions.items()} }")
    splits = pd.Series([task.splits[v] for v in task.volume_ids])
    lines.append("  split  " + ", ".join(f"{k}={int(v)}" for k, v in splits.value_counts().items()))
    if task.cohort:
        coh = pd.Series([task.cohort[v] for v in task.volume_ids])
        lines.append("  cohort " + ", ".join(f"{k}={int(v)}" for k, v in coh.value_counts().items()))
    if task.task_type == "binary":
        y = np.asarray(task.y).ravel()
        lines.append(f"  positive={int(y.sum())}/{len(y)}")
    elif task.task_type == "multilabel":
        y = np.asarray(task.y)
        rates = ", ".join(
            f"{name}={int(y[:, i].sum())}" for i, name in enumerate(task.label_names)
        )
        lines.append(f"  positives {rates}")
    elif task.task_type == "regression":
        y = np.asarray(task.y, dtype=float).ravel()
        lines.append(f"  mean={y.mean():.2f}  min={y.min():.1f}  max={y.max():.1f}")
    return "\n".join(lines)


def main() -> None:
    import sys

    if "--sync-labels" in sys.argv:
        for f in sync_label_files():
            print(f"[labels] {f}")
        return
    volumes = load_volumes()
    for name in TASK_NAMES:
        if name == "msd":
            continue
        task = load_task(name, volumes)
        print(task_summary(task))
        print()


if __name__ == "__main__":
    main()
