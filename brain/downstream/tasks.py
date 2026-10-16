"""Labeled brain.downstream tasks.

Classification and regression use the frozen-embedding heads. Segmentation
tasks point at the mask zip beside each volume; ``brain.downstream.seg_train``
trains a skip-connected decoder on a frozen ViT and scores per-volume Dice.

Splits are the ones stored in ``volumes.csv``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import DATASETS, PROCESSED_ROOT, SCRATCH

DEFAULT_VOLUMES = PROCESSED_ROOT / "volumes.csv"
# Extra tables for stores not yet copied into slices_zarr. AISD / INSTANCE live in the main store.
EXTRA_VOLUME_TABLES = ()

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

CLASSIFICATION_TASKS = (
    "cq500",
    "rsna",
    "rsna_ich",
    "oasis1_cdr",
    "oasis2_cdr",
    "oasis2_conversion",
    "abide_asd",
    "msd",
)
REGRESSION_TASKS = (
    "oasis1_age",
    "ixi_age",
    "pixar_age",
    "piop1_age",
    "piop2_age",
    "id1000_age",
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
)
# Frozen clf/reg suite (segmentation is ``seg3d`` / ``run_suite`` separately).
FROZEN_TASKS = CLASSIFICATION_TASKS + REGRESSION_TASKS
# Fast frozen-head tasks for a first smoke (both modalities, small n).
SMOKE_TASKS = ("oasis1_age", "oasis1_cdr", "cq500")
LESION_REGIONS = {"lesion": (1,)}
# AISD: remote=1, clear acute=2, blurred acute=3, invisible=4, infarct=5. Dice on {1,2,3,5}.
AISD_REGIONS = {"infarct": (1, 2, 3, 5)}
WMH_REGIONS = {"WMH": (1,)}
TASK_NAMES = CLASSIFICATION_TASKS + REGRESSION_TASKS + SEGMENTATION_TASKS
# Focal pathology: sample a finer axial grid so a 1–3 slice finding is not skipped.
FOCAL_SLICE_TASKS = frozenset((*SEGMENTATION_TASKS, "cq500", "rsna", "rsna_ich", "msd"))
FOCAL_SLICES = 64


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
    path = Path(path or DEFAULT_VOLUMES)
    frames = []
    if path.exists():
        frames.append(pd.read_csv(path, low_memory=False))
    for extra in EXTRA_VOLUME_TABLES:
        extra = Path(extra)
        if extra.exists() and extra.resolve() != path.resolve():
            frames.append(pd.read_csv(extra, low_memory=False))
    if not frames:
        raise FileNotFoundError(f"Missing volume table {path}")
    return pd.concat(frames, ignore_index=True).drop_duplicates("volume_id", keep="last")


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


def cq500_majority(reads_csv: Path | None = None) -> pd.DataFrame:
    """3-reader majority vote from Qure.ai reads.csv (0/1 per finding). EDH is omitted."""
    reads_csv = reads_csv or (DATASETS / "cq500_meta" / "reads.csv")
    df = pd.read_csv(reads_csv)
    out = pd.DataFrame({"name": df["name"].astype(str)})
    for lab in CQ500_LABELS:
        cols = [c for c in df.columns if c.endswith(":" + lab)]
        if not cols:
            raise ValueError(f"{reads_csv} has no reader columns for {lab}")
        votes = df[cols].apply(pd.to_numeric, errors="coerce")
        out[lab] = (votes.mean(axis=1) >= 0.5).astype(np.int64)
    return out


def load_cq500_task(
    volumes: pd.DataFrame,
    info_csv: Path | None = None,
    seg_info_csv: Path | None = None,
    threshold: float = 0.5,
    reads_csv: Path | None = None,
) -> TaskBundle:
    """CQ500 and Seg-CQ500 together. Labels are 3-reader majority; ``cohort`` is the dataset folder."""
    labels = cq500_majority(reads_csv)
    ct = volumes[volumes["dataset"].isin(["cq500", "seg_cq500"])].copy()
    ct["name"] = ct["volume_id"].map(_case_name)
    merged = ct.merge(labels[["name"] + CQ500_LABELS], on="name", how="inner")
    y = merged[CQ500_LABELS].to_numpy(dtype=np.int64)
    return _bundle_from_frame(
        "cq500",
        "multilabel",
        merged,
        y,
        CQ500_LABELS,
        modality="ct",
        cohort_col="dataset",
    )


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


def load_rsna_task(volumes: pd.DataFrame, labels: pd.DataFrame | None = None, name: str = "rsna") -> TaskBundle:
    labels = labels if labels is not None else build_rsna_volume_labels()
    rsna = volumes[volumes["dataset"] == "rsna_ich"].copy()
    merged = rsna.merge(labels, on="volume_id", how="inner")
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
    clinical_csv = clinical_csv or (DATASETS / "oasis" / "oasis1" / "oasis_cross-sectional.csv")
    clin = pd.read_csv(clinical_csv)
    merged = _oasis_sessions(volumes, "oasis1", clin, "ID")
    merged = merged.dropna(subset=["Age"])
    y = merged["Age"].to_numpy(dtype=np.float64)
    return _bundle_from_frame("oasis1_age", "regression", merged, y, ["Age"], modality="mri")


def load_oasis1_cdr_task(
    volumes: pd.DataFrame,
    clinical_csv: Path | None = None,
) -> TaskBundle:
    clinical_csv = clinical_csv or (DATASETS / "oasis" / "oasis1" / "oasis_cross-sectional.csv")
    clin = pd.read_csv(clinical_csv)
    merged = _oasis_sessions(volumes, "oasis1", clin, "ID")
    merged = merged.dropna(subset=["CDR"])
    y = (merged["CDR"].astype(float) > 0).to_numpy(dtype=np.int64)
    return _bundle_from_frame("oasis1_cdr", "binary", merged, y, ["CDR>0"], modality="mri")


def load_oasis2_cdr_task(
    volumes: pd.DataFrame,
    clinical_csv: Path | None = None,
) -> TaskBundle:
    clinical_csv = clinical_csv or (DATASETS / "oasis" / "oasis2" / "oasis_longitudinal.csv")
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
    clinical_csv = clinical_csv or (DATASETS / "oasis" / "oasis2" / "oasis_longitudinal.csv")
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


def load_abide_asd_task(
    volumes: pd.DataFrame,
    phenotypic: Path | None = None,
) -> TaskBundle:
    """ABIDE I: DX_GROUP 1 = ASD, 2 = control. Cohort is the scan site (FILE_ID prefix)."""
    phenotypic = phenotypic or (DATASETS / "abide1" / "Phenotypic_V1_0b.csv")
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
    xls = xls or (DATASETS / "ixi" / "IXI.xls")
    clin = pd.read_excel(xls, engine="xlrd")
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
    pool = DATASETS / "msd_braintumor" / "volume_labels.csv"
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


def load_task(
    name: str,
    volumes: pd.DataFrame,
    *,
    threshold: float = 0.5,
    msd_labels: Path | None = None,
    msd_target: str = "enhancing_high",
) -> TaskBundle:
    if name == "cq500":
        return load_cq500_task(volumes, threshold=threshold)
    if name in ("rsna", "rsna_ich"):
        return load_rsna_task(volumes, name=name)
    if name == "oasis1_age":
        return load_oasis1_age_task(volumes)
    if name == "oasis1_cdr":
        return load_oasis1_cdr_task(volumes)
    if name == "oasis2_cdr":
        return load_oasis2_cdr_task(volumes)
    if name == "oasis2_conversion":
        return load_oasis2_conversion_task(volumes)
    if name == "abide_asd":
        return load_abide_asd_task(volumes)
    if name == "ixi_age":
        return load_ixi_age_task(volumes)
    if name == "pixar_age":
        return load_bids_age_task(volumes, "pixar", DATASETS / "pixar" / "participants.tsv")
    if name == "piop1_age":
        return load_bids_age_task(volumes, "piop1", DATASETS / "piop1" / "participants.tsv")
    if name == "piop2_age":
        return load_bids_age_task(volumes, "piop2", DATASETS / "piop2" / "participants.tsv")
    if name == "id1000_age":
        return load_bids_age_task(volumes, "id1000", DATASETS / "id1000" / "participants.tsv")
    if name == "seg_cq500":
        return load_seg_cq500_task(volumes)
    if name == "brats2023_gli":
        return load_brats2023_gli_task(volumes)
    if name == "brats2023_men":
        return load_brats2023_men_task(volumes)
    if name == "brats2024_gli":
        return load_brats2024_gli_task(volumes)
    if name == "brats2023_ped":
        return load_brats2023_ped_task(volumes)
    if name == "atlas":
        return load_atlas_task(volumes)
    if name == "isles22":
        return load_isles22_task(volumes)
    if name == "wmh":
        return load_wmh_task(volumes)
    if name == "isles24":
        return load_isles24_task(volumes)
    if name == "aisd":
        return load_aisd_task(volumes)
    if name == "instance":
        return load_instance_task(volumes)
    if name == "msd":
        return load_msd_task(volumes, labels_csv=msd_labels, target=msd_target)
    raise KeyError(f"Unknown task {name}. Choices: {', '.join(TASK_NAMES)}")


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
    volumes = load_volumes()
    for name in TASK_NAMES:
        if name in ("msd", "rsna"):
            continue
        task = load_task(name, volumes)
        print(task_summary(task))
        print()


if __name__ == "__main__":
    main()
