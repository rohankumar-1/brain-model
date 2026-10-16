#!/usr/bin/env python3
"""Prepare shared-space 256×256×128 CT/MRI volumes → one compressed file per volume.

All datasets (CQ500, BraTS, OASIS, MSD, RSNA) share this pipeline. Each volume
is its own Zarr ZipStore (Blosc/zstd inside, zip stored so a slice chunk can
be read without unpacking the file). Files are grouped by dataset:

    $BRAIN_SCRATCH/data/processed/slices_zarr/<dataset>/<case_id>.zip
    $BRAIN_SCRATCH/data/processed/slices_zarr/<dataset>/<case_id>.<mask>.zip

Segmentation masks that exist (BraTS ``seg``, Seg-CQ500 ``ICH`` and ``brain``,
MSD tumor labels) are nearest-neighbor resampled onto the same grid and saved
beside the image. Slice sampling stays an index table (``image_path``, ``z``).
Splits are volume-level train / val / test (default 75 / 10 / 15).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

# Blosc via numcodecs is zarr2-oriented; zarr3 uses built-in codecs by default.
try:
    from numcodecs import Blosc

    BLOSC = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
except Exception:  # pragma: no cover
    BLOSC = None

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("BRAIN_SCRATCH", "/home/roku/orcd/scratch/brain-model"))
POOL = Path(os.environ.get("BRAIN_POOL", "/home/roku/orcd/pool"))
DATASETS = Path(os.environ.get("BRAIN_DATASETS", str(POOL / "datasets")))

# Raw archives live on pool; processed Zarr stays on scratch.
DEFAULT_RAW = DATASETS
DEFAULT_CQ500 = DATASETS / "seg_cq500" / "volumes"
DEFAULT_CQ500_EXTRA = DATASETS / "cq500"
DEFAULT_MSD_ROOT = DATASETS / "msd_braintumor" / "Task01_BrainTumour"
DEFAULT_BRATS_ROOT = DATASETS / "brats" / "extracted"
DEFAULT_OASIS_ROOT = DATASETS / "oasis" / "extracted"
DEFAULT_OUT = SCRATCH / "data" / "processed" / "slices_zarr"
DEFAULT_RSNA_ZIP = DATASETS / "rsna" / "rsna-ihd-dataset.zip"

OUT_XY = 256
OUT_Z = 128
ISO_SPACING = 1.0
SEED = 42

CT_HU_MIN = -200.0
CT_HU_MAX = 200.0

EMPTY_TISSUE_THRESH = 0.05
EMPTY_MIN_FRAC = 0.05

# Axial slice preference: Normal centered at mid-volume (paper-style μ≈mid, σ=50).
# Large σ still admits superior/inferior slices, but down-weights empty-ish extremes.
DEFAULT_SLICE_SIGMA = 50.0

MSD_CHANNELS = {"FLAIR": 0, "T1w": 1, "t1gd": 2, "T2w": 3}
DEFAULT_MSD_CHANNEL = "T1w"

# BraTS mpMRI contrasts to ingest as separate volumes (protocol diversity).
BRATS_CONTRASTS = ("t1n", "t1c", "t2w", "t2f")

ZARR_CHUNK_Z = 8  # 8 × 256 × 256 × 2 B ≈ 1 MiB float16 / chunk; random z reads stay cheap

SPLITS = ("train", "val", "test")
TRAIN_FRAC = 0.75
VAL_FRAC = 0.10
TEST_FRAC = 0.15


def dataset_family(dataset: str) -> str:
    """Coarse family for stratification (strip BraTS contrast suffix)."""
    ds = str(dataset)
    for contrast in BRATS_CONTRASTS:
        suf = f"_{contrast}"
        if ds.endswith(suf):
            return ds[: -len(suf)]
    if ds in ("seg_cq500", "cq500"):
        return "cq500"
    if ds.startswith("msd_"):
        return "msd_brats"
    return ds


def parse_brats_subject_contrast(case_id: str) -> tuple[str, str]:
    stem = str(case_id)
    for c in BRATS_CONTRASTS:
        for sep in (f"-{c}", f"_{c}"):
            if stem.endswith(sep):
                return stem[: -len(sep)], c
    return stem, ""


def make_dedup_key(modality: str, dataset: str, case_id: str) -> str:
    """Subject-level key for split assignment + cross-source dedup."""
    fam = dataset_family(dataset)
    if fam.startswith("brats"):
        subject, _ = parse_brats_subject_contrast(case_id)
        # Collapse year/cohort: BraTS-GLI-00324-000 shared across 2023/2024
        kind = "gli" if "_gli" in fam else "men" if "_men" in fam else fam
        return f"brats:{kind}:{subject}"
    if fam == "msd_brats":
        # MSD BRATS_001 style — treat as glioma-adjacent for dedup vs BraTS
        return f"msd:{case_id.replace('MSD-', '').replace('TS_', '')}"
    if fam == "cq500":
        return f"cq500:{case_id}"
    if fam.startswith("oasis"):
        return f"oasis:{case_id}"
    if fam == "rsna_ich":
        return f"rsna:{case_id}"
    return f"{modality}:{fam}:{case_id}"


def cohort_priority(dataset: str) -> int:
    """Higher wins when two volumes share (dedup_key, contrast)."""
    ds = str(dataset)
    if "brats2024_gli" in ds:
        return 50
    if "brats2023_gli" in ds:
        return 40
    if "brats2024_men" in ds:
        return 50
    if "brats2023_men" in ds:
        return 40
    if ds == "seg_cq500":
        return 20
    if ds == "cq500":
        return 10
    if ds.startswith("msd_"):
        return 5  # prefer real BraTS when both present
    return 0


def enrich_case(
    case_id: str,
    path: Path,
    dataset: str,
    modality: str,
    *,
    source_kind: str = "nifti",
    source_zip: str = "",
    source_members: tuple[str, ...] = (),
) -> dict:
    contrast = ""
    if dataset_family(dataset).startswith("brats"):
        _, contrast = parse_brats_subject_contrast(case_id)
    dedup = make_dedup_key(modality, dataset, case_id)
    subject = dedup
    return {
        "case_id": case_id,
        "path": str(path) if path else "",
        "dataset": dataset,
        "dataset_family": dataset_family(dataset),
        "modality": modality,
        "volume_id": f"{modality}:{case_id}",
        "dedup_key": dedup,
        "subject_id": subject,
        "contrast": contrast,
        "source_kind": source_kind,
        "source_zip": source_zip,
        "source_members": list(source_members),
        "priority": cohort_priority(dataset),
        "seg_path": "",
    }


def deduplicate_volumes(cases: list[dict]) -> list[dict]:
    """Keep highest-priority volume per (dedup_key, contrast, modality)."""
    best: dict[tuple[str, str, str], dict] = {}
    dropped = 0
    for c in cases:
        key = (c["dedup_key"], str(c.get("contrast") or ""), c["modality"])
        prev = best.get(key)
        if prev is None or int(c["priority"]) > int(prev["priority"]):
            if prev is not None:
                dropped += 1
            best[key] = c
        else:
            dropped += 1
    out = list(best.values())
    print(f"[dedup] kept={len(out)} dropped={dropped} (same subject/contrast)")
    return out


def stratified_volume_split(
    volume_rows: list[dict],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = SEED,
) -> dict[str, str]:
    """Volume-level train/val/test via subject ``dedup_key``.

    Stratifies unique subjects by (modality, dataset_family). All volumes that
    share a ``dedup_key`` receive the same split (BraTS multi-contrast, etc.).
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fracs must sum to 1, got {total}")

    rng = random.Random(seed)
    # subject -> (stratum, [volume_ids])
    subjects: dict[str, dict] = {}
    for r in volume_rows:
        dk = str(r["dedup_key"])
        stratum = (str(r["modality"]), str(r.get("dataset_family") or dataset_family(r["dataset"])))
        if dk not in subjects:
            subjects[dk] = {"stratum": stratum, "volume_ids": []}
        subjects[dk]["volume_ids"].append(r["volume_id"])

    by_stratum: dict[tuple[str, str], list[str]] = {}
    for dk, info in subjects.items():
        by_stratum.setdefault(info["stratum"], []).append(dk)

    lookup: dict[str, str] = {}
    for stratum, dks in sorted(by_stratum.items()):
        rng.shuffle(dks)
        n = len(dks)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        # Ensure leftovers go to train; clamp so train gets something when n small
        if n >= 3:
            n_test = min(max(1, n_test), n - 2)
            n_val = min(max(1, n_val), n - n_test - 1)
            n_train = n - n_test - n_val
        elif n == 2:
            n_train, n_val, n_test = 1, 0, 1
        else:
            n_train, n_val, n_test = n, 0, 0

        ordered = (
            [("test", n_test), ("val", n_val), ("train", n_train)]
        )
        i = 0
        counts = {"train": 0, "val": 0, "test": 0}
        for split_name, count in ordered:
            for _ in range(count):
                dk = dks[i]
                i += 1
                counts[split_name] += 1
                for vid in subjects[dk]["volume_ids"]:
                    lookup[vid] = split_name
        print(
            f"[split] {stratum[0]}/{stratum[1]}: subjects={n} "
            f"train={counts['train']} val={counts['val']} test={counts['test']}"
        )
    return lookup


def discover_ct_cases(*volumes_dirs: Path) -> list[tuple[str, Path, str]]:
    """Return (case_id, nifti_path, dataset_tag)."""
    cases: list[tuple[str, Path, str]] = []
    seen: set[str] = set()
    for volumes_dir in volumes_dirs:
        if volumes_dir is None or not volumes_dir.is_dir():
            continue
        tag = (
            "seg_cq500"
            if "seg_cq500" in str(volumes_dir).lower() or "Seg-CQ500" in str(volumes_dir)
            else "cq500"
        )
        for d in sorted(volumes_dir.iterdir()):
            if not d.is_dir():
                continue
            ct = d / "CT.nii"
            if not ct.exists():
                ct = d / "CT.nii.gz"
            if not ct.exists() or d.name in seen:
                continue
            seen.add(d.name)
            cases.append((d.name, ct, tag))
    return cases


def extract_msd_channel(src_4d: Path, dest_3d: Path, channel: int) -> None:
    """Write a single-channel 3D NIfTI from an MSD 4D volume (cached)."""
    import nibabel as nib

    if dest_3d.exists():
        return
    dest_3d.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.load(str(src_4d))
    data = np.asanyarray(nii.dataobj)
    if data.ndim != 4 or data.shape[-1] < channel + 1:
        raise ValueError(f"Expected 4D MSD volume at {src_4d}, got {data.shape}")
    vol = np.asarray(data[..., channel], dtype=np.float32)
    out = nib.Nifti1Image(vol, nii.affine, nii.header)
    nib.save(out, str(dest_3d))


def discover_msd_cases(
    msd_root: Path,
    channel_name: str = DEFAULT_MSD_CHANNEL,
    include_test_images: bool = True,
) -> list[tuple[str, Path, str]]:
    """Extract MSD channel to derived/ and return (case_id, path, dataset_tag)."""
    if not msd_root.is_dir():
        print(f"[msd] missing {msd_root}; skipping")
        return []
    if channel_name not in MSD_CHANNELS:
        raise ValueError(f"msd channel must be one of {list(MSD_CHANNELS)}")
    ch = MSD_CHANNELS[channel_name]
    derived = msd_root.parent / "derived" / channel_name
    cases: list[tuple[str, Path, str]] = []

    for split_dir, tag in (("imagesTr", "msd_brats"), ("imagesTs", "msd_brats_test")):
        if split_dir == "imagesTs" and not include_test_images:
            continue
        src_dir = msd_root / split_dir
        if not src_dir.is_dir():
            continue
        for src in sorted(src_dir.glob("BRATS_*.nii*")):
            if src.name.startswith("._"):
                continue
            stem = src.name.replace(".nii.gz", "").replace(".nii", "")
            vid = stem if tag == "msd_brats" else f"TS_{stem}"
            dest = derived / f"{vid}.nii.gz"
            try:
                extract_msd_channel(src, dest, ch)
            except Exception as e:
                print(f"[msd] skip {src.name}: {e}")
                continue
            cases.append((vid, dest, tag))
    print(f"[msd] {channel_name} channel: {len(cases)} volumes -> {derived}")
    return cases


def export_msd_labels(msd_root: Path, out_csv: Path) -> None:
    """Copy volume_labels.csv with volume_id = msd_braintumor:<name> for frozen eval."""
    src = Path(msd_root) / "volume_labels.csv"
    if not src.exists():
        src = Path(msd_root).parent / "volume_labels.csv"
    if not src.exists():
        print(f"[msd] no {src}; run data/download_msd_braintumor.py --skip-download")
        return
    df = pd.read_csv(src)
    df["volume_id"] = "msd_braintumor:" + df["name"].astype(str)
    q = df["enhancing_vox"].quantile([1 / 3, 2 / 3]).to_list()
    df["enhancing_tertile"] = pd.cut(
        df["enhancing_vox"],
        bins=[-0.5, q[0], q[1], float(df["enhancing_vox"].max()) + 1],
        labels=[0, 1, 2],
    ).astype(int)
    df["enhancing_high"] = (df["enhancing_vox"] >= q[1]).astype(int)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(
        f"[msd] wrote labels {out_csv} (n={len(df)}, "
        f"enhancing_high rate={df.enhancing_high.mean():.2f})"
    )


def discover_brats_cases(
    brats_root: Path,
    contrasts: tuple[str, ...] = BRATS_CONTRASTS,
) -> list[tuple[str, Path, str]]:
    """Discover BraTS NIfTIs under ``brats/extracted/{cohort}/...``.

    Each (case, contrast) is one volume. Dataset tags look like
    ``brats2023_gli_t1n``, ``brats2024_men_rt_t2w``, etc.
    """
    if not brats_root.is_dir():
        print(f"[brats] missing {brats_root}; skipping")
        return []

    cases: list[tuple[str, Path, str]] = []
    # Map top-level extract folder → tag prefix
    cohort_prefix = {
        "2023_GLI": "brats2023_gli",
        "2023_MEN": "brats2023_men",
        "2024_GLI": "brats2024_gli",
        "2024_GLI_add": "brats2024_gli",
        "2024_MEN_RT": "brats2024_men_rt",
    }

    for cohort_dir in sorted(brats_root.iterdir()):
        if not cohort_dir.is_dir():
            continue
        prefix = cohort_prefix.get(cohort_dir.name)
        if prefix is None:
            # tolerate nested challenge folder names
            prefix = f"brats_{cohort_dir.name.lower()}"
        for contrast in contrasts:
            pattern = f"*-{contrast}.nii.gz"
            for path in sorted(cohort_dir.rglob(pattern)):
                if path.name.startswith("._"):
                    continue
                # BraTS-GLI-00324-000-t1n.nii.gz → BraTS-GLI-00324-000_t1n
                stem = path.name.replace(".nii.gz", "")
                case_id = stem  # already includes contrast suffix
                tag = f"{prefix}_{contrast}"
                cases.append((case_id, path, tag))

    print(f"[brats] {len(cases)} volumes under {brats_root}")
    return cases


def discover_oasis_cases(oasis_root: Path) -> list[tuple[str, Path, str]]:
    """Discover OASIS-1/2 T1 volumes under ``oasis/extracted``.

    Prefers gain-field-corrected T88 processed MPRAGE when present; else RAW
    ``mpr-1`` Analyze/NIfTI. One volume per MR session.
    """
    if not oasis_root.is_dir():
        print(f"[oasis] missing {oasis_root}; skipping")
        return []

    cases: list[tuple[str, Path, str]] = []
    seen: set[str] = set()

    # OASIS-1 processed (best)
    for path in sorted(oasis_root.rglob("*_t88_gfc.img")):
        if path.name.startswith("._"):
            continue
        # OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.img → OAS1_0001_MR1
        parts = path.name.split("_")
        if len(parts) >= 3:
            sid = "_".join(parts[:3])  # OAS1_0001_MR1
        else:
            sid = path.stem
        if sid in seen:
            continue
        seen.add(sid)
        cases.append((sid, path, "oasis1"))

    # OASIS-1 RAW fallback for sessions without processed
    for path in sorted(oasis_root.rglob("*_mpr-1_anon.img")):
        if path.name.startswith("._"):
            continue
        parts = path.name.split("_")
        sid = "_".join(parts[:3]) if len(parts) >= 3 else path.stem
        if sid in seen:
            continue
        seen.add(sid)
        cases.append((sid, path, "oasis1"))

    # OASIS-2 RAW nifti (mpr-1 preferred)
    for path in sorted(oasis_root.rglob("mpr-1.nifti.img")):
        if path.name.startswith("._"):
            continue
        # .../OAS2_0001_MR1/RAW/mpr-1.nifti.img
        session = path.parent.parent.name  # OAS2_0001_MR1
        if not session.startswith("OAS2_"):
            session = path.parent.name
        if session in seen:
            continue
        seen.add(session)
        cases.append((session, path, "oasis2"))

    print(f"[oasis] {len(cases)} sessions under {oasis_root}")
    return cases


def resample_isotropic_ras(img, spacing: float = ISO_SPACING, nearest: bool = False):
    """Reorient to RAS and resample to isotropic spacing (mm)."""
    import SimpleITK as sitk

    img = sitk.DICOMOrient(img, "RAS")
    old_spacing = np.array(img.GetSpacing(), dtype=np.float64)
    old_size = np.array(img.GetSize(), dtype=np.int64)
    new_spacing = (float(spacing),) * 3
    new_size = [
        int(max(1, round(old_size[i] * old_spacing[i] / spacing))) for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize([int(s) for s in new_size])
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear)
    default = 0.0 if nearest else float(np.min(sitk.GetArrayViewFromImage(img)))
    resampler.SetDefaultPixelValue(default)
    return resampler.Execute(img)


def to_shared_space_sitk(
    img,
    *,
    iso_spacing: float = ISO_SPACING,
    out_xy: int = OUT_XY,
    out_z: int = OUT_Z,
    pad_value: float = 0.0,
    nearest: bool = False,
):
    """RAS + 1 mm iso + center crop/pad → sitk image of size (out_xy, out_xy, out_z).

    Center crop keeps the middle of the original volume (axial and in-plane).
    The returned image is the geometric reference for labels.
    """
    img = resample_isotropic_ras(img, spacing=iso_spacing, nearest=nearest)
    img = center_crop_pad_sitk(img, (out_xy, out_xy, out_z), pad_value=pad_value)
    size = tuple(int(s) for s in img.GetSize())
    if size != (out_xy, out_xy, out_z):
        raise RuntimeError(f"shared-space size {size} != {(out_xy, out_xy, out_z)}")
    return img


def resample_label_to_reference(seg_path: Path, reference) -> np.ndarray:
    """Nearest-neighbor resample of a label onto ``reference`` (shared-space image)."""
    import SimpleITK as sitk

    seg = load_sitk_image(seg_path)
    seg = sitk.DICOMOrient(seg, "RAS")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0.0)
    out = resampler.Execute(seg)
    arr = np.rint(sitk.GetArrayFromImage(out)).astype(np.uint8)
    ref_zyx = (
        int(reference.GetSize()[2]),
        int(reference.GetSize()[1]),
        int(reference.GetSize()[0]),
    )
    if arr.shape != ref_zyx:
        raise RuntimeError(
            f"mask shape {arr.shape} != reference {ref_zyx} for {seg_path}"
        )
    return arr


def center_crop_pad_sitk(img, out_size_xyz: tuple[int, int, int], pad_value: float):
    """Center crop/pad SimpleITK image to out_size (x, y, z)."""
    import SimpleITK as sitk

    size = list(img.GetSize())
    target = list(out_size_xyz)

    crop_lower, crop_upper = [], []
    need_crop = False
    for i in range(3):
        if size[i] > target[i]:
            lo = (size[i] - target[i]) // 2
            up = size[i] - target[i] - lo
            crop_lower.append(lo)
            crop_upper.append(up)
            need_crop = True
        else:
            crop_lower.append(0)
            crop_upper.append(0)
    if need_crop:
        img = sitk.Crop(img, crop_lower, crop_upper)
        size = list(img.GetSize())

    pad_lower, pad_upper = [], []
    need_pad = False
    for i in range(3):
        if size[i] < target[i]:
            lo = (target[i] - size[i]) // 2
            up = target[i] - size[i] - lo
            pad_lower.append(lo)
            pad_upper.append(up)
            need_pad = True
        else:
            pad_lower.append(0)
            pad_upper.append(0)
    if need_pad:
        img = sitk.ConstantPad(img, pad_lower, pad_upper, pad_value)
    return img


def normalize_ct_array(
    vol: np.ndarray,
    hu_min: float = -100.0,
    hu_max: float = 100.0,
) -> np.ndarray:
    """Soft-tissue / brain window → [0, 1]."""
    vol = np.clip(vol, hu_min, hu_max)
    return ((vol - hu_min) / (hu_max - hu_min)).astype(np.float32)


def normalize_mri_array(vol: np.ndarray) -> np.ndarray:
    """Foreground Z-score → clip ±3σ → [0, 1] (cross-volume comparable)."""
    vol = np.asarray(vol, dtype=np.float32)
    flat = vol.ravel()
    bg = float(np.percentile(flat, 2.0))
    span = float(flat.max()) - bg
    fg = vol > (bg + 1e-5 * (span if span > 0 else 1.0))
    if int(fg.sum()) < max(1000, flat.size // 100):
        fg = vol > float(np.percentile(flat, 10.0))
    if int(fg.sum()) == 0:
        return np.zeros_like(vol, dtype=np.float32)
    vals = vol[fg]
    mean = float(vals.mean())
    std = float(vals.std())
    if std < 1e-6:
        return np.zeros_like(vol, dtype=np.float32)
    clipped = np.clip((vol - mean) / std, -3.0, 3.0)
    return ((clipped + 3.0) / 6.0).astype(np.float32)


def slice_tissue_fraction(slc: np.ndarray, thresh: float = EMPTY_TISSUE_THRESH) -> float:
    return float((np.asarray(slc).squeeze() > thresh).mean())


def gaussian_slice_pdf(slice_idx: np.ndarray | float, mu: float, sigma: float) -> np.ndarray:
    """Unnormalized 1D Normal pdf at axial indices (relative scale via / pdf(mu))."""
    x = np.asarray(slice_idx, dtype=np.float64)
    if sigma <= 0:
        return np.ones_like(x, dtype=np.float64)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z)


def select_slices_gaussian(
    rows: list[dict],
    kept: list[np.ndarray],
    *,
    mu: float,
    sigma: float,
    rng: np.random.Generator,
    mode: str = "bernoulli",
    max_per_volume: int | None = None,
) -> tuple[list[dict], list[np.ndarray], list[dict]]:
    """Down-sample non-empty slices with a mid-skull Normal preference.

    ``bernoulli`` (default): keep slice i with p_i = pdf(i) / pdf(μ) ∈ (0, 1]
    so the mid-slice is always kept and extremes are kept less often but not
    never (σ=50 on a 128-deep volume still keeps ~40%+ near the ends).

    ``weighted``: sample without replacement up to ``max_per_volume`` using
    the same pdf as weights (requires max_per_volume).
    """
    if not rows:
        return [], [], []

    idxs = np.array([int(r["slice_idx"]) for r in rows], dtype=np.float64)
    pdf = gaussian_slice_pdf(idxs, mu, sigma)
    pdf_mu = float(gaussian_slice_pdf(mu, mu, sigma))
    p = np.clip(pdf / max(pdf_mu, 1e-12), 0.0, 1.0)

    dropped: list[dict] = []

    if mode == "weighted":
        n = len(rows)
        k = n if max_per_volume is None else min(n, int(max_per_volume))
        w = pdf / pdf.sum()
        chosen = set(rng.choice(n, size=k, replace=False, p=w).tolist())
        out_rows, out_kept = [], []
        for j, (r, slc) in enumerate(zip(rows, kept)):
            meta = dict(r)
            meta["gauss_p"] = float(p[j])
            if j in chosen:
                out_rows.append(meta)
                out_kept.append(slc)
            else:
                meta["drop_reason"] = "gaussian"
                dropped.append(meta)
        return out_rows, out_kept, dropped

    # bernoulli
    out_rows, out_kept = [], []
    u = rng.random(len(rows))
    for j, (r, slc) in enumerate(zip(rows, kept)):
        meta = dict(r)
        meta["gauss_p"] = float(p[j])
        if u[j] <= p[j]:
            out_rows.append(meta)
            out_kept.append(slc)
        else:
            meta["drop_reason"] = "gaussian"
            dropped.append(meta)
    # Guarantee ≥1 slice if any non-empty survived (edge case: all Bernoulli failed)
    if not out_rows and rows:
        j = int(np.argmax(p))
        meta = dict(rows[j])
        meta["gauss_p"] = float(p[j])
        out_rows = [meta]
        out_kept = [kept[j]]
        dropped = [d for d in dropped if d["slice_idx"] != meta["slice_idx"]]
    return out_rows, out_kept, dropped


def load_sitk_image(path: Path):
    """Load NIfTI as SimpleITK image; fall back via nibabel if direction is invalid."""
    import SimpleITK as sitk
    import nibabel as nib

    try:
        return sitk.ReadImage(str(path))
    except RuntimeError:
        nii = nib.as_closest_canonical(nib.load(str(path)))
        data = np.asanyarray(nii.dataobj, dtype=np.float32)
        data = np.squeeze(data)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D volume at {path}, got {data.shape}")
        arr = np.transpose(data, (2, 1, 0))
        img = sitk.GetImageFromArray(arr)
        zooms = [float(z) for z in nii.header.get_zooms()[:3]]
        img.SetSpacing((zooms[0], zooms[1], zooms[2]))
        img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        origin = nii.affine[:3, 3]
        img.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))
        return img


def volume_to_shared_space(
    path: Path,
    modality: str,
    iso_spacing: float = ISO_SPACING,
    out_xy: int = OUT_XY,
    out_z: int = OUT_Z,
    ct_hu_min: float = CT_HU_MIN,
    ct_hu_max: float = CT_HU_MAX,
    return_reference: bool = False,
):
    """Return float32 array (Z, 1, H, W) in shared RAS / 1 mm / 256×256×128 space.

    Always ``Z == out_z`` (default 128), taken from the **center** of the
    1 mm RAS volume. If ``return_reference``, also return the sitk image so
    labels can be resampled onto the exact same grid.
    """
    import SimpleITK as sitk

    img = load_sitk_image(path)
    pad_value = float(ct_hu_min) if modality == "ct" else 0.0
    img = to_shared_space_sitk(
        img,
        iso_spacing=iso_spacing,
        out_xy=out_xy,
        out_z=out_z,
        pad_value=pad_value,
        nearest=False,
    )

    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    if vol.shape != (out_z, out_xy, out_xy):
        raise RuntimeError(f"Unexpected shape {vol.shape} for {path} (want Z={out_z})")

    if modality == "ct":
        vol = normalize_ct_array(vol, hu_min=ct_hu_min, hu_max=ct_hu_max)
    else:
        vol = normalize_mri_array(vol)

    arr = vol[:, None, :, :].astype(np.float32)
    if return_reference:
        return arr, img
    return arr


def volume_to_shared_slices(*args, **kwargs) -> np.ndarray:
    """Alias — kept for older imports."""
    kwargs.pop("return_reference", None)
    return volume_to_shared_space(*args, **kwargs)


def nearest_label_to_shared_space(
    path: Path,
    iso_spacing: float = ISO_SPACING,
    out_xy: int = OUT_XY,
    out_z: int = OUT_Z,
    reference=None,
) -> np.ndarray:
    """Seg / mask on the same grid as the corresponding image.

    Prefer ``reference`` (the sitk image from ``volume_to_shared_space``) so
    voxel (z,y,x) in the mask is the same physical location as the image.
    """
    if reference is not None:
        return resample_label_to_reference(path, reference)
    img = load_sitk_image(path)
    img = to_shared_space_sitk(
        img,
        iso_spacing=iso_spacing,
        out_xy=out_xy,
        out_z=out_z,
        pad_value=0.0,
        nearest=True,
    )
    import SimpleITK as sitk

    vol = np.rint(sitk.GetArrayFromImage(img)).astype(np.uint8)
    if vol.shape != (out_z, out_xy, out_xy):
        raise RuntimeError(f"Unexpected label shape {vol.shape} for {path}")
    return vol


def find_brats_seg(nifti: Path) -> str:
    name = nifti.name
    for c in BRATS_CONTRASTS:
        token = f"-{c}.nii.gz"
        if token in name:
            cand = nifti.parent / name.replace(token, "-seg.nii.gz")
            if cand.exists():
                return str(cand)
    return ""


def find_cq500_seg(ct_path: Path) -> str:
    """Seg-CQ500 stores ``ICH_mask.nii.gz`` next to ``CT.nii``."""
    d = Path(ct_path).parent
    for name in ("ICH_mask.nii.gz", "ICH_mask.nii", "seg.nii.gz", "seg.nii", "mask.nii.gz"):
        cand = d / name
        if cand.exists():
            return str(cand)
    return ""


def find_msd_seg(nifti: Path) -> str:
    """MSD label next to the source tree (``labelsTr`` / ``labelsTs``)."""
    stem = nifti.name.replace(".nii.gz", "").replace(".nii", "")
    if stem.startswith("TS_"):
        stem = stem[3:]
    for parent in nifti.parents:
        for sub in ("labelsTr", "labelsTs"):
            for name in (f"{stem}.nii.gz", f"{stem}.nii"):
                cand = parent / sub / name
                if cand.exists():
                    return str(cand)
        if parent == parent.parent:
            break
    return ""


def find_case_masks(case: dict) -> list[tuple[str, str]]:
    """``(mask_name, path)`` labels to resample onto this volume's grid.

    BraTS writes the segmentation once, with the t1n contrast (shared geometry).
    Seg-CQ500 writes both the ICH mask and the brain mask.
    """
    dataset = str(case.get("dataset") or "")
    raw = case.get("path") or ""
    if not raw:
        return []
    path = Path(raw)
    fam = dataset_family(dataset)
    if fam.startswith("brats"):
        if case.get("contrast") and case.get("contrast") != "t1n":
            return []
        seg = find_brats_seg(path)
        return [("seg", seg)] if seg else []
    if dataset == "seg_cq500":
        found: list[tuple[str, str]] = []
        for name, candidates in (
            ("ICH", ("ICH_mask.nii.gz", "ICH_mask.nii")),
            ("brain", ("brain.nii.gz", "brain.nii")),
        ):
            for cand in candidates:
                p = path.parent / cand
                if p.exists():
                    found.append((name, str(p)))
                    break
        return found
    if fam.startswith("msd"):
        seg = find_msd_seg(path)
        return [("seg", seg)] if seg else []
    return []


def store_dataset(dataset: str) -> str:
    """One folder per dataset. BraTS contrasts share a cohort folder; Seg-CQ500 stays separate."""
    if str(dataset) == "seg_cq500":
        return "seg_cq500"
    return dataset_family(dataset)


def _safe_stem(name: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    stem = stem.strip("._") or "volume"
    return stem[:180]


def image_zip_path(out_dir: Path, dataset: str, case_id: str) -> Path:
    return Path(out_dir) / store_dataset(dataset) / f"{_safe_stem(case_id)}.zip"


def mask_zip_path(out_dir: Path, dataset: str, case_id: str, mask_name: str) -> Path:
    if mask_name == "seg" and dataset_family(dataset).startswith("brats"):
        subject, _ = parse_brats_subject_contrast(case_id)
        stem = _safe_stem(subject)
    else:
        stem = _safe_stem(case_id)
    return Path(out_dir) / store_dataset(dataset) / f"{stem}.{mask_name}.zip"


def _blosc(typesize: int):
    from zarr.codecs import BloscCodec

    return BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle", typesize=int(typesize))


def write_array_zip(path: Path, name: str, arr: np.ndarray, chunks: tuple[int, ...]) -> None:
    """Atomic ZipStore: Blosc chunks, zip method stored, so one chunk reads without unpacking."""
    from zarr.storage import ZipStore

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    store = ZipStore(tmp, mode="w", compression=0)
    try:
        root = zarr.open_group(store, mode="w")
        root.create_array(
            name,
            data=np.ascontiguousarray(arr),
            chunks=chunks,
            compressors=_blosc(arr.dtype.itemsize),
            overwrite=True,
        )
    except Exception:
        store.close()
        tmp.unlink(missing_ok=True)
        raise
    store.close()
    os.replace(tmp, path)


def read_array_zip(path: Path, name: str) -> np.ndarray:
    from zarr.storage import ZipStore

    store = ZipStore(path, mode="r")
    try:
        root = zarr.open_group(store, mode="r")
        return np.asarray(root[name][:])
    finally:
        store.close()


def zip_array_ok(path: Path, name: str, shape: tuple[int, ...], dtype) -> bool:
    if not Path(path).is_file() or Path(path).stat().st_size < 64:
        return False
    from zarr.storage import ZipStore

    try:
        store = ZipStore(path, mode="r")
        try:
            arr = zarr.open_group(store, mode="r")[name]
            return tuple(int(s) for s in arr.shape) == tuple(shape) and np.dtype(arr.dtype) == np.dtype(dtype)
        finally:
            store.close()
    except Exception:
        return False


def materialize_rsna_study(
    zip_path: Path,
    study_uid: str,
    member_names: list[str],
    work_dir: Path,
) -> Path:
    """Extract one RSNA study from zip → temporary NIfTI; caller must clean work_dir."""
    import SimpleITK as sitk

    study_dir = work_dir / "study" / study_uid.replace("/", "_")[:120]
    if study_dir.exists():
        shutil.rmtree(study_dir)
    study_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in member_names:
            dest = study_dir / Path(name).name
            with zf.open(name) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(study_dir))
    if not series_ids:
        files = sorted(str(p) for p in study_dir.glob("*.dcm"))
        if len(files) < 8:
            raise RuntimeError(f"too few DICOM slices ({len(files)})")
        reader.SetFileNames(files)
    else:
        best, best_n = series_ids[0], 0
        for sid in series_ids:
            files = reader.GetGDCMSeriesFileNames(str(study_dir), sid)
            if len(files) > best_n:
                best, best_n = sid, len(files)
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(study_dir), best))

    img = reader.Execute()
    tmp_nii = study_dir / "CT.nii.gz"
    sitk.WriteImage(img, str(tmp_nii))
    return tmp_nii


def build_rsna_study_index(
    zip_path: Path,
    cache_path: Path,
    max_studies: int | None = None,
) -> dict[str, list[str]]:
    """Map StudyInstanceUID → zip member paths (cached as JSON)."""
    import io

    import pydicom

    if cache_path.exists():
        print(f"[rsna] loading study index cache {cache_path}")
        with cache_path.open() as f:
            raw = json.load(f)
        if max_studies is not None:
            keys = sorted(raw.keys())[:max_studies]
            return {k: raw[k] for k in keys}
        return raw

    print(f"[rsna] indexing DICOM studies in {zip_path} (one-time; cached)")
    index: dict[str, list[str]] = {}
    keep: set[str] | None = None
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [
            info.filename
            for info in zf.infolist()
            if info.filename.lower().endswith(".dcm")
            and not info.filename.startswith("__MACOSX")
            and "/." not in info.filename
        ]
        print(f"[rsna] {len(names)} DICOM members")
        for name in tqdm(names, desc="rsna-index"):
            try:
                with zf.open(name) as src:
                    ds = pydicom.dcmread(io.BytesIO(src.read()), stop_before_pixels=True, force=True)
                uid = str(
                    getattr(ds, "StudyInstanceUID", None)
                    or getattr(ds, "PatientID", None)
                    or Path(name).stem
                )
            except Exception:
                uid = Path(name).stem
            if keep is not None and uid not in keep:
                continue
            index.setdefault(uid, []).append(name)
            if max_studies is not None and keep is None and len(index) >= max_studies:
                keep = set(index.keys())

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(index, f)
    print(f"[rsna] cached {len(index)} studies → {cache_path}")
    return index


def discover_rsna_cases(
    zip_path: Path,
    cache_path: Path,
    max_studies: int | None = None,
) -> list[dict]:
    if not zip_path.exists():
        print(f"[rsna] missing zip {zip_path}; skipping")
        return []
    index = build_rsna_study_index(zip_path, cache_path, max_studies=max_studies)
    members_dir = cache_path.parent / "rsna_members"
    members_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    for uid, members in sorted(index.items()):
        safe = "RSNA-" + uid.replace(".", "_")[:80]
        mem_path = members_dir / f"{safe}.json"
        if not mem_path.exists():
            with mem_path.open("w") as f:
                json.dump(members, f)
        c = enrich_case(
            safe,
            Path(""),
            "rsna_ich",
            "ct",
            source_kind="rsna_zip",
            source_zip=str(zip_path),
            source_members=(),
        )
        c["source_members_path"] = str(mem_path)
        cases.append(c)
    print(f"[rsna] {len(cases)} studies")
    return cases


def _rel_to_out(path: Path, out_dir: Path) -> str:
    return Path(path).resolve().relative_to(Path(out_dir).resolve()).as_posix()


def ssl_index_from_volume(
    vol: np.ndarray,
    *,
    case: dict,
    split: str,
    empty_tissue_thresh: float,
    empty_min_frac: float,
    slice_mu: float | None,
    slice_sigma: float,
    slice_sample: str,
    max_slices_per_vol: int | None,
    seed: int,
    out_z: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Bernoulli/empty-drop index rows from a shared-space volume (no pixel return)."""
    volume_id = case["volume_id"]
    base_meta = {
        "modality": case["modality"],
        "dataset": case["dataset"],
        "dataset_family": case.get("dataset_family", dataset_family(case["dataset"])),
        "volume_id": volume_id,
        "dedup_key": case["dedup_key"],
        "subject_id": case.get("subject_id", case["dedup_key"]),
        "contrast": case.get("contrast", ""),
        "source_path": case.get("path") or case.get("source_zip", ""),
        "source_kind": case.get("source_kind", "nifti"),
        "split": split,
        "vol_idx": int(case["vol_idx"]),
    }
    rows: list[dict] = []
    removed: list[dict] = []
    kept_flags: list[bool] = []
    for i in range(vol.shape[0]):
        frac = slice_tissue_fraction(vol[i], empty_tissue_thresh)
        meta = {**base_meta, "slice_idx": i, "z": i, "tissue_frac": frac}
        if frac < empty_min_frac:
            meta["drop_reason"] = "empty"
            removed.append(meta)
            continue
        rows.append(meta)
        kept_flags.append(True)

    gauss_dropped: list[dict] = []
    if slice_sample != "all" and rows:
        vol_seed = int(seed) ^ int(hashlib.md5(volume_id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(vol_seed)
        mu = float(slice_mu) if slice_mu is not None else 0.5 * (out_z - 1)
        rows, _, gauss_dropped = select_slices_gaussian(
            rows,
            kept_flags,
            mu=mu,
            sigma=float(slice_sigma),
            rng=rng,
            mode="weighted" if slice_sample == "weighted" else "bernoulli",
            max_per_volume=max_slices_per_vol,
        )

    tissue = [
        slice_tissue_fraction(vol[i], empty_tissue_thresh) for i in range(vol.shape[0])
    ]
    mid = int(vol.shape[0] // 2)
    volume_meta = {
        **base_meta,
        "n_slices_raw": int(vol.shape[0]),
        "n_slices_kept": len(rows),
        "n_slices_empty_dropped": len(removed),
        "n_slices_gauss_dropped": len(gauss_dropped),
        "tissue_frac_mid": float(tissue[mid]),
        "tissue_frac_lo": float(np.mean(tissue[: max(mid // 4, 1)])),
        "tissue_frac_hi": float(np.mean(tissue[-max(mid // 4, 1) :])),
        "status": "ok",
        "has_mask": False,
        "mask_array": "",
    }
    return rows, removed, gauss_dropped, volume_meta


def _process_one_case(args: tuple) -> dict:
    """Resample one volume into its dataset folder and return index rows (no pixels)."""
    (
        case,
        split,
        iso_spacing,
        out_xy,
        out_z,
        ct_hu_min,
        ct_hu_max,
        empty_tissue_thresh,
        empty_min_frac,
        slice_mu,
        slice_sigma,
        slice_sample,
        max_slices_per_vol,
        seed,
        work_root,
        out_dir,
    ) = args
    # One thread per worker. SimpleITK reads this when it is first imported.
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    volume_id = case["volume_id"]
    out_dir = Path(out_dir)
    image_path = image_zip_path(out_dir, case["dataset"], case["case_id"])
    image_shape = (out_z, 1, out_xy, out_xy)
    mask_shape = (out_z, out_xy, out_xy)
    tmp_study_dir: Path | None = None
    try:
        masks_spec = list(case.get("masks") or [])
        mask_jobs: list[tuple[str, str, Path, bool]] = []
        for name, seg_path in masks_spec:
            mpath = mask_zip_path(out_dir, case["dataset"], case["case_id"], name)
            mask_jobs.append(
                (name, seg_path, mpath, zip_array_ok(mpath, "mask", mask_shape, np.uint8))
            )
        image_ready = zip_array_ok(image_path, "image", image_shape, np.float16)
        need_geometry = (not image_ready) or any(not ok for *_, ok in mask_jobs)

        ref_img = None
        wrote_image = False
        if need_geometry:
            if case.get("source_kind") == "rsna_zip":
                work = Path(work_root) / f"w_{os.getpid()}"
                work.mkdir(parents=True, exist_ok=True)
                members = list(case.get("source_members") or [])
                mem_path = case.get("source_members_path")
                if not members and mem_path:
                    with open(mem_path) as f:
                        members = json.load(f)
                path = materialize_rsna_study(
                    Path(case["source_zip"]),
                    case["case_id"],
                    members,
                    work,
                )
                tmp_study_dir = path.parent
            else:
                path = Path(case["path"])
            vol, ref_img = volume_to_shared_space(
                path,
                case["modality"],
                iso_spacing=iso_spacing,
                out_xy=out_xy,
                out_z=out_z,
                ct_hu_min=ct_hu_min,
                ct_hu_max=ct_hu_max,
                return_reference=True,
            )
            if vol.shape != image_shape:
                raise RuntimeError(
                    f"{volume_id}: expected {image_shape}, got {vol.shape}"
                )
            if not image_ready:
                write_array_zip(
                    image_path,
                    "image",
                    np.asarray(vol, dtype=np.float16),
                    (min(ZARR_CHUNK_Z, out_z), 1, out_xy, out_xy),
                )
                wrote_image = True
        else:
            vol = read_array_zip(image_path, "image").astype(np.float32, copy=False)
            if vol.ndim == 3:
                vol = vol[:, None, :, :]

        mask_paths: dict[str, str] = {}
        for name, seg_path, mpath, ready in mask_jobs:
            if not ready:
                if ref_img is None:
                    raise RuntimeError(f"{volume_id}: missing geometry for mask {name}")
                mask = nearest_label_to_shared_space(
                    Path(seg_path),
                    iso_spacing=iso_spacing,
                    out_xy=out_xy,
                    out_z=out_z,
                    reference=ref_img,
                )
                if mask.shape != mask_shape:
                    raise RuntimeError(
                        f"{volume_id} mask {name}: {mask.shape} != {mask_shape}"
                    )
                write_array_zip(
                    mpath,
                    "mask",
                    np.asarray(mask, dtype=np.uint8),
                    (min(ZARR_CHUNK_Z, out_z), out_xy, out_xy),
                )
            mask_paths[name] = _rel_to_out(mpath, out_dir)
    except Exception as e:
        if tmp_study_dir is not None:
            shutil.rmtree(tmp_study_dir, ignore_errors=True)
        return {
            "error": f"{volume_id}: {e}",
            "rows": [],
            "removed": [],
            "gauss_dropped": [],
            "volume_meta": None,
            "vol_idx": case.get("vol_idx"),
        }

    if tmp_study_dir is not None:
        shutil.rmtree(tmp_study_dir, ignore_errors=True)

    rows, removed, gauss_dropped, volume_meta = ssl_index_from_volume(
        vol,
        case=case,
        split=split,
        empty_tissue_thresh=empty_tissue_thresh,
        empty_min_frac=empty_min_frac,
        slice_mu=slice_mu,
        slice_sigma=slice_sigma,
        slice_sample=slice_sample,
        max_slices_per_vol=max_slices_per_vol,
        seed=seed,
        out_z=out_z,
    )
    rel_image = _rel_to_out(image_path, out_dir)
    for row in rows:
        row["image_path"] = rel_image
    for row in removed + gauss_dropped:
        row["image_path"] = rel_image
    volume_meta["image_path"] = rel_image
    volume_meta["has_mask"] = bool(mask_paths)
    volume_meta["mask_seg"] = mask_paths.get("seg", "")
    volume_meta["mask_ich"] = mask_paths.get("ICH", "")
    volume_meta["mask_brain"] = mask_paths.get("brain", "")
    volume_meta["resumed"] = not wrote_image
    del vol
    return {
        "rows": rows,
        "removed": removed,
        "gauss_dropped": gauss_dropped,
        "volume_meta": volume_meta,
        "vol_idx": int(case["vol_idx"]),
        "wrote": wrote_image,
    }


def process_cases_parallel(
    cases: list[dict],
    split_lookup: dict[str, str],
    workers: int,
    iso_spacing: float,
    out_xy: int,
    out_z: int,
    *,
    out_dir: Path,
    ct_hu_min: float = CT_HU_MIN,
    ct_hu_max: float = CT_HU_MAX,
    empty_tissue_thresh: float = EMPTY_TISSUE_THRESH,
    empty_min_frac: float = EMPTY_MIN_FRAC,
    slice_mu: float | None = None,
    slice_sigma: float = DEFAULT_SLICE_SIGMA,
    slice_sample: str = "bernoulli",
    max_slices_per_vol: int | None = None,
    seed: int = SEED,
    work_root: Path | None = None,
    label: str | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Process volumes in parallel. Each worker writes its own zip files."""
    work_root = Path(work_root or (SCRATCH / "work" / "preprocess"))
    work_root.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir)
    rows: list[dict] = []
    removed: list[dict] = []
    gauss_dropped: list[dict] = []
    volume_rows: list[dict] = []
    n_ok = 0
    n_resume = 0
    label = label or (cases[0]["modality"] if cases else "vol")

    jobs = []
    for case in cases:
        split = split_lookup.get(case["volume_id"])
        if split is None:
            continue
        job_case = dict(case)
        job_case["source_members"] = []
        jobs.append(
            (
                job_case,
                split,
                iso_spacing,
                out_xy,
                out_z,
                ct_hu_min,
                ct_hu_max,
                empty_tissue_thresh,
                empty_min_frac,
                slice_mu,
                slice_sigma,
                slice_sample,
                max_slices_per_vol,
                seed,
                str(work_root),
                str(out_dir),
            )
        )

    def _consume(result: dict) -> None:
        nonlocal n_ok, n_resume
        if result.get("error"):
            print(f"[warn] skip {result['error']}")
            volume_rows.append(
                {
                    "volume_id": str(result["error"]).split(":", 1)[0],
                    "status": "error",
                    "error": result["error"],
                }
            )
            return
        if result.get("wrote"):
            n_ok += 1
        else:
            n_resume += 1
        if result.get("volume_meta"):
            volume_rows.append(result["volume_meta"])
        removed.extend(result.get("removed") or [])
        gauss_dropped.extend(result.get("gauss_dropped") or [])
        rows.extend(result.get("rows") or [])

    if not jobs:
        print(f"[{label}] wrote 0 new / resumed 0")
        return rows, removed, gauss_dropped, volume_rows

    if workers <= 1:
        for job in tqdm(jobs, desc=f"prep-{label}"):
            _consume(_process_one_case(job))
    else:
        ctx = mp.get_context("spawn")
        max_flight = max(workers * 2, workers)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            it = iter(jobs)
            in_flight: dict = {}

            def _submit() -> bool:
                job = next(it, None)
                if job is None:
                    return False
                fut = ex.submit(_process_one_case, job)
                in_flight[fut] = True
                return True

            for _ in range(min(max_flight, len(jobs))):
                if not _submit():
                    break
            pbar = tqdm(total=len(jobs), desc=f"prep-{label}")
            while in_flight:
                done, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    in_flight.pop(fut, None)
                    _consume(fut.result())
                    pbar.update(1)
                    _submit()
            pbar.close()

    print(f"[{label}] wrote {n_ok} new + {n_resume} resumed / {len(cases)} volumes")
    return rows, removed, gauss_dropped, volume_rows

def summarize_empty_removals(removed: list[dict], kept: list[dict]) -> pd.DataFrame:
    rem = pd.DataFrame(removed) if removed else pd.DataFrame(
        columns=["dataset", "split", "modality", "volume_id", "slice_idx"]
    )
    kep = pd.DataFrame(kept) if kept else pd.DataFrame(
        columns=["dataset", "split", "modality", "volume_id", "slice_idx"]
    )
    keys = ["dataset", "split"]
    if len(rem) == 0 and len(kep) == 0:
        return pd.DataFrame(columns=["dataset", "split", "kept", "removed", "produced", "removed_pct"])
    kep_n = kep.groupby(keys, dropna=False).size().rename("kept") if len(kep) else None
    rem_n = rem.groupby(keys, dropna=False).size().rename("removed") if len(rem) else None
    if kep_n is None:
        out = rem_n.to_frame()
        out["kept"] = 0
    elif rem_n is None:
        out = kep_n.to_frame()
        out["removed"] = 0
    else:
        out = pd.concat([kep_n, rem_n], axis=1)
    out = out.fillna(0).astype(int)
    out["produced"] = out["kept"] + out["removed"]
    out["removed_pct"] = np.where(
        out["produced"] > 0, 100.0 * out["removed"] / out["produced"], 0.0
    )
    return out.reset_index().sort_values(keys)


def _dir_nbytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cq500-dir", type=Path, default=DEFAULT_CQ500)
    p.add_argument("--cq500-extra-dir", type=Path, default=DEFAULT_CQ500_EXTRA)
    p.add_argument("--msd-root", type=Path, default=DEFAULT_MSD_ROOT)
    p.add_argument(
        "--msd-channel",
        type=str,
        default=DEFAULT_MSD_CHANNEL,
        choices=sorted(MSD_CHANNELS),
    )
    p.add_argument("--no-msd", action="store_true")
    p.add_argument(
        "--keep-msd",
        action="store_true",
        help="Keep MSD even when BraTS GLI is present (default: skip MSD to avoid glioma dup)",
    )
    p.add_argument("--brats-root", type=Path, default=DEFAULT_BRATS_ROOT)
    p.add_argument("--oasis-root", type=Path, default=DEFAULT_OASIS_ROOT)
    p.add_argument("--no-brats", action="store_true")
    p.add_argument("--no-oasis", action="store_true")
    p.add_argument(
        "--brats-contrasts",
        type=str,
        default=",".join(BRATS_CONTRASTS),
        help="Comma list of BraTS contrasts to include (default: t1n,t1c,t2w,t2f)",
    )
    p.add_argument("--rsna-zip", type=Path, default=DEFAULT_RSNA_ZIP)
    p.add_argument(
        "--rsna-index-cache",
        type=Path,
        default=SCRATCH / "work" / "rsna_ingest" / "study_index.json",
    )
    p.add_argument("--no-rsna", action="store_true")
    p.add_argument("--max-rsna", type=int, default=None, help="Cap RSNA studies")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--iso-spacing", type=float, default=ISO_SPACING)
    p.add_argument("--out-xy", type=int, default=OUT_XY)
    p.add_argument("--out-z", type=int, default=OUT_Z)
    p.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)
    p.add_argument("--test-frac", type=float, default=TEST_FRAC)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--ct-hu-min", type=float, default=CT_HU_MIN)
    p.add_argument("--ct-hu-max", type=float, default=CT_HU_MAX)
    p.add_argument("--empty-tissue-thresh", type=float, default=EMPTY_TISSUE_THRESH)
    p.add_argument("--empty-min-frac", type=float, default=EMPTY_MIN_FRAC)
    p.add_argument(
        "--slice-sample",
        type=str,
        default="bernoulli",
        choices=["all", "bernoulli", "weighted"],
    )
    p.add_argument("--slice-mu", type=float, default=None)
    p.add_argument("--slice-sigma", type=float, default=DEFAULT_SLICE_SIGMA)
    p.add_argument("--max-slices-per-vol", type=int, default=None)
    p.add_argument("--max-ct", type=int, default=None)
    p.add_argument("--max-mri", type=int, default=None)
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 4) // 2)),
    )
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip volumes whose zip already exists (also the default unless --overwrite)",
    )
    p.add_argument("--no-shuffle", action="store_true")
    return p.parse_args()


def main() -> None:
    from collections import Counter

    args = parse_args()

    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        print(f"[resume] keeping existing per-volume zips under {args.out_dir}")

    repo_link = ROOT / "data" / "processed" / "slices_zarr"
    if not repo_link.exists():
        try:
            repo_link.symlink_to(args.out_dir)
        except OSError:
            pass

    cases: list[dict] = []

    for vid, path, tag in discover_ct_cases(args.cq500_dir, args.cq500_extra_dir):
        cases.append(enrich_case(vid, path, tag, "ct"))

    if not args.no_rsna:
        cases.extend(
            discover_rsna_cases(
                args.rsna_zip,
                args.rsna_index_cache,
                max_studies=args.max_rsna,
            )
        )

    brats_raw: list[tuple[str, Path, str]] = []
    if not args.no_brats:
        contrasts = tuple(c.strip() for c in args.brats_contrasts.split(",") if c.strip())
        brats_raw = discover_brats_cases(args.brats_root, contrasts=contrasts)
        for vid, path, tag in brats_raw:
            cases.append(enrich_case(vid, path, tag, "mri"))

    if not args.no_oasis:
        for vid, path, tag in discover_oasis_cases(args.oasis_root):
            cases.append(enrich_case(vid, path, tag, "mri"))

    has_brats_gli = any(
        "brats2023_gli" in t or "brats2024_gli" in t for *_, t in brats_raw
    )
    skip_msd = args.no_msd or (has_brats_gli and not args.keep_msd)
    if skip_msd:
        reason = "--no-msd" if args.no_msd else "BraTS GLI present (use --keep-msd to force)"
        print(f"[msd] skipped ({reason})")
    else:
        for vid, path, tag in discover_msd_cases(
            args.msd_root, channel_name=args.msd_channel
        ):
            cases.append(enrich_case(f"MSD-{vid}", path, tag, "mri"))

    if args.max_ct is not None:
        ct = [c for c in cases if c["modality"] == "ct"][: args.max_ct]
        cases = [c for c in cases if c["modality"] != "ct"] + ct
    if args.max_mri is not None:
        mri = [c for c in cases if c["modality"] == "mri"][: args.max_mri]
        cases = [c for c in cases if c["modality"] != "mri"] + mri

    if not cases:
        raise SystemExit("No volumes discovered")

    n_before = len(cases)
    cases = deduplicate_volumes(cases)
    print(f"[catalog] {n_before} → {len(cases)} volumes after dedup")

    split_lookup = stratified_volume_split(
        cases,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    for c in cases:
        c["split"] = split_lookup[c["volume_id"]]
        c["masks"] = find_case_masks(c)
        c["seg_path"] = c["masks"][0][1] if c["masks"] else ""

    # Stable vol_idx 0..V-1 (write order ≠ batch order)
    cases.sort(key=lambda c: (c["modality"], c["dataset"], c["volume_id"]))
    for i, c in enumerate(cases):
        c["vol_idx"] = i

    print(
        f"[space] RAS + isotropic {args.iso_spacing}mm → "
        f"({args.out_xy},{args.out_xy},{args.out_z}); workers={args.workers}"
    )
    print(
        f"[split] train/val/test = "
        f"{args.train_frac:.0%}/{args.val_frac:.0%}/{args.test_frac:.0%}"
    )
    print(f"[paths] out={args.out_dir}")
    print(
        f"[zarr] one ZipStore per volume under <dataset>/*.zip  "
        f"n={len(cases)} dtype=float16 chunk_z={ZARR_CHUNK_Z}"
    )
    print(f"[paths] brats={args.brats_root} oasis={args.oasis_root}")
    print(f"[paths] rsna_zip={args.rsna_zip} (exists={args.rsna_zip.exists()})")
    print(f"[ct-window] HU [{args.ct_hu_min}, {args.ct_hu_max}] → [0,1]")
    print(
        f"[mri-norm] Z-score foreground ±3σ → [0,1]; "
        f"empty drop if tissue_frac<{args.empty_min_frac}"
    )
    print(f"[by dataset] {dict(sorted(Counter(c['dataset'] for c in cases).items()))}")
    mu = args.slice_mu if args.slice_mu is not None else 0.5 * (args.out_z - 1)
    print(
        f"[slice-sample] mode={args.slice_sample}  Normal(μ={mu:.1f}, σ={args.slice_sigma}) "
        f"(index table only; full 128-z stored once)"
    )

    labels_dir = args.out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    vol_df = pd.DataFrame(
        [
            {
                "vol_idx": c["vol_idx"],
                "volume_id": c["volume_id"],
                "dedup_key": c["dedup_key"],
                "subject_id": c["subject_id"],
                "modality": c["modality"],
                "dataset": c["dataset"],
                "dataset_family": c["dataset_family"],
                "contrast": c.get("contrast", ""),
                "split": c["split"],
                "source_kind": c.get("source_kind", "nifti"),
                "source_path": c.get("path") or c.get("source_zip", ""),
                "image_path": _rel_to_out(
                    image_zip_path(args.out_dir, c["dataset"], c["case_id"]),
                    args.out_dir,
                ),
                "seg_path": c.get("seg_path", ""),
                "priority": c.get("priority", 0),
            }
            for c in cases
        ]
    )
    vol_df.to_csv(args.out_dir / "volumes.csv", index=False)
    subj = (
        vol_df.groupby("dedup_key", as_index=False)
        .agg(
            split=("split", "first"),
            modality=("modality", "first"),
            dataset_family=("dataset_family", "first"),
            n_volumes=("volume_id", "nunique"),
        )
    )
    subj.to_csv(args.out_dir / "subjects.csv", index=False)
    print(f"[tables] volumes={len(vol_df)} subjects={len(subj)}")

    raw_giB = len(cases) * args.out_z * 1 * args.out_xy * args.out_xy * 2 / 1e9
    print(f"[zarr] raw_float16≈{raw_giB:.1f} GB if stored uncompressed")

    rows: list[dict] = []
    removed: list[dict] = []
    gauss_dropped_all: list[dict] = []
    volume_meta_all: list[dict] = []

    # RSNA is one shared zip and ~25k studies. Run the other datasets first so a
    # time limit still leaves CQ500, BraTS, and OASIS (and their masks) on disk.
    waves = (
        ("core", [c for c in cases if c.get("source_kind") != "rsna_zip"]),
        ("rsna", [c for c in cases if c.get("source_kind") == "rsna_zip"]),
    )
    for wave_name, subset in waves:
        if not subset:
            continue
        print(f"\n[{wave_name}] Processing {len(subset)} volumes...")
        r, rem, gdrop, vmeta = process_cases_parallel(
            subset,
            split_lookup,
            args.workers,
            args.iso_spacing,
            args.out_xy,
            args.out_z,
            out_dir=args.out_dir,
            ct_hu_min=args.ct_hu_min,
            ct_hu_max=args.ct_hu_max,
            empty_tissue_thresh=args.empty_tissue_thresh,
            empty_min_frac=args.empty_min_frac,
            slice_mu=mu,
            slice_sigma=args.slice_sigma,
            slice_sample=args.slice_sample,
            max_slices_per_vol=args.max_slices_per_vol,
            seed=args.seed,
            work_root=SCRATCH / "work" / "preprocess",
            label=wave_name,
        )
        rows.extend(r)
        removed.extend(rem)
        gauss_dropped_all.extend(gdrop)
        volume_meta_all.extend(vmeta)

    if volume_meta_all:
        meta_df = pd.DataFrame(volume_meta_all)
        keep_cols = [
            c
            for c in (
                "volume_id",
                "n_slices_raw",
                "n_slices_kept",
                "n_slices_empty_dropped",
                "n_slices_gauss_dropped",
                "status",
                "error",
                "has_mask",
                "image_path",
                "mask_seg",
                "mask_ich",
                "mask_brain",
                "tissue_frac_mid",
                "tissue_frac_lo",
                "tissue_frac_hi",
            )
            if c in meta_df.columns
        ]
        vol_df = vol_df.drop(
            columns=[c for c in keep_cols if c in vol_df.columns and c != "volume_id"],
            errors="ignore",
        ).merge(meta_df[keep_cols], on="volume_id", how="left")
        vol_df.to_csv(args.out_dir / "volumes.csv", index=False)

        ok = vol_df[vol_df.get("status", "ok") == "ok"] if "status" in vol_df.columns else vol_df
        if "n_slices_raw" in ok.columns and len(ok):
            z_ok = int((ok["n_slices_raw"] == args.out_z).sum())
            print(
                f"[geom] stored z=={args.out_z} for {z_ok}/{len(ok)} volumes "
                f"(center crop/pad of 1 mm RAS)"
            )
            if z_ok != len(ok):
                raise SystemExit("Some volumes did not store exactly 128 slices")
        if {"tissue_frac_mid", "tissue_frac_lo", "tissue_frac_hi"}.issubset(ok.columns) and len(ok):
            mid = float(ok["tissue_frac_mid"].mean())
            lo = float(ok["tissue_frac_lo"].mean())
            hi = float(ok["tissue_frac_hi"].mean())
            print(
                f"[geom] mean tissue_frac  mid={mid:.3f}  low-z={lo:.3f}  high-z={hi:.3f} "
                f"(mid should be highest — anatomy concentrated in the stack center)"
            )
        n_mask = int(ok["has_mask"].fillna(False).sum()) if "has_mask" in ok.columns else 0
        print(f"[labels] aligned masks written for {n_mask} volumes")
        mask_tbl = ok.loc[ok.get("has_mask", False) == True] if "has_mask" in ok.columns else ok.iloc[0:0]
        if len(mask_tbl):
            cols = [
                c
                for c in (
                    "vol_idx",
                    "volume_id",
                    "dedup_key",
                    "dataset",
                    "image_path",
                    "mask_seg",
                    "mask_ich",
                    "mask_brain",
                    "seg_path",
                    "split",
                )
                if c in mask_tbl.columns
            ]
            mask_tbl[cols].to_csv(labels_dir / "masks.csv", index=False)

    df = pd.DataFrame(rows)
    slice_cols = [
        "image_path",
        "vol_idx",
        "z",
        "volume_id",
        "dedup_key",
        "subject_id",
        "modality",
        "dataset",
        "dataset_family",
        "contrast",
        "split",
        "slice_idx",
        "tissue_frac",
        "source_path",
        "source_kind",
    ]
    for c in slice_cols:
        if c not in df.columns:
            df[c] = "" if c not in ("vol_idx", "z") else 0
    if len(df):
        df = df[slice_cols]
    df.to_csv(args.out_dir / "slices.csv", index=False)
    df.to_csv(args.out_dir / "manifest.csv", index=False)  # alias for older scripts
    try:
        df.to_parquet(args.out_dir / "slices.parquet", index=False)
    except Exception as e:
        print(f"[warn] parquet skipped: {e}")

    empty_summary = summarize_empty_removals(removed, rows)
    empty_summary.to_csv(args.out_dir / "empty_slice_removals.csv", index=False)

    print("\n[empty-slices] removals by dataset × split:")
    print(empty_summary.to_string(index=False))
    print("\n[ssl-index] by split × modality:")
    if len(df):
        print(df.groupby(["split", "modality"]).size())
        print(df.groupby(["dataset"]).agg(vols=("volume_id", "nunique"), slices=("z", "count")))

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        sa = set(df.loc[df.split == a, "dedup_key"]) if len(df) else set()
        sb = set(df.loc[df.split == b, "dedup_key"]) if len(df) else set()
        assert sa.isdisjoint(sb), f"dedup_key leak {a}/{b}"

    store_gb = _dir_nbytes(args.out_dir) / 1e9
    ratio = (raw_giB / store_gb) if store_gb > 0 else 0.0
    print(
        f"\n[zarr] images on disk {store_gb:.2f} GB  "
        f"raw_float16 {raw_giB:.2f} GB  compression≈{ratio:.1f}×"
    )
    if len(df):
        counts = df.groupby("split")["volume_id"].nunique().to_dict()
        print(
            f"[ok] volumes train={counts.get('train', 0)} "
            f"val={counts.get('val', 0)} test={counts.get('test', 0)} "
            f"ssl_slices={len(df)}"
        )

    if (not skip_msd) and args.msd_root.is_dir():
        export_msd_labels(args.msd_root, labels_dir / "msd_volume_labels.csv")

    if not args.no_rsna and args.rsna_zip.exists():
        try:
            with zipfile.ZipFile(args.rsna_zip, "r") as zf:
                cands = [
                    n
                    for n in zf.namelist()
                    if n.endswith("stage_2_train.csv") or n.endswith("stage_1_train.csv")
                ]
                if cands:
                    name = sorted(cands)[-1]
                    dest = labels_dir / "rsna_stage_2_train.csv"
                    with zf.open(name) as src, dest.open("wb") as out:
                        shutil.copyfileobj(src, out)
                    print(f"[labels] {dest}")
        except Exception as e:
            print(f"[rsna] label extract skipped: {e}")

    cq_meta = DEFAULT_RAW / "cq500_meta"
    if cq_meta.is_dir():
        for csv in cq_meta.glob("*.csv"):
            shutil.copy2(csv, labels_dir / csv.name)
            print(f"[labels] copied {csv.name}")

    n_oasis_csv = 0
    for oasis_csv in (args.oasis_root.parent).rglob("*clinical*.csv"):
        dest = labels_dir / oasis_csv.name
        if not dest.exists():
            shutil.copy2(oasis_csv, dest)
            print(f"[labels] {dest}")
            n_oasis_csv += 1
        if n_oasis_csv >= 8:
            break


if __name__ == "__main__":
    main()
