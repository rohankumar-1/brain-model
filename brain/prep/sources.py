"""Dataset adapters: each yields ``Case`` records; no pixels are read here."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import DATASETS, SCRATCH  # noqa: F401  (re-exported for prep.index)
from brain.prep.fomo import CONTRAST_SUFFIXES

BRATS_CONTRASTS = ("t1n", "t1c", "t2w", "t2f")
BRATS_COHORTS = {"2023_GLI": "brats2023_gli", "2023_MEN": "brats2023_men",
                 "2024_GLI": "brats2024_gli", "2024_GLI_add": "brats2024_gli",  # MEN_RT was never part of the store
                 "2023_PED": "brats2023_ped", "2024_MEN_RT": "brats2024_men_rt"}
COHORT_RANK = {"brats2024_gli": 50, "brats2023_gli": 40, "brats2024_men_rt": 30,
               "brats2023_men": 20, "brats2023_ped": 10}


# skull-stripped releases (dataset-name prefixes): BraTS, MSD, ATLAS R2/3, BrainMetShare, FOMO Yale mets, ISLES'22
STRIPPED_PREFIXES = ("brats", "msd_braintumor", "atlas", "brainmetshare", "fomo_yale_mets", "isles22")


@dataclass(frozen=True)
class Case:
    dataset: str
    case_id: str
    modality: str                                   # "ct" | "mri"
    path: Path | None = None                        # NIfTI / Analyze file
    masks: tuple[tuple[str, Path], ...] = ()        # (name, path)
    zip_path: str = ""                              # RSNA: source zip and member byte ranges
    members: tuple = ()
    ref: Path | None = None                         # co-registered sequence that decides this subject's grid

    @property
    def stripped(self) -> bool:
        """Known skull-stripped release. Extend with BRAIN_STRIPPED=prefix,prefix. Checked on the v6 smoke run: these
        (and only these) have brain/head volume ratio ~1 (``brain_head_ratio``; skull-on scans sit at 0.45-0.7)."""
        extra = tuple(x for x in os.environ.get("BRAIN_STRIPPED", "").split(",") if x)
        return self.dataset.startswith(STRIPPED_PREFIXES + extra)

    @property
    def folder(self) -> str:
        """Store folder: BraTS contrasts share their cohort folder."""
        for c in CONTRAST_SUFFIXES:
            if self.dataset.endswith(f"_{c}"):
                return self.dataset[: -len(c) - 1]
        return self.dataset

    @property
    def mask_id(self) -> str:
        """BraTS writes one segmentation per subject, next to its t1n volume."""
        return self.case_id.removesuffix("-t1n") if self.dataset.startswith("brats") else self.case_id


def _first(d: Path, *names: str) -> Path | None:
    return next((d / n for n in names if (d / n).exists()), None)


def cq500(*dirs: Path) -> list[Case]:
    """Seg-CQ500 (with ICH and brain masks) first, then extra CQ500 volumes not already seen."""
    cases, seen = [], set()
    for root in dirs:
        tag = "seg_cq500" if "seg_cq500" in str(root) else "cq500"
        for d in sorted(p for p in Path(root).iterdir() if p.is_dir() and p.name not in seen):
            ct = _first(d, "CT.nii", "CT.nii.gz")
            if ct is None:
                continue
            seen.add(d.name)
            masks = tuple((n, m) for n, names in (("ICH", ("ICH_mask.nii.gz", "ICH_mask.nii")),
                                                  ("brain", ("brain.nii.gz", "brain.nii")))
                          if tag == "seg_cq500" and (m := _first(d, *names)))
            cases.append(Case(tag, d.name, "ct", ct, masks))
    return cases


def cq500_dicom(root: Path = DATASETS / "cq500_dicom") -> list[Case]:
    """CQ500 from the original DICOMs (``brain.datasets.download_cq500``: one plain soft-tissue series per study zip),
    read by member offsets like RSNA. Replaces the third-party pre-converted NIfTIs as dataset ``cq500`` (same case
    ids, so the same volume ids); those were mirrored for some studies and mixed kernels / contrast series. Prep these
    with ``BRAIN_DICOM_READER=dcm2niix`` (CQ500 has gantry tilt)."""
    import zipfile

    cases = []
    for z in sorted(Path(root).glob("CQ500-CT-*.zip"), key=lambda p: int(p.stem.rsplit("-", 1)[1])):
        with zipfile.ZipFile(z) as zf:
            members = tuple((i.filename, i.header_offset, i.compress_size, i.compress_type) for i in zf.infolist()
                            if not i.filename.endswith("/"))
        if members:
            cases.append(Case("cq500", z.stem, "ct", zip_path=str(z), members=members))
    return cases


def oasis(root: Path = DATASETS / "oasis" / "extracted") -> list[Case]:
    """OASIS-1 and OASIS-2 raw MPRAGE (first scan of the session). OASIS-1 used to take the distributed
    Talairach-registered gain-field-corrected image (``*_t88_gfc``, cropped 176 x 208 x 176): every one of those
    registered poorly to MNI (template Dice ~0.83) and failed SynthSeg, while raw OASIS-2 registers like other
    cohorts (2026-10-21). The t88 image remains the fallback for a session without a raw scan."""
    cases, seen = [], set()

    def add(sid: str, path: Path, tag: str) -> None:
        if sid not in seen and not path.name.startswith("._"):
            seen.add(sid)
            cases.append(Case(tag, sid, "mri", path))

    for p in sorted(root.rglob("*_mpr-1_anon.img")):
        add("_".join(p.name.split("_")[:3]), p, "oasis1")
    for p in sorted(root.rglob("*_t88_gfc.img")):
        add("_".join(p.name.split("_")[:3]), p, "oasis1")
    for p in sorted(root.rglob("mpr-1.nifti.img")):
        session = p.parent.parent.name if p.parent.parent.name.startswith("OAS2_") else p.parent.name
        add(session, p, "oasis2")
    return cases


def brats(root: Path = DATASETS / "brats" / "extracted") -> list[Case]:
    """Local BraTS (GLI/MEN/PED + segs) plus FOMO BraTS-GEN (almost all GoAT) when extracted."""
    from brain.prep.fomo import FOMO_ROOT
    cases = brats_local(root)
    extracted = FOMO_ROOT / "extracted" / "PT009_BraTS-GEN"
    if extracted.exists() and any(extracted.rglob("*.nii.gz")):
        cases.extend(brats_fomo(FOMO_ROOT, root))
    return cases


def brats_local(root: Path = DATASETS / "brats" / "extracted") -> list[Case]:
    """Every (case, sequence) is one volume; the segmentation rides with the t1n volume."""
    cases = []
    for cohort in sorted(p for p in root.iterdir() if p.is_dir() and p.name in BRATS_COHORTS):
        for c in BRATS_CONTRASTS:
            for p in sorted(cohort.rglob(f"*-{c}.nii.gz")):
                if p.name.startswith("._"):
                    continue
                seg = p.parent / p.name.replace(f"-{c}.nii.gz", "-seg.nii.gz")
                masks = (("seg", seg),) if c == "t1n" and seg.exists() else ()
                t1n = p.parent / p.name.replace(f"-{c}.nii.gz", "-t1n.nii.gz")  # all four sequences share t1n's grid
                cases.append(Case(f"{BRATS_COHORTS[cohort.name]}_{c}", p.name.removesuffix(".nii.gz"), "mri", p, masks,
                                  ref=t1n if c != "t1n" and t1n.exists() else None))
    return cases


def local_brats_assets(root: Path = DATASETS / "brats" / "extracted") -> tuple[dict[str, Path], dict[str, str]]:
    """``BraTS-GLI-00005-000`` → (seg path, preferred cohort tag). 2024 beats 2023 when both exist."""
    segs, cohort = {}, {}
    if not root.exists():
        return segs, cohort
    for p in root.rglob("*.nii.gz"):
        if p.name.startswith("._"):
            continue
        top = None
        for part in p.parts:
            if part in BRATS_COHORTS:
                top = part
                break
        if top is None:
            continue
        tag = BRATS_COHORTS[top]
        if p.name.endswith("-seg.nii.gz"):
            key = p.name[: -len("-seg.nii.gz")]
            segs[key] = p
            if COHORT_RANK.get(tag, 0) >= COHORT_RANK.get(cohort.get(key, ""), 0):
                cohort[key] = tag
        elif p.name.endswith("-t1n.nii.gz"):
            key = p.name[: -len("-t1n.nii.gz")]
            if COHORT_RANK.get(tag, 0) >= COHORT_RANK.get(cohort.get(key, ""), 0):
                cohort[key] = tag
    return segs, cohort


def brats_fomo(fomo_root: Path, local_root: Path = DATASETS / "brats" / "extracted") -> list[Case]:
    """FOMO BraTS-GEN NIfTIs; original BraTS ids from mapping.tsv; segs from our extracted copy."""
    from brain.prep.fomo import (
        brats_dataset_tag,
        brats_stem,
        contrast_token,
        is_structural,
        load_holdout_families,
        read_mapping,
        resolve_extracted,
    )

    pt = "PT009_BraTS-GEN"
    mapping = read_mapping(fomo_root / pt / "mapping.tsv")
    if mapping.empty:
        mapping = read_mapping(fomo_root / "extracted" / pt / "mapping.tsv")
    if mapping.empty:
        # the corpus-wide mapping names the PT folder in ``dataset``; ``new_path`` is relative to it
        mapping = read_mapping(fomo_root / "mapping.tsv")
        if not mapping.empty:
            in_pt = pd.Series(False, index=mapping.index)
            for col in ("dataset", "new_path"):
                if col in mapping.columns:
                    in_pt |= mapping[col].astype(str).str.contains(pt, na=False)
            mapping = mapping[in_pt]
    held = load_holdout_families(fomo_root)
    segs, cohorts = local_brats_assets(local_root)
    cases, by_session = [], {}
    for r in mapping.itertuples(index=False):
        old_fn = str(getattr(r, "old_filename", "") or "")
        new_fn = str(getattr(r, "new_filename", "") or "")
        new_p = str(getattr(r, "new_path", "") or "")
        old_p = str(getattr(r, "old_path", "") or "")
        sub = str(getattr(r, "participant_id", "") or "")
        ses = str(getattr(r, "session_id", "") or "")
        if not is_structural(old_fn or new_fn):
            continue
        contrast = contrast_token(old_fn or new_fn, brats=True)
        if contrast not in BRATS_CONTRASTS:
            continue
        stem = brats_stem(old_fn) if old_fn else Path(new_fn).name
        key = stem
        for c in BRATS_CONTRASTS:
            if key.endswith("-" + c):
                key = key[: -len(c) - 1]
                break
        tag = brats_dataset_tag(old_p, old_fn, contrast, cohorts.get(key, ""))
        if tag.rsplit("_", 1)[0] in held or "fomo_brats" in held:
            continue
        fname = Path(new_fn).name if new_fn else Path(new_p).name
        path = resolve_extracted(fomo_root, pt, sub, ses, fname, contrast)
        if path is None:
            continue
        masks = (("seg", segs[key]),) if contrast == "t1n" and key in segs else ()
        case = Case(tag, stem, "mri", path, masks)
        cases.append(case)
        by_session.setdefault((sub, ses, tag.rsplit("_", 1)[0]), {})[contrast] = path
    # co-registered BraTS sequences share the t1n grid
    out = []
    for c in cases:
        fam = c.dataset.rsplit("_", 1)[0]
        sub = None
        # recover session from path: extracted/PT.../sub-x/ses-y/file
        parts = c.path.parts
        try:
            i = parts.index(pt)
            sub, ses = parts[i + 1], parts[i + 2]
        except ValueError:
            sub = ses = None
        t1n = by_session.get((sub, ses, fam), {}).get("t1n") if sub else None
        if t1n is not None and c.path != t1n:
            out.append(Case(c.dataset, c.case_id, c.modality, c.path, c.masks, ref=t1n))
        else:
            out.append(c)
    return out


def fomo(root: Path | None = None) -> list[Case]:
    """Non-BraTS FOMO folders we pulled. Structural sequences only; mapping holdouts dropped whole."""
    from brain.prep.fomo import (
        FOMO_ROOT,
        PULL_FOLDERS,
        SKIP_FOLDERS,
        contrast_token,
        family_for_row,
        is_structural,
        load_holdout_families,
        load_mappings,
        pt_of,
        resolve_extracted,
    )

    root = root or FOMO_ROOT
    held = load_holdout_families(root)
    maps = load_mappings(root)
    if maps.empty:
        return []
    cases = []
    seen = set()
    for r in maps.itertuples(index=False):
        new_p = str(getattr(r, "new_path", "") or "")
        old_p = str(getattr(r, "old_path", "") or "")
        new_fn = str(getattr(r, "new_filename", "") or "")
        old_fn = str(getattr(r, "old_filename", "") or "")
        sub = str(getattr(r, "participant_id", "") or "")
        ses = str(getattr(r, "session_id", "") or "")
        pt = str(getattr(r, "mapping_pt", "") or "") or pt_of(new_p)
        if not pt or pt in SKIP_FOLDERS or pt == "PT009_BraTS-GEN" or pt not in PULL_FOLDERS:
            continue
        fam = family_for_row({"new_path": new_p, "old_path": old_p, "mapping_pt": pt})
        if fam in held:
            continue
        if not is_structural(new_fn or old_fn or new_p):
            continue
        contrast = contrast_token(new_fn or old_fn)
        if contrast not in ("t1w", "t1c", "t2w", "flair", "pd"):
            continue
        fname = Path(new_fn).name if new_fn else Path(new_p).name
        path = resolve_extracted(root, pt, sub, ses, fname, contrast)
        if path is None:
            continue
        dataset = f"{fam}_{contrast}"
        # contrast must be in the filename: all sequences share folder ``fomo_<short>``
        case_id = f"{sub}_{ses}_{contrast}" if sub and ses else f"{Path(fname).stem}_{contrast}"
        key = (dataset, case_id)
        if key in seen:
            continue
        seen.add(key)
        cases.append(Case(dataset, case_id, "mri", path))
    return cases


def abide1(root: Path | None = None) -> list[Case]:
    """ABIDE I raw T1 (skull on, 24 site folders); diagnosis and age are in the phenotypic table."""
    roots = [root] if root is not None else [
        DATASETS / "abide1" / "RawData",
        SCRATCH / "datasets" / "abide1" / "RawData",
    ]
    for cand in roots:
        if cand is None:
            continue
        cases = [Case("abide1", f"{p.parents[3].name}_{p.parents[2].name}", "mri", p)
                 for p in sorted(Path(cand).glob("*/*/session_1/anat_1/mprage.nii.gz"))]
        if cases:
            return cases
    return []


def ixi(root: Path = DATASETS / "ixi") -> list[Case]:
    """IXI T1 and T2 (three hospitals); each file is one volume, subject id is the IXI number."""
    return [Case(f"ixi_{seq.lower()}", p.name.removesuffix(".nii.gz"), "mri", p)
            for seq in ("T1", "T2") for p in sorted((root / seq).glob(f"*-{seq}.nii.gz"))]


def isles22(root: Path = DATASETS / "isles22" / "ISLES-2022") -> list[Case]:
    """ISLES 2022: DWI, ADC and FLAIR per case; the lesion mask lives in DWI space and rides with the DWI volume.
    ADC and FLAIR take DWI's grid (``ref``) so the three sequences and the mask stay aligned."""
    cases = []
    for d in sorted(root.glob("sub-strokecase*")):
        s = f"{d.name}_ses-0001"
        dwi = d / "ses-0001" / "dwi" / f"{s}_dwi.nii.gz"
        msk = root / "derivatives" / d.name / "ses-0001" / f"{s}_msk.nii.gz"
        for seq, p in (("dwi", dwi), ("adc", d / "ses-0001" / "dwi" / f"{s}_adc.nii.gz"),
                       ("flair", d / "ses-0001" / "anat" / f"{s}_FLAIR.nii.gz")):
            if p.exists():
                cases.append(Case(f"isles22_{seq}", f"{d.name}_{seq}", "mri", p,
                                  (("lesion", msk),) if seq == "dwi" and msk.exists() else (),
                                  ref=dwi if seq != "dwi" and dwi.exists() else None))
    return cases


def atlas(root: Path = DATASETS / "atlas" / "ATLAS3_Training_Raw") -> list[Case]:
    """ATLAS R3.0 raw (skull-stripped) T1 with the chronic-stroke lesion mask; metadata CSV sits next to each T1."""
    cases = []
    for t1 in sorted(root.rglob("*_desc-brain_T1w.nii.gz")):
        mask = next(t1.parent.glob("*_label-lesion_desc-T1lesion_mask.nii.gz"), None)
        cases.append(Case("atlas", t1.name.split("_")[0], "mri", t1, (("lesion", mask),) if mask else ()))
    return cases


def bids_t1(tag: str, root: Path) -> list[Case]:
    """T1w scans of an OpenNeuro BIDS dataset (face-removed, skull on)."""
    first = {}  # one scan per subject (ID1000 repeats the T1w three times)
    for p in sorted(root.glob("sub-*/anat/*_T1w.nii.gz")):
        first.setdefault(p.parents[1].name, p)
    return [Case(tag, sub, "mri", p) for sub, p in first.items()]


def pixar(root: Path = DATASETS / "pixar") -> list[Case]:
    return bids_t1("pixar", root)


def piop1(root: Path = DATASETS / "piop1") -> list[Case]:
    return bids_t1("piop1", root)


def piop2(root: Path = DATASETS / "piop2") -> list[Case]:
    return bids_t1("piop2", root)


def id1000(root: Path = DATASETS / "id1000") -> list[Case]:
    return bids_t1("id1000", root)


def wmh(root: Path = DATASETS / "wmh" / "extracted") -> list[Case]:
    """WMH Segmentation Challenge 2017 (training + released test): raw FLAIR and the 3D T1 aligned to it.
    The WMH mask lives in FLAIR space and rides with the FLAIR; the aligned T1 takes the FLAIR grid (``ref``)."""
    cases = []
    for flair in sorted(root.rglob("orig/FLAIR.nii.gz")):
        subject = flair.parents[1]  # <split>/<site>[/<scanner>]/<id>
        cid = "_".join(subject.relative_to(root).parts)
        cid = "".join(ch if ch.isalnum() or ch in "_-" else "" for ch in cid.replace(" ", "").replace(".", ""))
        t1 = flair.with_name("T1.nii.gz")
        wmh_mask = subject / "wmh.nii.gz"
        cases.append(Case("wmh_flair", cid, "mri", flair, (("lesion", wmh_mask),) if wmh_mask.exists() else ()))
        if t1.exists():
            cases.append(Case("wmh_t1", cid, "mri", t1, ref=flair))
    return cases


def ms3seg(root: Path = DATASETS / "ms3seg" / "extracted" / "MS_100_patient_nifti") -> list[Case]:
    """MS3SEG (100 MS patients, 1.5T Toshiba, thick-slice clinical): axial FLAIR with the staged label map
    (``brain.prep.stage_new``: 1 ventricles, 2 normal WMH, 3 MS lesion), plus T1 and T2. The sagittal FLAIR is skipped."""
    cases = []
    if not root.exists():
        return cases
    for subject in sorted(p for p in root.iterdir() if p.is_dir()):
        sid = subject.name
        flair, seg = subject / f"{sid}_FLAIR.nii.gz", subject / f"{sid}_seg.nii.gz"
        if flair.exists():
            cases.append(Case("ms3seg_flair", sid, "mri", flair, (("seg", seg),) if seg.exists() else ()))
        for tag, name in (("t1", "T1WI"), ("t2", "T2WI")):
            p = subject / f"{sid}_{name}.nii.gz"
            if p.exists():
                cases.append(Case(f"ms3seg_{tag}", sid, "mri", p))
    return cases


VALDO_SEQ = {"T1": "t1", "T2": "t2", "FLAIR": "flair", "T2S": "t2s"}


def valdo(root: Path = DATASETS / "valdo" / "extracted") -> list[Case]:
    """VALDO 2021 (Task1 EPVS, Task2 microbleeds on T2*, Task3 lacunes). Every task's sequences are co-registered to
    its reference space (``space-T1`` / ``space-T2S``): the reference sequence decides the grid and carries the
    rater-union lesion mask staged by ``brain.prep.stage_new``; the other sequences take its grid (``ref``)."""
    cases = []
    if not root.exists():
        return cases
    pat = re.compile(r"_space-([A-Za-z0-9]+)_desc-(?!Rater|union)[A-Za-z0-9]+_(T1|T2|FLAIR|T2S)\.nii\.gz$")
    for task_dir in sorted(root.glob("Task*")):
        task = task_dir.name.lower().replace("task", "t")
        for subject in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            images = {}
            for f in subject.glob("*.nii.gz"):
                m = pat.search(f.name)
                if m:
                    space, seq = m.group(1).upper(), m.group(2).upper()
                    images[seq] = (f, space)
            if not images:
                continue
            space = next(iter(images.values()))[1]
            ref = images.get(space, (None,))[0]
            # segmentation targets: lacunes (Task 3) and microbleeds (Task 2); Task 1's PVS masks cover only slabs
            union = next((f for f in subject.glob("*_desc-union_*.nii.gz") if f.name.endswith(("_Lacunes.nii.gz", "_CMB.nii.gz"))), None)
            cid = f"{task}_{subject.name}"
            for seq, (f, _) in sorted(images.items()):
                is_ref = f == ref
                masks = (("lesion", union),) if is_ref and union is not None else ()
                cases.append(Case(f"valdo_{VALDO_SEQ[seq]}", cid, "mri", f, masks,
                                  ref=None if is_ref or ref is None else ref))
    return cases


# Official AISD test ids (GriffinLiang/AISD README). Numeric stems are zero-padded to 7.
AISD_TEST_IDS = (
    "0073410", "0072723", "0226290", "0537908", "0538058", "0091415", "0538780", "0073540",
    "0226188", "0226258", "0226314", "0091507", "0226298", "0538975", "0226257", "0226142",
    "0072681", "0091538", "0538983", "0537961", "0091646", "0072765", "0226137", "0091621",
    "0091458", "0021822", "0538319", "0226133", "0091657", "0537925", "0073489", "0538502",
    "0091476", "0226136", "0538532", "0073312", "0539025", "0226309", "0226307", "0091383",
    "0021092", "0537990", "0226299", "0073060", "0538505", "0073424", "0091534", "0226125",
    "0072691", "0538425", "0226199", "0226261",
)


def aisd_id(stem: str) -> str:
    s = "".join(ch for ch in str(stem) if ch.isalnum())
    return s.zfill(7) if s.isdigit() else s


def _pair_niftis(root: Path, dataset: str, modality: str, mask_name: str) -> list[Case]:
    """Match ``*.nii.gz`` images to a label file of the same stem (folder named label/mask, or ``*_label``)."""

    def stem(p: Path) -> str:
        s = p.name.removesuffix(".nii.gz")
        for tag in ("_label", "-label", "_mask", "-mask", "_seg", "-seg"):
            if s.lower().endswith(tag):
                return s[: -len(tag)]
        return s

    def is_label(p: Path) -> bool:
        blob = "/".join(x.lower() for x in p.parts)
        return any(k in blob for k in ("/label/", "/labels/", "/labelstr/", "/mask/", "/masks/")) or stem(p) != p.name.removesuffix(".nii.gz")

    niftis = [p for p in sorted(root.rglob("*.nii.gz")) if not p.name.startswith("._")]
    key = (lambda p: aisd_id(stem(p))) if dataset == "aisd" else stem
    labels = {key(p): p for p in niftis if is_label(p)}
    cases = []
    for p in niftis:
        if is_label(p):
            continue
        cid = key(p)
        mask = labels.get(cid)
        cases.append(Case(dataset, cid, modality, p, ((mask_name, mask),) if mask else ()))
    return cases


BM_SEQS = (("bravo", "brainmetshare_bravo"), ("t1_pre", "brainmetshare_t1_pre"),
           ("t1_gd", "brainmetshare_t1_gd"), ("flair", "brainmetshare_flair"))


def brainmetshare(root: Path = DATASETS / "brainmetshare" / "nifti") -> list[Case]:
    """BrainMetShare MRI (T1 GRE post / T1 SE pre / T1 SE post / T2 FLAIR). Segs ride with bravo."""
    cases = []
    for split in ("train", "test"):
        for sub in sorted((Path(root) / split).glob("Mets_*")):
            if not sub.is_dir():
                continue
            cid, bravo, seg = sub.name, sub / "bravo.nii.gz", sub / "seg.nii.gz"
            for fname, tag in BM_SEQS:
                p = sub / f"{fname}.nii.gz"
                if not p.exists():
                    continue
                masks = (("seg", seg),) if fname == "bravo" and seg.exists() else ()
                ref = bravo if fname != "bravo" and bravo.exists() else None
                cases.append(Case(tag, cid, "mri", p, masks, ref=ref))
    return cases


MSD_T1W_CHANNEL = 1  # Task01 4D order: FLAIR, T1w, T1gd, T2
DEFAULT_MSD_ROOT = DATASETS / "msd_braintumor" / "Task01_BrainTumour"


def ensure_msd_t1w(src: Path, channel: int = MSD_T1W_CHANNEL) -> Path:
    """Cache the T1w channel of an MSD 4D NIfTI as a 3D file (workers share the cache)."""
    src = Path(src)
    dest = src.parent.parent / "derived" / "T1w" / src.name
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    import nibabel as nib

    dest.parent.mkdir(parents=True, exist_ok=True)
    # nib.save keys off the suffix; ``.nii.gz.partial.PID`` is not a NIfTI path.
    tmp = dest.parent / f"{dest.name}.partial.{os.getpid()}.nii.gz"
    nii = nib.load(str(src))
    data = np.asanyarray(nii.dataobj)
    if data.ndim == 4:
        data = data[..., channel]
    elif data.ndim != 3:
        raise ValueError(f"Expected 3D or 4D MSD volume at {src}, got {data.shape}")
    try:
        nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), nii.affine, nii.header), str(tmp))
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def msd(root: Path = DEFAULT_MSD_ROOT) -> list[Case]:
    """Medical Decathlon Task01 train T1w (4D NIfTI, channel 1). Held-out frozen eval only."""
    root = Path(root)
    img_dir = root / "imagesTr"
    if not img_dir.is_dir():
        return []
    cases = []
    for src in sorted(img_dir.glob("BRATS_*.nii*")):
        if src.name.startswith("._"):
            continue
        stem = src.name.replace(".nii.gz", "").replace(".nii", "")
        seg = root / "labelsTr" / f"{stem}.nii.gz"
        if not seg.exists():
            seg = root / "labelsTr" / f"{stem}.nii"
        masks = (("seg", seg),) if seg.exists() else ()
        cases.append(Case("msd_braintumor", stem, "mri", src, masks))
    return cases


def sinoct(root: Path = DATASETS / "sinoct" / "reconstructed") -> list[Case]:
    """Keep-only reconstructed SinoCT series (RSNA overlap already dropped at pull). One case per series."""
    root = Path(root)
    if not root.exists():
        return []
    cases = []
    for d in sorted(root.glob("batch_*/series_*/reconstructed_image")):
        n = sum(1 for _ in d.glob("*.dcm")) + sum(1 for _ in d.glob("*.DCM"))
        if n < 8:
            continue
        series, batch = d.parent.name, d.parent.parent.name
        cases.append(Case("sinoct", f"{batch}_{series}", "ct", d))
    return cases


def instance(root: Path = DATASETS / "instance2022" / "extracted") -> list[Case]:
    """INSTANCE 2022 NCCT with ICH masks (MICCAI challenge training zip)."""
    return _pair_niftis(root, "instance", "ct", "ICH")


def bhsd(root: Path = DATASETS / "bhsd" / "extracted") -> list[Case]:
    """BHSD (MBH-Seg): 192 RSNA ICH studies reconstructed from DICOM positions by the BHSD authors, with voxel masks
    (1 EDH, 2 IPH, 3 IVH, 4 SAH, 5 SDH). ``label_192/images/<name>.nii.gz`` pairs with ``label_192/ground truths/
    <name>.nii.gz``; ``<name>`` is ``ID_<short>_ID_<rsna hash>``, so the case id is the RSNA study's (``RSNA-ID_<hash>``)
    and the index treats it as that study's subject (same split, flagged ``bhsd``)."""
    cases = []
    for img in sorted(Path(root).rglob("images/*.nii.gz")):
        mask = img.parent.parent / "ground truths" / img.name
        cid = "RSNA-ID_" + img.name.removesuffix(".nii.gz").rsplit("ID_", 1)[-1]
        cases.append(Case("bhsd", cid, "ct", img, (("ICH", mask),) if mask.exists() else ()))
    return cases


RSNA_ICA_DATASET = {"CTA": ("rsna_ica_cta", "ct"), "MRA": ("rsna_ica_mra", "mri"), "MRI T2": ("rsna_ica_t2w", "mri"),
                    "MRI T1post": ("rsna_ica_t1c", "mri")}


def rsna_ica(rsna_zip: Path = DATASETS / "rsna_ica" / "rsna-intracranial-aneurysm-detection.zip") -> list[Case]:
    """RSNA Intracranial Aneurysm Detection (2025; MIRA dataset 7, non-commercial): 4,348 series (CTA, TOF MRA,
    T2, post-contrast T1) read straight from the pool zip by byte offsets (``series/<SeriesUID>/<SOP>.dcm``; some MR
    series are one enhanced multi-frame file, so prep runs with ``BRAIN_DICOM_READER=dcm2niix``). The member index
    and ``train.csv`` are cached under ``work/rsna_ica``."""
    import io
    import json
    import zipfile

    if not rsna_zip.exists():
        return []
    work = SCRATCH / "work" / "rsna_ica"
    work.mkdir(parents=True, exist_ok=True)
    idx_path, csv_path = work / "series_members.json", work / "train.csv"
    if not idx_path.exists() or not csv_path.exists():
        with zipfile.ZipFile(rsna_zip) as zf:
            index: dict[str, list] = {}
            for i in zf.infolist():
                parts = i.filename.split("/")
                if len(parts) == 3 and parts[0] == "series" and parts[2]:
                    index.setdefault(parts[1], []).append([i.filename, i.header_offset, i.compress_size,
                                                           i.compress_type])
            csv_path.write_bytes(zf.read("train.csv"))
        idx_path.write_text(json.dumps(index))
    index = json.loads(idx_path.read_text())
    meta = pd.read_csv(csv_path)
    cases = []
    for uid, mod in zip(meta["SeriesInstanceUID"].astype(str), meta["Modality"].astype(str)):
        if uid in index and mod in RSNA_ICA_DATASET:
            ds, modality = RSNA_ICA_DATASET[mod]
            cases.append(Case(ds, uid, modality, zip_path=str(rsna_zip),
                              members=tuple(tuple(m) for m in sorted(index[uid]))))
    return cases


def hemseg(root: Path = DATASETS / "hemseg" / "extracted",
           rsna_zip: Path = DATASETS / "rsna" / "rsna-ihd-dataset.zip") -> list[Case]:
    """HemSeg-500: 525 RSNA studies with voxel masks of their one bleed (352 IPH-only, 173 IVH-only by RSNA's labels,
    so each binary mask is the scan's complete hemorrhage). The authors rebuilt each image with SimpleITK's series
    reader from a listed subset of the study's DICOM slices (``annotion_file_info_500.jsonl``); reading exactly those
    members from the RSNA archive reproduces their geometry, so the masks line up. Studies also in BHSD are skipped
    (BHSD's multi-class masks win)."""
    import json

    from brain.prep import dicom
    from brain.prep.index import bhsd_case_ids

    root = Path(root)
    info = root.parent / "annotion_file_info_500.jsonl"
    if not info.exists():
        return []
    work = SCRATCH / "work" / "rsna_ingest"
    index = dicom.build_offsets(rsna_zip, work / "study_index.json", work / "member_offsets.json")
    masks = {p.name.removesuffix(".nii.gz"): p for p in root.rglob("label/*.nii.gz")}
    skip = bhsd_case_ids()
    cases = []
    for line in info.read_text().splitlines():
        rec = json.loads(line)
        scan, uid = rec["scan_name"], "ID_" + rec["scan_name"]
        cid = "RSNA-" + uid
        if scan not in masks or uid not in index or cid in skip:
            continue
        want = {f"ID_{s}.dcm" for s in rec["slice_list"].values()}
        members = tuple(tuple(m) for m in index[uid] if Path(m[0]).name in want)
        if len(members) != len(want):
            continue                                 # a listed slice missing from the archive: geometry would differ
        cases.append(Case("hemseg", cid, "ct", masks=(("ICH", masks[scan]),), zip_path=str(rsna_zip), members=members))
    return cases


def aisd(root: Path = DATASETS / "aisd" / "extracted") -> list[Case]:
    """AISD NCCT with ischemic-infarct masks (labels 1/2/3/5 = lesion; 4 = invisible, ignored).

    Public zips are PNG slices; ``download_aisd.fetch`` stacks them into ``extracted/nifti``.
    """
    nifti = Path(root) / "nifti"
    src = nifti if nifti.is_dir() and any(nifti.glob("*.nii.gz")) else Path(root)
    return _pair_niftis(src, "aisd", "ct", "lesion")


def isles24(root: Path = DATASETS / "isles24" / "train") -> list[Case]:
    """ISLES'24 (acute stroke CT): raw NCCT with the follow-up-MRI infarct mask, the CTA-derived large-vessel-occlusion mask
    and the Circle of Willis labels (all in NCCT space), plus the CTA registered to NCCT (takes the NCCT grid via ``ref``)."""
    cases = []
    for d in sorted((root / "raw_data").glob("sub-stroke*")):
        s, der = d.name, root / "derivatives" / d.name
        ncct = d / "ses-01" / f"{s}_ses-01_ncct.nii.gz"
        if not ncct.exists():
            continue
        masks = tuple((n, p) for n, p in (("lesion", der / "ses-02" / f"{s}_ses-02_space-ncct_lesion-msk.nii.gz"),
                                          ("lvo", der / "ses-01" / f"{s}_ses-01_space-ncct_lvo-msk.nii.gz"),
                                          ("cow", der / "ses-01" / f"{s}_ses-01_space-ncct_cow-msk.nii.gz")) if p.exists())
        cases.append(Case("isles24_ncct", s, "ct", ncct, masks))
        cta = der / "ses-01" / f"{s}_ses-01_space-ncct_cta.nii.gz"
        if cta.exists():
            cases.append(Case("isles24_cta", s, "ct", cta, ref=ncct))
    return cases


def rsna(zip_path: Path = DATASETS / "rsna" / "rsna-ihd-dataset.zip", limit: int | None = None) -> list[Case]:
    """One case per study; member byte offsets are cached so workers never parse the zip directory."""
    from brain.prep import dicom

    work = SCRATCH / "work" / "rsna_ingest"
    index = dicom.build_offsets(zip_path, work / "study_index.json", work / "member_offsets.json")
    uids = sorted(index)[:limit]
    return [Case("rsna_ich", "RSNA-" + uid.replace(".", "_")[:80], "ct", zip_path=str(zip_path),
                 members=tuple(tuple(m) for m in index[uid])) for uid in uids]
