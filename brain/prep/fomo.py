"""FOMO260K mapping, structural-sequence filter, and leakage holdouts.

FOMO lives on scratch (the pool quota cannot hold it). Per-folder mappings join
FOMO BIDS names back to the original files; we hold out an entire FOMO source
(OpenNeuro dsXXXXXX, or a whole PT folder) when any of its rows match a
brain.downstream held-out family.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from brain.common.paths import SCRATCH

FOMO_REPO = "FOMO-MRI/FOMO260K"
FOMO_ROOT = Path(os.environ.get("BRAIN_FOMO", str(SCRATCH / "datasets" / "fomo260k")))

# Pulled for pretraining. BraTS-GEN replaces ``brats/extracted`` images.
PULL_FOLDERS = (
    "PT009_BraTS-GEN",
    "PT018_HBN",
    "PT035_Yale_Brain_Mets_Longitudinal",
    "PT027_NKI",
    "PT014_CoRR",
    "PT002_Nigerian_Clinical",
    "PT010_BrainLat",
    "PT024_M4Raw",
)
# Already in our store, or too leaky/large to ingest as a blob (OpenNeuro).
SKIP_FOLDERS = (
    "PT015_MSD_BrainTumor",
    "PT021_IXI",
    "PT028_OASIS1",
    "PT029_OASIS2",
    "PT030_OpenNeuro",
)

PT_SHORT = {
    "PT001_ClevelandCCF": "cleveland",
    "PT002_Nigerian_Clinical": "nigerian",
    "PT003_CUNMET": "cunmet",
    "PT004_ACPI": "acpi",
    "PT005_ADHD_200": "adhd200",
    "PT006_AHEAD": "ahead",
    "PT007_ATAG": "atag",
    "PT008_AdolescentBrainDevelopment": "abd",
    "PT009_BraTS-GEN": "brats",
    "PT010_BrainLat": "brainlat",
    "PT011_CFMM_7T": "cfmm",
    "PT012_CHBMP": "chbmp",
    "PT013_Calgary_Preschool": "calgary",
    "PT014_CoRR": "corr",
    "PT015_MSD_BrainTumor": "msd",
    "PT016_GSP": "gsp",
    "PT017_HBN-SSI": "hbnssi",
    "PT018_HBN": "hbn",
    "PT019_HCP_Wu_Minn_Test_Retest": "hcp_retest",
    "PT020_HCP_Wu_Minn": "hcp",
    "PT021_IXI": "ixi",
    "PT022_Bejing_Enhanced": "beijing",
    "PT023_Infant_Development_Brain": "infant",
    "PT024_M4Raw": "m4raw",
    "PT025_age_ility": "ageility",
    "PT026_MICA_MICs": "mica",
    "PT027_NKI": "nki",
    "PT028_OASIS1": "oasis1",
    "PT029_OASIS2": "oasis2",
    "PT030_OpenNeuro": "openneuro",
    "PT031_SLIM": "slim",
    "PT032_Tao_Wu": "taowu",
    "PT033_WAND": "wand",
    "PT034_Wayne": "wayne",
    "PT035_Yale_Brain_Mets_Longitudinal": "yale_mets",
    "PT036_Yale_HighRes": "yale_highres",
}

# Token after the last ``_`` of a nifti stem, lowercased.
STRUCTURAL_TOKENS = {
    "t1w", "t1", "t1n", "t1c", "t1ce", "t1gd", "mprage",
    "t2w", "t2", "t2f", "flair", "pd", "pdw",
}
BRATS_TOKEN = {"t1w": "t1n", "t1": "t1n", "t1n": "t1n", "mprage": "t1n",
               "t1c": "t1c", "t1ce": "t1c", "t1gd": "t1c",
               "t2w": "t2w", "t2": "t2w", "t2f": "t2f", "flair": "t2f",
               "pd": "pd", "pdw": "pd"}
FOMO_TOKEN = {"t1w": "t1w", "t1": "t1w", "t1n": "t1w", "mprage": "t1w",
              "t1c": "t1c", "t1ce": "t1c", "t1gd": "t1c",
              "t2w": "t2w", "t2": "t2w", "t2f": "flair", "flair": "flair",
              "pd": "pd", "pdw": "pd"}
CONTRAST_SUFFIXES = ("t1n", "t1c", "t2w", "t2f", "t1w", "flair", "pd")

# Entire original source is dropped from SSL if any row matches.
HOLD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cq500", re.compile(r"cq[-_]?500", re.I)),
    ("isles22", re.compile(r"isles[-_]?20?22|\bisles22\b", re.I)),
    ("isles24", re.compile(r"isles[-_]?20?24|\bisles24\b", re.I)),
    ("atlas", re.compile(r"\batlas[-_ ]?(?:r3|r3\.0|v2|3)\b|atlas3_training", re.I)),
    ("wmh", re.compile(r"wmh[-_ ]?(?:challenge|seg|2017)|\bwhite[-_ ]matter[-_ ]hyper", re.I)),
    ("instance", re.compile(r"instance[-_ ]?2022|\binstance2022\b", re.I)),
    ("aisd", re.compile(r"\baisd\b|acute[-_ ]ischemic[-_ ]stroke[-_ ]dataset", re.I)),
    ("brainmetshare", re.compile(r"brainmetshare|brain[-_ ]?met[-_ ]?share", re.I)),
)
DS_RE = re.compile(r"(ds\d{6})", re.I)
PT_RE = re.compile(r"(PT\d{3}_[A-Za-z0-9\-]+)")
BRATS_KIND_RE = re.compile(r"BraTS-(MEN-RT|GLI|MEN|PED|MET|GoAT|SSA)", re.I)


def nifti_token(name: str) -> str:
    stem = Path(str(name)).name.lower()
    for suf in (".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem.rsplit("_", 1)[-1].rsplit("-", 1)[-1]


def is_structural(name: str) -> bool:
    return nifti_token(name) in STRUCTURAL_TOKENS


def contrast_token(name: str, brats: bool = False) -> str:
    tok = nifti_token(name)
    table = BRATS_TOKEN if brats else FOMO_TOKEN
    return table.get(tok, tok)


def strip_contrast(dataset: str) -> str:
    ds = str(dataset)
    for c in CONTRAST_SUFFIXES:
        if ds.endswith("_" + c):
            return ds[: -len(c) - 1]
    return ds


def pt_of(path: str) -> str:
    m = PT_RE.search(str(path).replace("\\", "/"))
    return m.group(1) if m else ""


def pt_from_dataset(val: str) -> str:
    """``PT030_OpenNeuro/ds004146`` → ``PT030_OpenNeuro``."""
    s = str(val or "")
    if s.startswith("PT"):
        return s.split("/", 1)[0]
    return pt_of(s)


def short_of(pt: str) -> str:
    return PT_SHORT.get(pt, pt.lower().replace("pt", "pt", 1))


def pt_from_family(family: str) -> str:
    """``fomo_hbn`` / ``hbn`` → ``PT018_HBN``."""
    short = str(family).removeprefix("fomo_")
    for pt, s in PT_SHORT.items():
        if s == short:
            return pt
    raise KeyError(family)


def read_mapping(path: Path) -> pd.DataFrame:
    """Tab-separated FOMO mapping; sniffs the separator if a copy used spaces."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    head = path.read_bytes()[:4096]
    sep = "\t" if b"\t" in head.split(b"\n", 1)[0] else r"\s+"
    df = pd.read_csv(path, sep=sep, engine="python" if sep != "\t" else "c")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def mapping_paths(root: Path = FOMO_ROOT) -> list[Path]:
    found = []
    top = root / "mapping.tsv"
    if top.exists():
        found.append(top)
    for p in sorted(root.glob("PT*/mapping.tsv")):
        found.append(p)
    return found


def load_mappings(root: Path = FOMO_ROOT) -> pd.DataFrame:
    top = root / "mapping.tsv"
    if top.exists() and top.stat().st_size > 0:
        df = read_mapping(top)
        if not df.empty:
            df = df.copy()
            if "dataset" in df.columns:
                df["mapping_pt"] = df["dataset"].map(pt_from_dataset)
            elif "mapping_pt" not in df.columns or df["mapping_pt"].astype(str).eq("").all():
                src = df["new_path"] if "new_path" in df.columns else df.iloc[:, 1]
                df["mapping_pt"] = src.map(pt_from_dataset)
            return df
    frames = []
    for p in sorted(root.glob("PT*/mapping.tsv")):
        df = read_mapping(p)
        if df.empty:
            continue
        df = df.copy()
        df["mapping_pt"] = p.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _get(row, key: str) -> str:
    if isinstance(row, dict):
        return str(row.get(key) or "")
    if hasattr(row, "get"):
        return str(row.get(key) or "")
    return str(getattr(row, key, "") or "")


def family_for_row(row) -> str:
    """SSL family atom: one OpenNeuro ds, or one PT folder."""
    new_p = _get(row, "new_path")
    old_p = _get(row, "old_path")
    dscol = _get(row, "dataset") or _get(row, "mapping_pt")
    pt = pt_from_dataset(dscol) or pt_of(new_p) or pt_of(old_p)
    if pt == "PT030_OpenNeuro" or str(dscol).startswith("PT030_OpenNeuro"):
        m = DS_RE.search(dscol) or DS_RE.search(old_p) or DS_RE.search(new_p)
        dsid = m.group(1).lower() if m else "unknown"
        return f"fomo_openneuro_{dsid}"
    if pt:
        return "fomo_" + short_of(pt)
    return "fomo_unknown"


def blob_of(row) -> str:
    return " ".join(_get(row, c) for c in ("old_path", "old_filename", "new_path", "new_filename"))


def match_hold_family(blob: str) -> str | None:
    for fam, pat in HOLD_PATTERNS:
        if pat.search(blob):
            return fam
    return None


def holdout_from_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """One row per FOMO family that must stay out of the SSL index."""
    empty = pd.DataFrame(columns=["fomo_family", "held_out", "n", "example"])
    if df.empty:
        return empty
    cols = [c for c in ("dataset", "old_path", "old_filename", "new_path", "new_filename") if c in df.columns]
    text = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    recs = []
    for fam, pat in HOLD_PATTERNS:
        hit = text.str.contains(pat)
        if not hit.any():
            continue
        sub = df.loc[hit]
        for row in sub.itertuples(index=False):
            recs.append({"fomo_family": family_for_row(row._asdict() if hasattr(row, "_asdict") else row),
                         "held_out": fam,
                         "example": str(getattr(row, "old_path", "") or getattr(row, "old_filename", ""))[:240]})
    if not recs:
        return empty
    out = pd.DataFrame(recs)
    n = out.groupby(["fomo_family", "held_out"], as_index=False).agg(n=("example", "size"), example=("example", "first"))
    return n.sort_values(["held_out", "fomo_family"])


def write_holdout_lists(df: pd.DataFrame, root: Path = FOMO_ROOT) -> set[str]:
    root.mkdir(parents=True, exist_ok=True)
    hits = holdout_from_mapping(df)
    families = set(hits["fomo_family"].astype(str)) if len(hits) else set()
    txt = root / "holdout_datasets.txt"
    lines = ["# FOMO families held out of SSL because mapping.tsv overlaps a brain.downstream eval set",
             *[f for f in sorted(families)]]
    txt.write_text("\n".join(lines) + ("\n" if lines else ""))
    hits.to_csv(root / "holdout_report.csv", index=False)
    print(f"[fomo] hold-out families={len(families)} matched_rows={int(hits['n'].sum()) if len(hits) else 0} -> {txt}",
          flush=True)
    return families


def load_holdout_families(root: Path = FOMO_ROOT) -> set[str]:
    p = root / "holdout_datasets.txt"
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")}


def brats_kind(old_filename: str) -> str:
    m = BRATS_KIND_RE.search(str(old_filename))
    if not m:
        return "unk"
    k = m.group(1).lower().replace("-", "_")
    return k


def brats_year(old_path: str) -> str:
    p = str(old_path)
    if "2024" in p or "ISBI2024" in p:
        return "2024"
    if "2023" in p:
        return "2023"
    return ""


def brats_dataset_tag(old_path: str, old_filename: str, contrast: str, local_cohort: str = "") -> str:
    if local_cohort:
        return f"{local_cohort}_{contrast}"
    kind = brats_kind(old_filename)
    year = brats_year(old_path)
    if kind == "gli" and year:
        return f"brats{year}_gli_{contrast}"
    if kind == "men" and year:
        return f"brats{year}_men_{contrast}"
    if kind == "ped" and year:
        return f"brats{year}_ped_{contrast}"
    return f"brats_{kind}_{contrast}"


def brats_stem(old_filename: str) -> str:
    """``BraTS-GLI-00005-000-t1n.nii.gz`` → ``BraTS-GLI-00005-000-t1n``."""
    name = Path(str(old_filename)).name
    for suf in (".nii.gz", ".nii"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def session_zip(root: Path, pt: str, participant: str, session: str) -> Path:
    return root / pt / participant / f"{session}.zip"


def extracted_path(root: Path, pt: str, participant: str, session: str, filename: str) -> Path:
    return root / "extracted" / pt / participant / session / Path(filename).name


def resolve_extracted(root: Path, pt: str, participant: str, session: str, filename: str, contrast: str) -> Path | None:
    """Find the extracted nifti for one sequence; FOMO zips use T1w.nii.gz more often than the BIDS basename."""
    folder = root / "extracted" / pt / participant / session
    if filename:
        p = folder / Path(filename).name
        if p.exists():
            return p
    if not folder.is_dir():
        return None
    want = {contrast, BRATS_TOKEN.get(contrast, contrast), FOMO_TOKEN.get(contrast, contrast)}
    want.discard("")
    for p in folder.iterdir():
        tok = nifti_token(p.name)
        mapped = {tok, BRATS_TOKEN.get(tok, ""), FOMO_TOKEN.get(tok, "")}
        if want & mapped:
            return p
    return None
