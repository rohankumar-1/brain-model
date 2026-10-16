"""Build v4 index tables and leakage-safe splits from the processed store.

    python -m data.prep.index --root STORE --workers 12
    python -m data.prep.index --root STORE --keep-ssl-index   # add held-out zips without re-slicing SSL

Writes ``volumes.csv``, ``subjects.csv``, ``slices.parquet`` (pretraining index),
``empty_slice_removals.csv``. BHSD RSNA studies and held-out datasets are never
rows of ``slices.parquet``. Subject-level ``dedup_key`` is the split atom:
OASIS-2 sessions, IXI T1/T2, BraTS year/timepoint, and co-registered sequences
of one person always share a split. AISD keeps its official 52-case test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from data.prep import store
from data.prep.fomo import CONTRAST_SUFFIXES, load_holdout_families, strip_contrast
from data.prep.sources import DATASETS, SCRATCH

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = SCRATCH / "data" / "processed" / "slices_zarr"
V3_VOLUMES = SCRATCH / "data" / "processed" / "v3_index_backup" / "volumes.csv"
BHSD_ZIP = DATASETS / "bhsd" / "label_192.zip"
SEED, TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 42, 0.75, 0.10, 0.15
EMPTY_MIN, SLICE_SIGMA = 0.05, 50.0
SPLIT_RANK = {"test": 0, "val": 1, "train": 2}  # lower wins when merging inherited splits

# Entire families stay out of the pretraining index (downstream still gets train/val/test).
HELD_OUT = {"cq500", "isles22", "atlas", "wmh", "isles24", "instance", "aisd", "brainmetshare",
            "msd_braintumor"}


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
    return ds


def contrast_of(dataset: str, case_id: str) -> str:
    for c in CONTRAST_SUFFIXES:
        if dataset.endswith("_" + c) or case_id.endswith("-" + c):
            return c
    if dataset.endswith("_t1"):
        return "t1"
    if dataset.endswith("_t2"):
        return "t2"
    if dataset.endswith(("_dwi", "_adc", "_flair", "_ncct", "_cta")):
        return dataset.rsplit("_", 1)[1]
    return ""


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


def dedup_key(dataset: str, case_id: str) -> str:
    fam = dataset_family(dataset)
    if fam.startswith("brats"):
        return "brats:" + fam.removeprefix("brats_") + ":" + brats_patient(case_id)
    if fam == "cq500":
        return "cq500:" + case_id
    if fam == "rsna_ich":
        return "rsna:" + case_id
    if fam in ("oasis1", "oasis2"):
        return "oasis:" + oasis_subject(case_id)
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


def inherit_v3(path: Path) -> dict[str, set[str]]:
    """Map our subject key → splits seen in v3 (may be several: OASIS-2 leakage we will collapse)."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["dataset", "volume_id", "split", "modality"])
    found: dict[str, set[str]] = defaultdict(set)
    for r in df.itertuples(index=False):
        case = str(r.volume_id).split(":", 1)[-1]
        found[dedup_key(str(r.dataset), case)].add(str(r.split))
    return found


def gaussian_p(idx: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (idx - mu) / sigma
    pdf = np.exp(-0.5 * z * z)
    return np.clip(pdf / float(np.exp(0.0)), 0.0, 1.0)


def tissue_frac(slc: np.ndarray, modality: str) -> float:
    a = np.asarray(slc, dtype=np.float32).reshape(-1)
    return float((a > (-500 if modality == "ct" else -5.5)).mean())


def _index_one(args):
    """Read one volume, drop empty slices, Bernoulli mid-slab sample. Returns (kept, dropped, stats)."""
    image_path, volume_id_, modality, split, extra = args
    vol = store.read(Path(image_path), "image")
    if vol.ndim == 4:
        vol = vol[:, 0]
    z = vol.shape[0]
    mu = 0.5 * (z - 1)
    rows, empty = [], []
    fracs = []
    for i in range(z):
        frac = tissue_frac(vol[i], modality)
        fracs.append(frac)
        rec = {"volume_id": volume_id_, "z": i, "slice_idx": i, "tissue_frac": frac,
               "split": split, "modality": modality, **extra}
        (empty if frac < EMPTY_MIN else rows).append(rec)
    dropped = list(empty)
    if rows:
        idx = np.array([r["z"] for r in rows], dtype=np.float64)
        p = gaussian_p(idx, mu, SLICE_SIGMA)
        rng = np.random.default_rng(SEED ^ int(hashlib.md5(volume_id_.encode()).hexdigest()[:8], 16))
        keep, gauss = [], []
        for r, pi, u in zip(rows, p, rng.random(len(rows))):
            r = {**r, "gauss_p": float(pi)}
            (keep if u <= pi else gauss).append(r)
        if not keep:
            keep, gauss = [max(rows, key=lambda r: r.get("gauss_p", 0))], []
        dropped.extend({**r, "drop_reason": "gaussian"} for r in gauss)
        rows = keep
        for r in empty:
            r["drop_reason"] = "empty"
    mid = z // 2
    stats = {"n_slices_raw": z, "n_slices_kept": len(rows), "n_slices_empty_dropped": len(empty),
             "n_slices_gauss_dropped": len(dropped) - len(empty),
             "tissue_frac_mid": fracs[mid], "tissue_frac_lo": float(np.mean(fracs[: max(mid // 4, 1)])),
             "tissue_frac_hi": float(np.mean(fracs[-max(mid // 4, 1) :]))}
    return rows, dropped, stats


def index_ssl_volumes(root: Path, ssl_vols: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, list, pd.DataFrame]:
    """Slice-index SSL volumes. Returns (kept train/val slices, dropped rows, per-volume stats)."""
    jobs = [(root / r.image_path, r.volume_id, r.modality, r.split,
             {"dataset": r.dataset, "dataset_family": r.dataset_family, "dedup_key": r.dedup_key,
              "image_path": r.image_path, "vol_idx": i})
            for i, r in enumerate(ssl_vols.itertuples(index=False))]
    kept, dropped, stats_by = [], [], {}
    if not jobs:
        return pd.DataFrame(), [], pd.DataFrame()
    with mp.Pool(workers, maxtasksperchild=32) as pool:
        for k, (rows, drop, stats) in enumerate(pool.imap_unordered(_index_one, jobs), 1):
            kept.extend(rows)
            dropped.extend(drop)
            stats_by[rows[0]["volume_id"] if rows else drop[0]["volume_id"]] = stats
            if k % 500 == 0:
                print(f"[ssl] indexed {k}/{len(jobs)}", flush=True)
    slices = pd.DataFrame(kept)
    slices["pretrain"] = True
    slices = slices[slices["split"].isin(["train", "val"])].copy()
    return slices, dropped, pd.DataFrame.from_dict(stats_by, orient="index")


def load_ok_manifests(root: Path) -> pd.DataFrame:
    rows = {}
    for f in sorted(root.glob("manifest_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            r["case_id"] = str(r["case_id"])
            rows[(r["dataset"], r["case_id"])] = r
    ok = [r for r in rows.values() if r.get("status") == "ok"]
    return pd.DataFrame(ok)


def mask_rel(root: Path, folder: str, case_id: str, name: str) -> str:
    p = store.path_for(root, folder, case_id, name)
    return p.relative_to(root).as_posix() if p.exists() else ""


def folder_of(dataset: str) -> str:
    """BraTS mpMRI contrasts share a cohort folder; FOMO contrasts share ``fomo_<short>``.

    ISLES/WMH keep one folder per sequence (``wmh_flair`` must not collapse to ``wmh``).
    """
    ds = str(dataset)
    if ds.startswith("fomo_"):
        return strip_contrast(ds)
    for c in ("t1n", "t1c", "t2w", "t2f"):
        if ds.endswith("_" + c):
            return ds[: -len(c) - 1]
    return ds


def assign_splits(vols: pd.DataFrame, v3: dict[str, set[str]]) -> pd.DataFrame:
    strata = {k: (m, f) for k, m, f in vols[["dedup_key", "modality", "dataset_family"]].itertuples(index=False, name=None)}
    inherited, fresh = {}, []
    for k in vols["dedup_key"].unique():
        if k in v3:
            inherited[k] = merge_split(v3[k])
        else:
            fresh.append(k)
    inherited_n = len(inherited)
    assigned = stratified_subjects(fresh, strata) if fresh else {}
    assigned.update(inherited)
    vols = vols.copy()
    vols["split"] = vols["dedup_key"].map(assigned)
    print(f"[split] inherited v3 subjects={inherited_n} newly assigned={len(fresh)}", flush=True)
    return pin_official_splits(vols)


def inherit_current_splits(vols: pd.DataFrame, old_vols: pd.DataFrame) -> pd.DataFrame:
    """Keep splits already in this store; only assign keys that are new (AISD/INSTANCE)."""
    old = (old_vols.dropna(subset=["dedup_key", "split"]).drop_duplicates("dedup_key")
           .set_index("dedup_key")["split"].astype(str).to_dict())
    strata = {k: (m, f) for k, m, f in vols[["dedup_key", "modality", "dataset_family"]].itertuples(index=False, name=None)}
    fresh = [k for k in vols["dedup_key"].unique() if k not in old]
    assigned = stratified_subjects(fresh, strata) if fresh else {}
    assigned.update(old)
    vols = vols.copy()
    vols["split"] = vols["dedup_key"].map(assigned)
    print(f"[split] inherited current store subjects={len(old)} newly assigned={len(fresh)}", flush=True)
    return pin_official_splits(vols)


def pin_official_splits(vols: pd.DataFrame) -> pd.DataFrame:
    """AISD published a 52-case test set; do not let stratified assignment reshuffle it."""
    from data.download_aisd import splits_for
    from data.prep.sources import aisd_id

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


def assert_no_leak(vols: pd.DataFrame, slices: pd.DataFrame, bhsd: set[str]) -> None:
    """Hard stop if a subject crosses SSL splits or a BHSD/held-out study is in the SSL index."""
    ssl = slices.copy()
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        sa, sb = set(ssl.loc[ssl.split == a, "dedup_key"]), set(ssl.loc[ssl.split == b, "dedup_key"])
        leak = sa & sb
        assert not leak, f"dedup_key in SSL {a} and {b}: {list(leak)[:5]}"
    ssl_keys = set(ssl["dedup_key"])
    held = set(vols.loc[vols.dataset_family.isin(all_held_out()), "dedup_key"])
    assert not (ssl_keys & held), f"held-out subjects in SSL index: {list(ssl_keys & held)[:5]}"
    bhsd_keys = {"rsna:" + i for i in bhsd}
    assert not (ssl_keys & bhsd_keys), f"BHSD studies in SSL index: {list(ssl_keys & bhsd_keys)[:5]}"
    in_ssl = vols[vols.pretrain & vols.ssl_keep]
    for dk, g in in_ssl.groupby("dedup_key"):
        assert g["split"].nunique() == 1, dk
    # multi-sequence families share a split
    for fam in ("ixi", "isles22", "wmh", "isles24", "brats_gli", "brats_men", "brats_goat", "oasis2"):
        sub = vols[vols.dataset_family == fam]
        bad = sub.groupby("dedup_key")["split"].nunique()
        assert (bad == 1).all(), f"{fam} split conflict: {list(bad[bad > 1].index)[:5]}"
    print("[leak] ok: no subject in two SSL splits; BHSD and held-out families absent from slices.parquet", flush=True)


def build(root: Path, v3_path: Path, workers: int, keep_ssl_index: bool = False) -> None:
    bhsd = bhsd_case_ids()
    print(f"[bhsd] {len(bhsd)} RSNA studies with hemorrhage masks", flush=True)
    man = load_ok_manifests(root)
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
            "folder": folder,
            "image_path": img.relative_to(root).as_posix(),
            "priority": ssl_priority(ds, cid),
            "held_out": held,
            "bhsd": is_bhsd,
            "role": "heldout" if held or is_bhsd else "pretrain_eval",
            "mask_seg": mask_rel(root, folder, cid.removesuffix("-t1n") if ds.startswith("brats") else cid, "seg"),
            "mask_ich": mask_rel(root, folder, cid, "ICH"),
            "mask_brain": mask_rel(root, folder, cid, "brain"),
            "mask_lesion": mask_rel(root, folder, cid, "lesion"),
            "mask_lvo": mask_rel(root, folder, cid, "lvo"),
            "mask_cow": mask_rel(root, folder, cid, "cow"),
        })
    vols = pd.DataFrame(recs)
    old_vols = pd.read_csv(root / "volumes.csv", low_memory=False) if keep_ssl_index and (root / "volumes.csv").exists() else None
    if old_vols is not None:
        vols = inherit_current_splits(vols, old_vols)
    else:
        vols = assign_splits(vols, inherit_v3(v3_path))
    # one SSL volume per (subject, contrast): 2024 GLI beats 2023; OASIS-2 MR1 beats later visits
    vols = vols.sort_values(["dedup_key", "contrast", "priority", "case_id"], ascending=[True, True, False, True])
    ssl_keep = ~vols.duplicated(["dedup_key", "contrast", "modality"], keep="first")
    vols["ssl_keep"] = ssl_keep.to_numpy()
    vols["pretrain"] = (~vols["held_out"]) & (~vols["bhsd"])
    vols["has_mask"] = vols.filter(like="mask_").astype(str).ne("").any(axis=1)

    ssl_vols = vols[vols.pretrain & vols.ssl_keep].copy()
    print(f"[ssl] volumes eligible={len(ssl_vols)} / {len(vols)} (BHSD {int(vols.bhsd.sum())}, held-out {int(vols.held_out.sum())})", flush=True)

    dropped = None
    write_ssl = not keep_ssl_index
    if keep_ssl_index:
        old_vols_path, old_slices_path = root / "volumes.csv", root / "slices.parquet"
        if not old_vols_path.exists() or not old_slices_path.exists():
            raise SystemExit("--keep-ssl-index needs existing volumes.csv and slices.parquet")
        old_vols = pd.read_csv(old_vols_path, low_memory=False)
        for c in ("pretrain", "ssl_keep"):
            if c in old_vols.columns:
                old_vols[c] = old_vols[c].map(lambda x: str(x).lower() in ("true", "1"))
        old_ssl = set(old_vols.loc[old_vols["pretrain"] & old_vols["ssl_keep"], "volume_id"].astype(str))
        new_ssl = set(ssl_vols["volume_id"].astype(str))
        missing, extra = old_ssl - new_ssl, new_ssl - old_ssl
        if missing:
            raise SystemExit(
                f"--keep-ssl-index: existing SSL volumes disappeared (missing={len(missing)}); "
                "re-run without the flag")
        stat_cols = [c for c in old_vols.columns if c.startswith("n_slices") or c.startswith("tissue_frac")]
        vols = vols.merge(old_vols[["volume_id", *stat_cols]].drop_duplicates("volume_id"), on="volume_id", how="left")
        slices = pd.read_parquet(old_slices_path)
        print(f"[ssl] reused {old_slices_path} rows={len(slices)} extra={len(extra)}", flush=True)
        if extra:
            add = ssl_vols[ssl_vols["volume_id"].astype(str).isin(extra)]
            extra_slices, extra_drop, extra_stats = index_ssl_volumes(root, add, workers)
            vols = vols.merge(extra_stats, left_on="volume_id", right_index=True, how="left", suffixes=("", "_new"))
            for c in extra_stats.columns:
                neu = f"{c}_new"
                if neu in vols.columns:
                    vols[c] = vols[neu].combine_first(vols[c]) if c in vols.columns else vols[neu]
                    vols = vols.drop(columns=[neu])
            slices = pd.concat([slices, extra_slices], ignore_index=True)
            dropped = extra_drop
            write_ssl = True
            print(f"[ssl] indexed {len(extra)} new SSL volumes, slices now {len(slices)}", flush=True)
    else:
        slices, dropped, st = index_ssl_volumes(root, ssl_vols, workers)
        vols = vols.merge(st, left_on="volume_id", right_index=True, how="left")

    vols["vol_idx"] = np.arange(len(vols))
    vols["status"] = "ok"
    assert_no_leak(vols, slices, bhsd)

    subj = (vols.groupby("dedup_key", as_index=False)
            .agg(split=("split", "first"), modality=("modality", "first"),
                 dataset_family=("dataset_family", "first"), role=("role", "first"),
                 pretrain=("pretrain", "max"), n_volumes=("volume_id", "nunique"),
                 bhsd=("bhsd", "max")))

    root.mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    pd.Series(sorted(bhsd), name="case_id").to_csv(root / "labels" / "bhsd_studies.csv", index=False)
    vols.to_csv(root / "volumes.csv", index=False)
    subj.to_csv(root / "subjects.csv", index=False)
    if write_ssl:
        slices.to_parquet(root / "slices.parquet", index=False)
        slices.to_csv(root / "slices.csv", index=False)
        slices.to_csv(root / "manifest.csv", index=False)
        if dropped is not None:
            drop_path = root / "empty_slice_removals.csv"
            extra_drop = pd.DataFrame(dropped)
            if keep_ssl_index and drop_path.exists() and len(extra_drop):
                extra_drop = pd.concat([pd.read_csv(drop_path), extra_drop], ignore_index=True)
            extra_drop.to_csv(drop_path, index=False)

    print("\n[volumes] by family × split:")
    print(vols.groupby(["dataset_family", "split"]).size().unstack(fill_value=0).to_string())
    print("\n[ssl slices] by split × modality:")
    print(slices.groupby(["split", "modality"]).size().to_string())
    print(f"\n[ok] volumes={len(vols)} subjects={len(subj)} ssl_slices={len(slices)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--v3-volumes", type=Path, default=V3_VOLUMES)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--keep-ssl-index", action="store_true",
                    help="keep existing SSL slices; index only newly added SSL volumes")
    a = ap.parse_args()
    build(a.root, a.v3_volumes, a.workers, keep_ssl_index=a.keep_ssl_index)


if __name__ == "__main__":
    main()
