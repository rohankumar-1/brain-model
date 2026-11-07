"""Store-wide data QC (2026-11-03): measurements per volume and per expert mask (``scans`` / ``masks``, parquet shards
under ``<root>/labels/data_qc/``), the flags drawn from them (``report``) and the pixel check of the duplicate
candidates (``verify``). ``report`` and ``verify`` write CSVs to ``--out`` (default ``QC_DIR``), which
``brain.prep.decontam`` turns into the benchmark and training exclusion lists.

``scans``: one read of a central 16-slice slab per volume (2 of the 8-slice chunks, ~10% of the volume) with its
brain mask and SynthSeg map, for
  * contrast verification (MRI): median intensity of CSF, cortex and white matter (SynthSeg labels), whose ordering
    tells T1 (CSF < GM < WM), T2 (CSF > GM > WM) and FLAIR (CSF dark, GM > WM) apart;
  * CT calibration: air outside the head, brain parenchyma, CSF and bone percentiles in HU;
  * a duplicate fingerprint: the slab inside a fixed MNI-space box, rank-normalised and pooled to 8 x 24 x 24.
``masks``: every expert mask column (``mask_seg``, ``mask_ich``, ``mask_lesion``) of every volume, label by label:
volume (ml), connected components, share outside the brain mask, share in air, and the image inside the mask
against a 3-voxel ring around it (median and spread), read over the mask's own slices only.

``report`` (CPU; the duplicate search multiplies ~40k x 4.6k matrices):
  contrast   MRI whose tissue ordering (SynthSeg CSF / cortex / white matter medians) contradicts its contrast label:
             T1-like CSF < GM < WM (T1, T1c), T2-like CSF > GM > WM (T2, PD), FLAIR-like CSF < WM < GM.
  ct         CT whose Hounsfield scale is off: brain median outside 15-50 HU, air not near -1000, no bone above 300 HU.
  masks      expert-mask labels that are outliers within their own dataset / column / label: much of the mask in air
             or outside the brain, CT bleed labels mostly outside the blood range, or an image contrast against the
             surrounding ring far from the dataset's typical one (robust z beyond 4).
  duplicates scans whose fingerprint (rank-normalised central slab, contrast-free) nearly matches a scan of another
             subject (dedup_key), within modality and contrast; the threshold is set from pairs already known to be
             the same scan (the cq500 / seg_cq500 copies), and train-vs-held-out matches are listed separately.
``verify`` (2026-11-04): per candidate pair, the Spearman-like correlation (Pearson of within-scan ranks) of the voxel
intensities over a 32-slice central slab, inside the union of the two brain masks (foreground when a scan has none).
The same physical scan registered twice correlates far above two different people; random different-subject pairs of
the same modality / contrast give the null distribution the threshold comes from.

    python -m brain.prep.data_qc scans --root STORE --shard 0/8
    python -m brain.prep.data_qc masks --root STORE --shard 0/8
    python -m brain.prep.data_qc report [--root STORE] [--out DIR]   -> contrast_mismatch, ct_calibration_flags,
                                                                        mask_flags, duplicate_pairs (.csv)
    python -m brain.prep.data_qc verify [--min-sim 0.95] [--null 400]  -> duplicate_verified.csv, duplicate_null.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import PROCESSED_ROOT, REPO_ROOT
from brain.prep import store
from brain.prep.intensity import robust_stats

SLAB = 16
CSF, GM, WM = (4, 43, 24), (3, 42), (2, 41)            # FreeSurfer ids: ventricles + extra-ventricular CSF, cortex, WM
MASK_COLUMNS = ("mask_seg", "mask_ich", "mask_lesion")
QC_DIR = REPO_ROOT / "runs" / "diagnostics" / "data_qc"   # report / verify outputs (the decontamination inputs)
EXPECT = {"t1": "t1-like", "t1c": "t1-like", "t2": "t2-like", "pd": "t2-like", "flair": "flair-like"}
VERIFY_SLAB = 32


def _open(path: Path):
    import zarr
    from zarr.storage import ZipStore

    zs = ZipStore(path, mode="r")
    return zs, zarr.open_group(zs, mode="r")


def _arr(g, name: str, z0: int | None = None, z1: int | None = None) -> np.ndarray | None:
    if name not in set(g.array_keys()):
        return None
    a = g[name]
    sl = slice(z0, z1) if z0 is not None else slice(None)
    x = np.asarray(a[sl])
    if x.ndim == 4:
        x = x[:, 0] if x.shape[1] == 1 else x[..., 0]
    return x


def _pct(x: np.ndarray, q) -> list[float]:
    return np.percentile(x, q).astype(float).tolist() if x.size else [float("nan")] * len(q)


def scan_row(root: Path, r) -> dict:
    out = {"volume_id": r.volume_id}
    zs, g = _open(root / str(r.image_path))
    try:
        n = g["image"].shape[0]
        z0 = max(0, n // 2 - SLAB // 2)
        img = _arr(g, "image", z0, z0 + SLAB).astype(np.float32)
        brain = _arr(g, "brain", z0, z0 + SLAB)
    finally:
        zs.close()
    brain = brain > 0 if brain is not None else None
    lab = None
    sp = store.path_for(root, str(r.folder), str(r.case_id), "synthseg")
    if sp.exists():
        from brain.prep.synthseg import LABELS

        zs, g = _open(sp)
        try:
            idx = _arr(g, "mask", z0, z0 + SLAB)
        finally:
            zs.close()
        if idx is not None and idx.shape == img.shape:
            lab = LABELS[idx.astype(np.intp)]
    ct = str(r.modality).lower().startswith("ct")
    if lab is not None:
        for name, ids in (("csf", CSF), ("gm", GM), ("wm", WM)):
            v = img[np.isin(lab, ids)]
            out[f"{name}_med"], out[f"{name}_n"] = (float(np.median(v)) if v.size else float("nan")), int(v.size)
    if ct:
        head = img > -500
        out["air_p50"] = _pct(img[~head], [50])[0]
        out["bone_p99"] = _pct(img[head], [99])[0]
        if brain is not None and brain.any():
            out["brain_p10"], out["brain_p50"], out["brain_p90"] = _pct(img[brain], [10, 50, 90])
        out["min"], out["max"] = float(img.min()), float(img.max())
    # duplicate fingerprint: rank-normalised (contrast-free) slab, pooled; background stays 0
    fg = img > (-500 if ct else -5.5)
    v = np.zeros_like(img)
    if fg.any():
        x = img[fg]
        v[fg] = (np.argsort(np.argsort(x)) + 1) / x.size
    zc, yc, xc = (s - s % d for s, d in zip(v.shape, (8, 24, 24)))
    if zc and yc and xc:
        fp = v[:zc, :yc, :xc].reshape(8, zc // 8, 24, yc // 24, 24, xc // 24).mean(axis=(1, 3, 5))
        out["fp"] = fp.astype(np.float16).ravel().tobytes()
    out["z0"], out["n_slices"] = z0, n
    return out


def mask_rows(root: Path, r, column: str) -> list[dict]:
    path = root / str(getattr(r, column))
    zs, g = _open(path)
    try:
        m = _arr(g, "mask")
    finally:
        zs.close()
    if m is None:
        return [{"volume_id": r.volume_id, "column": column, "status": "no_array"}]
    zs_, gi = _open(root / str(r.image_path))
    try:
        if gi["image"].shape[0] != m.shape[0]:
            return [{"volume_id": r.volume_id, "column": column, "status": "shape_mismatch"}]
        nz = np.flatnonzero(m.reshape(m.shape[0], -1).any(1))
        if not nz.size:
            return [{"volume_id": r.volume_id, "column": column, "status": "empty"}]
        z0, z1 = max(0, nz[0] - 3), min(m.shape[0], nz[-1] + 4)
        img = _arr(gi, "image", z0, z1).astype(np.float32)
        brain = _arr(gi, "brain", z0, z1)
    finally:
        zs_.close()
    m = m[z0:z1]
    if img.shape != m.shape:
        return [{"volume_id": r.volume_id, "column": column, "status": "shape_mismatch"}]
    from scipy import ndimage

    ct = str(r.modality).lower().startswith("ct")
    rows = []
    for lab in [int(v) for v in np.unique(m) if v > 0]:
        sel = m == lab
        ring = ndimage.binary_dilation(sel, iterations=3) & ~(m > 0)
        _, ncomp = ndimage.label(sel)
        x, y = img[sel], img[ring]
        row = {"volume_id": r.volume_id, "column": column, "label": lab, "status": "ok",
               "ml": float(sel.sum()) / 1000.0, "n_comp": int(ncomp),
               "in_med": float(np.median(x)), "in_iqr": float(np.subtract(*np.percentile(x, [75, 25]))),
               "ring_med": float(np.median(y)) if y.size else float("nan"),
               "air_frac": float(np.mean(x < (-200 if ct else -5.5)))}
        if brain is not None:
            row["out_brain_frac"] = float(1 - np.mean(brain[sel] > 0))
        if ct:
            row["blood_frac"] = float(np.mean((x > 40) & (x < 90)))
        rows.append(row)
    return rows


def measure(a) -> None:
    k, n = (int(x) for x in a.shard.split("/"))
    v = pd.read_csv(a.root / "volumes.csv", low_memory=False).drop_duplicates("image_path")
    out_dir = a.root / "labels" / "data_qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, errors = [], 0
    if a.what == "scans":
        todo = v.iloc[k::n]
        todo = todo.head(a.limit) if a.limit else todo
        for r in todo.itertuples():
            try:
                rows.append(scan_row(a.root, r))
            except Exception as e:                              # one unreadable zip must not stop the shard
                rows.append({"volume_id": r.volume_id, "error": str(e)[:200]})
                errors += 1
    else:
        cols = [c for c in MASK_COLUMNS if c in v]
        todo = v[v[cols].notna().any(axis=1)].iloc[k::n]
        todo = todo.head(a.limit) if a.limit else todo
        for r in todo.itertuples():
            for c in cols:
                if pd.isna(getattr(r, c)):
                    continue
                try:
                    rows += mask_rows(a.root, r, c)
                except Exception as e:
                    rows.append({"volume_id": r.volume_id, "column": c, "status": "error", "error": str(e)[:200]})
                    errors += 1
    out = out_dir / f"{a.what}_{k:02d}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    print(f"[data_qc] {a.what} shard {k}/{n}: {len(rows)} rows ({errors} errors) -> {out}", flush=True)


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


# ---- report: flags from the measurements --------------------------------------------------------------------------

def load_measurements(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = ["volume_id", "dataset", "dataset_family", "modality", "contrast", "contrast_std", "dedup_key", "split",
            "held_out", "pretrain"]
    v = pd.read_csv(root / "volumes.csv", low_memory=False, usecols=cols).drop_duplicates("volume_id")
    s = pd.concat([pd.read_parquet(f) for f in sorted((root / "labels/data_qc").glob("scans_*.parquet"))])
    m = pd.concat([pd.read_parquet(f) for f in sorted((root / "labels/data_qc").glob("masks_*.parquet"))])
    return v, s.merge(v, on="volume_id", how="left"), m.merge(v, on="volume_id", how="left")


def contrast_flags(s: pd.DataFrame) -> pd.DataFrame:
    mri = s[~s["modality"].astype(str).str.lower().str.startswith("ct") & s["csf_med"].notna()].copy()
    c, g, w = mri["csf_med"], mri["gm_med"], mri["wm_med"]
    mri["seen"] = np.select([(c < g) & (g < w), (c > g) & (g > w), (c < w) & (w < g)],
                            ["t1-like", "t2-like", "flair-like"], "other")
    mri["label"] = mri["contrast_std"].fillna("unlabelled")
    mri["expected"] = mri["label"].map(EXPECT).fillna("?")
    mri["mismatch"] = (mri["expected"] != "?") & (mri["seen"] != mri["expected"])
    return mri


def ct_flags(s: pd.DataFrame) -> pd.DataFrame:
    c = s[s["modality"].astype(str).str.lower().str.startswith("ct")].copy()
    c["brain_off"] = ~c["brain_p50"].between(15, 50)
    c["air_off"] = c["air_p50"].notna() & ~c["air_p50"].between(-1100, -900)
    c["no_bone"] = c["bone_p99"] < 300
    c["flag"] = c["brain_off"] | c["air_off"] | c["no_bone"]
    return c


def _robust_z(x: pd.Series) -> pd.Series:
    med, mad = robust_stats(x.dropna().to_numpy())
    return (x - med) / mad if mad > 0 else x * 0.0


def mask_flags(m: pd.DataFrame) -> pd.DataFrame:
    ok = m[m["status"] == "ok"].copy()
    ok["contrast_gap"] = ok["in_med"] - ok["ring_med"]
    key = ["dataset", "column", "label"]
    ok["gap_z"] = ok.groupby(key)["contrast_gap"].transform(_robust_z)
    ok["air_bad"] = ok["air_frac"] > 0.10
    ok["out_bad"] = ok.groupby(key)["out_brain_frac"].transform(lambda x: x > max(0.3, x.quantile(0.75) + 3 * (
        x.quantile(0.75) - x.quantile(0.25))))
    ok["blood_bad"] = ok["blood_frac"].notna() & (ok["blood_frac"] < 0.10) & ok["column"].eq("mask_ich")
    ok["gap_bad"] = ok["gap_z"].abs() > 4
    ok["flag"] = ok[["air_bad", "out_bad", "blood_bad", "gap_bad"]].any(axis=1)
    return pd.concat([ok, m[m["status"] != "ok"]])


def duplicate_pairs(s: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    import torch

    s = s[s["fp"].notna()].copy()
    s["grp"] = np.where(s["modality"].astype(str).str.lower().str.startswith("ct"), "ct",
                        "mri_" + s["contrast_std"].fillna("unlabelled").astype(str))
    torch.set_num_threads(8)
    rows = []
    for grp, g in s.groupby("grp"):
        if len(g) < 2:
            continue
        x = torch.from_numpy(np.stack([np.frombuffer(b, dtype=np.float16) for b in g["fp"]]).astype(np.float32))
        x = x - x.mean(0, keepdim=True)                       # the template every registered brain shares
        x = torch.nn.functional.normalize(x, dim=1)
        key = g["dedup_key"].fillna(g["volume_id"]).astype(str).to_numpy()
        ids = g["volume_id"].to_numpy()
        for i0 in range(0, len(g), 2048):
            sim = x[i0:i0 + 2048] @ x.T
            sim[torch.arange(sim.shape[0]), torch.arange(i0, i0 + sim.shape[0])] = -1
            top, idx = sim.topk(min(6, len(g) - 1), dim=1)
            for a, (tv, ti) in enumerate(zip(top.tolist(), idx.tolist())):
                for t, j in zip(tv, ti):
                    if t < 0.9:
                        break
                    if key[i0 + a] != key[j] and ids[i0 + a] < ids[j]:
                        rows.append({"grp": grp, "a": ids[i0 + a], "b": ids[j], "sim": t})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    meta = v.set_index("volume_id")[["dataset", "split", "held_out"]]
    d = d.join(meta.add_suffix("_a"), on="a").join(meta.add_suffix("_b"), on="b")
    d["cross_split"] = (d["split_a"] != d["split_b"]) | (_is_true(d["held_out_a"]) != _is_true(d["held_out_b"]))
    return d.sort_values("sim", ascending=False)


def report(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    v, s, m = load_measurements(root)
    c = contrast_flags(s)
    c[c["mismatch"]][["volume_id", "dataset", "label", "seen", "csf_med", "gm_med", "wm_med"]].to_csv(
        out / "contrast_mismatch.csv", index=False)
    t = c.groupby(["dataset", "label"]).agg(n=("mismatch", "size"), mismatch=("mismatch", "mean"),
                                           seen=("seen", lambda x: x.value_counts().idxmax()))
    print("== contrast: datasets with >5% of labelled scans contradicting their label ==")
    print(t[(t["mismatch"] > 0.05) & (t["n"] >= 10)].round(3).sort_values("mismatch", ascending=False).to_string())
    print(f"   total {int(c['mismatch'].sum())} of {len(c)} checked MRI; unlabelled MRI read as: "
          f"{c[c['label'] == 'unlabelled']['seen'].value_counts().to_dict()}")
    k = ct_flags(s)
    k[k["flag"]][["volume_id", "dataset", "brain_p50", "air_p50", "bone_p99", "brain_off", "air_off", "no_bone"]].to_csv(
        out / "ct_calibration_flags.csv", index=False)
    print("\n== CT calibration flags by dataset ==")
    print(k.groupby("dataset")[["brain_off", "air_off", "no_bone", "flag"]].mean().round(3)
          .assign(n=k.groupby("dataset").size()).query("flag > 0").to_string())
    mk = mask_flags(m)
    mk[mk["flag"].fillna(False) | (mk["status"] != "ok")].to_csv(out / "mask_flags.csv", index=False)
    print("\n== expert masks: flagged share by dataset / column ==")
    agg = mk.groupby(["dataset", "column"]).agg(n=("volume_id", "nunique"), flagged=("flag", "mean"),
                                                air=("air_bad", "mean"), out=("out_bad", "mean"),
                                                blood=("blood_bad", "mean"), gap=("gap_bad", "mean"))
    print(agg[agg["flagged"] > 0].round(3).sort_values("flagged", ascending=False).head(30).to_string())
    print(f"   non-ok mask rows: {mk[mk['status'] != 'ok']['status'].value_counts().to_dict()}")
    d = duplicate_pairs(s, v)
    d.to_csv(out / "duplicate_pairs.csv", index=False)
    known = d[(d["dataset_a"].str.startswith("cq500") & d["dataset_b"].str.startswith("seg_cq500"))
              | (d["dataset_b"].str.startswith("cq500") & d["dataset_a"].str.startswith("seg_cq500"))] if len(d) else d
    print(f"\n== duplicates (cosine >= 0.9, different dedup_key): {len(d)} pairs; known cq500/seg_cq500 copies among "
          f"them: {len(known)} (similarity min {known['sim'].min() if len(known) else float('nan'):.3f})")
    if len(d):
        for thr in (0.95, 0.98, 0.99):
            x = d[d["sim"] >= thr]
            print(f"   >= {thr}: {len(x)} pairs, {int(x['cross_split'].sum())} across splits / held-out; "
                  f"top dataset pairs {x.groupby(['dataset_a', 'dataset_b']).size().nlargest(6).to_dict()}")


# ---- verify: pixel check of the duplicate candidates --------------------------------------------------------------

def _slab(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    zs, g = _open(path)
    try:
        n = g["image"].shape[0]
        z0 = max(0, n // 2 - VERIFY_SLAB // 2)
        img = _arr(g, "image", z0, z0 + VERIFY_SLAB).astype(np.float32)
        brain = _arr(g, "brain", z0, z0 + VERIFY_SLAB)
    finally:
        zs.close()
    return img, (brain > 0 if brain is not None else None)


def pair_corr(root: Path, a: str, b: str, paths: dict) -> float:
    """Rank correlation of two volumes' central slabs inside the union of their brain masks (NaN when the slabs
    differ in shape or share fewer than 1,000 voxels)."""
    ia, ba = _slab(root / paths[a])
    ib, bb = _slab(root / paths[b])
    if ia.shape != ib.shape:
        return float("nan")
    fg = lambda i: i > (-500 if i.min() < -100 else -5.5)  # noqa: E731
    m = (ba if ba is not None else fg(ia)) | (bb if bb is not None else fg(ib))
    if m.sum() < 1000:
        return float("nan")
    x, y = ia[m], ib[m]
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def verify(root: Path, qc_dir: Path, min_sim: float = 0.95, n_null: int = 400) -> None:
    v = pd.read_csv(root / "volumes.csv", low_memory=False,
                    usecols=["volume_id", "image_path", "modality", "contrast_std", "dedup_key", "dataset"])
    v = v.drop_duplicates("volume_id")
    paths = dict(zip(v["volume_id"], v["image_path"]))
    d = pd.read_csv(qc_dir / "duplicate_pairs.csv")
    cand = d[(d["sim"] >= min_sim) | ((d["sim"] >= 0.9) & (_is_true(d["held_out_a"]) != _is_true(d["held_out_b"])))].copy()
    cand["r"] = [pair_corr(root, x, y, paths) for x, y in zip(cand["a"], cand["b"])]
    # null: random pairs of different subjects within the same modality / contrast groups the candidates come from
    rng = np.random.default_rng(0)
    v["grp"] = np.where(v["modality"].astype(str).str.lower().str.startswith("ct"), "ct",
                        "mri_" + v["contrast_std"].fillna("unlabelled").astype(str))
    null = []
    for grp, k in cand["grp"].value_counts().items():
        pool = v[v["grp"] == grp]
        n = max(10, int(n_null * k / len(cand)))
        for _ in range(n):
            x, y = pool.sample(2, random_state=int(rng.integers(1 << 31)))["volume_id"].tolist()
            if pool.set_index("volume_id").loc[x, "dedup_key"] != pool.set_index("volume_id").loc[y, "dedup_key"]:
                null.append({"grp": grp, "a": x, "b": y, "r": pair_corr(root, x, y, paths)})
    null = pd.DataFrame(null)
    thr = float(np.nanpercentile(null["r"], 99.9)) if len(null) else 0.95
    cand["verified"] = cand["r"] > max(thr + 0.02, 0.9)
    cand.to_csv(qc_dir / "duplicate_verified.csv", index=False)
    null.to_csv(qc_dir / "duplicate_null.csv", index=False)
    print(f"null (random different-subject pairs, n={len(null)}): r median {null['r'].median():.3f}, "
          f"99.9th pct {thr:.3f}, max {null['r'].max():.3f}")
    print(f"candidates {len(cand)}: verified {int(cand['verified'].sum())} (r > {max(thr + 0.02, 0.9):.3f})")
    x = cand[cand["verified"]]
    print("verified by dataset pair:")
    print(x.groupby(["dataset_a", "dataset_b"]).size().sort_values(ascending=False).head(25).to_string())
    print("cross held-out verified:", int((_is_true(x["held_out_a"]) != _is_true(x["held_out_b"])).sum()),
          "| cross split (trained data):", int(x["cross_split"].sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    for name in ("scans", "masks"):
        p = sub.add_parser(name, help=f"measure {name} into <root>/labels/data_qc/{name}_<shard>.parquet")
        p.add_argument("--root", type=Path, default=PROCESSED_ROOT)
        p.add_argument("--shard", default="0/1")
        p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("report", help="flags from the measurement shards")
    p.add_argument("--root", type=Path, default=PROCESSED_ROOT)
    p.add_argument("--out", type=Path, default=QC_DIR)
    p = sub.add_parser("verify", help="pixel check of report's duplicate_pairs.csv")
    p.add_argument("--root", type=Path, default=PROCESSED_ROOT)
    p.add_argument("--qc-dir", type=Path, default=QC_DIR)
    p.add_argument("--min-sim", type=float, default=0.95)
    p.add_argument("--null", type=int, default=400)
    a = ap.parse_args()
    if a.what == "report":
        report(a.root, a.out)
    elif a.what == "verify":
        verify(a.root, a.qc_dir, a.min_sim, a.null)
    else:
        measure(a)


if __name__ == "__main__":
    main()
