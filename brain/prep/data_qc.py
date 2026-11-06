"""Store-wide data QC (2026-11-03, before scaling): measurements per volume and per expert mask, written as parquet
shards under ``<root>/labels/data_qc/``; ``runs/diagnostics/data_qc_report.py`` turns them into flags.

``scans``: one read of a central 16-slice slab per volume (2 of the 8-slice chunks, ~10% of the volume) with its
brain mask and SynthSeg map, for
  * contrast verification (MRI): median intensity of CSF, cortex and white matter (SynthSeg labels), whose ordering
    tells T1 (CSF < GM < WM), T2 (CSF > GM > WM) and FLAIR (CSF dark, GM > WM) apart;
  * CT calibration: air outside the head, brain parenchyma, CSF and bone percentiles in HU;
  * a duplicate fingerprint: the slab inside a fixed MNI-space box, rank-normalised and pooled to 8 x 24 x 24.
``masks``: every expert mask column (``mask_seg``, ``mask_ich``, ``mask_lesion``) of every volume, label by label:
volume (ml), connected components, share outside the brain mask, share in air, and the image inside the mask
against a 3-voxel ring around it (median and spread), read over the mask's own slices only.

    python -m brain.prep.data_qc scans --root STORE --shard 0/8
    python -m brain.prep.data_qc masks --root STORE --shard 0/8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brain.prep import store

SLAB = 16
CSF, GM, WM = (4, 43, 24), (3, 42), (2, 41)            # FreeSurfer ids: ventricles + extra-ventricular CSF, cortex, WM
MASK_COLUMNS = ("mask_seg", "mask_ich", "mask_lesion")


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["scans", "masks"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
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


if __name__ == "__main__":
    main()
