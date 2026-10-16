"""Download AISD (acute ischemic stroke NCCT) into the pool and preprocess it there.

Images and masks are the public Google Drive zips from
https://github.com/GriffinLiang/AISD. The release is one folder of 8-bit PNG
slices per case (not NIfTI); ``fetch`` stacks those into ``extracted/nifti``.
Raw files and the processed Zarr store both live under ``$BRAIN_DATASETS/aisd``.
The official 52-case test split is pinned; the rest is 90/10 train/val.

    python -m data.download_aisd --prep
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

from data.prep.sources import AISD_TEST_IDS, DATASETS, aisd_id

IMAGE_ID = "157f9aE3ZhRSdIuIbP2PRG8ub9JJWvMGk"
MASK_ID = "1d08fFpEvK4D6YTKfRlNuv_OlIxigZxl6"
OUT = DATASETS / "aisd"
STORE = OUT / "processed"
# AISD ships display PNGs, no geometry. Typical head NCCT: ~0.5 mm in-plane, 5 mm slices.
PNG_SPACING_XYZ = (0.5, 0.5, 5.0)
# 8-bit is a brain window, not native HU. Map 0..255 onto 0..80 HU (WW=80, WL=40).
PNG_HU_WIDTH = 80.0


def gdrive(file_id: str, out: Path) -> Path:
    if out.exists() and zipfile.is_zipfile(out):
        print(f"[aisd] already have {out} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)
        return out
    import gdown
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[aisd] gdown {file_id} -> {out.name}", flush=True)
    gdown.download(id=file_id, output=str(out), fuzzy=True, resume=True)
    if not out.exists() or not zipfile.is_zipfile(out):
        raise SystemExit(f"{out} is not a zip (Google Drive may have returned an HTML confirm page)")
    print(f"[aisd] wrote {out} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)
    return out


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[aisd] unzip {archive.name} -> {dest}", flush=True)
    subprocess.run(["unzip", "-n", "-q", str(archive), "-d", str(dest)], check=True)


def _png_key(p: Path):
    return (int(p.stem), p.name) if p.stem.isdigit() else (10**9, p.name)


def stack_png_dir(case_dir: Path) -> np.ndarray:
    pngs = sorted(case_dir.glob("*.png"), key=_png_key)
    if not pngs:
        raise FileNotFoundError(f"no PNGs in {case_dir}")
    slices = []
    for p in pngs:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
        slices.append(np.asarray(arr[0] if arr.ndim == 3 else arr))
    return np.stack(slices, axis=0)


def png_to_hu(vol: np.ndarray) -> np.ndarray:
    """Windowed 8-bit NCCT → HU-like: tissue 0..80, extra-cranial air −1024.

    AISD PNGs are display-windowed, not native HU. A brain window maps 0..255 onto
    0..80 HU. Holes-filled foreground keeps ventricular zeros at 0 HU (CSF) instead
    of punching air through the ventricles.
    """
    hu = vol.astype(np.float32) * (PNG_HU_WIDTH / 255.0)
    # Per-slice so ventricles (2D holes) stay CSF; a 3D fill would leave them open at the vault.
    fg = np.stack([ndi.binary_fill_holes(sl > 0) for sl in vol], axis=0)
    hu[~fg] = -1024.0
    return hu


def write_nifti(path: Path, array: np.ndarray, spacing: tuple[float, float, float] = PNG_SPACING_XYZ) -> None:
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(tuple(float(s) for s in spacing))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), useCompression=True)


def _convert_one(image_dir: Path, mask_dir: Path | None, img_out: Path, mask_out: Path | None) -> str:
    if img_out.exists() and (mask_out is None or mask_out.exists()):
        return str(img_out)
    vol = stack_png_dir(image_dir)
    write_nifti(img_out, png_to_hu(vol))
    if mask_dir is not None and mask_out is not None:
        mask = stack_png_dir(mask_dir)
        if mask.shape != vol.shape:
            raise ValueError(f"mask shape {mask.shape} != image {vol.shape} for {image_dir.name}")
        write_nifti(mask_out, mask.astype(np.uint8))
    return str(img_out)


def pngs_to_nifti(extracted: Path, workers: int = 8) -> Path:
    """Stack ``image/<id>/*.png`` (+ masks) into ``extracted/nifti/<id>.nii.gz``."""
    image_root, mask_root, out = extracted / "image", extracted / "mask", extracted / "nifti"
    if not image_root.is_dir():
        return out
    jobs = []
    for case in sorted(p for p in image_root.iterdir() if p.is_dir()):
        cid = aisd_id(case.name)
        md = mask_root / case.name
        if not md.is_dir():
            md = mask_root / cid
        has_mask = md.is_dir()
        jobs.append((case, md if has_mask else None, out / f"{cid}.nii.gz",
                     out / f"{cid}_label.nii.gz" if has_mask else None))
    if not jobs:
        return out
    print(f"[aisd] stacking {len(jobs)} PNG cases -> {out} ({workers} workers)", flush=True)
    n_done = 0
    if workers <= 1 or len(jobs) == 1:
        for job in jobs:
            _convert_one(*job)
            n_done += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_convert_one, *job) for job in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                fut.result()
                n_done += 1
                if i == 1 or i == len(jobs) or i % 50 == 0:
                    print(f"[aisd] nifti {i}/{len(jobs)}", flush=True)
    print(f"[aisd] wrote {n_done} volumes under {out}", flush=True)
    return out


def fetch(workers: int = 8) -> Path:
    dest = OUT / "extracted"
    image, mask = OUT / "image.zip", OUT / "mask.zip"
    gdrive(IMAGE_ID, image)
    gdrive(MASK_ID, mask)
    if not (dest / "image").is_dir() and not any(dest.rglob("*.nii.gz")):
        extract(image, dest)
        extract(mask, dest)
    pngs_to_nifti(dest, workers=workers)
    n = len(list((dest / "nifti").glob("*.nii.gz"))) if (dest / "nifti").is_dir() else len(list(dest.rglob("*.nii.gz")))
    print(f"[aisd] {n} NIfTI files under {dest}", flush=True)
    return dest


def splits_for(case_ids: list[str], seed: int = 42) -> dict[str, str]:
    """Official 52-case test set; 10% of the remainder is val."""
    test = {aisd_id(c) for c in AISD_TEST_IDS}
    rest = [c for c in case_ids if c not in test]
    rng = random.Random(seed)
    rng.shuffle(rest)
    n_val = max(1, int(round(len(rest) * 0.10))) if rest else 0
    out = {c: "test" for c in case_ids if c in test}
    out.update({c: "val" for c in rest[:n_val]})
    out.update({c: "train" for c in rest[n_val:]})
    return out


def write_volumes(store_root: Path = STORE) -> Path:
    """Absolute-path ``volumes.csv`` so downstream can overlay this store on the main table."""
    from data.prep import store as st
    from data.prep.run import read_manifests
    from data.prep.sources import aisd as discover

    man = read_manifests(store_root)
    # jsonl round-trips numeric stems as ints ("72762"); store folders use 7-digit ids.
    ok = {aisd_id(c) for c in man.loc[man.status == "ok", "case_id"]}
    cases = [c for c in discover() if c.case_id in ok]
    split = splits_for([c.case_id for c in cases])
    rows = []
    for c in cases:
        img = st.path_for(store_root, c.folder, c.case_id)
        mask = st.path_for(store_root, c.folder, c.mask_id, "lesion")
        if not img.exists():
            continue
        rows.append({
            "volume_id": f"aisd:{c.case_id}",
            "dedup_key": f"aisd:{c.case_id}",
            "case_id": c.case_id,
            "modality": "ct",
            "dataset": "aisd",
            "dataset_family": "aisd",
            "folder": c.folder,
            "split": split[c.case_id],
            "image_path": str(img.resolve()),
            "mask_lesion": str(mask.resolve()) if mask.exists() else "",
            "has_mask": bool(mask.exists()),
            "held_out": True,
            "status": "ok",
        })
    df = pd.DataFrame(rows)
    store_root.mkdir(parents=True, exist_ok=True)
    path = store_root / "volumes.csv"
    df.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(df.groupby("split").size().to_string() if len(df) else "[aisd] no volumes", flush=True)
    print(f"[aisd] volumes -> {path} ({len(df)})", flush=True)
    return path


def prep(store: Path = STORE, workers: int = 16) -> None:
    cmd = [sys.executable, "-m", "data.prep.run", "--root", str(store), "--datasets", "aisd",
           "--workers", str(workers), "--z-anchor", "auto"]
    print(f"[aisd] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    write_volumes(store)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    root = fetch(workers=a.workers)
    from data.prep import sources
    cases = sources.aisd(root)
    n_mask = sum(1 for c in cases if c.masks)
    print(f"[aisd] {len(cases)} volumes, {n_mask} with infarct masks", flush=True)
    if not cases:
        raise SystemExit(f"no NCCT volumes found under {root}")
    if a.prep:
        prep(STORE, a.workers)


if __name__ == "__main__":
    main()
