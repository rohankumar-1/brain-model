#!/usr/bin/env python3
"""Preview interval slices from a few preprocessed volumes per dataset.

Reads the per-volume ZipStores from ``data/preprocess.py``:

    <root>/<dataset>/<case>.zip
    <root>/<dataset>/<case>.{ICH,brain}.zip
    <root>/<dataset>/<subject>.seg.zip

One PNG per dataset. Each picked volume is a row of ``plt.imshow`` panels at
evenly spaced axial indices. A mask, when present, is the next row on the
same indices.

    python data/inspect_volumes.py \\
        --root $BRAIN_SCRATCH/data/processed/slices_zarr \\
        --per-dataset 2 --step 16
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
import zarr
from zarr.storage import ZipStore

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("BRAIN_SCRATCH", "/home/roku/orcd/scratch/brain-model"))
DEFAULT_ROOT = SCRATCH / "data" / "processed" / "slices_zarr"

MASK_KINDS = ("seg", "ICH", "brain")
BRATS_CONTRASTS = ("t1n", "t1c", "t2w", "t2f")
SKIP_DIRS = {"labels", "inspect"}


def is_image_zip(path: Path) -> bool:
    name = path.name
    if not name.endswith(".zip") or ".partial." in name:
        return False
    return not any(name.endswith(f".{kind}.zip") for kind in MASK_KINDS)


def subject_stem(stem: str) -> str:
    for contrast in BRATS_CONTRASTS:
        for sep in (f"-{contrast}", f"_{contrast}"):
            if stem.endswith(sep):
                return stem[: -len(sep)]
    return stem


def mask_paths_for(image_zip: Path) -> list[tuple[str, Path]]:
    stem = image_zip.name[: -len(".zip")]
    parent = image_zip.parent
    subject = subject_stem(stem)
    found: list[tuple[str, Path]] = []
    for kind in MASK_KINDS:
        candidates = [parent / f"{stem}.{kind}.zip"]
        if subject != stem:
            candidates.append(parent / f"{subject}.{kind}.zip")
        for cand in candidates:
            if cand.is_file():
                found.append((kind, cand))
                break
    return found


def discover(root: Path, per_dataset: int) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    if not root.is_dir():
        return out
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
        images = sorted(p for p in folder.glob("*.zip") if is_image_zip(p))
        if not images:
            continue
        out[folder.name] = images[:per_dataset]
    return out


def read_slices(path: Path, key: str, indices: list[int]) -> list[np.ndarray]:
    store = ZipStore(path, mode="r")
    try:
        arr = zarr.open_group(store, mode="r")[key]
        depth = int(arr.shape[0])
        slices = []
        for z in indices:
            if z < 0 or z >= depth:
                raise IndexError(f"{path.name} z={z} outside depth {depth}")
            slices.append(np.asarray(arr[z]).squeeze())
        return slices
    finally:
        store.close()


def slice_indices(depth: int, step: int) -> list[int]:
    step = max(1, int(step))
    start = min(step // 2, depth - 1)
    idxs = list(range(start, depth, step))
    return idxs or [depth // 2]


def _row_label(name: str, limit: int = 42) -> str:
    if len(name) <= limit:
        return name
    return name[: limit - 1] + "…"


def render_dataset(
    dataset: str,
    images: list[Path],
    indices: list[int],
    save_path: Path,
) -> None:
    rows: list[tuple[str, list[np.ndarray], str]] = []
    for image in images:
        label = _row_label(image.stem)
        rows.append((label, read_slices(image, "image", indices), "image"))
        for kind, mask_path in mask_paths_for(image):
            rows.append(
                (f"{label}  {kind}", read_slices(mask_path, "mask", indices), "mask")
            )

    n_rows = len(rows)
    n_cols = len(indices)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.15 * n_cols, 2.25 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    for r, (label, slices, kind) in enumerate(rows):
        vmax = 1.0 if kind == "image" else max(1.0, float(np.max(slices)))
        for c, (z, slc) in enumerate(zip(indices, slices)):
            ax = axes[r, c]
            ax.imshow(slc, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
            title = f"z={z}"
            if c == 0:
                title = f"{label}\n{title}"
            ax.set_title(title, fontsize=8, loc="left")
            ax.axis("off")
    fig.suptitle(dataset, fontsize=14)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Processed root (<dataset>/*.zip)")
    p.add_argument("--per-dataset", type=int, default=2)
    p.add_argument("--step", type=int, default=16, help="Axial stride through the 128-slice volume")
    p.add_argument("--depth", type=int, default=128)
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="PNG directory (default: <root>/inspect)",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset folder names (default: all present)",
    )
    args = p.parse_args()

    root = args.root
    if not root.is_dir():
        raise SystemExit(f"No processed volumes under {root}")

    picked = discover(root, max(1, args.per_dataset))
    if args.datasets.strip():
        want = {name.strip() for name in args.datasets.split(",") if name.strip()}
        picked = {k: v for k, v in picked.items() if k in want}
    if not picked:
        raise SystemExit(f"No per-volume image zips under {root}")

    indices = slice_indices(args.depth, args.step)
    save_dir = args.save_dir or (root / "inspect")
    print(f"[inspect] root={root}")
    print(f"[inspect] z={indices}")
    for dataset, images in picked.items():
        dest = save_dir / f"{dataset}.png"
        render_dataset(dataset, images, indices, dest)
        masks = sum(len(mask_paths_for(img)) for img in images)
        print(f"[inspect] {dataset}: {len(images)} volumes, {masks} mask rows → {dest}")


if __name__ == "__main__":
    main()
