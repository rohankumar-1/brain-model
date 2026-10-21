#!/usr/bin/env python3
"""One SynthStrip example per completed family → notebooks/inspect_out/mask_*.png"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.pop("WORLD_SIZE", None)
os.environ.pop("RANK", None)
os.environ.pop("LOCAL_RANK", None)
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path.home() / ".local/lib/python3.12/site-packages")]

import matplotlib.pyplot as plt
import numpy as np

from brain.common.paths import SCRATCH
from brain.common.windowing import PRESETS, asinh_window
from brain.prep import store
from brain.prep.index import dataset_family
from brain.prep.remask import MASK_TAILS, SKIP_DIRS, is_image_zip

STORE = SCRATCH / "data" / "processed" / "slices_zarr"
OUT = ROOT / "notebooks" / "inspect_out"
# Main remask excluded these; their .brain.zip files are still classical.
PENDING_PREFIX = ("fomo_",)
PENDING_EXACT = {"rsna_ich"}


def _win(vol, kind):
    c, w = (PRESETS["ct"]["soft"] if str(kind).startswith("ct") else PRESETS["mri"]["tissue"])
    x = asinh_window(vol, c, w)
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def ortho(vol):
    z, y, x = (s // 2 for s in vol.shape)
    return vol[z], vol[:, y, :], np.rot90(vol[:, :, x])


def pick(folder: Path, n_scan: int = 40):
    cands = []
    for p in sorted(folder.glob("*.zip")):
        if not is_image_zip(p):
            continue
        bp = p.with_name(p.stem + ".brain.zip")
        if not bp.is_file():
            continue
        try:
            battrs = store.read_attrs(bp, "mask")
        except Exception:
            continue
        if battrs.get("brain_method") != "synthstrip":
            continue
        iattrs = {}
        try:
            iattrs = store.read_attrs(p, "image")
        except Exception:
            pass
        frac = float(battrs.get("brain_frac") or 0)
        cands.append((abs(frac - 0.10), p, bp, iattrs, battrs))
        if len(cands) >= n_scan:
            break
    if not cands:
        return None
    cands.sort()
    _, p, bp, iattrs, battrs = cands[0]
    return p, bp, iattrs, battrs


def overlay(gray, mask):
    rgb = np.stack([gray, gray, gray], axis=-1)
    m = mask.astype(bool)
    rgb[m, 0] = np.clip(rgb[m, 0] * 0.55 + 0.45, 0, 1)
    rgb[m, 1] *= 0.55
    rgb[m, 2] *= 0.55
    return rgb


def draw_row(axes, vol, brain, kind):
    g = _win(vol, kind)
    m = brain.astype(bool) if brain is not None else np.zeros(vol.shape, bool)
    ga, gc, gs = ortho(g)
    ma, mc, ms = ortho(m)
    panels = [(ga, ma, "axial"), (gc, mc, "coronal"), (gs, ms, "sagittal")]
    for ax, (im, mk, title) in zip(axes, panels):
        ax.imshow(overlay(im, mk))
        if mk.any():
            ax.contour(mk, levels=[0.5], colors="lime", linewidths=0.7)
        ax.set_title(title, fontsize=8)
        ax.axis("off")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_fam = {}
    for d in sorted(p for p in STORE.iterdir() if p.is_dir() and p.name not in SKIP_DIRS):
        if d.name.startswith(PENDING_PREFIX) or d.name in PENDING_EXACT:
            continue
        fam = dataset_family(d.name)
        if fam in by_fam:
            continue
        hit = pick(d)
        if hit is None:
            continue
        by_fam[fam] = (d.name, *hit)

    print(f"families with synthstrip: {len(by_fam)}")
    rows = []
    for fam in sorted(by_fam):
        folder, path, bp, iattrs, battrs = by_fam[fam]
        vol = np.asarray(store.read(path, "image"), dtype=np.float32).squeeze()
        brain = np.asarray(store.read(bp, "mask"), dtype=np.uint8).squeeze() > 0
        kind = str(iattrs.get("kind") or iattrs.get("modality") or "mri")
        frac = float(battrs.get("brain_frac") or brain.mean())
        ok = battrs.get("brain_ok")
        rows.append((fam, folder, path.stem, kind, frac, ok, vol, brain))
        print(f"  {fam:16s} {folder}/{path.stem}  {kind:8s} frac={frac:.3f} ok={ok}")

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9, 1.55 * n))
    if n == 1:
        axes = np.array([axes])
    for i, (fam, folder, cid, kind, frac, ok, vol, brain) in enumerate(rows):
        draw_row(axes[i], vol, brain, kind)
        axes[i, 0].set_ylabel(
            f"{fam}\n{cid[:22]}\n{kind}  {frac:.2f}",
            fontsize=7, rotation=0, ha="right", va="center",
        )
    fig.suptitle(
        "SynthStrip masks on format-5 volumes (red wash + lime contour). "
        "One typical case per completed family. FOMO / RSNA not remasked yet.",
        fontsize=10,
    )
    fig.tight_layout()
    all_path = OUT / "mask_completed.png"
    fig.savefig(all_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", all_path)

    for fam, folder, cid, kind, frac, ok, vol, brain in rows:
        fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
        draw_row(ax, vol, brain, kind)
        fig.suptitle(f"{fam}  {folder}/{cid}  {kind}  brain_frac={frac:.3f}  ok={ok}", fontsize=9)
        fig.tight_layout()
        p = OUT / f"mask_{fam}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
    print("per-family pngs", n)


if __name__ == "__main__":
    main()
