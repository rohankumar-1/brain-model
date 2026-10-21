#!/usr/bin/env python3
"""Same cases as render_masks.py, but overlay a derived skull / scalp-skull shell.

We do not store a skull mask. CT: cortical bone (HU >= 300). MRI: head envelope minus
the SynthStrip brain (scalp + calvarium + extra-axial). Stripped releases have almost none.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.pop("WORLD_SIZE", None)
os.environ.pop("RANK", None)
os.environ.pop("LOCAL_RANK", None)
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(Path.home() / ".local/lib/python3.12/site-packages")]

import matplotlib.pyplot as plt
import numpy as np
from render_masks import OUT, PENDING_EXACT, PENDING_PREFIX, STORE, ortho, pick

from brain.common.windowing import PRESETS, asinh_window
from brain.prep import store
from brain.prep.brainmask import _fill_axial, _largest
from brain.prep.index import dataset_family
from brain.prep.remask import SKIP_DIRS


def _win(vol, kind):
    c, w = (PRESETS["ct"]["soft"] if str(kind).startswith("ct") else PRESETS["mri"]["tissue"])
    x = asinh_window(vol, c, w)
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def head_mask(vol: np.ndarray, kind: str) -> np.ndarray:
    fg = vol > (-500.0 if str(kind).startswith("ct") else -5.5)
    return _fill_axial(_largest(fg))


def skull_mask(vol: np.ndarray, brain: np.ndarray, kind: str) -> tuple[np.ndarray, str]:
    """Return (mask, label). CT uses bone HU; MRI uses head minus brain."""
    brain = brain.astype(bool)
    head = head_mask(vol, kind)
    if str(kind).startswith("ct"):
        bone = (vol >= 300.0) & head
        if float(bone.mean()) >= 0.005:
            return bone, "CT bone HU≥300"
        return head & ~brain, "CT head\\brain (no HU≥300)"
    shell = head & ~brain
    return shell, "MRI head\\brain"


def overlay_cyan(gray, mask):
    rgb = np.stack([gray, gray, gray], axis=-1)
    m = mask.astype(bool)
    rgb[m, 0] *= 0.45
    rgb[m, 1] = np.clip(rgb[m, 1] * 0.45 + 0.55, 0, 1)
    rgb[m, 2] = np.clip(rgb[m, 2] * 0.45 + 0.55, 0, 1)
    return rgb


def draw_row(axes, vol, skull, kind):
    g = _win(vol, kind)
    ga, gc, gs = ortho(g)
    sa, sc, ss = ortho(skull.astype(bool))
    for ax, im, mk, title in zip(axes, (ga, gc, gs), (sa, sc, ss), ("axial", "coronal", "sagittal")):
        ax.imshow(overlay_cyan(im, mk))
        if mk.any():
            ax.contour(mk, levels=[0.5], colors="yellow", linewidths=0.6)
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

    rows = []
    for fam in sorted(by_fam):
        folder, path, bp, iattrs, battrs = by_fam[fam]
        vol = np.asarray(store.read(path, "image"), dtype=np.float32).squeeze()
        brain = np.asarray(store.read(bp, "mask"), dtype=np.uint8).squeeze() > 0
        kind = str(iattrs.get("kind") or iattrs.get("modality") or "mri")
        skull, label = skull_mask(vol, brain, kind)
        frac = float(skull.mean())
        rows.append((fam, folder, path.stem, kind, label, frac, vol, skull))
        print(f"  {fam:16s} {folder}/{path.stem}  {label:20s} frac={frac:.3f}")

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9, 1.55 * n))
    if n == 1:
        axes = np.array([axes])
    for i, (fam, folder, cid, kind, label, frac, vol, skull) in enumerate(rows):
        draw_row(axes[i], vol, skull, kind)
        axes[i, 0].set_ylabel(
            f"{fam}\n{cid[:22]}\n{frac:.2f}",
            fontsize=7, rotation=0, ha="right", va="center",
        )
    fig.suptitle(
        "Derived skull masks on the same cases. Cyan + yellow contour. "
        "CT: HU≥300 bone. MRI: head envelope minus SynthStrip brain (scalp+calvarium). "
        "Not stored — computed from the volume.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT / "skull_completed.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    for fam, folder, cid, kind, label, frac, vol, skull in rows:
        fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
        draw_row(ax, vol, skull, kind)
        fig.suptitle(f"{fam}  {folder}/{cid}  {label}  frac={frac:.3f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"skull_{fam}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print("wrote", n, "skull panels")


if __name__ == "__main__":
    main()
