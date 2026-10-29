"""QC figure: where the crop landed, native-value histogram, and the processed slices under several windows."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from brain.prep.pipeline import Result
from brain.prep.sources import Case
from brain.common.windowing import PRESETS, asinh_window

Z_LEVELS = (24, 44, 64, 84, 104)


def _source_panel(ax, res: Result, ct: bool) -> None:
    src, xyz = res.debug["src"], res.debug["center_xyz"]
    ix, iy, iz = src.TransformPhysicalPointToContinuousIndex(tuple(xyz))
    sl = np.asarray(sitk_view(src)[int(np.clip(round(iz), 0, src.GetSize()[2] - 1))])
    lo, hi = (-100, 300) if ct else np.percentile(sl, [1, 99.5])
    ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi)
    sx, sy = src.GetSpacing()[:2]
    for (cx, cy), color, label in [((ix, iy), "red", "new"), (src.TransformPhysicalPointToContinuousIndex(
            (*res.debug["old_xy"], xyz[2]))[:2], "deepskyblue", "old")]:
        ax.add_patch(Rectangle((cx - 128 / sx, cy - 128 / sy), 256 / sx, 256 / sy, fill=False, ec=color, lw=1.5, label=label))
        ax.plot(cx, cy, "+", color=color, ms=10)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("source slice, 256 mm crop boxes", fontsize=8)


def sitk_view(img):
    import SimpleITK as sitk

    return sitk.GetArrayViewFromImage(img)


def overview(png: Path, case: Case, res: Result) -> None:
    ct = case.modality == "ct"
    presets = PRESETS["ct"] if ct else PRESETS["mri"]
    names = ["brain", "subdural", "bone"] if ct else ["tissue", "wide"]
    zs = list(Z_LEVELS)
    overlays = {**res.masks, **({"brain": res.brain} if res.brain is not None else {})}
    lesion = next((res.masks[n] for n in ("ICH", "seg", "lesion") if n in res.masks), None)
    if lesion is not None and lesion.any():
        zs[2] = int((lesion > 0).sum((1, 2)).argmax())
        zs.sort()
    fig = plt.figure(figsize=(14, 2.6 * (1 + len(names))))
    gs = fig.add_gridspec(1 + len(names), len(zs))
    _source_panel(fig.add_subplot(gs[0, 0:2]), res, ct)

    ax = fig.add_subplot(gs[0, 2:4])
    v = res.vol.astype(np.float32)
    v = v[v > -1023.5] if ct else v[v > -5.5]
    ax.hist(v, bins=120, log=True, color="gray")
    ax.set_title("native values in foreground (" + ("HU" if ct else "robust z") + ")", fontsize=8)

    ax = fig.add_subplot(gs[0, 4])
    ax.axis("off")
    i = res.info
    ax.text(0, 1, f"{case.dataset}/{case.case_id}\nmethod: {i['center_method']}  gap: {i['center_gap_mm']}\n"
                  f"shift: {i['shift_mm']:.1f} mm ({i['shift_x_mm']:.0f}, {i['shift_y_mm']:.0f})\n"
                  f"spacing: {i['spacing_mm']}\nfg: {i['fg_frac']:.2f}  brain in slab: {i['brain_cover']:.2f}\n"
                  f"z_anchor: {i['z_anchor']}  QC FLAG: {i['qc_flag']}\n"
                  f"reg: {i.get('reg_method', 'off')} rot {i.get('reg_rot_deg', 0):.0f} deg scale {i.get('reg_scale', 1):.2f}\n"
                  f"brain: {i.get('brain_method', 'none')} frac {i.get('brain_frac', 0):.3f}", va="top", fontsize=8)

    for r, name in enumerate(names, start=1):
        c, w = presets[name]
        for k, z in enumerate(zs):
            ax = fig.add_subplot(gs[r, k])
            ax.imshow(asinh_window(res.vol[z].astype(np.float32), c, w), cmap="gray", vmin=-2.5, vmax=2.5)
            for mname, color in (("brain", "lime"), ("ICH", "red"), ("seg", "yellow"),
                                 ("lesion", "red"), ("lvo", "cyan"), ("cow", "orange")):
                m = overlays.get(mname)
                if m is not None and m[z].any():
                    ax.contour(m[z] > 0, levels=[0.5], colors=color, linewidths=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name} (c={c:g}, w={w:g})  z={z}", fontsize=7)
    fig.tight_layout()
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=70)
    plt.close(fig)
