"""Clinical measures from a SynthSeg label map (and the image), on the store's 1 mm RAS grid (arrays (Z, Y, X);
x -> right, y -> anterior, z -> superior). All are measured inside the image (ratios, densities, distances between
structures of the same scan), so rotation, flips and registration scale do not distort them.

    python -m brain.multitask.measures --workers 48     # every SynthSeg-labelled volume -> labels/synthseg/measures.csv

``midline_shift``  septum pellucidum's distance from the ideal midline at the foramen of Monro, as radiologists measure
                   it. Ideal midline: the line through the interhemispheric boundary at the brain's anterior and
                   posterior poles (falx attachments; mass effect barely moves them), from SynthSeg's hemisphere
                   labels. Septum: per row, the midpoint between the lateral ventricles' medial walls, on axial slices
                   where the third ventricle and both lateral ventricles are present. Template-grid mm.
``evans_index``    maximal frontal-horn width / maximal brain width on the same axial slices (> 0.3: ventriculomegaly).
``ct_gray_white``  CT: median HU of the (eroded) cerebral cortex / median HU of the (eroded) cerebral white matter.
``ct_calcification``  CT: share of (eroded) brain-mask voxels above 100 HU (basal ganglia / pineal / choroid / vascular).
``ct_skull_hu``    CT: median HU of bone (> 300 HU) outside the brain mask: opportunistic bone density.
"""

from __future__ import annotations

import numpy as np

# FreeSurfer ids (SynthSeg)
L_HEMI, R_HEMI = (2, 3), (41, 42)           # cerebral white matter + cortex
L_LV, R_LV, V3 = (4,), (43,), (14,)


def _mask(seg: np.ndarray, ids, labels: np.ndarray | None) -> np.ndarray:
    """``seg`` holds FreeSurfer ids, or indices into ``labels`` (the stored SynthSeg zips)."""
    if labels is not None:
        ids = np.searchsorted(labels, ids)
    return np.isin(seg, ids)


def _pole_boundary(lh: np.ndarray, rh: np.ndarray, rows: np.ndarray) -> tuple[float, float] | None:
    """Interhemispheric boundary (x, y) over ``rows`` (y indices): the gap between the left hemisphere's right-most
    and the right hemisphere's left-most voxel, per row, medianed."""
    xs, ys = [], []
    for y in rows:
        lx, rx = np.nonzero(lh[:, y, :])[1], np.nonzero(rh[:, y, :])[1]
        if lx.size and rx.size:
            xs.append(0.5 * (lx.max() + rx.min()))
            ys.append(y)
    if len(xs) < 3:
        return None
    return float(np.median(xs)), float(np.median(ys))


def midline_shift(seg: np.ndarray, labels: np.ndarray | None = None, pole_mm: int = 12) -> dict:
    """``seg`` (Z, Y, X). Returns ``shift_mm`` (absolute), ``signed_mm`` (+ = septum displaced to the right) and the
    pieces, or NaNs when the anatomy needed is missing (no third ventricle / lateral ventricles in view)."""
    nan = {"shift_mm": float("nan"), "signed_mm": float("nan"), "n_rows": 0}
    lv_l, lv_r, v3 = _mask(seg, L_LV, labels), _mask(seg, R_LV, labels), _mask(seg, V3, labels)
    lh, rh = _mask(seg, L_HEMI, labels), _mask(seg, R_HEMI, labels)
    zs = [z for z in np.nonzero(v3.any(axis=(1, 2)))[0] if lv_l[z].any() and lv_r[z].any()]
    if not zs:
        return nan
    hemi = lh | rh
    ys = np.nonzero(hemi[zs].any(axis=(0, 2)))[0]
    if ys.size < 2 * pole_mm:
        return nan
    ant = _pole_boundary(lh[zs], rh[zs], np.arange(ys.max() - pole_mm, ys.max() + 1))
    post = _pole_boundary(lh[zs], rh[zs], np.arange(ys.min(), ys.min() + pole_mm + 1))
    if ant is None or post is None or ant[1] == post[1]:
        return nan
    (xa, ya), (xp, yp) = ant, post
    slope = (xa - xp) / (ya - yp)                      # x of the ideal line as a function of y
    d = []
    for z in zs:
        for y in np.nonzero(lv_l[z].any(axis=1) & lv_r[z].any(axis=1))[0]:
            lx, rx = np.nonzero(lv_l[z, y])[0], np.nonzero(lv_r[z, y])[0]
            if lx.max() < rx.min():                    # left ventricle's medial wall left of the right one's
                septum = 0.5 * (lx.max() + rx.min())
                d.append(septum - (xp + slope * (y - yp)))
    if not d:
        return nan
    # perpendicular distance: x offset times cos of the line's angle to the y axis
    signed = float(np.median(d)) / float(np.sqrt(1.0 + slope**2))
    return {"shift_mm": abs(signed), "signed_mm": signed, "n_rows": len(d), "line": (xp, yp, slope)}


def symmetry_plane(shell: np.ndarray, coarse: int = 2) -> tuple[float, float, float] | None:
    """Mid-sagittal plane ``x = a + b*y + c*z`` (grid voxels) about which ``shell`` (Z, Y, X) bool -- the skull on CT,
    the head outline on MRI: rigid, bilaterally symmetric, not moved by intracranial mass effect -- is most
    mirror-symmetric. Small-angle reflection ``x -> 2 (a + b y + c z) - x``; the score is the overlap of a smoothed
    shell with its reflection, searched coarse-to-fine over offset (+-20 mm), yaw and roll (+-15 deg)."""
    from scipy.ndimage import gaussian_filter, map_coordinates

    if shell.sum() < 2000:
        return None
    s = gaussian_filter(shell[::coarse, ::coarse, ::coarse].astype(np.float32), 1.0)
    z, y, x = np.nonzero(s > 0.2)
    w = s[z, y, x]
    xc = (shell.shape[2] - 1) / 2 / coarse

    def score(a, b, c):
        xr = 2 * (a + b * y + c * z) - x
        return float((w * map_coordinates(s, [z, y, xr], order=1, mode="constant")).sum())

    best = (xc, 0.0, 0.0)
    for span_a, span_t, n in ((10.0, 0.27, 9), (2.0, 0.06, 7), (0.5, 0.015, 5)):    # +-20 mm / +-15 deg, refined
        a0, b0, c0 = best
        grid = [(a0 + da, b0 + db, c0 + dc) for da in np.linspace(-span_a, span_a, n)
                for db in np.linspace(-span_t, span_t, n) for dc in np.linspace(-span_t, span_t, n)]
        best = max(grid, key=lambda p: score(*p))
    a, b, c = best
    return a * coarse, b, c                           # b, c are slopes: unchanged by the coarse grid


def midline_shift_plane(seg: np.ndarray, shell: np.ndarray, labels: np.ndarray | None = None) -> dict:
    """Signed perpendicular displacement (mm, + = toward the patient's right) of midline structures from the shell's
    symmetry plane. Grid indices run toward the right (x) and anterior (y).

    * ``shift_mm``: the septum pellucidum (midpoint between the lateral ventricles' medial walls), medianed over every
      row with both ventricles on slices with the third ventricle -- the first version, kept for comparison;
    * ``frontal_mm``: the same restricted to the frontal horns (rows anterior to the third ventricle's front, i.e.
      the foramen of Monro), where radiologists measure: median over all such rows;
    * ``frontal_max_mm``: the largest per-slice frontal-horn median (radiologists report the maximal shift);
    * ``v3_mm``: the third ventricle's centroid, medianed over its slices;
    * ``max_mm``: the larger of ``frontal_max_mm`` and ``v3_mm`` (absolute values)."""
    keys = ("shift_mm", "signed_mm", "frontal_mm", "frontal_max_mm", "v3_mm", "max_mm")
    nan = {**{k: float("nan") for k in keys}, "n_rows": 0, "n_frontal": 0}
    plane = symmetry_plane(shell)
    if plane is None:
        return nan
    a, b, c = plane
    norm = float(np.sqrt(1.0 + b * b + c * c))
    lv_l, lv_r, v3 = _mask(seg, L_LV, labels), _mask(seg, R_LV, labels), _mask(seg, V3, labels)
    zs = np.nonzero(v3.any(axis=(1, 2)))[0]
    if not len(zs):
        return {**nan, "plane": plane}
    v3_front = int(np.nonzero(v3.any(axis=(0, 2)))[0].max())          # most anterior third-ventricle row
    rows, v3d = [], []
    for z in zs:
        vy, vx = np.nonzero(v3[z])
        v3d.append((vx.mean() - (a + b * vy.mean() + c * z)) / norm)
        for y in np.nonzero(lv_l[z].any(axis=1) & lv_r[z].any(axis=1))[0]:
            lx, rx = np.nonzero(lv_l[z, y])[0], np.nonzero(lv_r[z, y])[0]
            if lx.max() < rx.min():
                rows.append((z, y, (0.5 * (lx.max() + rx.min()) - (a + b * y + c * z)) / norm))
    out = {**nan, "plane": plane, "n_rows": len(rows), "v3_mm": float(np.median(v3d))}
    if rows:
        r = np.array(rows)
        out["signed_mm"] = float(np.median(r[:, 2]))
        out["shift_mm"] = abs(out["signed_mm"])
        fr = r[r[:, 1] > v3_front]
        out["n_frontal"] = len(fr)
        if len(fr):
            out["frontal_mm"] = float(np.median(fr[:, 2]))
            per_slice = [np.median(fr[fr[:, 0] == z, 2]) for z in np.unique(fr[:, 0])]
            out["frontal_max_mm"] = float(max(per_slice, key=abs))
    out["max_mm"] = float(np.nanmax([abs(out["frontal_max_mm"]), abs(out["v3_mm"])]))
    return out


def head_shell(img: np.ndarray, modality: str) -> np.ndarray:
    """The rigid symmetric reference: skull (> 300 HU) on CT, the head outline (foreground boundary) on MRI."""
    from scipy.ndimage import binary_erosion, binary_fill_holes

    if str(modality).lower().startswith("ct"):
        return img > 300
    head = binary_fill_holes(img > -5.5)
    return head & ~binary_erosion(head, iterations=3)


def evans_index(seg: np.ndarray, labels: np.ndarray | None = None) -> float:
    """Frontal horns: the lateral ventricles anterior to their own centroid. Widths are left-most to right-most voxel
    in a row; both maxima are taken over the axial slices holding frontal horn."""
    lv = _mask(seg, L_LV + R_LV, labels)
    hemi = _mask(seg, L_HEMI + R_HEMI, labels)
    if lv.sum() < 100:
        return float("nan")
    yc = np.nonzero(lv)[1].mean()
    fh = lv.copy()
    fh[:, : int(np.ceil(yc)), :] = False
    zs = np.nonzero(fh.any(axis=(1, 2)))[0]
    if not zs.size:
        return float("nan")

    def width(m):
        w = 0
        for z in zs:
            for y in np.nonzero(m[z].any(axis=1))[0]:
                xs = np.nonzero(m[z, y])[0]
                w = max(w, xs.max() - xs.min() + 1)
        return w

    bw = width(hemi)
    return float(width(fh) / bw) if bw else float("nan")


def _eroded(m: np.ndarray, it: int = 1) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    return binary_erosion(m, iterations=it)


def ct_measures(img: np.ndarray, brain: np.ndarray, seg: np.ndarray | None, labels: np.ndarray | None = None) -> dict:
    """CT image in HU (Z, Y, X), its brain mask and (optionally) SynthSeg labels."""
    b = _eroded(np.asarray(brain) > 0, 2)
    out = {"ct_gray_white": float("nan"), "ct_calcification": float("nan"), "ct_skull_hu": float("nan")}
    if b.sum() < 1000:
        return out
    out["ct_calcification"] = float((img[b] > 100).mean())
    head = img > -500
    bone = head & (img > 300) & ~(np.asarray(brain) > 0)
    if bone.sum() > 1000:
        out["ct_skull_hu"] = float(np.median(img[bone]))
    if seg is not None:
        gm, wm = _eroded(_mask(seg, (3, 42), labels)), _eroded(_mask(seg, (2, 41), labels))
        g, w = img[gm & (img > 0) & (img < 80)], img[wm & (img > 0) & (img < 80)]
        if g.size > 500 and w.size > 500 and np.median(w) > 0:
            out["ct_gray_white"] = float(np.median(g) / np.median(w))
    return out


def _measure_one(args) -> dict:
    from brain.prep import store

    vid, image_path, seg_path, modality, root = args
    from brain.prep.synthseg import LABELS

    try:
        seg = store.read_image(root / seg_path, "mask")
        out = {"volume_id": vid, **midline_shift(seg, LABELS), "evans_index": evans_index(seg, LABELS)}
        if str(modality).lower().startswith("ct"):
            a = store.read_many(root / image_path, ("image", "brain"))
            out.update(ct_measures(store.squeeze_channel(a["image"]).astype(np.float32), a.get("brain", np.ones(1)),
                                   seg, LABELS))
        return out
    except Exception as e:   # noqa: BLE001  one unreadable volume should not stop the pass
        return {"volume_id": vid, "error": str(e)[:200]}


def main() -> None:
    import argparse
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    from brain.multitask.labels import STORE, SYNTHSEG
    from brain.prep import store

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sel = pd.read_csv(SYNTHSEG / "select.csv")
    if a.limit:
        sel = sel.head(a.limit)
    args = [(r.volume_id, r.image_path, store.path_for("", r.folder, str(r.case_id), "synthseg"), r.modality, STORE)
            for r in sel.itertuples()]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_measure_one, args, chunksize=16))
    out = pd.DataFrame(rows)
    out.to_csv(SYNTHSEG / "measures.csv", index=False)
    print(out.describe().T.round(3).to_string())
    print("errors:", int(out["error"].notna().sum()) if "error" in out else 0)


if __name__ == "__main__":
    main()
