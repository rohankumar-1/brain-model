"""Labels derived from lesion masks and from the ventricles, per stored scan (2026-11-12).

From a lesion mask (haemorrhage: BHSD, HemSeg, RSNA pseudo-masks; tumour: BraTS):

* side: millilitres left / right of the scan's brain-mask centroid (the convention of ``brain.downstream.mask_labels``;
  store volumes are on the 1 mm RAS grid as (Z, Y, X), x increasing toward the patient's right, so x below the
  midline is the patient's left);
* compartment: millilitres inside anatomical priors on the shared grid (voxels SynthSeg labels as that compartment in
  >= 30% of 20 IXI T1 scans, as ``mask_labels.anatomy_priors``): ventricles, posterior fossa, deep grey matter,
  the rest of the intracranial prior (hemispheric) and outside it (extra-axial). Priors, not the scan's own SynthSeg,
  because SynthSeg is unreliable on bleeding CT and on tumours;
* volume and the number of connected components.

From the scan alone, midline shift: the septum pellucidum's distance from the grid's midsagittal plane (the store
grid is registered to the template, so the plane is fixed; a mass moves the septum, not the skull). The septum is
the thin tissue gap between the two lateral ventricles in each axial row of the frontal horns and bodies; the
ventricles come from the scan's SynthSeg (MRI) or from CSF density inside the brain mask (CT, -5..20 HU). Rows are
searched inside the lateral-ventricle prior widened by ``MLS_REACH`` mm so a shifted septum is still found.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from brain.prep import store

GRID_X = 256
GRID_MID_X = (GRID_X - 1) / 2.0
MIN_COMPONENT_ML = 0.05             # components smaller than 50 voxels are mask noise, not lesions
SIDE_CLASSES = ("left", "right", "bilateral")
SIDE_UNILATERAL = 0.9               # >= 90% of the lesion on one side: that side, else bilateral -- the unilateral
                                    # rule of the side benchmarks (instance_side, atlas_lesion_side, isles24_lvo_side);
                                    # Seg-CQ500 reads could not calibrate it (its stored copies are left-right
                                    # inconsistent, see analysis/lesion_measures_validate.py)
LOCATION_CLASSES = ("hemispheric", "deep grey matter", "intraventricular", "infratentorial", "extra-axial")
VENTRICLES = (4, 5, 43, 44, 14, 15)
LATERAL_VENTRICLES = (4, 43)
POSTERIOR_FOSSA = (7, 8, 46, 47, 16, 15)
DEEP_GREY = (10, 11, 12, 13, 49, 50, 51, 52, 26, 58, 28, 60)
MLS_REACH = 15                      # mm: the widest septum shift searched for
MLS_GAP = (1, 8)                    # mm: septum (tissue between the ventricles) width in a row; wider = atria / horns apart
MLS_MIN_ROWS = 12                   # rows with a septum needed for a measurement
CT_CSF_HU = (-5.0, 20.0)


def compartment_priors(path: Path, ixi_synthseg: list[Path] | None = None) -> dict[str, np.ndarray]:
    """{ventricles, lateral_ventricles, posterior_fossa, deep_grey, intracranial}: boolean (Z, Y, X) priors on the
    store grid, built once from IXI T1 SynthSeg masks (``ixi_synthseg``) and cached at ``path``."""
    path = Path(path)
    if not path.exists():
        from brain.prep.synthseg import LABELS

        have = [Path(p) for p in ixi_synthseg or [] if Path(p).exists()]
        if len(have) < 10:
            raise SystemExit(f"{path} missing and only {len(have)} IXI SynthSeg masks given to build it (need 10)")
        acc = {k: [] for k in ("ventricles", "lateral_ventricles", "posterior_fossa", "deep_grey", "intracranial")}
        for zp in have:
            seg = LABELS[np.asarray(store.read_image(zp, "mask")).astype(int)]
            acc["ventricles"].append(np.isin(seg, VENTRICLES))
            acc["lateral_ventricles"].append(np.isin(seg, LATERAL_VENTRICLES))
            acc["posterior_fossa"].append(np.isin(seg, POSTERIOR_FOSSA))
            acc["deep_grey"].append(np.isin(seg, DEEP_GREY))
            acc["intracranial"].append(seg > 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: np.mean(v, 0) >= 0.3 for k, v in acc.items()}, n=len(have))
    with np.load(path) as z:
        return {k: z[k].astype(bool) for k in z.files if k != "n"}


def midline_x(brain: np.ndarray) -> float:
    """x of the brain-mask centroid (the side reference of ``mask_labels``)."""
    xs = np.nonzero(brain.any(axis=(0, 1)))[0]
    if not xs.size:
        return GRID_MID_X
    w = brain.sum(axis=(0, 1)).astype(np.float64)
    return float((np.arange(brain.shape[2]) * w).sum() / max(w.sum(), 1.0))


def lesion_stats(mask: np.ndarray, brain: np.ndarray, priors: dict[str, np.ndarray]) -> dict[str, float]:
    """Volume, side split, compartments and components of a boolean lesion mask (Z, Y, X)."""
    from scipy.ndimage import label

    out = {"ml": float(mask.sum() / 1000.0)}
    mid = midline_x(brain)
    x = np.nonzero(mask)[2]
    out["left_ml"], out["right_ml"] = float((x < mid).sum() / 1000.0), float((x >= mid).sum() / 1000.0)
    rest = mask.copy()
    for name, key in (("intraventricular", "ventricles"), ("infratentorial", "posterior_fossa"),
                      ("deep grey matter", "deep_grey"), ("hemispheric", "intracranial")):
        inside = rest & priors[key]
        out[f"{name}_ml"] = float(inside.sum() / 1000.0)
        rest &= ~inside
    out["extra-axial_ml"] = float(rest.sum() / 1000.0)
    lab, n = label(mask, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(0)
    out["components"] = int((sizes >= MIN_COMPONENT_ML * 1000).sum())
    return out


def side_class(left_ml: float, right_ml: float, unilateral: float = SIDE_UNILATERAL) -> int:
    """Index into ``SIDE_CLASSES``: left / right when that side holds >= ``unilateral`` of the lesion, else bilateral."""
    tot = left_ml + right_ml
    if tot <= 0:
        return -1
    return 0 if left_ml / tot >= unilateral else 1 if right_ml / tot >= unilateral else 2


def location_class(stats: dict[str, float]) -> int:
    """Index into ``LOCATION_CLASSES``: the compartment holding most of the lesion."""
    ml = [stats.get(f"{c}_ml", 0.0) for c in LOCATION_CLASSES]
    return int(np.argmax(ml)) if max(ml) > 0 else -1


def _plane(x0) -> tuple[float, float, float]:
    """A midsagittal plane ``x = a + b*y + c*z`` from a scalar x (a plane parallel to the grid's) or (a, b, c)."""
    return (float(x0), 0.0, 0.0) if np.isscalar(x0) else tuple(float(t) for t in x0)


def _septum_rows(vent: np.ndarray, region: np.ndarray, x0) -> np.ndarray:
    """Signed septum offsets (mm, + = toward the patient's right) of every axial row of ``region`` where two
    ventricle runs are separated by a ``MLS_GAP`` tissue gap within ``MLS_REACH`` mm of the plane ``x0`` (scalar or
    (a, b, c), see ``_plane``), measured perpendicular to it."""
    a, b, c = _plane(x0)
    norm = float(np.sqrt(1.0 + b * b + c * c))
    zs, ys = np.nonzero(region.any(axis=2))
    out = []
    for z, y in zip(zs, ys):
        xm = a + b * y + c * z
        lo, hi = max(int(np.floor(xm - MLS_REACH - 25)), 0), min(int(np.ceil(xm + MLS_REACH + 25)), vent.shape[2])
        row = vent[z, y, lo:hi]
        if row.sum() < 4:
            continue
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]       # runs [start, end)
        best = None
        for k in range(len(starts) - 1):
            if ends[k] - starts[k] < 2 or ends[k + 1] - starts[k + 1] < 2:
                continue
            gap = starts[k + 1] - ends[k]
            if not MLS_GAP[0] <= gap <= MLS_GAP[1]:
                continue
            cx = lo + (ends[k] + starts[k + 1] - 1) / 2.0
            if abs(cx - xm) <= MLS_REACH and (best is None or abs(cx - xm) < abs(best - xm)):
                best = cx
        if best is not None:
            out.append((best - xm) / norm)
    return np.asarray(out, dtype=np.float64)


def midline_shift(vent: np.ndarray, priors: dict[str, np.ndarray], x0=GRID_MID_X) -> dict[str, float]:
    """Septum offset from the midsagittal plane ``x0`` (scalar x or a plane (a, b, c) such as
    ``measures.symmetry_plane`` of the skull): {mls_mm (|median|), mls_p90 (90th percentile of |offset|), mls_signed
    (median, + = septum pushed to the patient's right), mls_rows}; NaN with fewer than ``MLS_MIN_ROWS`` rows."""
    from scipy.ndimage import maximum_filter

    region = maximum_filter(priors["lateral_ventricles"], size=(3, 5, 2 * MLS_REACH + 1))
    s = _septum_rows(vent, region, x0)
    if len(s) < MLS_MIN_ROWS:
        return {"mls_mm": float("nan"), "mls_p90": float("nan"), "mls_signed": float("nan"), "mls_rows": len(s)}
    med = float(np.median(s))
    return {"mls_mm": abs(med), "mls_p90": float(np.percentile(np.abs(s), 90)), "mls_signed": med,
            "mls_rows": int(len(s))}


def ventricles_ct(image_hu: np.ndarray, brain: np.ndarray) -> np.ndarray:
    return brain & (image_hu > CT_CSF_HU[0]) & (image_hu < CT_CSF_HU[1])


def ventricles_synthseg(synthseg_mask: np.ndarray) -> np.ndarray:
    from brain.prep.synthseg import LABELS

    return np.isin(LABELS[np.asarray(synthseg_mask).astype(int)], LATERAL_VENTRICLES)
