"""Native-geometry ("ngeo") CT copy: native slices, 0.5 mm in-plane, next to the 1 mm template-grid store.

Contract: runs/diagnostics/ngeo/CONTRACT.md section 2. Per CT case one extra zip ``<case>.ngeo.zip`` beside the image zip:

* ``image`` (S, 1, R, C) int16 HU (values < -500 HU set to -1024, clipped at 3071). Axis 0 runs along the slice normal
  of the acquisition, inferior -> superior; R, C are a fixed 512 x 512 field at 0.5 mm, centred on the brain. Rows run
  anterior -> posterior and columns right -> left (the radiological display of an axial DICOM), along the acquisition's
  own in-plane axes (no rotation into the template).
* ``brain`` (S, R, C) uint8: the 1 mm brain mask mapped onto this grid (nearest).
* label masks with a native source: ``<case>.<name>.ngeo.zip`` (array ``mask``, (S, R, C) uint8), resampled in-plane
  from the source mask; template-grid-only labels map from the 1 mm grid (nearest).

Geometry. A case is read into a ``SliceStack``: 2D slices with their own physical origin (``p``) and shared in-plane step
vectors (``a`` per row, ``b`` per column). Slices are sorted along the normal and de-duplicated, then ``plan`` turns
their positions into an evenly spaced set of target slices: thick series (spacing >= 2.5 mm) keep their native slices
(a slice within 15% of a target position is copied, never interpolated through-plane); thin series are box-averaged
into ~5 mm slabs (``synth_thick``); in mixed series (RSNA: 2.5 mm posterior fossa + 5 mm above) the thin part is
box-averaged onto the thick grid; a missing slice is filled by linear interpolation and counted (``n_gaps``). Every
target slice is resampled in-plane only, by a 2D affine per source slice, which also removes gantry-tilt shear (the
in-plane shift per slice) since the output slices are stacked along the normal.

Template mapping. The 1 mm store's ``reg_affine`` maps stored-grid physical points to the source scan *as prep read
it* ("store frame"). For NIfTI sources and dcm2niix reads (CQ500, RSNA-ICA) that is the true scanner frame. For RSNA
ICH / HemSeg prep used SimpleITK's series reader, which places slice k at ``origin + k * spacing * normal`` with
spacing = |IPP_last - IPP_first| / (N - 1): wrong for tilted (sheared) and unevenly spaced series. ``q`` holds every
source slice's origin in that store frame (recovered from the stored ``native_*`` attrs and the slice order), so labels
computed on the 1 mm grid map back exactly, slice by slice. ``affine_template`` is the least-squares affine of that
per-slice map (exact for evenly spaced series, sheared or not; ``tpl_resid_mm`` reports the misfit otherwise).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

H_MM = 0.5                    # in-plane spacing
RC = 512                      # rows = cols
THICK_MIN_MM = 2.5            # spacing at or above which a series is "thick" (native slices kept)
SLAB_MM = 5.0                 # target slab of a thin series
AIR_HU, FLOOR_HU, MAX_HU = -500.0, -1024.0, 3071.0
DUP_TOL_MM = 0.2              # slices closer than this along the normal are repeats
SNAP_FRAC = 0.15              # a native slice within this fraction of the spacing is copied as is
NGEO_REV = 1
KIND = "ct_hu_ngeo"
RANGE_MARGIN_MM = 5.0         # around the brain's extent along the normal


# ----------------------------------------------------------------------------------------------------------- stacks

@dataclass
class SliceStack:
    """2D slices in physical space (ITK / LPS mm). ``pix[k, i, j]`` sits at ``p[k] + i * a + j * b``."""

    pix: np.ndarray                         # (K, H, W) float32
    p: np.ndarray                           # (K, 3) origin (pixel [k, 0, 0]) of each slice
    a: np.ndarray                           # (3,) step per row index
    b: np.ndarray                           # (3,) step per column index
    q: np.ndarray | None = None             # (K, 3) the same origins in the 1 mm store's source frame (None: = p)
    thick: float = 0.0                      # nominal SliceThickness from the header (0: unknown)
    info: dict = field(default_factory=dict)
    src: np.ndarray | None = None           # (K,) index of each slice in its source (DICOM member), kept by ``take``
    aq: np.ndarray | None = None            # in-plane steps in the store frame (None: = a, b)
    bq: np.ndarray | None = None

    @property
    def K(self) -> int:
        return int(self.p.shape[0])

    @property
    def normal(self) -> np.ndarray:
        return slice_normal(self.a, self.b)

    @property
    def qq(self) -> np.ndarray:
        return self.p if self.q is None else self.q

    @property
    def aqq(self) -> np.ndarray:
        return self.a if self.aq is None else self.aq

    @property
    def bqq(self) -> np.ndarray:
        return self.b if self.bq is None else self.bq

    def take(self, idx) -> "SliceStack":
        idx = np.asarray(idx, dtype=int)
        return SliceStack(self.pix[idx] if len(self.pix) else self.pix, self.p[idx], self.a, self.b,
                          None if self.q is None else self.q[idx], self.thick, dict(self.info),
                          None if self.src is None else self.src[idx], self.aq, self.bq)


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def slice_normal(a, b) -> np.ndarray:
    """Unit normal of the slice plane, pointing superior (LPS +z); for a non-axial plane, along its largest component."""
    n = unit(np.cross(a, b))
    ref = 2 if abs(n[2]) >= 0.3 else int(np.argmax(np.abs(n)))
    return n if n[ref] >= 0 else -n


def sort_dedupe(st: SliceStack, tol: float = DUP_TOL_MM) -> tuple[SliceStack, int]:
    """Slices sorted inferior -> superior along the normal; a slice within ``tol`` of the previous kept one is a repeat
    (the first in input order is kept)."""
    keep = _dedupe_index(st, tol)
    return st.take(keep), int(st.K - len(keep))


def stack_tilt_deg(st: SliceStack) -> float:
    """Angle between the slice stacking direction (IPP steps) and the plane normal: gantry tilt (shear)."""
    if st.K < 2:
        return 0.0
    d = np.diff(st.p, axis=0)
    n = st.normal
    ok = np.abs(d @ n) > DUP_TOL_MM
    if not ok.any():
        return 0.0
    ang = np.degrees(np.arccos(np.clip(np.abs(d[ok] @ n) / np.linalg.norm(d[ok], axis=1), -1.0, 1.0)))
    return float(np.median(ang))


# ------------------------------------------------------------------------------------------------------ slice plan

@dataclass
class Plan:
    T: np.ndarray                           # (S,) target positions along the normal (evenly spaced)
    weights: list                           # S lists of (source slice, weight)
    D: float                                # slice_mm
    thin: bool                              # thin series -> synthetic slabs
    thick_mm: float
    n_gaps: int = 0
    n_synth: int = 0                        # targets averaged from more than one slice
    uneven: bool = False
    snap_max_mm: float = 0.0


def plan(t, lo: float = -np.inf, hi: float = np.inf, slab_mm: float = SLAB_MM, thick_hint: float = 0.0) -> Plan:
    """Even target slices over sorted, de-duplicated slice positions ``t`` (mm along the normal), kept within
    [lo, hi]. See the module docstring for the rules."""
    t = np.asarray(t, dtype=np.float64)
    if len(t) < 2:
        raise ValueError(f"need at least 2 slices, got {len(t)}")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("positions must be sorted and unique")
    cls = dt >= THICK_MIN_MM * 0.95
    thin = dt[~cls].sum() > dt[cls].sum()
    if thin:
        d = float(np.median(dt[~cls]))
        n = max(1, int(round(slab_mm / d)))
        D = n * d
        uneven = bool(np.any(np.abs(dt - d) > 0.1 * d))
    else:
        # the coarsest spacing class covering >= 25% of the length (2.5 mm + 5 mm series: 5 mm, the 2.5 mm part is
        # box-averaged onto it rather than the 5 mm part interpolated)
        dtt = dt[cls]
        D = float(np.median(dtt))
        for c in sorted(set(np.round(dtt / 0.25) * 0.25), reverse=True):
            near = np.abs(dtt - c) <= 0.1 * c
            if dtt[near].sum() >= 0.25 * dt.sum():
                D = float(np.median(dtt[near]))
                break
        uneven = bool(np.any(np.abs(dt - D) > 0.1 * D))
    left, right = np.r_[dt[0], dt], np.r_[dt, dt[-1]]
    s_lo, s_hi = t - np.minimum(left, D) / 2, t + np.minimum(right, D) / 2
    # a slice stands for a whole slab when it has the slab's width or a neighbour at the slab spacing (the first 5 mm
    # slice after a 2.5 mm block); only those are copied
    native_ok = ((s_hi - s_lo) >= (1 - SNAP_FRAC) * D) | (np.abs(left - D) <= SNAP_FRAC * D) \
        | (np.abs(right - D) <= SNAP_FRAC * D)
    if thin:
        anchor = s_lo[0] + D / 2
    else:
        nat = np.flatnonzero(native_ok)
        anchor = float(t[nat[len(nat) // 2]]) if len(nat) else float(t[len(t) // 2])
    j0, j1 = math.floor((s_lo[0] - anchor) / D) - 1, math.ceil((s_hi[-1] - anchor) / D) + 1
    T, W = [], []
    n_gaps = n_synth = 0
    snap = 0.0
    for j in range(j0, j1 + 1):
        Tj = anchor + j * D
        if Tj < lo - 1e-6 or Tj > hi + 1e-6:
            continue
        k = int(np.argmin(np.abs(t - Tj)))
        if not thin and native_ok[k] and abs(t[k] - Tj) <= SNAP_FRAC * D:
            T.append(Tj)
            W.append([(k, 1.0)])
            snap = max(snap, abs(t[k] - Tj))
            continue
        ov = np.clip(np.minimum(s_hi, Tj + D / 2) - np.maximum(s_lo, Tj - D / 2), 0.0, None)
        if ov.sum() >= (0.99 if thin else 0.5) * D:            # thin: whole slabs only
            ks = np.flatnonzero(ov > 1e-6)
            w = ov[ks] / ov[ks].sum()
            T.append(Tj)
            W.append([(int(a), float(b)) for a, b in zip(ks, w)])
            n_synth += int(len(ks) > 1)
        elif s_lo[0] <= Tj - D / 2 and Tj + D / 2 <= s_hi[-1]:      # a hole inside the series: interpolate, count it
            hi_k = int(np.searchsorted(t, Tj))
            lo_k = hi_k - 1
            f = (Tj - t[lo_k]) / (t[hi_k] - t[lo_k])
            T.append(Tj)
            W.append([(lo_k, float(1 - f)), (hi_k, float(f))])
            n_gaps += 1
    if not T:
        raise ValueError("no target slice inside the head range")
    thick_mm = D if thin or not (0.5 * D <= thick_hint <= 1.5 * D) else float(thick_hint)
    return Plan(np.asarray(T), W, float(D), bool(thin), float(thick_mm), n_gaps, n_synth, uneven, float(snap))


# ------------------------------------------------------------------------------------------------------- the grid

@dataclass
class NGrid:
    """The ngeo grid: voxel (s, r, c) -> ``O + s * D * n + r * H * u_r + c * H * u_c`` (true scanner frame)."""

    O: np.ndarray
    n: np.ndarray
    u_r: np.ndarray
    u_c: np.ndarray
    D: float
    S: int
    h: float = H_MM
    rc: int = RC

    def affine(self) -> np.ndarray:
        A = np.eye(4)
        A[:3, 0], A[:3, 1], A[:3, 2], A[:3, 3] = self.D * self.n, self.h * self.u_r, self.h * self.u_c, self.O
        return A

    def slice_origin(self, s) -> np.ndarray:
        return self.O + np.asarray(s, dtype=np.float64)[..., None] * self.D * self.n


def inplane_axes(a, b, n) -> tuple[np.ndarray, np.ndarray]:
    """(u_r, u_c): the slice's own in-plane axes, exactly orthonormal to each other and to ``n``. Columns run toward
    patient left (LPS +x: the in-plane axis closer to x), rows ``n x u_c`` (posterior for an axial slice)."""
    ah, bh = unit(a), unit(b)
    col = ah if abs(ah[0]) >= abs(bh[0]) else bh
    u_c = unit(col - (col @ n) * n)
    u_c = u_c if u_c[0] >= 0 else -u_c
    return unit(np.cross(n, u_c)), u_c


def make_grid(st: SliceStack, pl: Plan, center) -> NGrid:
    """Grid whose slice s lies in the plane at ``pl.T[s]`` and whose in-plane field is centred on ``center``."""
    n = st.normal
    u_r, u_c = inplane_axes(st.a, st.b, n)
    c = np.asarray(center, dtype=np.float64)
    half = (RC - 1) / 2 * H_MM
    O = c - (c @ n) * n + pl.T[0] * n - half * u_r - half * u_c
    return NGrid(O, n, u_r, u_c, pl.D, len(pl.T))


# --------------------------------------------------------------------------------------------- in-plane resampling

def _pinv_ab(a, b) -> np.ndarray:
    return np.linalg.pinv(np.stack([a, b], axis=1))            # (2, 3): physical offset -> (i, j)


def slice_transform(st: SliceStack, k: int, origin: np.ndarray, g: NGrid) -> tuple[np.ndarray, np.ndarray]:
    """2D affine (matrix, offset) mapping output (r, c) of a target slice with physical ``origin`` to (i, j) on source
    slice ``k`` (the offset along the normal is dropped: in-plane only)."""
    L = _pinv_ab(st.a, st.b)
    M = np.stack([L @ (g.h * g.u_r), L @ (g.h * g.u_c)], axis=1)
    off = L @ (origin - st.p[k])
    return M, off


def _antialias(src: np.ndarray, st: SliceStack, h: float) -> np.ndarray:
    sig = [0.5 * math.sqrt(max(0.0, (h / s) ** 2 - 1.0)) if s < 0.9 * h else 0.0
           for s in (np.linalg.norm(st.a), np.linalg.norm(st.b))]
    return ndimage.gaussian_filter(src, sig, mode="nearest") if any(sig) else src


def resample_image(st: SliceStack, pl: Plan, g: NGrid, cval: float = FLOOR_HU) -> np.ndarray:
    """(S, RC, RC) float32: each target slice = weighted sum of its source slices, each resampled in-plane (linear)."""
    out = np.empty((g.S, g.rc, g.rc), dtype=np.float32)
    cache: dict[int, np.ndarray] = {}
    for s, wk in enumerate(pl.weights):
        origin = g.slice_origin(s)
        acc = np.zeros((g.rc, g.rc), dtype=np.float32)
        for k, w in wk:
            M, off = slice_transform(st, k, origin, g)
            src = cache.get(k)
            if src is None:
                src = _antialias(st.pix[k].astype(np.float32, copy=False), st, g.h)
                cache = {kk: v for kk, v in cache.items() if kk >= k - 16}   # bounded: slices are visited in order
                cache[k] = src
            acc += w * ndimage.affine_transform(src, M, off, output_shape=(g.rc, g.rc), order=1, mode="constant",
                                                cval=cval, prefilter=False)
        out[s] = acc
    return out


def resample_labels(st: SliceStack, pl: Plan, g: NGrid) -> np.ndarray:
    """(S, RC, RC) uint8 from a label stack aligned with the image's source slices: nearest in-plane; a target made of
    several slices takes the weighted majority label (background competes)."""
    out = np.zeros((g.S, g.rc, g.rc), dtype=np.uint8)
    for s, wk in enumerate(pl.weights):
        origin = g.slice_origin(s)
        if len(wk) == 1 or max(w for _, w in wk) >= 0.999:
            k = max(wk, key=lambda kw: kw[1])[0]
            M, off = slice_transform(st, k, origin, g)
            out[s] = ndimage.affine_transform(st.pix[k], M, off, output_shape=(g.rc, g.rc), order=0,
                                              mode="constant", cval=0, prefilter=False)
            continue
        votes: dict[int, np.ndarray] = {}
        for k, w in wk:
            M, off = slice_transform(st, k, origin, g)
            lab = ndimage.affine_transform(st.pix[k], M, off, output_shape=(g.rc, g.rc), order=0, mode="constant",
                                           cval=0, prefilter=False).astype(np.uint8)
            for v in np.unique(lab):
                votes.setdefault(int(v), np.zeros((g.rc, g.rc), dtype=np.float32))
                votes[int(v)] += w * (lab == v)
        labs = sorted(votes)
        if 0 not in votes:
            labs = [0] + labs
            votes[0] = np.zeros((g.rc, g.rc), dtype=np.float32)
        score = np.stack([votes[v] for v in labs])
        out[s] = np.asarray(labs, dtype=np.uint8)[np.argmax(score, axis=0)]
    return out


def to_hu_int16(vol: np.ndarray) -> np.ndarray:
    """int16 HU in the store's format-7 CT encoding (``store.encode_image``: < -500 HU -> -1024, 1 HU steps), clipped
    to the store's CT range."""
    return encode_hu(vol)[0]


def encode_hu(vol: np.ndarray) -> tuple[np.ndarray, dict]:
    """(int16 array, attrs ``dtype_rev`` / ``scale`` / ``air_below``) so ``store.read`` decodes it like an image zip."""
    from brain.prep import store

    v = np.clip(np.asarray(vol, dtype=np.float32), FLOOR_HU, MAX_HU)
    return store.encode_image(v, "ct_hu", air_below=AIR_HU)


# ----------------------------------------------------------------------------------- store frame and template maps

def _affine_from(origin, spacing, direction) -> np.ndarray:
    A = np.eye(4)
    A[:3, :3] = np.asarray(direction, dtype=np.float64).reshape(3, 3) @ np.diag(np.asarray(spacing, dtype=np.float64))
    A[:3, 3] = np.asarray(origin, dtype=np.float64)
    return A


def grid_affine(attrs: dict) -> np.ndarray:
    """1 mm stored grid: index (x, y, z) -> template-grid physical mm."""
    return _affine_from(attrs["grid_origin"], attrs["grid_spacing"], attrs["grid_direction"])


def reg_matrix(attrs: dict) -> np.ndarray:
    return np.asarray(attrs["reg_affine"], dtype=np.float64).reshape(4, 4)


def store_to_template(attrs: dict) -> np.ndarray:
    """store-frame physical -> stored 1 mm grid physical (inverse of ``reg_affine``)."""
    return np.linalg.inv(reg_matrix(attrs))


def sitk_store_origins(p_files: np.ndarray, a: np.ndarray, b: np.ndarray, attrs: dict,
                       instance: np.ndarray | None = None) -> tuple[np.ndarray | None, dict]:
    """Origins of each file's slice in the frame SimpleITK's series reader gave it (the 1 mm store's source frame).

    ``p_files`` are the IPPs of exactly the files SimpleITK read (one series, duplicates included), in member order.
    ITK orders the files (GDCM sorts along the normal n0 = row x col, but falls back to instance / file order when
    positions repeat), puts the first file's IPP at the origin and steps by |IPP_last - IPP_first| / (N - 1) along
    n0; prep then reoriented that grid to RAS (``native_*`` attrs). The axis permutation / flips come from matching the
    stored direction columns to the DICOM axes; the file order is the candidate whose first IPP and spacing reproduce
    the stored grid."""
    K = len(p_files)
    size = np.asarray(attrs.get("native_size", [0, 0, 0]), dtype=int)
    sp_nat = np.asarray(attrs["native_spacing"], dtype=np.float64)
    Dn = np.asarray(attrs["native_direction"], dtype=np.float64).reshape(3, 3)
    A = _affine_from(attrs["native_origin"], attrs["native_spacing"], attrs["native_direction"])
    rowc, colc = unit(b), unit(a)                       # DICOM row cosine (along columns j), column cosine (rows i)
    n0 = unit(np.cross(rowc, colc))
    axes = [colc, rowc, n0]
    match = []
    for m in range(3):
        dots = np.array([Dn[:, m] @ v for v in axes])
        r = int(np.argmax(np.abs(dots)))
        if abs(dots[r]) < 0.99:
            return None, {"store_frame": "unmatched_axes"}
        match.append((r, 1.0 if dots[r] > 0 else -1.0))
    if sorted(r for r, _ in match) != [0, 1, 2]:
        return None, {"store_frame": "unmatched_axes"}
    slice_ax = next(m for m in range(3) if match[m][0] == 2)
    if size[slice_ax] != K:
        return None, {"store_frame": f"slice_count_{size[slice_ax]}_vs_{K}"}
    t0 = p_files @ n0
    inst = np.arange(K) if instance is None else np.asarray(instance)
    cands = {"position": np.lexsort((inst, t0)), "instance": np.lexsort((np.arange(K), inst)),
             "member": np.arange(K)}

    def origins(kz_of_file):
        idx = np.zeros((K, 3))
        for m, (r, sgn) in enumerate(match):
            if r == 2:
                idx[:, m] = kz_of_file if sgn > 0 else size[m] - 1 - kz_of_file
            else:
                idx[:, m] = 0.0 if sgn > 0 else size[m] - 1
        return (A[:3, :3] @ idx.T).T + A[:3, 3]

    best = None
    for name, order in cands.items():
        for rev in (False, True):
            o = order[::-1] if rev else order
            dz = np.linalg.norm(p_files[o[-1]] - p_files[o[0]]) / max(K - 1, 1)
            kz = np.empty(K)
            kz[o] = np.arange(K)
            q = origins(kz)
            err = float(np.linalg.norm(q[o[0]] - p_files[o[0]])) + abs(dz - sp_nat[slice_ax])
            if best is None or err < best[1] - 1e-6:
                best = (q, err, f"{name}_{'desc' if rev else 'asc'}")
    q, err, how = best
    info = {"store_frame": "sitk_series", "store_frame_check_mm": err, "store_frame_order": how}
    if err > 1.0:
        return None, info | {"store_frame": "origin_mismatch"}
    return q, info


def store_to_true_fn(st: SliceStack):
    """A function mapping store-frame points (N, 3) to the true scanner frame (identity when the frames agree): a
    point's offset along the store-frame normal picks its (fractional) slice, its in-plane coordinates on that slice
    carry over."""
    if (st.q is None or np.allclose(st.q, st.p, atol=1e-3)) and st.aq is None and st.bq is None:
        return lambda X: np.asarray(X, dtype=np.float64)
    nq = unit(np.cross(st.aqq, st.bqq))
    tau = st.qq @ nq
    o = np.argsort(tau)
    tau, P, Qo = tau[o], st.p[o], st.qq[o]
    Lq = np.linalg.pinv(np.stack([st.aqq, st.bqq], axis=1))
    ab = np.stack([st.a, st.b], axis=1)
    K = len(tau)

    def f(X):
        X = np.asarray(X, dtype=np.float64)
        x = X @ nq
        d0, d1 = max(tau[1] - tau[0], 1e-6) if K > 1 else 1.0, max(tau[-1] - tau[-2], 1e-6) if K > 1 else 1.0
        k = np.interp(x, tau, np.arange(K))
        k = np.where(x < tau[0], (x - tau[0]) / d0, np.where(x > tau[-1], K - 1 + (x - tau[-1]) / d1, k))
        lo = np.clip(np.floor(k).astype(int), 0, max(K - 2, 0))
        fr = (k - lo)[:, None]
        hi = np.minimum(lo + 1, K - 1)
        pk = P[lo] * (1 - fr) + P[hi] * fr
        qk = Qo[lo] * (1 - fr) + Qo[hi] * fr
        ij = (Lq @ (X - qk).T).T
        return pk + ij @ ab.T

    return f


def store_slice_origins(st: SliceStack, pl: Plan, g: NGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Q (S, 3), Ur, Uc): target pixel (s, r, c) sits at ``Q[s] + r * Ur + c * Uc`` in the store frame. A true point
    X on source slice k has in-plane coordinates L (X - p_k) and store position q_k + [aq bq] L (X - p_k)."""
    L = _pinv_ab(st.a, st.b)
    Bq = np.stack([st.aqq, st.bqq], axis=1)
    BL = Bq @ L
    Q = np.empty((g.S, 3))
    for s, wk in enumerate(pl.weights):
        o = g.slice_origin(s)
        pw = sum(w * st.p[k] for k, w in wk)
        qw = sum(w * st.qq[k] for k, w in wk)
        Q[s] = qw + BL @ (o - pw)
    return Q, BL @ (g.h * g.u_r), BL @ (g.h * g.u_c)


def fit_slice_affine(Q: np.ndarray, Ur: np.ndarray, Uc: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares affine voxel -> store-frame mm of a per-slice origin list, and its worst residual (mm)."""
    s = np.arange(len(Q), dtype=np.float64)
    X = np.stack([s, np.ones_like(s)], axis=1)
    coef, *_ = np.linalg.lstsq(X, Q, rcond=None)              # (2, 3): slope, intercept
    resid = float(np.linalg.norm(X @ coef - Q, axis=1).max()) if len(Q) else 0.0
    A = np.eye(4)
    A[:3, 0], A[:3, 1], A[:3, 2], A[:3, 3] = coef[0], Ur, Uc, coef[1]
    return A, resid


def map_template_array(arr_zyx: np.ndarray, attrs: dict, Q: np.ndarray, Ur: np.ndarray, Uc: np.ndarray, g: NGrid,
                       order: int = 0, stride: int = 1, cval: float = 0.0) -> np.ndarray:
    """A 1 mm stored-grid array (Z, Y, X) sampled at every ngeo voxel through the exact per-slice store-frame map
    (``stride``: every n-th row / column only)."""
    M = np.linalg.inv(grid_affine(attrs)) @ store_to_template(attrs)        # store phys -> 1 mm grid index (x, y, z)
    ax = np.arange(0, g.rc, stride, dtype=np.float64)
    rr, cc = np.meshgrid(ax, ax, indexing="ij")
    out = np.zeros((g.S, len(ax), len(ax)), dtype=arr_zyx.dtype if order == 0 else np.float32)
    du, dc = M[:3, :3] @ Ur, M[:3, :3] @ Uc
    for s in range(g.S):
        base = M[:3, :3] @ Q[s] + M[:3, 3]
        xyz = base[:, None, None] + du[:, None, None] * rr + dc[:, None, None] * cc
        out[s] = ndimage.map_coordinates(arr_zyx, [xyz[2], xyz[1], xyz[0]], order=order, mode="constant", cval=cval,
                                         prefilter=False)
    return out


def corr_with_1mm(img: np.ndarray, image_1mm_zyx: np.ndarray, attrs: dict, Q: np.ndarray, Ur, Uc, g: NGrid,
                  stride: int = 4) -> float:
    """Frame check: correlation of the ngeo image with the 1 mm stored image sampled at the same voxels (through the
    per-slice map), in a soft-tissue window over the head. A wrong frame (offset, mirror, shear) drops it."""
    ref = map_template_array(np.asarray(image_1mm_zyx, dtype=np.float32), attrs, Q, Ur, Uc, g, order=1, stride=stride,
                             cval=FLOOR_HU)
    x = img[:, ::stride, ::stride].astype(np.float32)
    head = (x > AIR_HU) | (ref > AIR_HU)
    if head.sum() < 100:
        return float("nan")
    a, b = np.clip(x[head], -50, 150), np.clip(ref[head], -50, 150)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mask_stack(mask_img, st: SliceStack) -> SliceStack:
    """A label image (SimpleITK, any grid in the store frame) sampled nearest at every source pixel of ``st`` -> a label
    stack aligned with the image's source slices."""
    import SimpleITK as sitk

    arr = sitk.GetArrayViewFromImage(mask_img)
    A = _affine_from(mask_img.GetOrigin(), mask_img.GetSpacing(), mask_img.GetDirection())
    Ai = np.linalg.inv(A)
    H, W = st.pix.shape[1:]
    ii, jj = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    da, db = Ai[:3, :3] @ st.aqq, Ai[:3, :3] @ st.bqq
    out = np.zeros(st.pix.shape, dtype=np.uint8)
    for k in range(st.K):
        base = Ai[:3, :3] @ st.qq[k] + Ai[:3, 3]
        xyz = base[:, None, None] + da[:, None, None] * ii + db[:, None, None] * jj
        out[k] = ndimage.map_coordinates(arr, [np.rint(xyz[2]), np.rint(xyz[1]), np.rint(xyz[0])], order=0,
                                         mode="constant", cval=0, prefilter=False)
    return SliceStack(out, st.p, st.a, st.b, st.q, st.thick, {}, None, st.aq, st.bq)


def sample_physical(img, g: NGrid, order: int = 1, nsub: int = 1, cval: float = 0.0) -> np.ndarray:
    """A SimpleITK volume in the true scanner frame sampled at the ngeo voxels: ``nsub`` points spread through each
    slab's thickness, averaged (order 1) or majority-voted per label (order 0)."""
    import SimpleITK as sitk

    arr = sitk.GetArrayViewFromImage(img)
    Ai = np.linalg.inv(_affine_from(img.GetOrigin(), img.GetSpacing(), img.GetDirection()))
    ax = np.arange(g.rc, dtype=np.float64)
    rr, cc = np.meshgrid(ax, ax, indexing="ij")
    du, dc, dn = Ai[:3, :3] @ (g.h * g.u_r), Ai[:3, :3] @ (g.h * g.u_c), Ai[:3, :3] @ (g.D * g.n)
    offs = (np.arange(nsub) + 0.5) / nsub - 0.5
    out = np.zeros((g.S, g.rc, g.rc), dtype=np.float32 if order else np.uint8)
    for s in range(g.S):
        base = Ai[:3, :3] @ g.slice_origin(s) + Ai[:3, 3]
        acc, votes = None, {}
        for o in offs:
            xyz = (base + o * dn)[:, None, None] + du[:, None, None] * rr + dc[:, None, None] * cc
            v = ndimage.map_coordinates(arr, [xyz[2], xyz[1], xyz[0]], order=order, mode="constant", cval=cval,
                                        prefilter=False)
            if order:
                acc = v if acc is None else acc + v
            else:
                for lab in np.unique(v):
                    votes[int(lab)] = votes.get(int(lab), 0) + (v == lab)
        if order:
            out[s] = acc / len(offs)
        else:
            labs = sorted(votes)
            out[s] = np.asarray(labs, dtype=np.uint8)[np.argmax(np.stack([votes[k] for k in labs]), axis=0)]
    return out


# ---------------------------------------------------------------------------------------------- range and centring

def head_range_and_center(st: SliceStack, attrs: dict, brain_zyx: np.ndarray | None) -> tuple[float, float, np.ndarray, dict]:
    """[lo, hi] along the normal (true frame) and the in-plane centre: the 1 mm grid's 160 mm slab along its central
    column, widened to the brain's extent (+5 mm); the centre is the brain's centroid (else the grid centre)."""
    to_true = store_to_true_fn(st)
    n = st.normal
    G, R = grid_affine(attrs), reg_matrix(attrs)
    gs = np.asarray(attrs["grid_size"], dtype=np.float64)
    col = np.stack([np.full(2, (gs[0] - 1) / 2), np.full(2, (gs[1] - 1) / 2), np.array([0.0, gs[2] - 1]),
                    np.ones(2)], axis=0)
    ends = to_true(((R @ G @ col)[:3]).T)
    t_ends = ends @ n
    lo, hi = float(t_ends.min()), float(t_ends.max())
    center = to_true(((R @ G @ np.r_[(gs - 1) / 2, 1.0])[:3])[None])[0]
    info = {"range_grid_mm": [lo, hi]}
    if brain_zyx is not None and brain_zyx.any():
        z, y, x = np.nonzero(brain_zyx[::2, ::2, ::2])
        pts = np.stack([x * 2.0, y * 2.0, z * 2.0, np.ones_like(x, dtype=np.float64)], axis=0)
        P = to_true(((R @ G @ pts)[:3]).T)
        tb = P @ n
        lo, hi = min(lo, float(tb.min()) - RANGE_MARGIN_MM), max(hi, float(tb.max()) + RANGE_MARGIN_MM)
        center = P.mean(0)
        info["range_brain_mm"] = [float(tb.min()), float(tb.max())]
    return lo, hi, center, info


# --------------------------------------------------------------------------------------------------------- build

@dataclass
class NgeoResult:
    image: np.ndarray                       # (S, RC, RC) int16
    brain: np.ndarray | None                # (S, RC, RC) uint8
    masks: dict                             # name -> (S, RC, RC) uint8
    attrs: dict
    grid: NGrid
    plan: Plan
    Q: np.ndarray                           # (S, 3) store-frame position of pixel (0, 0) of every slice
    Ur: np.ndarray                          # store-frame step per row / column
    Uc: np.ndarray


def build(st: SliceStack, attrs: dict, brain_zyx: np.ndarray | None = None, native_masks: dict | None = None,
          template_masks: dict | None = None, extra: dict | None = None,
          image_1mm_zyx: np.ndarray | None = None) -> NgeoResult:
    """Everything after reading: dedupe -> range -> plan -> grid -> resample -> template maps -> attrs.

    ``st`` must carry ``q`` when the 1 mm store read the scan in another frame. ``native_masks``: name -> label
    SliceStack aligned with ``st`` (before dedupe). ``template_masks``: name -> 1 mm stored-grid (Z, Y, X) arrays."""
    order_keep = _dedupe_index(st)
    st_s = st.take(order_keep)
    n_dup = st.K - st_s.K
    tilt = stack_tilt_deg(st_s)
    lo, hi, center, rinfo = head_range_and_center(st_s, attrs, brain_zyx)
    t = st_s.p @ st_s.normal
    pl = plan(t, lo, hi, thick_hint=st.thick)
    g = make_grid(st_s, pl, center)
    img, enc = encode_hu(resample_image(st_s, pl, g))
    Q, Ur, Uc = store_slice_origins(st_s, pl, g)
    A_store, resid_store = fit_slice_affine(Q, Ur, Uc)
    T2 = store_to_template(attrs)
    A_tpl = T2 @ A_store
    resid_tpl = resid_store * float(np.linalg.norm(T2[:3, :3], 2))
    brain = None
    if brain_zyx is not None:
        brain = (map_template_array(np.asarray(brain_zyx, dtype=np.uint8), attrs, Q, Ur, Uc, g) > 0).astype(np.uint8)
    masks = {}
    for name, ms in (native_masks or {}).items():
        masks[name] = resample_labels(ms.take(order_keep), pl, g)
    for name, arr in (template_masks or {}).items():
        masks[name] = map_template_array(np.asarray(arr), attrs, Q, Ur, Uc, g).astype(np.uint8)
    out_attrs = {
        "kind": KIND, "ngeo_rev": NGEO_REV, **enc,
        "spacing_mm": [pl.D, H_MM, H_MM], "slice_mm": pl.D, "thick_mm": pl.thick_mm,
        "affine_scanner": g.affine().ravel().tolist(), "affine_template": A_tpl.ravel().tolist(),
        "synth_thick": bool(pl.thin), "n_synth": int(pl.n_synth),
        "assembly": {"n_dup_removed": int(n_dup), "n_gaps": int(pl.n_gaps), "uneven": bool(pl.uneven),
                     "tilt_deg": round(float(tilt), 3), "deshear": bool(tilt > 0.1)},
        "snap_max_mm": pl.snap_max_mm, "tpl_resid_mm": float(resid_tpl),
        # exact per-slice template position of pixel (0, 0) (use instead of affine_template when tpl_resid_mm is large)
        "tpl_slice_origin": [[round(float(v), 3) for v in (T2[:3, :3] @ q + T2[:3, 3])] for q in Q],
        "head_range_mm": [float(lo), float(hi)], "n_src_slices": int(st.K),
        "src_inplane_mm": [float(np.linalg.norm(st.a)), float(np.linalg.norm(st.b))],
        **rinfo, **{k: v for k, v in st.info.items() if not k.startswith('_')}, **(extra or {}),
    }
    out_attrs |= qc(img, brain, masks, pl.D)
    if image_1mm_zyx is not None:
        out_attrs["qc_corr_1mm"] = corr_with_1mm(img, image_1mm_zyx, attrs, Q, Ur, Uc, g)
    return NgeoResult(img, brain, masks, out_attrs, g, pl, Q, Ur, Uc)


def _dedupe_index(st: SliceStack, tol: float = DUP_TOL_MM) -> np.ndarray:
    t = st.p @ st.normal
    order = np.argsort(t, kind="stable")
    keep, last = [], -np.inf
    for k in order:
        if t[k] - last > tol:
            keep.append(int(k))
            last = t[k]
    return np.asarray(keep, dtype=int)


def qc(img: np.ndarray, brain: np.ndarray | None, masks: dict, slice_mm: float = 1.0) -> dict:
    """Mask-on-tissue checks: brain voxels in soft tissue (-20..120 HU), air (< -200) and bone (> 300); lesion voxels
    in soft tissue; tissue on the field's border (a cut head)."""
    out = {}
    border = np.concatenate([img[:, 0, :], img[:, -1, :], img[:, :, 0], img[:, :, -1]], axis=1)
    out["qc_edge_tissue"] = float((border > -300).mean())
    if brain is not None and brain.any():
        v = img[brain > 0]
        out |= {"qc_brain_tissue": float(((v >= -20) & (v <= 120)).mean()), "qc_brain_air": float((v < -200).mean()),
                "qc_brain_bone": float((v > 300).mean()), "qc_brain_ml": float(brain.sum() * H_MM * H_MM * slice_mm / 1000.0)}
    for name, m in masks.items():
        if m.any():
            v = img[m > 0]
            out[f"qc_{name}_tissue"] = float(((v >= -20) & (v <= 150)).mean())
            out[f"qc_{name}_vox"] = int((m > 0).sum())
    return out


# ------------------------------------------------------------------------------------------------------- readers

def stack_from_sitk(img) -> SliceStack:
    """A SimpleITK volume as a slice stack: the slice axis is the coarsest one (or, nearly isotropic, the one closest
    to superior-inferior); slices keep the volume's physical frame."""
    sp = np.asarray(img.GetSpacing(), dtype=np.float64)
    Dm = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    o = np.asarray(img.GetOrigin(), dtype=np.float64)
    m = int(np.argmax(sp)) if sp.max() / sp.min() > 1.2 else int(np.argmax(np.abs(Dm[2])))
    eh, ew = [e for e in range(3) if e != m]
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(img).astype(np.float32)          # (z, y, x): numpy axis of ITK axis e is 2 - e
    pix = np.ascontiguousarray(np.moveaxis(arr, [2 - m, 2 - eh, 2 - ew], [0, 1, 2]))
    K = pix.shape[0]
    p = o[None] + np.arange(K)[:, None] * (sp[m] * Dm[:, m])[None]
    return SliceStack(pix, p, sp[eh] * Dm[:, eh], sp[ew] * Dm[:, ew], None, 0.0, {"slice_axis": m, "_axes": (m, eh, ew)})


def nifti_store_frame(st: SliceStack, img, attrs: dict, image_1mm_zyx: np.ndarray | None) -> dict:
    """The 1 mm store's frame for a volume source read by ``geometry.read``. Normally the same as this read (nothing
    to do). Volumes stored by the pre-2026-10-25 nibabel fallback reader (seg_cq500 among the CT) carry a ``native_*``
    grid in another frame and index order: the stored grid indexes the same voxels up to axis flips; the flips are
    chosen by correlating the source with the 1 mm image through each candidate. Sets ``st.q / aq / bq``."""
    size = np.asarray(img.GetSize())
    if "native_size" not in attrs or list(attrs["native_size"]) != list(size):
        return {"store_frame": "true"}
    A_now = _affine_from(img.GetOrigin(), img.GetSpacing(), img.GetDirection())
    A_nat = _affine_from(attrs["native_origin"], attrs["native_spacing"], attrs["native_direction"])
    corners = np.array([[i, j, k, 1] for i in (0, size[0] - 1) for j in (0, size[1] - 1) for k in (0, size[2] - 1)],
                       dtype=np.float64).T
    if np.abs((A_now - A_nat) @ corners).max() < 0.5:
        return {"store_frame": "true"}
    if image_1mm_zyx is None:
        return {"store_frame": "native_mismatch_unresolved"}
    m, eh, ew = st.info["_axes"]
    M = np.linalg.inv(grid_affine(attrs)) @ store_to_template(attrs)
    ks = np.arange(0, st.K, 2)
    step = max(1, min(st.pix.shape[1:]) // 64)
    ii, jj = np.meshgrid(np.arange(0, st.pix.shape[1], step), np.arange(0, st.pix.shape[2], step), indexing="ij")
    src = st.pix[ks][:, ::step, ::step]
    best = None
    for fx in (0, 1):
        for fy in (0, 1):
            for fz in (0, 1):
                flip = np.array([fx, fy, fz])
                F = np.eye(4)                                        # current index -> stored index
                for ax in range(3):
                    if flip[ax]:
                        F[ax, ax], F[ax, 3] = -1.0, size[ax] - 1
                B = A_nat @ F
                idx = np.zeros((4, len(ks), *ii.shape))
                idx[m] = ks[:, None, None]
                idx[eh], idx[ew], idx[3] = ii[None], jj[None], 1.0
                xyz = np.tensordot(M @ B, idx, axes=1)
                ref = ndimage.map_coordinates(np.asarray(image_1mm_zyx, np.float32), [xyz[2], xyz[1], xyz[0]],
                                              order=1, mode="constant", cval=FLOOR_HU, prefilter=False)
                head = (src > AIR_HU) & (ref > AIR_HU)
                c = float(np.corrcoef(np.clip(src[head], -50, 150), np.clip(ref[head], -50, 150))[0, 1]) \
                    if head.sum() > 200 else -1.0
                if best is None or c > best[0]:
                    best = (c, B, flip)
    c, B, flip = best
    info = {"store_frame": "native_flip_" + "".join("xyz"[i] for i in range(3) if flip[i]) if flip.any()
            else "native_same_index", "store_frame_corr": c}
    if not c >= 0.6:
        return info | {"store_frame": "native_mismatch_unresolved"}
    st.q = (B[:3, m][None] * np.arange(st.K)[:, None]) + B[:3, 3][None]
    st.aq, st.bq = B[:3, eh].copy(), B[:3, ew].copy()
    return info


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


@dataclass
class DicomSlice:
    ipp: np.ndarray
    iop: np.ndarray
    ps: tuple
    thick: float
    series: str
    shape: tuple
    blob_idx: int
    instance: int


def parse_headers(blobs: list[bytes]) -> list[DicomSlice]:
    import pydicom

    out = []
    for i, b in enumerate(blobs):
        try:
            ds = pydicom.dcmread(io.BytesIO(b), stop_before_pixels=True, force=True)
            if "ImagePositionPatient" not in ds or "ImageOrientationPatient" not in ds:
                continue
            if int(ds.get("NumberOfFrames", 1) or 1) > 1:
                continue
            itype = "\\".join(str(x) for x in (ds.get("ImageType") or [])).upper()
            if "LOCALIZER" in itype:
                continue
            out.append(DicomSlice(np.asarray([float(x) for x in ds.ImagePositionPatient]),
                                  np.asarray([float(x) for x in ds.ImageOrientationPatient]),
                                  tuple(float(x) for x in ds.PixelSpacing), _f(ds.get("SliceThickness")),
                                  str(ds.get("SeriesInstanceUID", "")), (int(ds.Rows), int(ds.Columns)), i,
                                  int(_f(ds.get("InstanceNumber"), 0))))
        except Exception:                                    # noqa: BLE001  a non-image / broken member
            continue
    return out


def pick_series(hdrs: list[DicomSlice]) -> tuple[list[DicomSlice], dict]:
    """The largest series (SimpleITK's choice), then within it the majority orientation / matrix / pixel size."""
    by: dict[str, list] = {}
    for h in hdrs:
        by.setdefault(h.series, []).append(h)
    if not by:
        raise RuntimeError("no image slices")
    series = max(by.values(), key=len)
    key = lambda h: (tuple(np.round(h.iop, 3)), h.shape, tuple(np.round(h.ps, 4)))  # noqa: E731
    groups: dict = {}
    for h in series:
        groups.setdefault(key(h), []).append(h)
    main = max(groups.values(), key=len)
    return main, {"n_series": len(by), "n_files_series": len(series), "n_geom_dropped": len(series) - len(main)}


def decode_hu(blob: bytes) -> np.ndarray:
    """Pixels of one CT slice in HU (rescale applied; the RSNA unsigned-12-bit wrap repaired)."""
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(blob), force=True)
    x = ds.pixel_array.astype(np.float32)
    slope, icpt = _f(ds.get("RescaleSlope"), 1.0) or 1.0, _f(ds.get("RescaleIntercept"), 0.0)
    if int(ds.get("BitsStored", 16)) == 12 and int(ds.get("PixelRepresentation", 1)) == 0 and icpt > -100:
        # RSNA ICH: some unsigned 12-bit slices carry intercept 0 with values wrapped at 4096 (known dataset quirk)
        x = x + 1000.0
        x[x >= 4096] -= 4096
        icpt = -1000.0
    return x * slope + icpt


def repair_ct(pix: np.ndarray) -> tuple[np.ndarray, dict]:
    """Volume-level CT repairs as prep does them: a +1024 offset (air near 0), headers rescaling pixels that are
    already HU (``dicom.stored_is_hu`` pattern), then the store's range."""
    s = pix[::2, ::4, ::4]
    info = {"offset_repaired": False}
    if np.percentile(s, 1) > -200 and ((s > 1000) & (s < 1100)).mean() > 0.05:
        pix = pix - 1024.0
        info["offset_repaired"] = True
    return np.clip(pix, FLOOR_HU, MAX_HU).astype(np.float32, copy=False), info


def dicom_geometry(blobs: list[bytes]) -> tuple[SliceStack, list[DicomSlice], dict]:
    """Position-based assembly of one series, headers only: every file of the chosen series / geometry group in input
    order (duplicates included; ``build`` sorts and de-duplicates), pixels not decoded yet (``decode_stack``)."""
    hdrs = parse_headers(blobs)
    main, info = pick_series(hdrs)
    h0 = main[0]
    b = h0.iop[:3] * h0.ps[1]                                # along increasing column index j (DICOM row cosine)
    a = h0.iop[3:] * h0.ps[0]                                # along increasing row index i (column cosine)
    p = np.stack([h.ipp for h in main])
    thick = float(np.median([h.thick for h in main]))
    st = SliceStack(np.zeros((0, *h0.shape), np.float32), p, a, b, None, thick, {},
                    np.asarray([h.blob_idx for h in main], dtype=int))
    return st, main, info


def decode_stack(st: SliceStack, blobs: list[bytes]) -> tuple[SliceStack, dict]:
    """Decode the pixels of every slice of ``st`` (HU), then the CT repairs (header rescale of already-HU pixels,
    +1024 offset, range)."""
    import pydicom

    from brain.prep.dicom import stored_is_hu

    pix = np.stack([decode_hu(blobs[i]) for i in st.src]).astype(np.float32)
    ds = pydicom.dcmread(io.BytesIO(blobs[int(st.src[0])]), stop_before_pixels=True, force=True)
    slope, icpt = _f(ds.get("RescaleSlope"), 1.0) or 1.0, _f(ds.get("RescaleIntercept"), 0.0)
    undone = bool(stored_is_hu(pix[::2, ::2, ::2], slope, icpt))
    if undone:
        pix = (pix - icpt) / slope
    pix, info = repair_ct(pix)
    out = SliceStack(pix, st.p, st.a, st.b, st.q, st.thick, dict(st.info), st.src)
    return out, info | {"rescale_undone": undone}
