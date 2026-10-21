"""GPU (PyTorch) version of ``registration.register``: same inputs, same rigid -> affine recipe, same acceptance rules.

The SimpleITK fit spends ~18 s per case on the CPU and was 83% of prep time. This one optimises the same transform
family on the GPU in about a second:

* fixed image = the 2 mm template (brain-only for skull-stripped scans), moving image = ``registration._regview``
  of the scan (2 mm isotropic, CT clamped to [-300, 700] HU, MRI to its 0.5-99.5 percentiles);
* metric = mutual information from a soft (Gaussian Parzen) joint histogram, 32 bins: differentiable, and like Mattes
  MI it needs no intensity relationship, so CT and every MRI contrast fit the same T1 template;
* transform ``p -> A (p - c) + c + t`` about the template brain center ``c``, started from the centering translation;
  rigid (rotation vector + translation), then affine (rigid times ``I + S``);
* coarse to fine: 8, 4, 2 mm fixed grids with matched smoothing of the moving image; regular-step gradient descent
  in millimetre-scaled parameters (as SimpleITK); a random third of the fixed points per step at 2 mm, all of them at
  the coarse levels; the best pose seen on each level is kept;
* acceptance as before: each stage must be sane (singular values in [0.65, 1.5], rotation <= 35 deg) and raise the
  metric over the previous pose (evaluated on every fixed point), otherwise the previous pose is kept.

``reg_backend = "torch"`` is recorded on every volume; ``reg_mi`` is this metric (higher is better), not Mattes.
"""

from __future__ import annotations

import math
import time

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from brain.prep import registration as R

BINS = 32
LEVELS_RIGID = ((4, 120), (2, 100), (1, 80))   # (fixed downsampling of the 2 mm template, steps)
LEVELS_AFFINE = ((4, 100), (2, 100), (1, 80))   # scale is only well-behaved at 8 mm: start there too
SAMPLE_FRAC = 1 / 3


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _geom(img: sitk.Image, dev) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(origin (3,), index->physical matrix (3, 3), its inverse) in float64."""
    m = np.array(img.GetDirection()).reshape(3, 3) @ np.diag(img.GetSpacing())
    to = lambda a: torch.as_tensor(np.asarray(a, np.float64), device=dev)  # noqa: E731
    return to(img.GetOrigin()), to(m), to(np.linalg.inv(m))


def _unit(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / max(hi - lo, 1e-6)).astype(np.float32)


class _Fixed:
    """Fixed-image pyramid: voxel values in [0, 1] and physical positions of the voxel centers."""

    def __init__(self, img: sitk.Image, dev):
        arr = torch.from_numpy(_unit(sitk.GetArrayFromImage(img))).to(dev)[None, None]
        origin, m, _ = _geom(img, dev)
        self.levels = {}
        for f in (1, 2, 4):
            a = arr if f == 1 else F.avg_pool3d(arr, f, f)
            nz, ny, nx = a.shape[-3:]
            zz, yy, xx = torch.meshgrid(*(torch.arange(n, device=dev, dtype=torch.float64) * f + (f - 1) / 2
                                          for n in (nz, ny, nx)), indexing="ij")
            idx = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
            self.levels[f] = (a.reshape(-1), origin + idx @ m.T)


class _Moving:
    """Moving-image pyramid for trilinear sampling at physical points."""

    def __init__(self, img: sitk.Image, dev):
        arr = torch.from_numpy(_unit(sitk.GetArrayFromImage(img))).to(dev)[None, None]
        self.origin, _, self.inv = _geom(img, dev)
        self.levels = {f: (arr if f == 1 else F.avg_pool3d(arr, f, f, ceil_mode=True)) for f in (1, 2, 4)}

    def sample(self, pts: torch.Tensor, f: int) -> tuple[torch.Tensor, torch.Tensor]:
        vol = self.levels[f]
        idx = (pts - self.origin) @ self.inv.T                     # fine (2 mm) voxel index, (x, y, z)
        idx = (idx - (f - 1) / 2) / f                              # index into the pooled level
        size = torch.tensor(vol.shape[-3:][::-1], device=pts.device, dtype=pts.dtype)  # (x, y, z)
        g = 2 * idx / (size - 1).clamp_min(1) - 1
        inside = (g.abs() <= 1).all(-1)
        v = F.grid_sample(vol, g.float().view(1, 1, 1, -1, 3), mode="bilinear", align_corners=True)
        return v.view(-1), inside


def _soft_hist(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    centers = (torch.arange(BINS, device=v.device, dtype=v.dtype) + 0.5) / BINS
    k = torch.exp(-0.5 * ((v[:, None] - centers) * BINS) ** 2)   # sigma = one bin
    return k / k.sum(1, keepdim=True).clamp_min(1e-12) * w[:, None]


def mutual_information(fv: torch.Tensor, mv: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    w = valid.to(fv.dtype)
    p = _soft_hist(fv, w).T @ _soft_hist(mv, torch.ones_like(w))
    p = p / p.sum().clamp_min(1e-12)
    pf, pm = p.sum(1, keepdim=True), p.sum(0, keepdim=True)
    return (p * (torch.log(p + 1e-10) - torch.log(pf @ pm + 1e-10))).sum()


def _rotation(rv: torch.Tensor) -> torch.Tensor:
    """Rodrigues: rotation vector (3,) -> (3, 3)."""
    theta = rv.norm().clamp_min(1e-12)
    k = rv / theta
    kx = torch.zeros(3, 3, dtype=rv.dtype, device=rv.device)
    kx[0, 1], kx[0, 2], kx[1, 0], kx[1, 2], kx[2, 0], kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    eye = torch.eye(3, dtype=rv.dtype, device=rv.device)
    return eye + torch.sin(theta) * kx + (1 - torch.cos(theta)) * (kx @ kx)


class _Pose(torch.nn.Module):
    def __init__(self, t0: np.ndarray, center, dev, A0: np.ndarray | None = None):
        super().__init__()
        self.center = torch.as_tensor(np.asarray(center, np.float64), device=dev)
        self.base = torch.as_tensor(np.eye(3) if A0 is None else A0, device=dev, dtype=torch.float64)
        self.rv = torch.nn.Parameter(torch.full((3,), 1e-6, dtype=torch.float64, device=dev))
        self.t = torch.nn.Parameter(torch.as_tensor(t0, dtype=torch.float64, device=dev).clone())
        self.s = torch.nn.Parameter(torch.zeros(3, 3, dtype=torch.float64, device=dev))
        self.affine = False

    def matrix(self) -> torch.Tensor:
        a = _rotation(self.rv) @ self.base
        if self.affine:
            a = a @ (torch.eye(3, dtype=a.dtype, device=a.device) + self.s)
        return a

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        return (pts - self.center) @ self.matrix().T + self.center + self.t


def _metric(pose: _Pose, fixed: _Fixed, moving: _Moving, level: int = 1) -> float:
    with torch.no_grad():
        fv, pts = fixed.levels[level]
        mv, inside = moving.sample(pose(pts), level)
        return float(mutual_information(fv, mv, inside))


RADIUS_MM = 80.0      # rotation and linear terms are scaled by a head radius so every parameter moves ~1 mm per unit
STEP_MM = {4: 4.0, 2: 2.0, 1: 1.0}   # first step length per level (fixed downsampling factor -> mm)
MIN_STEP_MM = 0.02
CHECK_EVERY = 10
PITCH_STARTS_DEG = (12.0, -12.0, 24.0, -24.0)   # rigid restarts when the fit from the identity is rejected


def _pitch(deg: float) -> np.ndarray:
    """Rotation about the physical x (left-right) axis."""
    r = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(r), -math.sin(r)], [0.0, math.sin(r), math.cos(r)]])


def _fit(pose: _Pose, fixed: _Fixed, moving: _Moving, levels, seed: int) -> int:
    """Regular-step gradient descent (as SimpleITK's optimizer) in millimetre-scaled parameters: each step moves a
    fixed length along the *whole* normalised gradient, so a nearly flat direction gets a nearly zero share (Adam's
    per-parameter normalisation would walk it at full speed); the step halves whenever the direction reverses.
    Every ``CHECK_EVERY`` steps the pose is scored on all points of the level and the best is kept."""
    gen = torch.Generator(device=fixed.levels[1][0].device).manual_seed(seed)
    params = [(pose.rv, RADIUS_MM), (pose.t, 1.0)] + ([(pose.s, RADIUS_MM)] if pose.affine else [])
    steps = 0
    for f, n in levels:
        fv_all, pts_all = fixed.levels[f]
        best, best_state = _metric(pose, fixed, moving, f), [p.detach().clone() for p, _ in params]
        step, prev = STEP_MM[f], None
        for i in range(n):
            if f == 1:
                pick = torch.randperm(len(pts_all), generator=gen, device=pts_all.device)[: int(SAMPLE_FRAC * len(pts_all))]
                fv, pts = fv_all[pick], pts_all[pick]
            else:
                fv, pts = fv_all, pts_all
            for p, _ in params:
                p.grad = None
            mv, inside = moving.sample(pose(pts), f)
            (-mutual_information(fv, mv, inside)).backward()
            g = torch.cat([(p.grad / sc).reshape(-1) for p, sc in params])       # gradient in mm units
            norm = g.norm()
            if not torch.isfinite(norm) or norm == 0:
                break
            if prev is not None and float(g @ prev) < 0:
                step *= 0.5
            if step < MIN_STEP_MM:
                break
            prev = g
            with torch.no_grad():
                off = 0
                for p, sc in params:
                    k = p.numel()
                    p -= (step * g[off:off + k] / norm).view_as(p) / sc
                    off += k
            steps += 1
            if (i + 1) % CHECK_EVERY == 0 or i + 1 == n:
                m = _metric(pose, fixed, moving, f)
                if m > best:
                    best, best_state = m, [p.detach().clone() for p, _ in params]
        m = _metric(pose, fixed, moving, f)
        if m > best:
            best, best_state = m, [p.detach().clone() for p, _ in params]
        with torch.no_grad():
            for (p, _), v in zip(params, best_state):
                p.copy_(v)
    return steps


def _to_sitk(pose: _Pose) -> sitk.AffineTransform:
    tf = sitk.AffineTransform(3)
    c = pose.center.detach().cpu().numpy()
    tf.SetCenter(tuple(float(v) for v in c))
    tf.SetMatrix(tuple(float(v) for v in pose.matrix().detach().cpu().numpy().ravel()))
    tf.SetTranslation(tuple(float(v) for v in pose.t.detach().cpu().numpy()))
    return tf


def register(moving: sitk.Image, template: R.Template, modality: str, init_center_mm, mode: str = "affine",
             seed: int = 0, stripped: bool = False) -> R.RegResult:
    """Drop-in for ``registration.register`` (same arguments, same ``RegResult`` and info keys).

    The GPU is shared with other prep workers and the SynthStrip batches, so an out-of-memory error waits and retries,
    then runs the same fit on the CPU; it never degrades to the bare centering translation."""
    if mode not in R.MODES[:2]:
        raise ValueError(f"registration mode must be one of {R.MODES[:2]}, got {mode!r}")
    dev = _device()
    for attempt in range(4 if dev.type == "cuda" else 1):
        try:
            res = _register(moving, template, modality, init_center_mm, mode, seed, stripped, dev)
            res.info["reg_oom_retries"] = attempt
            return res
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            time.sleep(3 + 5 * attempt)
    res = _register(moving, template, modality, init_center_mm, mode, seed, stripped, torch.device("cpu"))
    res.info["reg_oom_retries"] = 4
    return res


def _register(moving, template, modality, init_center_mm, mode, seed, stripped, dev) -> R.RegResult:
    use_brain = stripped and template.image_brain is not None
    fixed = _Fixed(template.image_brain if use_brain else template.image, dev)
    mov = _Moving(R._regview(moving, modality), dev)
    shift = np.asarray(init_center_mm, float) - np.asarray(template.center_mm, float)
    start = sitk.Euler3DTransform()
    start.SetCenter(template.center_mm)
    start.SetTranslation(tuple(float(v) for v in shift))
    info: dict = {"reg_mode": mode, "reg_method": "translation", "reg_template": "brain" if use_brain else "head",
                  "reg_backend": "torch", "reg_device": dev.type}
    best: sitk.Transform = start
    torch.manual_seed(seed)
    try:
        pose = _Pose(shift, template.center_mm, dev)
        mi0 = _metric(pose, fixed, mov)
        steps = _fit(pose, fixed, mov, LEVELS_RIGID, seed)
        mi_r, rigid = _metric(pose, fixed, mov), _to_sitk(pose)
        info |= {"reg_mi_init": mi0, "reg_mi": mi0, "reg_iters": steps, "reg_starts": 1}
        if not (R._sane(rigid) and mi_r > mi0):
            # large head tilts (CT gantry tilt, flexed necks) can leave the fit from the identity in a wrong basin:
            # restart from pitched poses (about the left-right axis) and keep the best acceptable one
            for k, pitch in enumerate(PITCH_STARTS_DEG, start=1):
                cand = _Pose(shift, template.center_mm, dev, A0=_pitch(pitch))
                info["reg_iters"] += _fit(cand, fixed, mov, LEVELS_RIGID, seed + k)
                info["reg_starts"] += 1
                m, tf = _metric(cand, fixed, mov), _to_sitk(cand)
                if R._sane(tf) and m > mi0 and (not (R._sane(rigid) and mi_r > mi0) or m > mi_r):
                    pose, mi_r, rigid = cand, m, tf
        if R._sane(rigid) and mi_r > mi0:
            best, info["reg_method"], info["reg_mi"] = rigid, "rigid", mi_r
            if mode == "affine":
                aff = _Pose(pose.t.detach().cpu().numpy(), template.center_mm, dev,
                            A0=pose.matrix().detach().cpu().numpy())
                aff.affine = True
                info["reg_iters"] += _fit(aff, fixed, mov, LEVELS_AFFINE, seed + 1)
                mi_a = _metric(aff, fixed, mov)
                fit = _to_sitk(aff)
                if R._sane(fit) and mi_a > mi_r:
                    best, info["reg_method"], info["reg_mi"] = fit, "affine", mi_a
    except torch.OutOfMemoryError:
        raise                                          # handled by register(): retry, then CPU
    except RuntimeError as e:                          # numerical failure: keep the best pose so far
        info["reg_error"] = str(e)[:120]
    p = R.pose_summary(best)
    moved = np.array(best.TransformPoint(tuple(template.center_mm))) - np.asarray(init_center_mm, float)
    info |= {"reg_rot_deg": p["rot_deg"], "reg_scale": p["scale"], "reg_aniso": p["aniso"],
             "reg_shift_mm": float(np.linalg.norm(moved)), "reg_ok": info["reg_method"] != "translation"}
    return R.RegResult(best, info)
