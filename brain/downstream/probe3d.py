"""Frozen-token probes that measure what a pretrained encoder keeps, without a task head per dataset.

Every volume is encoded once and every encoder reads the same volume, so the comparison is paired.
The default patch grid is the 3D encoder's token grid (``brain.common.grid.TOKEN_GRID``: 20 x 16 x 16 tokens of
16 x 16 x 8 voxels on the 256 x 256 x 160 volume), flattened (Z, Y, X). ``--fine-grid`` keeps the tiled 8 x 8 x 4
encode on ``FINE_TOKEN_GRID`` (40 x 32 x 32). The 2D slice model is averaged over the 8 slices of each coarse patch.

1. Dense lesion probe: from one token, is there lesion inside its patch? Linear and one-hidden-layer
   probes per held-out dataset, scored by average precision against the prevalence (lift) and AUROC.
2. Fingerprint probe: how well a volume-pooled embedding recovers the modality and the source
   dataset. High values mean the latent space spends capacity on scanner and protocol.
3. Location probe: how well one token predicts where in the volume it came from. A latent space that
   knows anatomy places its tokens; one that does not has near-zero R2.

Controls: ``--pixel`` (intensity statistics per patch) and ``--random-init`` (same 3D architecture,
untrained). A useful encoder must beat both on the lesion probe.

    python -m brain.downstream.probe3d --ckpt runs/a.pt --ckpt runs/b.pt --pixel --random-init --out runs/probes/x.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from brain.common.grid import FINE_PATCH_XYZ, FINE_TOKEN_GRID, PATCH_XYZ, TOKEN_GRID
from brain.common.paths import PROCESSED_ROOT
from brain.prep import store
from brain.downstream.embed3d import _crop_to_patch, read_volume_xyz, token_gate
from brain.downstream.embeddings import load_frozen_encoder, resolve_path
from brain.common.windowing import asinh_window, windows_for

GRID = TOKEN_GRID  # (Z, Y, X) tokens
FINE_GRID = FINE_TOKEN_GRID
N_TOK = GRID[0] * GRID[1] * GRID[2]
LABEL_MASK_COLS = ("mask_seg", "mask_lesion", "mask_ich")  # mask_brain is not a label
TILE_WINDOW = (128, 128, 64)  # the training global-crop size, in voxels (X, Y, Z)
TILE_STRIDE = (64, 64, 32)
LESION_FAMILIES = ("atlas", "isles22", "isles24", "wmh", "cq500", "aisd", "instance")
SEEN_FAMILIES = ("brats_gli",)  # images were in pretraining, masks never were
DEFAULT_ROOT = str(PROCESSED_ROOT)


# ----------------------------------------------------------------------------- data

@dataclass(frozen=True)
class Case:
    vid: str
    family: str
    modality: str
    image: Path
    mask: Path | None
    split: str


def load_cases(root: Path, families, split: str, n: int, seed: int, *, masked: bool) -> list[Case]:
    df = pd.read_csv(root / "volumes.csv", low_memory=False)
    df = df[df.status == "ok"]
    out = []
    for fam in families:
        d = df[(df.dataset_family == fam) & (df.split == split)]
        if masked:
            cols = [c for c in LABEL_MASK_COLS if c in d.columns]
            d = d[d[cols].fillna("").astype(str).ne("").any(axis=1)] if cols else d.iloc[:0]
        d = d.sample(min(n, len(d)), random_state=seed)
        for r in d.itertuples():
            m = next((x for x in (getattr(r, c, None) for c in LABEL_MASK_COLS) if isinstance(x, str) and x), None)
            out.append(Case(r.volume_id, fam, r.modality, resolve_path(root, r.image_path),
                            resolve_path(root, m) if (masked and m) else None, split))
    return out


def lesion_grid(mask_zyx: np.ndarray, family: str, grid: tuple[int, int, int] = GRID,
                patch_xyz: tuple[int, int, int] = PATCH_XYZ) -> np.ndarray:
    """Lesion fraction inside each patch, (prod(grid),) in token order. WMH label 2 is not lesion."""
    if family == "wmh":
        m = mask_zyx == 1
    elif family == "aisd":
        m = np.isin(mask_zyx, (1, 2, 3, 5))
    else:
        m = mask_zyx > 0
    z, y, x = grid
    pz, py, px = patch_xyz[2], patch_xyz[1], patch_xyz[0]
    return m.reshape(z, pz, y, py, x, px).mean(axis=(1, 3, 5)).reshape(-1).astype(np.float32)


def token_positions(grid: tuple[int, int, int] = GRID) -> np.ndarray:
    """(prod(grid), 3) normalized (z, y, x) of each token in [0, 1]."""
    zz, yy, xx = np.meshgrid(*[np.linspace(0, 1, n) for n in grid], indexing="ij")
    return np.stack([zz, yy, xx], -1).reshape(-1, 3).astype(np.float32)


# ----------------------------------------------------------------------------- providers

class Provider:
    """``tokens(vol_xyz, modality) -> (T, D) float, (T,) bool`` for one encoder."""

    name: str
    patch_xyz = PATCH_XYZ
    grid = GRID
    ckpt: Path | None = None       # for experiment records; None for controls
    variant: str | None = None

    def tokens(self, vol: torch.Tensor, modality: str) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Brain3D(Provider):
    """3D encoder tokens. With ``window`` the volume is encoded as overlapping training-size windows."""

    def __init__(self, enc, name: str, window=None, stride=None, fine_pool: bool | None = None,
                 fine_grid: bool | None = None):
        self.enc, self.name, self.window, self.stride = enc, name, window, stride
        self.fine_pool, self.fine_grid = fine_pool, fine_grid
        if fine_grid:
            self.patch_xyz, self.grid = FINE_PATCH_XYZ, FINE_GRID
        else:
            self.patch_xyz, self.grid = PATCH_XYZ, GRID

    def tokens(self, vol, modality):
        seqs, _, fg = self.enc.embed_windows(
            vol, windows_for("canonical", modality), modality=modality,
            window_xyz=self.window, stride_xyz=self.stride,
            fine_pool=self.fine_pool, fine_grid=self.fine_grid)
        # the model's own global embeddings for this volume (windows concatenated), when it has prefix tokens
        self.last_global = {}
        if self.enc.last_cls is not None:
            self.last_global["cls"] = self.enc.last_cls.float().reshape(-1).cpu().numpy()
        if self.enc.last_reg is not None:                      # (n_windows, R, D) -> keep the register axis
            r = self.enc.last_reg.float().cpu().numpy()
            self.last_global["registers"] = np.transpose(r, (1, 0, 2)).reshape(r.shape[1], -1)   # (R, n_win * D)
        return torch.cat(list(seqs), -1), fg


class Slice2D(Provider):
    """2D ViT patch tokens per axial slice, averaged over the 8 slices of each 3D patch."""

    def __init__(self, enc, name: str, batch: int = 64):
        self.model, self.name, self.batch = enc.model, name, batch

    @torch.no_grad()
    def tokens(self, vol, modality):
        zyx = vol.permute(2, 1, 0)[:, None]  # (Z, 1, Y, X)
        fg = token_gate(_crop_to_patch(vol.float(), PATCH_XYZ)[0], modality, PATCH_XYZ)
        feats = []
        for c, w in windows_for("canonical", modality):
            x, outs = asinh_window(zyx, c, w), []
            for i in range(0, x.shape[0], self.batch):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                    outs.append(self.model.forward_features(x[i:i + self.batch])[1].float())
            p = torch.cat(outs)  # (Z, Y*X patches, C)
            z, n, ch = p.shape
            feats.append(p.view(GRID[0], z // GRID[0], n, ch).mean(1).reshape(N_TOK, ch))
        return torch.cat(feats, -1), fg


class Pixel(Provider):
    """Control: mean, std, min, max of the windowed intensity inside each patch."""

    name = "control/pixel"

    def __init__(self, name: str = "control/pixel", patch_xyz=PATCH_XYZ, grid=GRID):
        self.name = name
        self.patch_xyz, self.grid = tuple(patch_xyz), tuple(grid)

    @torch.no_grad()
    def tokens(self, vol, modality):
        px, py, pz = self.patch_xyz
        z, y, x = self.grid
        v = _crop_to_patch(vol.float(), self.patch_xyz)[0]
        fg = token_gate(v, modality, self.patch_xyz)
        feats = []
        for c, w in windows_for("canonical", modality):
            p = asinh_window(v, c, w).permute(2, 1, 0).reshape(z, pz, y, py, x, px)
            p = p.permute(0, 2, 4, 1, 3, 5).reshape(z * y * x, -1)
            feats.append(torch.stack([p.mean(1), p.std(1), p.amin(1), p.amax(1)], -1))
        return torch.cat(feats, -1), fg


def build_providers(ckpts, device, *, pixel: bool, random_init: bool, tiled: bool = False,
                    fine: bool = False, fine_grid: bool = False) -> list[Provider]:
    from brain.downstream.embed3d import FrozenEncoder3D, load_brain3d

    out, first3d = [], None
    for path in ckpts:
        enc = load_frozen_encoder(Path(path), device)
        name = f"{Path(path).parent.name}/{Path(path).stem}"
        start = len(out)
        if isinstance(enc, FrozenEncoder3D):
            out.append(Brain3D(enc, name, fine_pool=False, fine_grid=False))
            if fine:
                out.append(Brain3D(enc, name + "@fine", fine_pool=True, fine_grid=False))
            if fine_grid:
                out.append(Brain3D(enc, name + "@finegrid", fine_pool=False, fine_grid=True))
            if tiled:
                out.append(Brain3D(enc, name + "@tiled", TILE_WINDOW, TILE_STRIDE, fine_pool=False, fine_grid=False))
            first3d = first3d or Path(path)
        else:
            out.append(Slice2D(enc, name))
        for prov in out[start:]:
            prov.ckpt, prov.variant = Path(path), (prov.name.split("@", 1)[1] if "@" in prov.name else None)
    if random_init and first3d is not None:
        ckpt = torch.load(first3d, map_location=device, weights_only=False)
        torch.manual_seed(0)
        out.append(Brain3D(load_brain3d(ckpt, device, pretrained=False), "control/random3d",
                           fine_pool=False, fine_grid=False))
        if fine_grid:
            torch.manual_seed(0)
            out.append(Brain3D(load_brain3d(ckpt, device, pretrained=False), "control/random3d@finegrid",
                               fine_pool=False, fine_grid=True))
    if pixel:
        out.append(Pixel())
        if fine_grid:
            out.append(Pixel(name="control/pixel@finegrid", patch_xyz=FINE_PATCH_XYZ, grid=FINE_GRID))
    return out


# ----------------------------------------------------------------------------- extraction

@dataclass
class Volume:
    case: Case
    feat: torch.Tensor  # (n_fg, D) float16, on the device
    frac: np.ndarray | None  # (n_fg,) lesion fraction
    idx: np.ndarray  # (n_fg,) token index
    pooled: np.ndarray  # (D,) mean over foreground tokens
    grid: tuple[int, int, int] = GRID
    globals: dict = field(default_factory=dict)   # "cls": (D,), "registers": (R, D) for prefix-token encoders
    norm_ratio: np.ndarray | None = None          # (n_fg,) token L2 norm / the volume's median token norm


def extract(providers: list[Provider], cases: list[Case], device) -> dict[str, list[Volume]]:
    """One read per volume; every provider embeds it."""
    out = {p.name: [] for p in providers}
    t0 = time.time()
    for i, c in enumerate(cases):
        vol = torch.from_numpy(np.ascontiguousarray(read_volume_xyz(c.image))).to(device)
        mask = store.read(c.mask, "mask") if c.mask is not None else None
        frac_by: dict[tuple, np.ndarray] = {}
        for p in providers:
            feat, fg = p.tokens(vol, c.modality)
            keep = fg.bool().cpu().numpy()
            idx = np.flatnonzero(keep)
            if idx.size == 0:
                continue
            if mask is not None:
                key = (p.grid, p.patch_xyz)
                if key not in frac_by:
                    frac_by[key] = lesion_grid(mask, c.family, grid=p.grid, patch_xyz=p.patch_xyz)
                frac = frac_by[key]
            else:
                frac = None
            f = feat[torch.from_numpy(idx).to(device)]
            norms = f.float().norm(dim=-1)
            out[p.name].append(Volume(c, f.half(), None if frac is None else frac[idx], idx,
                                      f.float().mean(0).cpu().numpy(), grid=p.grid,
                                      globals=dict(getattr(p, "last_global", {}) or {}),
                                      norm_ratio=(norms / norms.median().clamp_min(1e-6)).cpu().numpy()))
        if (i + 1) % 50 == 0 or i + 1 == len(cases):
            print(f"[probe] {i + 1}/{len(cases)} volumes, {time.time() - t0:.0f}s", flush=True)
    return out


# ----------------------------------------------------------------------------- probes

def _fit_token_probe(x: torch.Tensor, y: torch.Tensor, hidden: int, seed: int, steps: int = 300) -> torch.nn.Module:
    torch.manual_seed(seed)
    d = x.shape[1]
    net = torch.nn.Linear(d, 1) if hidden == 0 else torch.nn.Sequential(
        torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))
    net = net.to(x.device)
    pos = y.float().mean().clamp_min(1e-6)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).clamp(1.0, 100.0))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2 if hidden == 0 else 3e-3, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss_fn(net(x).squeeze(-1), y.float()).backward()
        opt.step()
    return net.eval()


def lesion_probe(train: list[Volume], test: list[Volume], hidden: int, seed: int = 0) -> dict:
    dev = train[0].feat.device
    xtr = torch.cat([v.feat for v in train]).float()
    ytr = torch.from_numpy(np.concatenate([v.frac for v in train]) > 0).to(dev)
    xte = torch.cat([v.feat for v in test]).float()
    yte = np.concatenate([v.frac for v in test]) > 0
    if yte.sum() == 0 or ytr.sum() == 0:
        return {"ap": float("nan"), "auroc": float("nan"), "prevalence": float(yte.mean()), "lift": float("nan")}
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-4)
    net = _fit_token_probe((xtr - mu) / sd, ytr, hidden, seed)
    with torch.no_grad():
        s = net((xte - mu) / sd).squeeze(-1).cpu().numpy()
    ap = float(average_precision_score(yte, s))
    cuts = np.cumsum([len(v.frac) for v in test])[:-1]
    return {"ap": ap, "auroc": float(roc_auc_score(yte, s)), "prevalence": float(yte.mean()),
            "lift": ap / float(yte.mean()), "_scores": np.split(s, cuts), "_labels": np.split(yte, cuts)}


def global_view(v: Volume, view: str) -> np.ndarray:
    """``mean`` (patch-token mean), ``cls``, ``registers`` (all, concatenated), ``cls+registers``, ``reg<i>``."""
    if view == "mean":
        return v.pooled
    if view == "registers":
        return v.globals["registers"].reshape(-1)
    if view == "cls+registers":
        return np.concatenate([v.globals["cls"], v.globals["registers"].reshape(-1)])
    if view.startswith("reg") and view[3:].isdigit():
        return v.globals["registers"][int(view[3:])]
    return v.globals[view]


def global_views(vols: list[Volume]) -> list[str]:
    g = vols[0].globals if vols else {}
    views = ["mean"] + (["cls"] if "cls" in g else []) + (["registers"] if "registers" in g else [])
    return views + (["cls+registers"] if "cls" in g and "registers" in g else [])


def fingerprint_probe(vols: list[Volume], seed: int = 0, view: str = "mean") -> dict:
    x = StandardScaler().fit_transform(np.stack([global_view(v, view) for v in vols]))
    out = {}
    for key, labels in (("modality", [v.case.modality for v in vols]), ("dataset", [v.case.family for v in vols])):
        y = np.asarray(labels)
        counts = pd.Series(y).value_counts()
        keep = np.isin(y, counts[counts >= 5].index)
        if len(np.unique(y[keep])) < 2:
            out[f"{key}_bal_acc"] = out[f"{key}_chance"] = float("nan")
            continue
        folds = StratifiedKFold(5, shuffle=True, random_state=seed)
        pred = cross_val_predict(LogisticRegression(max_iter=2000, C=0.1), x[keep], y[keep], cv=folds)
        out[f"{key}_bal_acc"] = float(balanced_accuracy_score(y[keep], pred))
        out[f"{key}_chance"] = 1.0 / len(np.unique(y[keep]))
        out[f"_{key}_rows"] = (y[keep], pred)
    return out


def location_probe(vols: list[Volume], per_volume: int = 200, seed: int = 0) -> dict:
    rng, pos = np.random.default_rng(seed), token_positions(vols[0].grid if vols else GRID)
    xs, ys, groups = [], [], []
    for i, v in enumerate(vols):
        pick = rng.choice(len(v.idx), min(per_volume, len(v.idx)), replace=False)
        xs.append(v.feat[torch.from_numpy(pick).to(v.feat.device)].float().cpu().numpy())
        ys.append(pos[v.idx[pick]])
        groups += [i] * len(pick)
    x, y, g = np.concatenate(xs), np.concatenate(ys), np.asarray(groups)
    half = g % 2 == 0  # volumes alternate between fit and held-out
    sc = StandardScaler().fit(x[half])
    pred = Ridge(alpha=10.0).fit(sc.transform(x[half]), y[half]).predict(sc.transform(x[~half]))
    r2 = [float(r2_score(y[~half][:, k], pred[:, k])) for k in range(3)]
    ev = np.linalg.eigvalsh(np.cov(x[~half], rowvar=False).astype(np.float64)).clip(0)
    held = ~half
    owner = np.unique(g[held])  # one sufficient statistic per held-out volume, for the bootstrap
    stats = [(y[held & (g == i)], pred[g[held] == i], x[held & (g == i)]) for i in owner]
    return {"r2_z": r2[0], "r2_y": r2[1], "r2_x": r2[2], "r2_mean": float(np.mean(r2)),
            "token_participation_ratio": float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12)), "dim": int(x.shape[1]),
            "_volumes": stats}


def _r2_mean(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([r2_score(true[:, k], pred[:, k]) for k in range(3)]))


def _participation(xs: list[np.ndarray]) -> float:
    n = sum(len(a) for a in xs)
    s1 = sum(a.sum(0, dtype=np.float64) for a in xs)
    s2 = sum(a.T.astype(np.float64) @ a.astype(np.float64) for a in xs)
    mean = s1 / n
    ev = np.linalg.eigvalsh((s2 - n * np.outer(mean, mean)) / max(n - 1, 1)).clip(0)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))


METRICS = ("lift_lin", "lift_mlp", "auroc_lin", "dataset_acc", "loc_r2", "pr")


def bootstrap_probes(raw: dict, reps: int = 200, seed: int = 0) -> dict[str, np.ndarray]:
    """Resample test volumes (lesion, location) or rows (fingerprint) and recompute every headline metric.

    The draws depend only on the seed and the group sizes, so encoders probed on the same cases get the
    same resamples, and differences between them are paired.
    """
    out = {m: np.full(reps, np.nan) for m in METRICS}
    les = raw["lesion"]
    for r in range(reps):
        lift = {"linear": [], "mlp": []}
        auc = []
        for fam, (lin, mlp) in sorted(les.items()):
            n = len(lin["_scores"])
            if fam not in LESION_FAMILIES or n == 0:
                continue
            idx = np.random.default_rng([seed, r, sum(map(ord, fam)), n]).integers(0, n, n)
            y = np.concatenate([lin["_labels"][i] for i in idx])
            if y.sum() == 0 or y.all():
                continue
            for kind, d in (("linear", lin), ("mlp", mlp)):
                sc = np.concatenate([d["_scores"][i] for i in idx])
                lift[kind].append(average_precision_score(y, sc) / y.mean())
                if kind == "linear":
                    auc.append(roc_auc_score(y, sc))
        if lift["linear"]:
            out["lift_lin"][r], out["lift_mlp"][r], out["auroc_lin"][r] = (
                np.mean(lift["linear"]), np.mean(lift["mlp"]), np.mean(auc))
        if "_dataset_rows" in raw["fingerprint"]:
            y, p = raw["fingerprint"]["_dataset_rows"]
            i = np.random.default_rng([seed, r, 1, len(y)]).integers(0, len(y), len(y))
            out["dataset_acc"][r] = balanced_accuracy_score(y[i], p[i])
        vols = raw["location"]["_volumes"]
        i = np.random.default_rng([seed, r, 2, len(vols)]).integers(0, len(vols), len(vols))
        out["loc_r2"][r] = _r2_mean(np.concatenate([vols[k][0] for k in i]), np.concatenate([vols[k][1] for k in i]))
        out["pr"][r] = _participation([vols[k][2] for k in i])
    return out


def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if not str(k).startswith("_")}
    return d


def register_probe(vols: list[Volume], seed: int = 0) -> dict:
    """What each register carries: modality / dataset decodability of every register on its own, how alike the
    registers are (mean pairwise cosine; near 1 = redundant), and their norm relative to the patch tokens."""
    regs = np.stack([v.globals["registers"] for v in vols])            # (V, R, D)
    n_reg = regs.shape[1]
    per = {f"reg{r}": fingerprint_probe(vols, seed, view=f"reg{r}") for r in range(n_reg)}
    unit = regs / np.linalg.norm(regs, axis=-1, keepdims=True).clip(1e-6)
    cos = np.einsum("vrd,vsd->vrs", unit, unit)
    off = cos[:, ~np.eye(n_reg, dtype=bool)]
    patch_norm = np.array([float(np.linalg.norm(v.pooled)) for v in vols])
    out = {"n_registers": n_reg, "pairwise_cosine": float(off.mean()),
           "norm_vs_patch_mean": float(np.mean(np.linalg.norm(regs, axis=-1).mean(1) / patch_norm.clip(1e-6)))}
    for k in ("modality_bal_acc", "dataset_bal_acc"):
        vals = [per[r][k] for r in per]
        out[f"{k}_mean"], out[f"{k}_max"], out[f"{k}_min"] = (float(np.nanmean(vals)), float(np.nanmax(vals)),
                                                              float(np.nanmin(vals)))
        out[f"{k}_per_register"] = {f"reg{r}": float(x) for r, x in enumerate(vals)}
    return out


def token_norm_probe(vols: list[Volume], thresh: float = 2.0) -> dict:
    """High-norm outlier tokens (Darcet et al., "Vision Transformers Need Registers"): share of foreground tokens whose
    norm exceeds ``thresh`` x the volume median, the largest ratio, and the norms' spread. Models without registers
    tend to repurpose a few tokens as global scratch space; registers should remove them."""
    ratios = [v.norm_ratio for v in vols if v.norm_ratio is not None and v.norm_ratio.size]
    if not ratios:
        return {}
    allr = np.concatenate(ratios)
    return {"outlier_frac": float(np.mean(allr > thresh)), "outlier_threshold": thresh,
            "max_ratio_mean": float(np.mean([r.max() for r in ratios])),
            "p99_ratio": float(np.percentile(allr, 99)), "cv_mean": float(np.mean([r.std() for r in ratios]))}


def run_probes(name: str, lesion_train, lesion_test, fingerprint_vols, reps: int = 200) -> dict:
    res, raw = {"lesion": {}}, {"lesion": {}}
    for fam in LESION_FAMILIES + SEEN_FAMILIES:
        tr = [v for v in lesion_train if v.case.family == fam]
        te = [v for v in lesion_test if v.case.family == fam]
        if not tr or not te:
            continue
        lin, mlp = lesion_probe(tr, te, 0), lesion_probe(tr, te, 256)
        raw["lesion"][fam] = (lin, mlp)
        res["lesion"][fam] = {"linear": lin, "mlp": mlp, "n_train": len(tr), "n_test": len(te)}
    for kind in ("linear", "mlp"):
        for key in ("ap", "lift", "auroc"):
            vals = [res["lesion"][f][kind][key] for f in LESION_FAMILIES if f in res["lesion"]]
            res["lesion"][f"mean_{kind}_{key}"] = float(np.nanmean(vals))
    raw["fingerprint"], raw["location"] = fingerprint_probe(fingerprint_vols), location_probe(fingerprint_vols)
    res["fingerprint"], res["location"] = raw["fingerprint"], raw["location"]
    # every global embedding the model has (CLS, registers), read out the same way as the patch-token mean
    res["fingerprint_views"] = {view: fingerprint_probe(fingerprint_vols, view=view)
                                for view in global_views(fingerprint_vols) if view != "mean"}
    if fingerprint_vols and "registers" in fingerprint_vols[0].globals:
        res["registers"] = register_probe(fingerprint_vols)
    res["token_norm"] = token_norm_probe(fingerprint_vols + lesion_test)
    res = _strip(res)
    if reps:
        boot = bootstrap_probes(raw, reps)
        res["ci"] = {m: {"lo": float(np.nanpercentile(v, 2.5)), "hi": float(np.nanpercentile(v, 97.5))}
                     for m, v in boot.items()}
        res["boot"] = {m: [round(float(x), 5) for x in v] for m, v in boot.items()}
    return res


def paired_table(results: dict, ref: str) -> str:
    """Each encoder minus ``ref`` on the shared resamples, as mean difference and 95% interval."""
    names = [n for n in results if "boot" in results[n]]
    key = next((n for n in names if ref in n), None)
    if key is None:
        return f"[paired] no encoder matches --ref {ref!r}"
    base = {m: np.asarray(v) for m, v in results[key]["boot"].items()}
    lines = [f"paired difference vs {key} (95% bootstrap interval; * = excludes 0)",
             f"{'encoder':44s} " + " ".join(f"{m:>22s}" for m in METRICS)]
    for n in names:
        if n == key:
            continue
        cells = []
        for m in METRICS:
            d = np.asarray(results[n]["boot"][m]) - base[m]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            cells.append(f"{np.nanmean(d):+7.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ' '}")
        lines.append(f"{n:44s} " + " ".join(f"{c:>22s}" for c in cells))
    return "\n".join(lines)


def format_summary(results: dict) -> str:
    cols = ("lift(lin)", "lift(mlp)", "AUC(lin)", "dataset acc", "modality acc", "loc R2", "PR")
    lines = [f"{'encoder':44s} " + " ".join(f"{c:>20s}" for c in cols)]
    for name, r in results.items():
        les, fp, loc, ci = r["lesion"], r["fingerprint"], r["location"], r.get("ci", {})
        row = ((les["mean_linear_lift"], "lift_lin"), (les["mean_mlp_lift"], "lift_mlp"),
               (les["mean_linear_auroc"], "auroc_lin"), (fp["dataset_bal_acc"], "dataset_acc"),
               (fp["modality_bal_acc"], None), (loc["r2_mean"], "loc_r2"), (loc["token_participation_ratio"], "pr"))
        cells = []
        for v, m in row:
            cells.append(f"{v:.3f} [{ci[m]['lo']:.2f},{ci[m]['hi']:.2f}]" if m in ci else f"{v:.3f}")
        lines.append(f"{name:44s} " + " ".join(f"{c:>20s}" for c in cells))
    return "\n".join(lines)


def record_probes(providers, results: dict, args, started: str, n_cases: dict) -> int:
    """One experiment record per (encoder, probe, head); CIs from the paired bootstrap; draws stay in ``--out``."""
    from brain.downstream.records import make_record, model_info, provenance, write_record

    data = provenance(args.root) | n_cases
    protocol = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("ckpt", "out")}
    art = {"probe_json": str(args.out)}
    n = 0
    for p in providers:
        r = results.get(p.name)
        if r is None:
            continue
        if p.ckpt is not None:
            model = model_info(p.ckpt, variant=p.variant)
        else:
            model = model_info(None, name=p.name)
        if p.name.startswith("control/random3d") and providers:
            model["architecture_from"] = next((str(q.ckpt) for q in providers if q.ckpt is not None), None)
        ci = r.get("ci", {})

        def put(task: str, head: str, test: dict, ci_map: dict | None = None):
            nonlocal n
            m = dict(test)
            for metric, key in (ci_map or {}).items():
                if key in ci:
                    m[f"{metric}_ci"] = [ci[key]["lo"], ci[key]["hi"]]
            write_record(make_record(suite="probe3d", kind="probe", task=task, task_type="token_probe", head=head,
                                     model=model, data=data, protocol=protocol | {"probe": task, "head": head},
                                     metrics={"test": m}, started_utc=started, artifacts=art))
            n += 1

        les = r["lesion"]
        for fam, v in les.items():
            if isinstance(v, dict):
                for kind in ("linear", "mlp"):
                    put(f"probe:lesion:{fam}", kind, {**v[kind], "n_train": v["n_train"], "n_test": v["n_test"],
                                                      "seen_in_pretraining": fam in SEEN_FAMILIES})
        for kind, tag in (("linear", "lin"), ("mlp", "mlp")):
            put("probe:lesion:mean", kind, {k: les[f"mean_{kind}_{k}"] for k in ("ap", "lift", "auroc")},
                {"lift": f"lift_{tag}", **({"auroc": "auroc_lin"} if kind == "linear" else {})})
        put("probe:fingerprint", "logreg_cv5", r["fingerprint"], {"dataset_bal_acc": "dataset_acc"})
        for view, fp in (r.get("fingerprint_views") or {}).items():
            put("probe:fingerprint", f"logreg_cv5:{view}", fp)
        if r.get("registers"):
            put("probe:registers", "per_register", r["registers"])
        if r.get("token_norm"):
            put("probe:token_norm", "stats", r["token_norm"])
        put("probe:location", "ridge", r["location"], {"r2_mean": "loc_r2", "token_participation_ratio": "pr"})
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", action="append", default=[], help="3D BrainEncoder or 2D ViT checkpoint (repeatable)")
    ap.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    ap.add_argument("--pixel", action="store_true")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--tiled", action="store_true",
                    help="also embed each 3D ckpt as overlapping training-size windows (train/test context diagnostic)")
    ap.add_argument("--fine", action="store_true",
                    help="also embed each 3D ckpt with half-size patches (tiled), averaged onto the coarse grid")
    ap.add_argument("--fine-grid", action="store_true",
                    help="also embed each 3D ckpt with tiled 8x8x4 patches kept on the 32x32x32 grid")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--n-fingerprint", type=int, default=20, help="test volumes per dataset family")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=200, help="bootstrap resamples of the test volumes (0 = off)")
    ap.add_argument("--ref", default=None, help="substring of an encoder name; print paired differences against it")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from brain.downstream.records import now_utc

    started = now_utc()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    providers = build_providers(args.ckpt, device, pixel=args.pixel, random_init=args.random_init,
                                tiled=args.tiled, fine=args.fine, fine_grid=args.fine_grid)
    fams = LESION_FAMILIES + SEEN_FAMILIES
    train = load_cases(args.root, fams, "train", args.n_train, args.seed, masked=True)
    test = load_cases(args.root, fams, "test", args.n_test, args.seed, masked=True)
    all_fams = sorted(pd.read_csv(args.root / "volumes.csv", low_memory=False, usecols=["dataset_family"]).dataset_family.unique())
    fing = load_cases(args.root, all_fams, "test", args.n_fingerprint, args.seed, masked=False)
    print(f"[probe] encoders={[p.name for p in providers]} lesion train={len(train)} test={len(test)} fingerprint={len(fing)}")
    ex = {k: extract(providers, cases, device) for k, cases in (("train", train), ("test", test), ("fing", fing))}
    results = {p.name: run_probes(p.name, ex["train"][p.name], ex["test"][p.name], ex["fing"][p.name], args.reps)
               for p in providers}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    n = record_probes(providers, results, args, started, n_cases={"lesion_train": len(train), "lesion_test": len(test),
                                                                 "fingerprint": len(fing)})
    print(f"[record] {n} experiment records written")
    print(format_summary(results))
    if args.ref and args.reps:
        print("\n" + paired_table(results, args.ref))
    for name, r in results.items():
        print(f"\n[{name}] per-dataset linear lift / mlp lift / prevalence")
        for fam, v in r["lesion"].items():
            if isinstance(v, dict):
                print(f"  {fam:10s} {v['linear']['lift']:7.1f} {v['mlp']['lift']:7.1f}  prev={v['linear']['prevalence']:.4f}"
                      f"  n={v['n_train']}/{v['n_test']}")


if __name__ == "__main__":
    main()
