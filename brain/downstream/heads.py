"""Volume-level heads on frozen slice embeddings (N volumes, S slices, D dims).

Default probes are linear (mean+max logreg / ridge, C or alpha on val).
``xgboost_mean`` and ``slice_attn_mlp`` remain available via ``--heads``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from brain.common.optim import lr_at
from brain.common.grid import DEPTH

DEFAULT_HEADS = "logreg_pool"
LOGREG_CS = np.logspace(-2, 2, 9)
RIDGE_ALPHAS = np.logspace(-1, 4, 12)
_XGB = dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, n_jobs=4, tree_method="hist")
VOLUME_DEPTH = DEPTH


def mean_pool(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    w = mask[..., None].astype(np.float32)
    return (x * w).sum(1) / np.clip(w.sum(1), 1, None)


def mean_max_pool(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hi = np.where(mask[..., None], x, -np.inf).max(1)
    return np.concatenate([mean_pool(x, mask), np.where(np.isfinite(hi), hi, 0.0)], axis=1)


def flat_pool(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep every slot as its own feature. Invalid slots are zero."""
    w = mask[..., None].astype(np.float32)
    return (x * w).reshape(x.shape[0], -1)


def _pipe(model: Any) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def _predict_proba(pipe, z: np.ndarray, multilabel: bool) -> np.ndarray:
    p = pipe.predict_proba(z)
    if isinstance(p, list):
        return np.column_stack([q[:, 1] for q in p])
    return p if multilabel else p[:, 1]


@dataclass
class SklearnHead:
    name: str
    pipeline: Any
    pool: str  # "mean" | "mean_max"
    multilabel: bool = False
    regression: bool = False
    grid: np.ndarray | None = None
    grid_param: str = ""
    grid_higher_better: bool = True
    chosen_: float | None = field(default=None, init=False, repr=False)

    def inputs(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.pool == "mean_max":
            return mean_max_pool(x, mask)
        if self.pool == "flat":
            return flat_pool(x, mask)
        return mean_pool(x, mask)

    def _val_score(self, pipe, xv: np.ndarray, yv: np.ndarray) -> float:
        if self.regression:
            return float(mean_absolute_error(yv, pipe.predict(xv)))
        try:
            return float(roc_auc_score(yv, _predict_proba(pipe, xv, self.multilabel)))
        except ValueError:
            return float("nan")

    def fit(self, x, y, mask, x_val=None, y_val=None, mask_val=None, z=None, z_val=None) -> None:
        xt = self.inputs(x, mask)
        if self.grid is None or x_val is None or mask_val is None or len(x_val) == 0:
            self.pipeline.fit(xt, y)
            return
        xv = self.inputs(x_val, mask_val)
        best, best_p, best_pipe = None, None, None
        for p in self.grid:
            pipe = clone(self.pipeline)
            pipe.set_params(**{self.grid_param: float(p)})
            pipe.fit(xt, y)
            s = self._val_score(pipe, xv, y_val)
            if not np.isfinite(s):
                continue
            if best is None or (s > best if self.grid_higher_better else s < best):
                best, best_p, best_pipe = s, float(p), pipe
        if best_pipe is None:
            self.pipeline.fit(xt, y)
            return
        self.pipeline, self.chosen_ = best_pipe, best_p

    def predict_scores(self, x, mask, z=None) -> np.ndarray:
        zt = self.inputs(x, mask)
        if self.regression:
            return np.asarray(self.pipeline.predict(zt), dtype=np.float64).ravel()
        return _predict_proba(self.pipeline, zt, self.multilabel)


def make_logreg_pool(multilabel: bool, seed: int) -> SklearnHead:
    lr = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced", random_state=seed)
    if multilabel:
        model, param = OneVsRestClassifier(lr), "model__estimator__C"
    else:
        model, param = lr, "model__C"
    return SklearnHead(
        "logreg_pool", _pipe(model), "mean_max", multilabel,
        grid=LOGREG_CS, grid_param=param, grid_higher_better=True,
    )


def make_ridge() -> SklearnHead:
    return SklearnHead(
        "ridge", _pipe(Ridge()), "mean_max", regression=True,
        grid=RIDGE_ALPHAS, grid_param="model__alpha", grid_higher_better=False,
    )


def make_xgboost(multilabel: bool, regression: bool, seed: int) -> SklearnHead:
    from xgboost import XGBClassifier, XGBRegressor

    if regression:
        return SklearnHead("xgboost_mean", _pipe(XGBRegressor(random_state=seed, **_XGB)),
                           "mean_max", regression=True)
    clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=seed, **_XGB)
    return SklearnHead("xgboost_mean", _pipe(OneVsRestClassifier(clf) if multilabel else clf),
                       "mean_max", multilabel)


class SliceAttnMLP(nn.Module):
    """LayerNorm, learned slot pos, anatomical z in the scorer, attention pool, MLP."""

    def __init__(self, dim: int, n_slices: int, n_out: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pos = nn.Parameter(torch.zeros(n_slices, dim))
        self.score = nn.Linear(dim + 1, 1)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(x) + self.pos[: x.size(1)]
        if z is None:
            z = torch.linspace(0, 1, x.size(1), device=x.device, dtype=h.dtype).expand(x.size(0), -1)
        else:
            z = z.to(dtype=h.dtype) / (VOLUME_DEPTH - 1)
        w = self.score(torch.cat([h, z.unsqueeze(-1)], dim=-1)).squeeze(-1)
        w = w.masked_fill(~mask, -1e9).softmax(dim=1)
        return self.mlp((w.unsqueeze(-1) * h).sum(1))


class TokenAttnMLP(nn.Module):
    """Attention pool over a 3D token grid. Position is the normalized token center."""

    def __init__(self, dim: int, n_out: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim + 3, 1)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        w = self.score(torch.cat([h, xyz.to(dtype=h.dtype)], dim=-1)).squeeze(-1)
        w = w.masked_fill(~mask, -1e9).softmax(dim=1)
        return self.mlp((w.unsqueeze(-1) * h).sum(1))


class AttnEnsemble:
    """``n_seeds`` attention pools, each early-stopped on val; scores are averaged.

    ``spatial=False`` is the 2D slice head. ``spatial=True`` scores native 3D tokens
    with their (x, y, z) centers.
    """

    def __init__(self, dim: int, n_slices: int, n_out: int, multilabel: bool, seed: int,
                 n_seeds: int = 5, epochs: int = 100, patience: int = 15, lr: float = 1e-3,
                 batch_size: int = 32, regression: bool = False, spatial: bool = False,
                 model_factory=None, half: bool = False):
        self.model_factory, self.half = model_factory, bool(half)
        self.spatial = bool(spatial)
        self.name = "token_attn_mlp" if self.spatial else "slice_attn_mlp"
        self.args = (dim, n_out) if self.spatial else (dim, n_slices, n_out)
        self.multilabel, self.regression = multilabel, regression
        self.seed, self.n_seeds = seed, n_seeds
        self.epochs, self.patience, self.lr, self.bs = epochs, patience, lr, batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: list[nn.Module] = []

    def _host_tokens(self, n_tok: int) -> bool:
        """32^3 grids are too large to pin in GPU memory; only minibatches move over."""
        return bool(self.spatial) and int(n_tok) > 8192

    def _t(self, a, dtype=torch.float32, device=None):
        device = self.device if device is None else device
        arr = np.asarray(a)
        same = isinstance(a, np.ndarray) and a.flags.c_contiguous and arr.dtype == {
            torch.float16: np.float16, torch.float32: np.float32, torch.bool: np.bool_,
        }.get(dtype)
        try:
            t = torch.from_numpy(a) if same else torch.as_tensor(arr, dtype=dtype)
        except (ValueError, RuntimeError, TypeError):
            t = torch.as_tensor(arr, dtype=dtype)
        if t.dtype != dtype:
            t = t.to(dtype)
        return t if t.device == device else t.to(device)

    def _z(self, z, n: int, s: int, device=None) -> torch.Tensor:
        if self.spatial:
            if z is None or np.asarray(z).ndim != 3:
                raise ValueError("token attention needs (N, T, 3) coordinates")
            return self._t(z, device=device)
        if z is None:
            from brain.downstream.embeddings import slot_z
            z = slot_z(n, s)
        return self._t(z, device=device)

    def _gpu_batch(self, t: torch.Tensor, idx) -> torch.Tensor:
        b = t[idx]
        if b.device != self.device:
            b = b.to(self.device)
        if b.is_floating_point() and b.dtype != torch.float32:
            b = b.float()
        return b

    def _logits(self, model: nn.Module, x, m, z) -> torch.Tensor:
        outs = []
        n = len(x)
        for i in range(0, n, self.bs):
            sl = slice(i, min(i + self.bs, n))
            outs.append(model(self._gpu_batch(x, sl), self._gpu_batch(m, sl), self._gpu_batch(z, sl)))
        return torch.cat(outs)

    def _fit_one(self, seed, x, y, m, z, xv, yv, mv, zv) -> nn.Module:
        torch.manual_seed(seed)
        cls = TokenAttnMLP if self.spatial else SliceAttnMLP
        model = (self.model_factory() if self.model_factory else cls(*self.args)).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-2)
        loss_fn = nn.MSELoss() if self.regression else nn.BCEWithLogitsLoss()
        n, steps = len(y), self.epochs * -(-len(y) // self.bs)
        best, best_state, bad, step = float("inf"), None, 0, 0
        perm_dev = x.device
        for _ in range(self.epochs):
            model.train()
            for idx in torch.randperm(n, device=perm_dev).split(self.bs):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, steps, self.lr, warmup=min(50, steps // 10))
                loss = loss_fn(model(self._gpu_batch(x, idx), self._gpu_batch(m, idx), self._gpu_batch(z, idx)),
                               self._gpu_batch(y, idx))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
            model.eval()
            with torch.no_grad():
                v = float(loss_fn(self._logits(model, xv, mv, zv), self._gpu_batch(yv, slice(None))))
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            elif (bad := bad + 1) >= self.patience:
                break
        model.load_state_dict(best_state)
        return model.eval()

    def fit(self, x, y, mask, x_val, y_val, mask_val, z=None, z_val=None) -> None:
        y, y_val = (np.asarray(a, dtype=np.float32).reshape(len(a), -1) for a in (y, y_val))
        host = torch.device("cpu") if self._host_tokens(x.shape[1]) else self.device
        data = (
            self._t(x, torch.float16 if self.half else torch.float32, device=host), self._t(y, device=host),
            self._t(mask, torch.bool, device=host), self._z(z, len(x), x.shape[1], device=host),
            self._t(x_val, torch.float16 if self.half else torch.float32, device=host), self._t(y_val, device=host),
            self._t(mask_val, torch.bool, device=host), self._z(z_val, len(x_val), x_val.shape[1], device=host),
        )
        self.models = [self._fit_one(self.seed + i, *data) for i in range(self.n_seeds)]

    @torch.no_grad()
    def predict_scores(self, x, mask, z=None) -> np.ndarray:
        host = torch.device("cpu") if self._host_tokens(x.shape[1]) else self.device
        xt = self._t(x, torch.float16 if self.half else torch.float32, device=host)
        mt = self._t(mask, torch.bool, device=host)
        zt = self._z(z, len(x), x.shape[1], device=host)
        logits = torch.stack([self._logits(m, xt, mt, zt) for m in self.models]).mean(0)
        p = logits if self.regression else logits.sigmoid()
        p = p.cpu().numpy()
        return p.ravel() if self.regression or (not self.multilabel and p.shape[1] == 1) else p


def build_heads(*, latent_dim: int, max_slices: int, n_out: int, multilabel: bool, seed: int,
                which: list[str], task_type: str, spatial: bool = False) -> list:
    """Heads valid for the task type, in the order requested."""
    reg = task_type == "regression"
    makers = {
        "logreg_pool": lambda: make_logreg_pool(multilabel, seed),
        "slice_attn_mlp": lambda: AttnEnsemble(
            latent_dim, max_slices, n_out, multilabel, seed, regression=reg, spatial=spatial,
        ),
        "token_attn_mlp": lambda: AttnEnsemble(
            latent_dim, max_slices, n_out, multilabel, seed, regression=reg, spatial=True,
        ),
        "xgboost_mean": lambda: make_xgboost(multilabel, reg, seed),
        "ridge": make_ridge,
    }
    if reg:
        default = ["ridge"]
        picked = [h for h in which if h in makers] if which != DEFAULT_HEADS.split(",") else default
        picked = picked or default
    else:
        picked = [h for h in which if h in makers and h != "ridge"] or DEFAULT_HEADS.split(",")
    return [makers[h]() for h in picked]
