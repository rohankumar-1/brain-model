"""Volume-level heads on frozen embeddings (N volumes, T tokens, D dims; the benchmark's pooled readouts pass T = 1).

- ``SklearnHead``: a linear probe on the mean token (logistic regression / ridge, C or alpha chosen on validation);
- ``MLPProbe``: BrainDINO's frozen-feature MLP head on the mean token;
- ``ProbeEnsemble``: seeds of a torch token-grid probe (``brain.downstream.token_probes.ConvProbe``), early-stopped;
- ``AttentiveProbe`` (``make_attn_probe``, head ``attn_probe``): learned queries cross-attending over a scan's foreground
  tokens, then linear (or a small MLP), as a ProbeEnsemble. An EXTRA readout (``metrics.EXTRA_READOUTS``): reported
  beside the benchmark, never pinned and never in a headline score.

Every head has ``fit`` / ``predict_scores`` and a cross-fitted ``predict_val`` (``brain.downstream.cv``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from brain.common.optim import lr_at
from brain.downstream.metrics import ATTN_PROBE, _safe_auc

LOGREG_CS = np.logspace(-2, 2, 9)
RIDGE_ALPHAS = np.logspace(-1, 4, 12)
HOST_BYTES = 4e9          # token arrays above this stay in host memory; only minibatches move to the GPU


def mean_pool(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    w = mask[..., None].astype(np.float32)
    return (x * w).sum(1) / np.clip(w.sum(1), 1, None)


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
    multilabel: bool = False
    regression: bool = False
    grid: np.ndarray | None = None
    grid_param: str = ""
    grid_higher_better: bool = True
    chosen_: float | None = field(default=None, init=False, repr=False)
    cross_: tuple | None = field(default=None, init=False, repr=False)   # per val half: the fit tuned on the other

    def inputs(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return mean_pool(x, mask)

    def _val_score(self, pipe, xv: np.ndarray, yv: np.ndarray) -> float:
        if self.regression:
            return float(mean_absolute_error(yv, pipe.predict(xv)))
        return _safe_auc(yv, _predict_proba(pipe, xv, self.multilabel))

    def _pick(self, fitted: list, xv: np.ndarray, yv: np.ndarray) -> tuple[float | None, Any]:
        best, best_p, best_pipe = None, None, None
        for p, pipe in fitted:
            s = self._val_score(pipe, xv, yv)
            if not np.isfinite(s):
                continue
            if best is None or (s > best if self.grid_higher_better else s < best):
                best, best_p, best_pipe = s, p, pipe
        return best_p, best_pipe

    def fit(self, x, y, mask, x_val=None, y_val=None, mask_val=None, z=None, z_val=None, val_half=None) -> None:
        """Fit every grid value on train and keep the best on validation (what predicts test). With ``val_half``
        (0 / 1 per validation volume) also the best on each half, which ``predict_val`` scores the other half with."""
        xt = self.inputs(x, mask)
        self.cross_ = None
        if self.grid is None or x_val is None or mask_val is None or len(x_val) == 0:
            self.pipeline.fit(xt, y)
            return
        xv, yv = self.inputs(x_val, mask_val), np.asarray(y_val)
        fitted = []
        for p in self.grid:
            pipe = clone(self.pipeline)
            pipe.set_params(**{self.grid_param: float(p)})
            fitted.append((float(p), pipe.fit(xt, y)))
        best_p, best_pipe = self._pick(fitted, xv, yv)
        if best_pipe is None:
            best_pipe = self.pipeline.fit(xt, y)
        if val_half is not None:
            h = np.asarray(val_half)
            self.cross_ = tuple(self._pick(fitted, xv[h != k], yv[h != k])[1] or best_pipe for k in (0, 1))
        self.pipeline, self.chosen_ = best_pipe, best_p

    def _scores(self, pipe, zt: np.ndarray) -> np.ndarray:
        if self.regression:
            return np.asarray(pipe.predict(zt), dtype=np.float64).ravel()
        return _predict_proba(pipe, zt, self.multilabel)

    def predict_scores(self, x, mask, z=None) -> np.ndarray:
        return self._scores(self.pipeline, self.inputs(x, mask))

    def predict_val(self, x, mask, val_half, z=None) -> np.ndarray:
        """Validation scores none of which comes from a fit tuned on that volume (``fit`` with ``val_half``)."""
        if self.cross_ is None:
            return self.predict_scores(x, mask, z)
        zt, h = self.inputs(x, mask), np.asarray(val_half)
        out = None
        for k, pipe in enumerate(self.cross_):
            if (h == k).any():
                s = self._scores(pipe, zt[h == k])
                out = np.zeros((len(h), *s.shape[1:]), dtype=np.float64) if out is None else out
                out[h == k] = s
        return out


def make_logreg(multilabel: bool, seed: int, name: str = "linear") -> SklearnHead:
    lr = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced", random_state=seed)
    if multilabel:
        model, param = OneVsRestClassifier(lr), "model__estimator__C"
    else:
        model, param = lr, "model__C"
    return SklearnHead(name, _pipe(model), multilabel, grid=LOGREG_CS, grid_param=param, grid_higher_better=True)


def make_ridge(name: str = "linear") -> SklearnHead:
    return SklearnHead(name, _pipe(Ridge(solver="svd")), regression=True, grid=RIDGE_ALPHAS,
                       grid_param="model__alpha", grid_higher_better=False)


class ProbeEnsemble:
    """``n_seeds`` torch token probes from ``model_factory``, each early-stopped on val; scores are averaged. Probes
    take ``(tokens, mask, xyz)``: 3D tokens with their normalised centres."""

    def __init__(self, model_factory, multilabel: bool, seed: int, *, name: str = "tok_conv", n_seeds: int = 5,
                 epochs: int = 100, patience: int = 15, lr: float = 1e-3, batch_size: int = 32,
                 regression: bool = False, half: bool = False, standardize: bool = False):
        self.model_factory, self.half, self.name = model_factory, bool(half), name
        # ``standardize``: regression targets z-scored on train (scores mapped back); off for tok_conv (as pinned)
        self.standardize = bool(standardize)
        self.y_mu, self.y_sd = np.zeros(1, np.float32), np.ones(1, np.float32)
        self.multilabel, self.regression = multilabel, regression
        self.seed, self.n_seeds = seed, n_seeds
        self.epochs, self.patience, self.lr, self.bs = epochs, patience, lr, batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: list[nn.Module] = []
        self.stop_half_: list[int] | None = None

    def _host(self, x) -> torch.device:
        """Where the token arrays live: host memory for big (memory-mapped) caches, else the GPU."""
        return torch.device("cpu") if np.asarray(x).nbytes > HOST_BYTES else self.device

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
        if z is None or np.asarray(z).ndim != 3:
            raise ValueError("token attention needs (N, T, 3) coordinates")
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
        model = self.model_factory().to(self.device)
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

    def fit(self, x, y, mask, x_val, y_val, mask_val, z=None, z_val=None, val_half=None) -> None:
        """``n_seeds`` probes early-stopped on validation; with ``val_half`` (0 / 1 per validation volume) seed i
        stops on half i % 2 only, so ``predict_val`` can score each half with the seeds that never saw it."""
        y, y_val = (np.asarray(a, dtype=np.float32).reshape(len(a), -1) for a in (y, y_val))
        self.y_mu, self.y_sd = np.zeros(y.shape[1], np.float32), np.ones(y.shape[1], np.float32)
        if self.regression and self.standardize:
            self.y_mu, sd = y.mean(0), y.std(0)
            self.y_sd = np.where(sd > 0, sd, 1.0).astype(np.float32)
            y, y_val = (y - self.y_mu) / self.y_sd, (y_val - self.y_mu) / self.y_sd
        host = self._host(x)
        data = (
            self._t(x, torch.float16 if self.half else torch.float32, device=host), self._t(y, device=host),
            self._t(mask, torch.bool, device=host), self._z(z, len(x), x.shape[1], device=host),
            self._t(x_val, torch.float16 if self.half else torch.float32, device=host), self._t(y_val, device=host),
            self._t(mask_val, torch.bool, device=host), self._z(z_val, len(x_val), x_val.shape[1], device=host),
        )
        self.stop_half_ = None
        h = None if val_half is None else np.asarray(val_half)
        if h is None or self.n_seeds < 2 or len(np.unique(h)) < 2:
            self.models = [self._fit_one(self.seed + i, *data) for i in range(self.n_seeds)]
            return
        self.stop_half_ = [i % 2 for i in range(self.n_seeds)]
        sel = [torch.as_tensor(h == k, device=data[4].device) for k in (0, 1)]
        self.models = [self._fit_one(self.seed + i, *data[:4], *(t[sel[k]] for t in data[4:]))
                       for i, k in enumerate(self.stop_half_)]

    @torch.no_grad()
    def _all_logits(self, x, mask, z) -> torch.Tensor:
        """(models, N, out) logits."""
        host = self._host(x)
        xt = self._t(x, torch.float16 if self.half else torch.float32, device=host)
        mt = self._t(mask, torch.bool, device=host)
        zt = self._z(z, len(x), x.shape[1], device=host)
        return torch.stack([self._logits(m, xt, mt, zt) for m in self.models])

    def _to_scores(self, logits: torch.Tensor) -> np.ndarray:
        p = (logits if self.regression else logits.sigmoid()).cpu().numpy()
        if self.regression:
            p = p * self.y_sd + self.y_mu
        return p.ravel() if self.regression or (not self.multilabel and p.shape[1] == 1) else p

    def predict_scores(self, x, mask, z=None) -> np.ndarray:
        return self._to_scores(self._all_logits(x, mask, z).mean(0))

    def predict_val(self, x, mask, val_half, z=None) -> np.ndarray:
        """Each validation volume scored by the seeds that early-stopped on the other half."""
        if self.stop_half_ is None:
            return self.predict_scores(x, mask, z)
        lg, h = self._all_logits(x, mask, z), torch.as_tensor(np.asarray(val_half))
        own = torch.as_tensor(self.stop_half_)
        w = (own[:, None] != h[None, :]).to(lg.dtype).to(lg.device)               # (models, N): seeds blind to it
        return self._to_scores((lg * w[..., None]).sum(0) / w.sum(0)[:, None])


class AttentiveProbe(nn.Module):
    """Attentive probe: ``n_queries`` learned queries cross-attend (multi-head) over a scan's tokens, the padded and
    gated-out (air) tokens masked, then a linear layer (``hidden`` 0) or the token probes' small MLP. Tokens are
    LayerNorm-ed and projected to ``width`` first (a few hundred training scans: a full-width attention would overfit
    and, at 5k tokens x D, cost more than the conv probe); the normalised token centres enter the keys through a
    zero-initialised MLP (``pos``), so the probe can tell left from right (RoPE tokens carry no absolute position)."""

    def __init__(self, dim: int, n_out: int, n_queries: int = 1, width: int = 256, heads: int = 8, hidden: int = 0,
                 pos: bool = True, dropout: float = 0.1):
        super().__init__()
        self.norm, self.proj = nn.LayerNorm(dim), nn.Linear(dim, width)
        self.pos = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, width)) if pos else None
        if self.pos is not None:
            nn.init.zeros_(self.pos[-1].weight)
            nn.init.zeros_(self.pos[-1].bias)
        self.query = nn.Parameter(torch.randn(int(n_queries), width) * 0.02)
        self.attn = nn.MultiheadAttention(width, int(heads), batch_first=True)
        d = int(n_queries) * width
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_out)) if not hidden else nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, xyz: torch.Tensor | None = None,
                return_attn: bool = False):
        h = self.proj(self.norm(x.float()))
        if self.pos is not None and xyz is not None:
            h = h + self.pos(xyz.float())
        pad = ~mask.bool()
        empty = pad.all(1)
        if bool(empty.any()):                      # a scan with no kept token attends its first one (zeros)
            pad = pad.clone()
            pad[empty, 0] = False
        q = self.query.unsqueeze(0).expand(len(x), -1, -1)
        o, w = self.attn(q, h, h, key_padding_mask=pad, need_weights=return_attn)
        y = self.out(self.out_norm(o.reshape(len(x), -1)))
        return (y, w) if return_attn else y


def make_attn_probe(dim: int, n_out: int, multilabel: bool, seed: int, regression: bool, n_queries: int = 1,
                    hidden: int = 0, n_seeds: int = 5) -> ProbeEnsemble:
    """The attentive probe as the head ``attn_probe`` (``metrics.ATTN_PROBE``): ``n_seeds`` seeds trained like the conv
    probe (AdamW, warmup + cosine, early stopping on the validation loss, seeds alternating validation halves so
    ``predict_val`` is cross-fitted), regression targets z-scored on train."""
    return ProbeEnsemble(lambda: AttentiveProbe(dim, n_out, n_queries=n_queries, hidden=hidden), multilabel, seed,
                         name=ATTN_PROBE, n_seeds=n_seeds, regression=regression, half=True, standardize=True)


class MLPProbe:
    """BrainDINO's frozen-feature head (Wu et al. 2026, arXiv 2604.27277, Methods 4.3 and their runners/): LayerNorm ->
    Linear(D, D) -> GELU -> Linear(D, K) on one pooled embedding per volume. Classification: K-way cross-entropy
    (K = 2 for binary), Adam lr 1e-4, weight decay 1e-5; regression: MSE, AdamW lr 1e-3, weight decay 1e-4. Batch 4,
    at most 8 epochs, the epoch with the best validation score kept (AUC; MAE for regression), stopped after 5 without
    improvement. Deviations, for our tasks: multilabel tasks (no BrainDINO counterpart) use per-label BCE; regression
    targets are z-scored on train (BrainDINO regresses raw years); two seeds, each validated on one half of the
    validation set, so ``predict_val`` is cross-fitted like the other heads (test scores average the two)."""

    def __init__(self, n_out: int, multilabel: bool, seed: int, regression: bool = False, n_seeds: int = 2,
                 epochs: int = 8, patience: int = 5, batch_size: int = 4):
        self.name = "mlp"
        self.n_out, self.multilabel, self.regression = n_out, multilabel, regression
        self.seed, self.n_seeds, self.epochs, self.patience, self.bs = seed, n_seeds, epochs, patience, batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: list[nn.Module] = []
        self.stop_half_: list[int] | None = None
        self.y_mu, self.y_sd = 0.0, 1.0

    def _k(self) -> int:
        return self.n_out if (self.regression or self.multilabel) else 2

    def _net(self, d: int) -> nn.Module:
        return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, self._k()))

    def _scores(self, logits: torch.Tensor) -> np.ndarray:
        if self.regression:
            return (logits[:, 0] * self.y_sd + self.y_mu).double().cpu().numpy()
        if self.multilabel:
            return logits.sigmoid().double().cpu().numpy()
        return logits.softmax(-1)[:, 1].double().cpu().numpy()

    def _val(self, model: nn.Module, xv: torch.Tensor, yv: np.ndarray) -> float:
        """Higher is better: AUC, or minus the MAE."""
        model.eval()
        with torch.no_grad():
            s = self._scores(model(xv))
        if self.regression:
            return -float(np.mean(np.abs(s - yv.ravel())))
        return _safe_auc(yv if self.multilabel else yv.ravel(), s)

    def _fit_one(self, seed: int, x: torch.Tensor, y: torch.Tensor, xv: torch.Tensor, yv: np.ndarray) -> nn.Module:
        torch.manual_seed(seed)
        model = self._net(x.shape[1]).to(self.device)
        if self.regression:
            opt, loss_fn = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4), nn.MSELoss()
        else:
            opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
            loss_fn = nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss()
        g = torch.Generator(device="cpu").manual_seed(seed)
        best, best_state, bad = -float("inf"), None, 0
        for _ in range(self.epochs):
            model.train()
            for idx in torch.randperm(len(x), generator=g).split(self.bs):
                idx = idx.to(self.device)
                loss = loss_fn(model(x[idx]), y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            v = self._val(model, xv, yv) if len(xv) else -float("inf")
            if best_state is None or v > best:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            elif (bad := bad + 1) >= self.patience:
                break
        model.load_state_dict(best_state)
        return model.eval()

    def _x(self, x, mask) -> torch.Tensor:
        return torch.as_tensor(mean_pool(np.asarray(x, np.float32), np.asarray(mask)), device=self.device)

    def fit(self, x, y, mask, x_val, y_val, mask_val, z=None, z_val=None, val_half=None) -> None:
        xt, xv = self._x(x, mask), self._x(x_val, mask_val)
        y, y_val = np.asarray(y, np.float32), np.asarray(y_val, np.float32)
        if self.regression:
            self.y_mu, self.y_sd = float(y.mean()), float(y.std() or 1.0)
            yt = torch.as_tensor(((y - self.y_mu) / self.y_sd).reshape(len(y), -1), device=self.device)
        elif self.multilabel:
            yt = torch.as_tensor(y.reshape(len(y), -1), device=self.device)
        else:
            yt = torch.as_tensor(y.ravel(), dtype=torch.long, device=self.device)
        h = None if val_half is None else np.asarray(val_half)
        self.stop_half_ = None
        if h is None or len(np.unique(h)) < 2:
            self.models = [self._fit_one(self.seed + i, xt, yt, xv, y_val) for i in range(self.n_seeds)]
            return
        self.stop_half_ = [i % 2 for i in range(self.n_seeds)]
        self.models = [self._fit_one(self.seed + i, xt, yt, xv[torch.as_tensor(h == k, device=self.device)],
                                     y_val[h == k]) for i, k in enumerate(self.stop_half_)]

    @torch.no_grad()
    def _all_logits(self, x, mask) -> torch.Tensor:
        xt = self._x(x, mask)
        return torch.stack([m(xt) for m in self.models])

    def predict_scores(self, x, mask, z=None) -> np.ndarray:
        lg = self._all_logits(x, mask)
        if self.regression:
            return self._scores(lg.mean(0))
        return np.mean([self._scores(t) for t in lg], axis=0)        # average probabilities, not logits

    def predict_val(self, x, mask, val_half, z=None) -> np.ndarray:
        """Each validation volume scored by the seeds that stopped on the other half."""
        if self.stop_half_ is None:
            return self.predict_scores(x, mask, z)
        lg, h = self._all_logits(x, mask), np.asarray(val_half)
        per = np.stack([self._scores(t) for t in lg])                # (seeds, N[, L])
        w = (np.asarray(self.stop_half_)[:, None] != h[None, :]).astype(np.float64)
        w = w.reshape(w.shape + (1,) * (per.ndim - 2))
        return (per * w).sum(0) / w.sum(0)
