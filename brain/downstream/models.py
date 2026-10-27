"""The models the evaluation suite encodes with: our checkpoints and published encoders, behind one interface.

A ``ModelRef`` is everything the suite needs to know about a model, wherever it comes from:

* ``model_id``        the records id: ``<run>/<file>@<fingerprint[:10]>`` for a checkpoint, ``<name>@<fingerprint[:10]>``
                      for a registry model
* ``cache_key``       directory name of its embedding caches and run directories
* ``fingerprint``     identity of the weights (size plus first / last MiB of the file)
* ``training_class``  what it learned from, which decides what it may be compared with (``TRAINING_CLASSES``)
* ``seen_tasks``      tasks whose labels it trained on (left out of its comparisons)
* ``info()``          the records' model section; ``load(device)`` a frozen encoder; ``pretrain_overlap(...)``

``resolve_model(target)`` takes a name in ``configs/bench/models.yaml`` (our BrainQuery checkpoints, ``BQ-...``;
published encoders and baselines, each with an adapter class) or a checkpoint path (ours: SSL, multitask, joint,
single-task, random init; shown by its registry name when it has one, else as ``BQ <run>/<file>``).

Every encoder the suite uses has ``encode_volume(path, volume_id, modality, window) -> VolumeEmbedding``: patch tokens
(T, D) with normalised positions and a foreground mask, plus optional global vectors (CLS, registers). Our 3D encoder
(``embed3d.FrozenEncoder3D``) returns its 20 x 16 x 16 token grid; an encoder with only a global embedding returns it
as one token and as ``cls``, so the suite reads it out with its global heads. Published encoders subclass
``ExternalEncoder`` and implement ``encode`` on the stored volume (1 mm MNI grid, 256 x 256 x 160; CT in HU, MRI as a
robust z-score), doing their own resampling and intensity normalisation there.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from brain.common.paths import REPO_ROOT, resolve

MODELS_YAML = REPO_ROOT / "configs" / "bench" / "models.yaml"

# What a model learned from. Comparisons are made within a class, or across classes with the label overlap removed:
#   ssl          images only (our LeJEPA / MAE encoders, BrainIAC, Brain2Vec, CT-FM, ...)
#   supervised   labels of other tasks in pretraining (our multitask / joint models, SFCN, BrainFM, ...)
#   specialist   trained end to end on the evaluated task itself (single-task models, nnU-Net): a per-task ceiling
#   baseline     no learning or no pretraining (random init, hand-crafted features): the floor
TRAINING_CLASSES = ("ssl", "supervised", "specialist", "baseline")


@dataclass
class VolumeEmbedding:
    tokens: torch.Tensor                 # (T, D) per window, concatenated on D
    xyz: torch.Tensor                    # (T, 3) token centres, normalised to [0, 1]
    keep: torch.Tensor                   # (T,) foreground tokens
    cls: torch.Tensor | None = None      # (D',) global embedding, windows concatenated
    registers: torch.Tensor | None = None


@dataclass
class ModelRef:
    name: str
    model_id: str
    cache_key: str
    fingerprint: str
    training_class: str
    ckpt: Path | None = None             # our checkpoints
    spec: dict = field(default_factory=dict)       # registry models: their models.yaml entry
    seen_tasks: tuple[str, ...] = ()

    @property
    def is_checkpoint(self) -> bool:
        return self.ckpt is not None

    def load(self, device: torch.device):
        """A frozen encoder with ``encode_volume``."""
        if self.ckpt is not None:
            from brain.downstream.embed3d import is_brain3d_state, load_brain3d

            ck = torch.load(self.ckpt, map_location=device, weights_only=False)
            state = {str(k).removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
            if not is_brain3d_state(state):
                raise SystemExit(f"{self.ckpt}: not a 3D BrainEncoder checkpoint (the 2D slice ViT was retired)")
            return load_brain3d(ck, device)
        mod, _, cls = str(self.spec["adapter"]).partition(":")
        return getattr(importlib.import_module(mod), cls)(self.spec, device)

    def info(self) -> dict:
        """The records' ``model`` section."""
        from brain.downstream.records import model_info

        if self.ckpt is not None:
            info = model_info(self.ckpt)
        else:
            info = {"model_id": self.model_id, "ckpt": self.spec.get("weights"), "fingerprint": self.fingerprint,
                    "run_name": self.name, "file": None, "method": self.spec.get("method", "external"),
                    "control": self.training_class == "baseline", "variant": None,
                    "provenance": {"reference": self.spec.get("reference"), "adapter": self.spec.get("adapter"),
                                   "pretrain_datasets": self.spec.get("pretrain_datasets")}}
        return info | {"training_class": self.training_class, "seen_tasks": list(self.seen_tasks),
                       "modalities": list(self.spec.get("modalities") or ["ct", "mri"])}

    def pretrain_overlap(self, volume_ids, volumes) -> float | None:
        """Fraction of a task's volumes the model may have seen in pretraining. Ours: the SSL pool flags of the
        index; a registry model: volumes of the dataset families its entry declares (None when it declares none)."""
        from brain.downstream.records import pretrain_overlap

        if self.ckpt is not None:
            return pretrain_overlap(volume_ids, volumes)
        fams = self.spec.get("pretrain_datasets")
        if fams is None or volumes is None or "dataset_family" not in getattr(volumes, "columns", ()):
            return None
        fam = volumes.set_index("volume_id")["dataset_family"].astype(str)
        ids = [v for v in volume_ids if v in fam.index]
        return float(np.mean([fam[v] in set(fams) for v in ids])) if ids else None


def registry(path: Path = MODELS_YAML) -> dict[str, dict]:
    """``configs/bench/models.yaml``: name -> entry."""
    return (yaml.safe_load(Path(path).read_text()) or {}).get("models") or {} if Path(path).exists() else {}


def _checkpoint_class(ckpt: Path) -> tuple[str, tuple[str, ...]]:
    """(training class, trained task ids) of one of our checkpoints, from its own config."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    cfg = ck.get("config") or {}
    method = (ck.get("provenance") or {}).get("method") or cfg.get("method")
    if method == "random_init":
        return "baseline", ()
    if "tasks" not in ck:                                     # an SSL encoder: no task heads
        return "ssl", ()
    from brain.multitask.registry import load_registry

    trained = tuple(load_registry(cfg["data"].get("registry")).subset(list(ck["tasks"])).train_ids(cfg.get("tasks")))
    tk = cfg.get("tasks") or {}
    single = bool(tk.get("ids")) or (bool(tk.get("groups")) and len(trained) <= 6)   # one task or one small family
    return ("specialist" if single else "supervised"), trained


def _checkpoint_ref(p: Path, name: str | None = None, spec: dict | None = None) -> ModelRef:
    from brain.downstream.records import model_info

    p = p.resolve()
    cls, trained = _checkpoint_class(p)
    mid = model_info(p)["model_id"]
    return ModelRef(name=name or f"BQ {p.parent.name}/{p.stem}", model_id=mid, cache_key=f"{p.parent.name}_{p.stem}",
                    fingerprint=mid.rpartition("@")[2], training_class=cls, ckpt=p, seen_tasks=trained,
                    spec={"modalities": ["ct", "mri"], **(spec or {})})


def bq_names() -> dict[Path, str]:
    """Our named checkpoints: resolved path -> registry name (``BQ-...``)."""
    return {resolve(e["ckpt"]).resolve(): n for n, e in registry().items() if e.get("ckpt")}


def resolve_model(target: str | Path) -> ModelRef:
    """A registry name (``BQ-A1``, ``regional_stats``) or a checkpoint path (absolute, or relative to the repo)."""
    from brain.downstream.records import fingerprint

    reg = registry()
    if str(target) in reg and reg[str(target)].get("ckpt"):           # one of our checkpoints, by its BQ name
        p = resolve(reg[str(target)]["ckpt"])
        if not p.is_file():
            raise SystemExit(f"{target}: checkpoint {p} does not exist (yet)")
        return _checkpoint_ref(p, str(target), reg[str(target)])
    p = Path(target)
    p = p if p.is_absolute() else resolve(p)
    if p.is_file():
        return _checkpoint_ref(p, bq_names().get(p.resolve()))
    if str(target) not in reg:
        raise SystemExit(f"{target}: neither a checkpoint file nor a model in {MODELS_YAML} ({', '.join(reg)})")
    spec = dict(reg[str(target)])
    cls = str(spec.get("class", ""))
    if cls not in TRAINING_CLASSES:
        raise SystemExit(f"model {target}: class {cls!r} is not one of {TRAINING_CLASSES}")
    weights = spec.get("weights")
    fp = fingerprint(resolve(weights)) if weights else f"v{spec.get('version', 1)}"
    return ModelRef(name=str(target), model_id=f"{target}@{fp[:10]}", cache_key=f"ext_{target}", fingerprint=fp,
                    training_class=cls, spec=spec, seen_tasks=tuple(spec.get("seen_tasks") or ()))


class ExternalEncoder:
    """Base class of the adapters for registry models. Subclasses set ``dim`` and implement ``encode``; they may set
    ``needs_brain`` to receive the stored brain mask (skull stripping), and ``modalities``."""

    kind = "external"
    representation = "tokens"
    needs_brain = False
    dim = 0

    def __init__(self, spec: dict, device: torch.device):
        self.spec, self.device = spec, device
        self.modalities = tuple(spec.get("modalities") or ("ct", "mri"))

    def encode(self, vol_xyz: np.ndarray, modality: str, brain: np.ndarray | None, attrs: dict) -> dict:
        """``vol_xyz`` (X, Y, Z) float32 on the store's 1 mm MNI-registered grid; ``brain`` the stored brain mask (with
        ``needs_brain``); ``attrs`` the volume's stored attributes (``raw_intensity`` undoes the MRI z-score;
        ``native_spacing`` / ``thick_mm`` describe the acquisition; ``reg_affine`` maps the grid to the source scan).
        Returns ``pooled`` (D,) and optionally ``tokens`` (T, D) with ``xyz`` (T, 3) in [0, 1] and ``keep`` (T,)."""
        raise NotImplementedError

    @staticmethod
    def raw_intensity(vol_xyz: np.ndarray, modality: str, attrs: dict) -> np.ndarray:
        """The scan's own intensity scale: CT is already in HU; MRI is stored as a robust z-score of the (N4-corrected)
        intensities, ``(x - raw_median) / raw_mad``, with the background set to -6, which this inverts."""
        if str(modality).lower().startswith("ct") or "raw_mad" not in attrs:
            return vol_xyz
        return vol_xyz * float(attrs["raw_mad"]) + float(attrs["raw_median"])

    # ---- geometry: the stored grid in world (RAS mm, template space) and back, for models with their own input grid
    @staticmethod
    def grid_affine(attrs: dict) -> np.ndarray:
        """4 x 4 from a stored-grid voxel index (x, y, z) to RAS mm in the template space the store is registered to
        (``grid_*`` attrs are ITK geometry, i.e. LPS)."""
        a = np.eye(4)
        a[:3, :3] = np.asarray(attrs["grid_direction"], float).reshape(3, 3) * np.asarray(attrs["grid_spacing"], float)
        a[:3, 3] = np.asarray(attrs["grid_origin"], float)
        return np.diag([-1.0, -1.0, 1.0, 1.0]) @ a

    @staticmethod
    def sample(src: torch.Tensor, affine: np.ndarray, ras: torch.Tensor, mode: str = "bilinear",
               fill: float | None = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """``src`` (C, A0, A1, A2), whose voxel index maps to RAS mm by ``affine``, sampled (trilinear, or ``nearest``)
        at the points ``ras`` (P, 3): (C, P) values (``fill`` outside; None: the nearest edge value) and (P,) whether
        each point lies inside."""
        inv = torch.as_tensor(np.linalg.inv(affine), dtype=torch.float32, device=src.device)
        idx = ras.float() @ inv[:3, :3].T + inv[:3, 3]
        n = torch.tensor(src.shape[1:], dtype=torch.float32, device=src.device)
        inside = ((idx > -0.5) & (idx < n - 0.5)).all(-1)
        g = (2 * idx / (n - 1).clamp_min(1) - 1).flip(-1)       # grid_sample reads (A2, A1, A0) as (x, y, z)
        f = 0.0 if fill is None else float(fill)
        out = torch.nn.functional.grid_sample((src.float() - f)[None], g.view(1, 1, 1, -1, 3), mode=mode,
                                              padding_mode="border" if fill is None else "zeros", align_corners=True)
        return out.view(src.shape[0], -1) + f, inside

    def resample(self, vol_xyz: np.ndarray, attrs: dict, affine: np.ndarray, shape: tuple[int, int, int],
                 fill: float = 0.0, mode: str = "bilinear") -> torch.Tensor:
        """The stored volume on another grid (``shape``; voxel index -> RAS mm by ``affine``), on the device."""
        ax = [torch.arange(s, dtype=torch.float32, device=self.device) for s in shape]
        idx = torch.stack(torch.meshgrid(*ax, indexing="ij"), -1).view(-1, 3)
        a = torch.as_tensor(affine, dtype=torch.float32, device=self.device)
        src = torch.as_tensor(np.ascontiguousarray(vol_xyz), dtype=torch.float32, device=self.device)[None]
        out, _ = self.sample(src, self.grid_affine(attrs), idx @ a[:3, :3].T + a[:3, 3], mode=mode, fill=fill)
        return out.view(*shape)

    def to_token_grid(self, fmap: torch.Tensor, affine: np.ndarray, vol_xyz: np.ndarray, modality: str,
                      attrs: dict) -> dict:
        """A model's feature map ``fmap`` (C, A0, A1, A2; voxel index -> RAS mm by ``affine``) sampled at our token
        centres, so its tokens take the same readouts as ours (``tok_conv``, plane / grid / global PCA): ``tokens``
        (5120, C) in our (Z, Y, X) order, ``xyz`` as our encoder gives them, ``keep`` = our foreground gate and inside
        the model's field of view."""
        from brain.common.grid import GRID_XYZ, PATCH_XYZ
        from brain.downstream.embed3d import _crop_coords, token_gate

        coords, _ = _crop_coords(self.device, (0, 0, 0), GRID_XYZ, (1.0, 1.0, 1.0), PATCH_XYZ)
        coords = coords[0]                                     # token centres, mm from the grid corner, (x, y, z)
        g = torch.as_tensor(self.grid_affine(attrs), dtype=torch.float32, device=self.device)
        feats, inside = self.sample(fmap, affine, (coords - 0.5) @ g[:3, :3].T + g[:3, 3], fill=None)
        lo, hi = coords.min(0).values, coords.max(0).values
        fg = token_gate(torch.as_tensor(np.ascontiguousarray(vol_xyz), dtype=torch.float32, device=self.device),
                        modality, PATCH_XYZ)
        return {"tokens": feats.T.cpu(), "xyz": ((coords - lo) / (hi - lo).clamp_min(1e-6)).cpu(),
                "keep": (fg & inside).cpu()}

    @torch.no_grad()
    def encode_volume(self, path: Path, volume_id: str, modality: str, window: str) -> VolumeEmbedding:
        from brain.prep import store

        names = ("image", "brain") if self.needs_brain else ("image",)
        arrs = store.read_many(path, names)
        attrs = store.read_attrs(path, "image")
        vol = np.transpose(store.squeeze_channel(arrs["image"]).astype(np.float32), (2, 1, 0))
        brain = (np.transpose(store.squeeze_channel(arrs["brain"]) > 0, (2, 1, 0))
                 if self.needs_brain and arrs.get("brain") is not None else None)
        out = self.encode(vol, modality, brain, attrs)
        pooled = torch.as_tensor(np.asarray(out["pooled"], dtype=np.float32))
        tokens = out.get("tokens")
        if tokens is None:                                     # a global-only encoder: one token, read out globally
            return VolumeEmbedding(tokens=pooled[None], xyz=torch.zeros(1, 3), keep=torch.ones(1, dtype=torch.bool),
                                   cls=pooled)
        return VolumeEmbedding(tokens=torch.as_tensor(np.asarray(tokens, dtype=np.float32)),
                               xyz=torch.as_tensor(np.asarray(out["xyz"], dtype=np.float32)),
                               keep=torch.as_tensor(np.asarray(out["keep"], dtype=bool)), cls=pooled)
