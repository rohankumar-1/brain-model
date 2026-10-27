"""Embed a full native volume with the 3D encoder. Dropped tokens stay zero."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from brain.data.gate import token_keep
from brain.data.coords import crop_bbox_mm, patch_size_mm, token_centers_mm
from brain.data.volumes import meta_id, n_meta
from brain.models.vit3d import VIT_SIZES, BrainEncoder
from brain.common.windowing import asinh_window, windows_for
from brain.common.grid import GRID_XYZ, PATCH_XYZ


DEFAULT_GATE = {"ct_hu": (-500.0, 200.0), "mri_lo": -5.5}
FINE_TILE = ((128, 128, 64), (64, 64, 32))  # training crop size and stride, voxels (x, y, z)


_VOLUME_META: dict[str, int] | None = None


def volume_meta() -> dict[str, int]:
    """volume_id -> META id from the store index (cached), for encoders trained with metadata."""
    global _VOLUME_META
    if _VOLUME_META is None:
        import pandas as pd

        from brain.downstream.tasks import DEFAULT_VOLUMES

        v = pd.read_csv(DEFAULT_VOLUMES, usecols=lambda c: c in ("volume_id", "contrast", "modality", "dataset"),
                        low_memory=False)
        _VOLUME_META = {str(i): meta_id(c, m, d) for i, c, m, d in
                        zip(v.volume_id, v.get("contrast", [None] * len(v)), v.modality, v.dataset)}
    return _VOLUME_META


def embed_batch_size() -> int:
    """How many same-size crops share one encoder forward. Tiled 8x8x4 is 27 windows; 8 at a time is the win."""
    return max(1, int(os.environ.get("BRAIN_EMBED_BATCH", "8")))


def fine_pool_on() -> bool:
    """``BRAIN_FINE_POOL=1`` makes embed_tokens default to fine-patch tokens averaged onto the coarse grid."""
    return os.environ.get("BRAIN_FINE_POOL", "") == "1"


def fine_grid_on() -> bool:
    """``BRAIN_FINE_GRID=1`` keeps the native 8x8x4 token grid (no 2x2x2 pool onto 16^3)."""
    return os.environ.get("BRAIN_FINE_GRID", "") == "1"


def fine_patch_xyz(coarse: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(max(1, int(p) // 2) for p in coarse)


def token_gate(cropped_xyz: torch.Tensor, modality: str, patch_xyz: tuple[int, int, int],
               gate: dict | None = None) -> torch.Tensor:
    """Foreground flag per patch token, (T,), in the encoder's (Z, Y, X) token order."""
    gate = gate or DEFAULT_GATE
    mod = torch.tensor([0 if str(modality).lower().startswith("ct") else 1], device=cropped_xyz.device)
    keep = token_keep(
        cropped_xyz.view(1, 1, *cropped_xyz.shape), mod, patch_xyz,
        ct_hu=tuple(gate.get("ct_hu", DEFAULT_GATE["ct_hu"])), mri_lo=float(gate.get("mri_lo", DEFAULT_GATE["mri_lo"])),
        mode=str(gate.get("mode", "center")),
    )
    return keep[0].reshape(-1)


def is_brain3d_state(state: dict) -> bool:
    for k, w in state.items():
        if str(k).endswith("patch_embed.proj.weight") and getattr(w, "ndim", 0) == 5:
            return True
    return False


class FrozenEncoder3D:
    kind = "brain3d"
    representation = "tokens"

    def __init__(self, model: BrainEncoder, patch_xyz: tuple[int, int, int], native_xyz: tuple[int, int, int],
                 spacing_mm: tuple[float, float, float]):
        self.model = model
        self.patch_xyz = tuple(int(v) for v in patch_xyz)
        self.native_xyz = tuple(int(v) for v in native_xyz)
        self.spacing_mm = tuple(float(v) for v in spacing_mm)
        self.dim = int(model.embed_dim)
        self.gate = dict(DEFAULT_GATE)
        # Blocks whose outputs are returned (concatenated before the final tokens) by embed_tokens.
        self.tap_blocks: tuple[int, ...] = ()
        self.tap_norm = False          # layer-normalise tapped (pre-norm residual) features, as the final tokens are
        self.taps_only = False         # drop the final tokens (and the CLS) and keep only the tapped blocks
        self.meta: int | None = None   # the next volume's META id (``volume_meta``); None = unknown
        self.task_tokens: torch.Tensor | None = None   # a task-token model's trained-task tokens (task_token_prefix)
        # CLS embeddings (models with a CLS token): of the last single-pass encode, per window of the last
        # ``embed_windows`` call, and per volume of the last ``extract_volume_embeddings_3d`` run.
        self._last_cls: torch.Tensor | None = None
        self.last_cls: torch.Tensor | None = None
        self.cls_bank: np.ndarray | None = None
        self._last_reg: torch.Tensor | None = None       # register outputs, (B, R, D), of the last encode
        self.last_reg: torch.Tensor | None = None        # (n_windows, R, D) for the last volume
        self.reg_bank: np.ndarray | None = None          # (N, n_windows * R * D) over a whole extraction

    @torch.no_grad()
    def embed_tokens(
        self, vol_xyz: torch.Tensor, c: float, w: float, modality: str = "mri",
        window_xyz: tuple[int, int, int] | None = None, stride_xyz: tuple[int, int, int] | None = None,
        fine_pool: bool | None = None, fine_grid: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Native-grid tokens. ``vol_xyz`` is (X, Y, Z). Returns ``(T, D)``, normalized centers, the keep mask.

        With ``window_xyz`` the volume is encoded as overlapping windows of that size (the training crop
        size) and each token is the mean over the windows that cover it. Without it, one pass over the volume.

        With ``fine_pool`` (default: ``BRAIN_FINE_POOL``) the volume is encoded with patches half as large on
        every axis, as tiled training-size windows, and the 2 x 2 x 2 fine tokens under each coarse position are
        averaged (the way training ties its fine crops to a coarse position). The result lies on the usual grid.

        With ``fine_grid`` (default: ``BRAIN_FINE_GRID``) the same tiled 8 x 8 x 4 encode is kept on the
        32 x 32 x 32 grid. ``fine_grid`` wins if both flags are set.
        """
        use_grid = fine_grid_on() if fine_grid is None else fine_grid
        use_pool = fine_pool_on() if fine_pool is None else fine_pool
        if fine_pool is None and getattr(self, "force_fine_pool", False):
            use_pool = True
        if use_grid and window_xyz is None:
            return self._embed_fine_tiled(vol_xyz, c, w, modality)
        if use_pool and window_xyz is None:
            return self._embed_fine_pooled(vol_xyz, c, w, modality)
        native = vol_xyz.float()
        cropped, start, size = _crop_to_patch(native, self.patch_xyz)
        fg = token_gate(cropped, modality, self.patch_xyz, self.gate)
        x = asinh_window(cropped, c, w)
        if window_xyz is None:
            tokens, coords = self._encode(x, start, size, fg)
        else:
            tokens, coords = self._encode_tiled(x, start, size, fg, window_xyz, stride_xyz or window_xyz)
        lo, hi = coords.min(dim=0).values, coords.max(dim=0).values
        return tokens, (coords - lo) / (hi - lo).clamp_min(1e-6), fg

    def _embed_fine_tiled(self, vol_xyz, c, w, modality):
        """Tiled 8x8x4 tokens on the native fine grid. Restores ``patch_xyz``."""
        coarse = self.patch_xyz
        fine = fine_patch_xyz(coarse)
        cropped, _, _ = _crop_to_patch(vol_xyz.float(), fine)
        self.patch_xyz = fine
        try:
            return self.embed_tokens(
                cropped, c, w, modality, window_xyz=FINE_TILE[0], stride_xyz=FINE_TILE[1],
                fine_pool=False, fine_grid=False,
            )
        finally:
            self.patch_xyz = coarse

    def _embed_fine_pooled(self, vol_xyz, c, w, modality):
        coarse = self.patch_xyz
        fine = fine_patch_xyz(coarse)
        cropped, start, size = _crop_to_patch(vol_xyz.float(), coarse)
        fg = token_gate(cropped, modality, coarse, self.gate)
        tok, _, _ = self._embed_fine_tiled(cropped, c, w, modality)
        nx, ny, nz = (size[0] // coarse[0], size[1] // coarse[1], size[2] // coarse[2])
        sx, sy, sz = coarse[0] // fine[0], coarse[1] // fine[1], coarse[2] // fine[2]
        d = tok.shape[-1]
        pooled = tok.float().view(nz, sz, ny, sy, nx, sx, d).mean(dim=(1, 3, 5)).reshape(-1, d)
        pooled = pooled * fg.unsqueeze(-1)
        coords, _ = _crop_coords(vol_xyz.device, start, size, self.spacing_mm, coarse)
        coords = coords[0]
        lo, hi = coords.min(dim=0).values, coords.max(dim=0).values
        return pooled.to(tok.dtype), (coords - lo) / (hi - lo).clamp_min(1e-6), fg

    def _encode_batch(self, x, starts, size, keep):
        """``x`` is (B, X, Y, Z), ``starts`` are B origin triples, ``keep`` is (B, T)."""
        view = x.unsqueeze(1)
        coords, pmms = zip(*[_crop_coords(view.device, st, size, self.spacing_mm, self.patch_xyz) for st in starts])
        coords, pmm = torch.cat(coords, 0), torch.cat(pmms, 0)
        self.model.eval()
        amp = dict(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda)
        meta = None
        if getattr(self.model, "meta_emb", None) is not None:   # a model trained with metadata: the scan's, or unknown
            meta = torch.full((view.shape[0],), int(self.meta or 0), dtype=torch.long, device=view.device)
        tt = None if self.task_tokens is None else self.task_tokens.to(view.device)
        with torch.autocast(**amp):
            out = self.model(view, coords, pmm, keep=keep, patch_xyz=self.patch_xyz, taps=self.tap_blocks, meta=meta,
                             task_tokens=tt)
        taps = [t.float() for t in out["taps"]]                 # bf16 under autocast; float32 like the final tokens
        if self.tap_norm:
            taps = [torch.nn.functional.layer_norm(t, t.shape[-1:]) for t in taps]
        tok = torch.cat(taps if self.taps_only else [*taps, out["tokens"]], dim=-1)
        self._last_cls = None if self.taps_only else out.get("cls")
        self._last_reg = out.get("registers")
        return tok, coords, out["patch"]

    def _embed_patch_batch(self, x, starts, size, keep):
        """Patch embedding only (no transformer). Same grid as ``_encode_batch``."""
        view = x.unsqueeze(1)
        coords, pmms = zip(*[_crop_coords(view.device, st, size, self.spacing_mm, self.patch_xyz) for st in starts])
        coords, pmm = torch.cat(coords, 0), torch.cat(pmms, 0)
        self.model.eval()
        amp = dict(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda)
        with torch.autocast(**amp):
            tokens, _ = self.model._embed(view, coords, pmm, self.patch_xyz, zyx=False)
        return tokens * keep.unsqueeze(-1).to(dtype=tokens.dtype), coords

    def embed_patch(self, vol_xyz: torch.Tensor, c: float, w: float, modality: str = "mri",
                    window_xyz: tuple[int, int, int] | None = None,
                    stride_xyz: tuple[int, int, int] | None = None) -> torch.Tensor:
        """Pre-transformer patch embeddings on the ``embed_tokens`` grid (decoder skip)."""
        cropped, start, size = _crop_to_patch(vol_xyz.float(), self.patch_xyz)
        fg = token_gate(cropped, modality, self.patch_xyz, self.gate)
        x = asinh_window(cropped, c, w)
        if window_xyz is None:
            pe, _ = self._embed_patch_batch(x.unsqueeze(0), (start,), size, fg.view(1, -1))
            return pe[0]
        pe, _ = self._encode_tiled(x, start, size, fg, window_xyz, stride_xyz or window_xyz, patch_only=True)
        return pe

    def _encode(self, x, start, size, fg):
        tok, coords, _ = self._encode_batch(x.unsqueeze(0), (start,), size, fg.view(1, -1))
        return tok[0], coords[0]

    def embed_volumes(self, vols, c: float, w: float, modality: str = "mri"):
        """One forward for several same-size full volumes (BraTS sequences)."""
        cropped, starts, keeps, size = [], [], [], None
        for v in vols:
            cr, st, sz = _crop_to_patch(v.float(), self.patch_xyz)
            cropped.append(asinh_window(cr, c, w))
            starts.append(st)
            keeps.append(token_gate(cr, modality, self.patch_xyz, self.gate))
            size = sz
        toks, _, _ = self._encode_batch(torch.stack(cropped), starts, size, torch.stack(keeps))
        return toks

    def embed_patches(self, vols, c: float, w: float, modality: str = "mri"):
        """Patch embeddings for several same-size full volumes (BraTS sequences)."""
        cropped, starts, keeps, size = [], [], [], None
        for v in vols:
            cr, st, sz = _crop_to_patch(v.float(), self.patch_xyz)
            cropped.append(asinh_window(cr, c, w))
            starts.append(st)
            keeps.append(token_gate(cr, modality, self.patch_xyz, self.gate))
            size = sz
        pe, _ = self._embed_patch_batch(torch.stack(cropped), starts, size, torch.stack(keeps))
        return pe

    def embed_windows(self, vol_xyz, windows, modality: str = "mri", **kwargs):
        """Same volume under several asinh windows, one batched forward when the encode is a single pass."""
        seqs, xyz, fg = [], None, None
        tiled = kwargs.get("window_xyz") is not None
        fine = kwargs.get("fine_pool")
        if fine is None:
            fine = fine_pool_on()
        grid = kwargs.get("fine_grid")
        if grid is None:
            grid = fine_grid_on()
        if tiled or fine or grid or len(windows) <= 1:
            clses, regs = [], []
            for c, w in windows:
                self._last_cls = self._last_reg = None
                seq, xyz, fg = self.embed_tokens(vol_xyz, c, w, modality=modality, **kwargs)
                seqs.append(seq)
                single = not (tiled or fine or grid)              # a tiled encode has no single CLS / registers
                clses.append(self._last_cls[0] if single and self._last_cls is not None else None)
                regs.append(self._last_reg[0] if single and self._last_reg is not None else None)
            self.last_cls = torch.stack(clses) if clses and all(x is not None for x in clses) else None
            self.last_reg = torch.stack(regs) if regs and all(x is not None for x in regs) else None
            return torch.stack(seqs), xyz, fg
        native = vol_xyz.float()
        cropped, start, size = _crop_to_patch(native, self.patch_xyz)
        fg = token_gate(cropped, modality, self.patch_xyz, self.gate)
        xs = torch.stack([asinh_window(cropped, c, w) for c, w in windows])
        keep = fg.unsqueeze(0).expand(len(windows), -1).contiguous()
        toks, coords, _ = self._encode_batch(xs, (start,) * len(windows), size, keep)
        self.last_cls = self._last_cls                      # (n_windows, D) or None
        self.last_reg = self._last_reg                      # (n_windows, R, D) or None
        lo, hi = coords[0].min(dim=0).values, coords[0].max(dim=0).values
        xyz = (coords[0] - lo) / (hi - lo).clamp_min(1e-6)
        return toks, xyz, fg

    @torch.no_grad()
    def encode_volume(self, path: Path, volume_id: str, modality: str, window: str):
        """The suite's encoder interface (``brain.downstream.models``): one stored volume under the ``window`` asinh
        windows, tokens concatenated over windows. ``BRAIN_EVAL_TILE=1`` encodes overlapping training-size windows
        (128 x 128 x 64 mm, half overlap) instead of the whole volume at once."""
        from brain.downstream.models import VolumeEmbedding

        x = torch.from_numpy(np.ascontiguousarray(read_volume_xyz(path))).to(next(self.model.parameters()).device)
        self.meta = volume_meta().get(str(volume_id)) if getattr(self.model, "meta_emb", None) is not None else None
        tile = os.environ.get("BRAIN_EVAL_TILE", "") == "1"
        seqs, xyz, fg = self.embed_windows(x, windows_for(window, modality), modality=modality,
                                           **({"window_xyz": (128, 128, 64), "stride_xyz": (64, 64, 32)} if tile else {}))
        return VolumeEmbedding(tokens=torch.cat(list(seqs), dim=-1), xyz=xyz, keep=fg,
                               cls=None if self.last_cls is None else self.last_cls.reshape(-1),
                               registers=None if self.last_reg is None else self.last_reg.reshape(-1))

    def _encode_tiled(self, x, start, size, fg, window_xyz, stride_xyz, patch_only: bool = False):
        px, py, pz = self.patch_xyz
        grid = (size[0] // px, size[1] // py, size[2] // pz)  # (nx, ny, nz) tokens
        win = tuple(min(int(a), int(b)) for a, b in zip(window_xyz, size))
        stride = tuple(int(v) for v in stride_xyz)
        if any(w % p or s % p for w, s, p in zip(win, stride, self.patch_xyz)):
            raise ValueError(f"window {win} and stride {stride} must be multiples of the patch {self.patch_xyz}")
        origins = [sorted({*range(0, size[a] - win[a] + 1, stride[a]), size[a] - win[a]}) for a in range(3)]
        fg_zyx = fg.view(grid[2], grid[1], grid[0])
        jobs = []
        for ox in origins[0]:
            for oy in origins[1]:
                for oz in origins[2]:
                    tx, ty, tz = ox // px, oy // py, oz // pz
                    nx, ny, nz = win[0] // px, win[1] // py, win[2] // pz
                    keep = fg_zyx[tz:tz + nz, ty:ty + ny, tx:tx + nx].reshape(-1)
                    sub = x[ox:ox + win[0], oy:oy + win[1], oz:oz + win[2]]
                    jobs.append((ox, oy, oz, tx, ty, tz, nx, ny, nz, keep, sub))
        acc = count = None
        bs = embed_batch_size()
        for i in range(0, len(jobs), bs):
            chunk = jobs[i:i + bs]
            xs = torch.stack([j[-1] for j in chunk])
            starts = [(start[0] + j[0], start[1] + j[1], start[2] + j[2]) for j in chunk]
            keeps = torch.stack([j[-2] for j in chunk])
            if patch_only:
                toks, _ = self._embed_patch_batch(xs, starts, win, keeps)
            else:
                toks, _, _ = self._encode_batch(xs, starts, win, keeps)
            if acc is None:
                acc = torch.zeros(grid[2], grid[1], grid[0], toks.shape[-1], device=toks.device)
                count = torch.zeros(grid[2], grid[1], grid[0], 1, device=toks.device)
            for tok, j in zip(toks, chunk):
                _, _, _, tx, ty, tz, nx, ny, nz, _, _ = j
                acc[tz:tz + nz, ty:ty + ny, tx:tx + nx] += tok.float().view(nz, ny, nx, -1)
                count[tz:tz + nz, ty:ty + ny, tx:tx + nx] += 1
        full, _ = _crop_coords(x.device, start, size, self.spacing_mm, self.patch_xyz)
        return (acc / count.clamp_min(1)).reshape(-1, acc.shape[-1]), full[0]


def task_token_prefix(ckpt: dict) -> torch.Tensor | None:
    """(T, D) inputs of a task-token BrainQuery's trained-task tokens (``BrainQuery.task_queries`` of its written
    tasks), or None. Its patch tokens attended them throughout training, so frozen transfer encodes with them in the
    sequence too: without, the features shift (~40% for a warm start) and the model is scored on features it never
    produced."""
    st = ckpt.get("multitask") or {}
    if "tt_write" not in st or not bool(st["tt_write"].any()):
        return None
    from brain.multitask.data import TYPE_ID
    from brain.multitask.registry import load_registry

    reg = load_registry((ckpt.get("config") or {}).get("data", {}).get("registry")).subset(list(ckpt["tasks"]))
    ids = st["tt_write"].nonzero()[:, 0].cpu()
    types = torch.tensor([TYPE_ID[reg.tasks[int(i)].type] for i in ids])
    return (st["task_emb.weight"].float().cpu()[ids] + st["type_emb.weight"].float().cpu()[types]).contiguous()


def load_brain3d(ckpt: dict, device: torch.device, pretrained: bool = True) -> FrozenEncoder3D:
    """``pretrained=False`` keeps the architecture and drops the weights (a random-init control)."""
    state = {str(k).removeprefix("_orig_mod.").removeprefix("module."): v for k, v in ckpt["model"].items()
             if not str(k).removeprefix("module.").startswith(("anchor.", "predictor."))}
    cfg = ckpt.get("config") or {}
    m, v = cfg.get("model") or {}, cfg.get("views") or {}
    size = str(m.get("size", "tiny"))
    if size not in VIT_SIZES:
        dim = int(state["patch_embed.proj.weight"].shape[0])
        size = next(k for k, spec in VIT_SIZES.items() if spec["embed_dim"] == dim)
    patch = tuple(int(x) for x in (v.get("patch_xyz") or v.get("global_patch_xyz") or (16, 16, 8)))
    native = tuple(int(x) for x in (cfg.get("data") or {}).get("volume_size_xyz", GRID_XYZ))
    spacing = tuple(float(x) for x in (cfg.get("data") or {}).get("default_spacing_mm", (1.0, 1.0, 1.0)))
    projector_bn = any("projector" in str(k) and str(k).endswith("running_mean") for k in state)
    proj_norm = "batch" if projector_bn else str(m.get("proj_norm", "layer"))
    net = BrainEncoder(
        size=size, patch_xyz=patch, in_chans=int(m.get("in_chans", 1)),
        proj_dim=int(m.get("proj_dim", 128)), proj_hidden=m.get("proj_hidden"),
        qk_norm=bool(m.get("qk_norm", False)), rope_base_mm=float(m.get("rope_base_mm", 100.0)),
        checkpoint_blocks=False, proj_norm=proj_norm,
        dense_dim=int(state["token_head.net.6.weight"].shape[0]) if "token_head.net.6.weight" in state else 0,
        dense_hidden=int(m.get("dense_hidden", 512)),
        n_cls=int(m.get("n_cls", 0)), n_registers=int(m.get("n_registers", 0)),
        mlp=str(m.get("mlp", "gelu")), layer_scale=m.get("layer_scale"),
        rope_min_period_mm=m.get("rope_min_period_mm"),
        rope_fp32=bool(m.get("rope_fp32", False)), fp32_residual=bool(m.get("fp32_residual", False)),
        n_meta=n_meta(m),
    ).to(device)
    moe_blocks = {int(k.split(".")[1]) for k in state if ".mlp.experts." in k}
    if moe_blocks:                    # mixture-of-experts top blocks (multitask ``model.moe``): rebuild before loading
        n_exp = 1 + max(int(k.split(".")[4]) for k in state if ".mlp.experts." in k)
        net.enable_moe(net.depth - min(moe_blocks), n_exp, int((m.get("moe") or {}).get("top_k", 2)))
    if pretrained:
        net.load_state_dict(state)
    net.eval().requires_grad_(False)
    # A model trained on 8 x 8 x 4 tokens (MAE) is read out on the usual 16 x 16 x 8 grid: it encodes at its own patch
    # in training-size windows and the 2 x 2 x 2 fine tokens under each grid position are averaged (``fine_pool``),
    # so every frozen head, cache and probe works unchanged.
    native_fine = patch == fine_patch_xyz(PATCH_XYZ)
    enc = FrozenEncoder3D(net, PATCH_XYZ if native_fine else patch, native, spacing)
    enc.force_fine_pool = native_fine
    enc.task_tokens = task_token_prefix(ckpt) if pretrained else None
    gate = v.get("gate") or (cfg.get("loss") or {}).get("parenchyma")
    if isinstance(gate, dict):
        enc.gate = gate
    return enc


def read_volume_xyz(path: Path) -> np.ndarray:
    from brain.prep import store

    return np.transpose(store.read_image(path).astype(np.float32), (2, 1, 0))


def _crop_to_patch(vol: torch.Tensor, patch_xyz: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
    """Center-crop (X, Y, Z) so each axis is a multiple of the patch."""
    px, py, pz = patch_xyz
    nx, ny, nz = (int(v) for v in vol.shape[-3:])
    tx, ty, tz = (nx // px) * px, (ny // py) * py, (nz // pz) * pz
    if min(tx, ty, tz) <= 0:
        raise ValueError(f"volume {(nx, ny, nz)} is smaller than patch {patch_xyz}")
    x0, y0, z0 = (nx - tx) // 2, (ny - ty) // 2, (nz - tz) // 2
    return vol[..., x0:x0 + tx, y0:y0 + ty, z0:z0 + tz], (x0, y0, z0), (tx, ty, tz)


def _crop_coords(device, start, size, spacing_mm, patch_xyz):
    bbox = crop_bbox_mm(start, size, spacing_mm, (0.0, 0.0, 0.0)).to(device)
    coords = token_centers_mm(bbox, size, patch_xyz).unsqueeze(0).to(device)
    pmm = patch_size_mm(bbox, size, patch_xyz).unsqueeze(0).to(device)
    return coords, pmm


@torch.no_grad()
def extract_volume_embeddings(
    enc, image_paths: dict[str, str], volume_ids: list[str], slices_root: Path,
    modalities: dict[str, str] | None = None, window: str = "canonical",
) -> dict:
    """Every volume through ``enc.encode_volume``: tokens (N, T, D), foreground mask (N, T), positions (N, T, 3), and
    the global banks ``cls`` (N, D') / ``registers`` (N, R') when the encoder has them."""
    from brain.downstream.embeddings import resolve_path

    modalities = modalities or {}
    sample_mod = modalities.get(volume_ids[0], "mri") if volume_ids else "mri"
    embs = coords = mask = cls_bank = reg_bank = None
    n = len(volume_ids)
    for i, vid in enumerate(volume_ids):
        e = enc.encode_volume(resolve_path(slices_root, image_paths[vid]), str(vid), modalities.get(vid, sample_mod),
                              window)
        feat = e.tokens
        if embs is None:
            t, d = feat.shape
            # float16 for the fine grid or a large task (RSNA: 25k x 5120 x 384 would be ~200 GB as float32)
            dtype = np.float16 if t > 8192 or n * t * d * 4 > 16 * 2**30 else np.float32
            embs = np.zeros((n, t, d), dtype=dtype)
            coords = np.zeros((n, t, 3), dtype=np.float32)
            mask = np.ones((n, t), dtype=bool)
        embs[i] = feat.detach().float().cpu().numpy()
        coords[i] = e.xyz.detach().float().cpu().numpy()
        mask[i] = e.keep.detach().cpu().numpy()
        for bank, vec in (("cls", e.cls), ("reg", e.registers)):
            if vec is None:
                continue
            v = vec.reshape(-1).float().cpu().numpy()
            if bank == "cls":
                cls_bank = np.zeros((n, v.shape[0]), dtype=np.float32) if cls_bank is None else cls_bank
                cls_bank[i] = v
            else:
                reg_bank = np.zeros((n, v.shape[0]), dtype=np.float32) if reg_bank is None else reg_bank
                reg_bank[i] = v
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == n:
            print(f"[embed] {i + 1}/{n} volumes window={window} tokens={feat.shape[0]} dim={feat.shape[1]} "
                  f"encoder={getattr(enc, 'kind', '?')}")
    return {"mus": embs, "mask": mask, "z": coords, "cls": cls_bank, "registers": reg_bank}
