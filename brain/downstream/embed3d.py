"""Our frozen 3D encoders: load a checkpoint (``load_brain3d``), embed full native volumes. Dropped tokens stay zero."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from brain.data.gate import token_keep
from brain.data.coords import crop_bbox_mm, patch_size_mm, token_centers_mm
from brain.data.volumes import meta_id, n_meta
from brain.models.vit3d import VIT_SIZES, BrainEncoder
from brain.common.windowing import asinh_window, encoder_input, windows_for
from brain.models.window_bank import attach_bank, bank_of, encoder_in_chans
from brain.common.grid import GRID_XYZ


DEFAULT_GATE = {"ct_hu": (-500.0, 200.0), "mri_lo": -5.5}


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


def token_gate(cropped_xyz: torch.Tensor, modality: str, patch_xyz: tuple[int, int, int],
               gate: dict | None = None) -> torch.Tensor:
    """Foreground flag per patch token, (T,), in the encoder's (Z, Y, X) token order."""
    gate = gate or DEFAULT_GATE
    mod = torch.tensor([0 if str(modality).lower().startswith("ct") else 1], device=cropped_xyz.device)
    keep = token_keep(
        cropped_xyz.view(1, 1, *cropped_xyz.shape), mod, patch_xyz,
        ct_hu=tuple(gate.get("ct_hu", DEFAULT_GATE["ct_hu"])), mri_lo=float(gate.get("mri_lo", DEFAULT_GATE["mri_lo"])),
        mode=str(gate.get("mode", "center")), min_frac=float(gate.get("min_frac", 0.25)),
    )
    return keep[0].reshape(-1)


def is_brain3d_state(state: dict) -> bool:
    for k, w in state.items():
        if str(k).endswith("patch_embed.proj.weight") and getattr(w, "ndim", 0) == 5:
            return True
    return False


class FrozenEncoder3D:
    """One of our 3D encoders, frozen: the suite's ``encode_volume`` and the token-grid encodes seg3d / probe3d use."""

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
        self.meta: int | None = None   # the next volume's META id (``volume_meta``); None = unknown
        self.task_tokens: torch.Tensor | None = None   # a task-token model's trained-task tokens (task_token_prefix)
        # an encoder with a learned window bank (brain.models.window_bank: its windows are in its state dict) windows
        # every scan with it in one pass, whatever ``window``; so does a three-input encoder with the role windows
        self.bank = bank_of(model)
        self.in_chans = int(getattr(model, "in_chans", 0) or model.patch_embed.proj.in_channels)
        self.role_channels = self.bank is None and self.in_chans == 3
        self.single_pass = self.bank is not None or self.role_channels
        self.meta_off = False          # encode as if the scan's metadata (its contrast) were unknown
        # the CLS (n_windows, D) and register (n_windows, R, D) outputs of the last ``embed_windows`` call, or None
        self.last_cls: torch.Tensor | None = None
        self.last_reg: torch.Tensor | None = None

    @torch.no_grad()
    def _window(self, cropped: torch.Tensor, c: float, w: float, modality: str) -> torch.Tensor:
        """The encoder input of a native-unit (X, Y, Z) crop: one asinh window (c, w), or (``windowing.encoder_input``,
        ``c`` / ``w`` ignored) the window bank's K channels / the three role channels (C, X, Y, Z)."""
        if not self.single_pass:
            return asinh_window(cropped, c, w)
        mod = torch.tensor([0 if str(modality).lower().startswith("ct") else 1], device=cropped.device)
        return encoder_input(cropped[None, None], mod, self.in_chans, self.bank)[0]

    def _encode_batch(self, x, starts, size, keep) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """``x`` is (B, X, Y, Z) or (B, C, X, Y, Z), ``starts`` are B origin triples, ``keep`` is (B, T). Returns the
        tokens (B, T, D), their centres (B, T, 3) in mm and the model's outputs."""
        view = x if x.dim() == 5 else x.unsqueeze(1)                   # (B, C, X, Y, Z)
        coords, pmms = zip(*[_crop_coords(view.device, st, size, self.spacing_mm, self.patch_xyz) for st in starts])
        coords, pmm = torch.cat(coords, 0), torch.cat(pmms, 0)
        self.model.eval()
        amp = dict(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda)
        meta = None
        if getattr(self.model, "meta_emb", None) is not None:   # a model trained with metadata: the scan's, or unknown
            meta = torch.full((view.shape[0],), int(self.meta or 0), dtype=torch.long, device=view.device)
        tt = None if self.task_tokens is None else self.task_tokens.to(view.device)
        with torch.autocast(**amp):
            out = self.model(view, coords, pmm, keep=keep, patch_xyz=self.patch_xyz, meta=meta, task_tokens=tt)
        return out["tokens"], coords, out

    @torch.no_grad()
    def embed_windows(self, vol_xyz: torch.Tensor, windows, modality: str = "mri"):
        """The volume (X, Y, Z, native units) under each (centre, width) asinh window, in one batched forward: tokens
        (n_windows, T, D), normalised token centres (T, 3) and the foreground gate (T,). Sets ``last_cls`` /
        ``last_reg``."""
        cropped, start, size = _crop_to_patch(vol_xyz.float(), self.patch_xyz)
        fg = token_gate(cropped, modality, self.patch_xyz, self.gate)
        xs = torch.stack([self._window(cropped, c, w, modality) for c, w in windows])
        keep = fg.unsqueeze(0).expand(len(windows), -1).contiguous()
        toks, coords, out = self._encode_batch(xs, (start,) * len(windows), size, keep)
        self.last_cls, self.last_reg = out.get("cls"), out.get("registers")
        lo, hi = coords[0].min(dim=0).values, coords[0].max(dim=0).values
        return toks, (coords[0] - lo) / (hi - lo).clamp_min(1e-6), fg

    def embed_tokens(self, vol_xyz: torch.Tensor, c: float, w: float, modality: str = "mri"):
        """One window: tokens (T, D), normalised centres (T, 3), the foreground gate (T,)."""
        toks, xyz, fg = self.embed_windows(vol_xyz, [(c, w)], modality)
        return toks[0], xyz, fg

    @torch.no_grad()
    def embed_volumes(self, vols, c: float, w: float, modality: str = "mri"):
        """One forward for several same-size full volumes (BraTS sequences): tokens (n_volumes, T, D)."""
        cropped, starts, keeps, size = [], [], [], None
        for v in vols:
            cr, st, sz = _crop_to_patch(v.float(), self.patch_xyz)
            cropped.append(self._window(cr, c, w, modality))
            starts.append(st)
            keeps.append(token_gate(cr, modality, self.patch_xyz, self.gate))
            size = sz
        return self._encode_batch(torch.stack(cropped), starts, size, torch.stack(keeps))[0]

    @torch.no_grad()
    def encode_volume(self, path: Path, volume_id: str, modality: str, window: str):
        """The suite's encoder interface (``brain.downstream.models``): one stored volume under the ``window`` asinh
        windows, tokens concatenated over windows."""
        from brain.downstream.models import VolumeEmbedding

        x = torch.from_numpy(np.ascontiguousarray(read_volume_xyz(path))).to(next(self.model.parameters()).device)
        with_meta = getattr(self.model, "meta_emb", None) is not None and not self.meta_off
        self.meta = volume_meta().get(str(volume_id)) if with_meta else None
        windows = [(0.0, 1.0)] if self.single_pass else windows_for(window, modality)   # bank / roles: one pass
        seqs, xyz, fg = self.embed_windows(x, windows, modality=modality)
        return VolumeEmbedding(tokens=torch.cat(list(seqs), dim=-1), xyz=xyz, keep=fg,
                               cls=None if self.last_cls is None else self.last_cls.reshape(-1),
                               registers=None if self.last_reg is None else self.last_reg.reshape(-1))


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
    native = tuple(int(x) for x in (cfg.get("data") or {}).get("volume_size_xyz", GRID_XYZ))
    spacing = tuple(float(x) for x in (cfg.get("data") or {}).get("default_spacing_mm", (1.0, 1.0, 1.0)))
    if m.get("arch") == "resnet3d":          # convolutional single-task baseline (brain.models.resnet3d)
        from brain.models.resnet3d import ResNetEncoder

        r = m.get("resnet") or {}
        patch = tuple(int(x) for x in (v.get("patch_xyz") or (16, 16, 8)))
        net = ResNetEncoder(widths=r.get("widths", (64, 128, 240, 384)), blocks=r.get("blocks", (2, 2, 2, 2)),
                            in_chans=encoder_in_chans(m), patch_xyz=patch, checkpoint_blocks=False).to(device)
        attach_bank(net, m)
        if pretrained:
            net.load_state_dict(state)
        enc = FrozenEncoder3D(net.eval().requires_grad_(False), patch, native, spacing)
        if isinstance(v.get("gate"), dict):           # the token gate it trained with, as for the ViT below
            enc.gate = v["gate"]
        return enc
    size = str(m.get("size", "tiny"))
    if size not in VIT_SIZES:
        dim = int(state["patch_embed.proj.weight"].shape[0])
        size = next(k for k, spec in VIT_SIZES.items() if spec["embed_dim"] == dim)
    patch = tuple(int(x) for x in (v.get("patch_xyz") or v.get("global_patch_xyz") or (16, 16, 8)))
    projector_bn = any("projector" in str(k) and str(k).endswith("running_mean") for k in state)
    proj_norm = "batch" if projector_bn else str(m.get("proj_norm", "layer"))
    net = BrainEncoder(
        size=size, patch_xyz=patch, in_chans=encoder_in_chans(m),
        proj_dim=int(m.get("proj_dim", 128)), proj_hidden=m.get("proj_hidden"),
        qk_norm=bool(m.get("qk_norm", False)), rope_base_mm=float(m.get("rope_base_mm", 100.0)),
        checkpoint_blocks=False, proj_norm=proj_norm,
        dense_dim=int(state["token_head.net.6.weight"].shape[0]) if "token_head.net.6.weight" in state else 0,
        dense_hidden=int(m.get("dense_hidden", 512)),
        n_cls=int(m.get("n_cls", 0)), n_registers=int(m.get("n_registers", 0)),
        mlp=str(m.get("mlp", "gelu")), layer_scale=m.get("layer_scale"),
        rope_min_period_mm=m.get("rope_min_period_mm"),
        rope_fp32=bool(m.get("rope_fp32", False)), fp32_residual=bool(m.get("fp32_residual", False)),
        n_meta=n_meta(m), conv_stem=m.get("conv_stem") or None,
    ).to(device)
    if any(str(k).startswith("fusion.") for k in state):     # study-level input (R14-F): rebuild before loading
        fu = m.get("fusion") or {}
        net.enable_fusion(int(fu.get("after_block", net.depth - 4)), fu.get("heads"))
    attach_bank(net, m)               # model.window_bank: its learned windows load with the weights
    if pretrained:
        net.load_state_dict(state)
    net.eval().requires_grad_(False)
    enc = FrozenEncoder3D(net, patch, native, spacing)
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
            # float16 for a large task (RSNA: 25k x 5120 x 384 would be ~200 GB as float32)
            dtype = np.float16 if n * t * d * 4 > 16 * 2**30 else np.float32
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
