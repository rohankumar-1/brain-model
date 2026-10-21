"""Slot readout of a multitask checkpoint for the frozen suite.

``attach`` gives the frozen encoder of a multitask checkpoint a ``slot_features`` method: the volume is encoded once,
then every trained task is asked on its own (one slot per query, as in training, where a query held at most
``tasks.k_max`` tasks) and the suite gets two global views next to the encoder's own heads:

    slots    the trained tasks' slot vectors, concatenated (n_trained x slot width)
    taskout  the model's prediction for each trained task: binary logit, standardised regression value, multiclass
             class probabilities (segmentation tasks have no scalar output and contribute only through ``slots``)
"""

from __future__ import annotations

import numpy as np
import torch

from brain.multitask.data import gate_and_window
from brain.multitask.model import MultitaskModel
from brain.multitask.registry import load_registry


def is_multitask(ckpt_path) -> bool:
    """Cheap check (memory-mapped load, no tensor reads)."""
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:   # noqa: BLE001  not a zip checkpoint: not ours
        return False
    return "multitask" in ck


def attach(enc, ckpt: dict, device: torch.device) -> None:
    cfg = ckpt["config"]
    reg = load_registry(cfg["data"].get("registry")).subset(list(ckpt["tasks"]))   # the checkpoint's task order
    model = MultitaskModel(enc.model, reg, cfg["model"]).to(device)     # shares the frozen encoder's weights
    missing, unexpected = model.load_state_dict(
        {k: v for k, v in ckpt["multitask"].items() if not k.startswith("backbone.")}, strict=False)
    if [k for k in missing if not k.startswith("backbone.")] or unexpected:
        raise SystemExit(f"multitask state mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    model.eval().requires_grad_(False)
    trained = [reg.index[t.id] for t in reg.trained]
    task = torch.tensor(trained, device=device).unsqueeze(1)          # (T, 1): one task per query
    gate = (cfg.get("views") or {}).get("gate") or {}
    patch = tuple(model.backbone.patch_xyz)
    kinds = [reg[t].type for t in trained]
    n_cls = [len(reg[t].classes) for t in trained]

    @torch.no_grad()
    def slot_features(vol_xyz: torch.Tensor, modality: str) -> tuple[np.ndarray, np.ndarray]:
        raw = vol_xyz.float().permute(2, 1, 0)[None, None]           # (1, 1, Z, Y, X) as in training
        mod = torch.tensor([0 if str(modality).lower().startswith("ct") else 1], device=raw.device)
        x, keep = gate_and_window(raw, mod, patch, gate)
        amp = dict(device_type=raw.device.type, dtype=torch.bfloat16, enabled=raw.is_cuda)
        with torch.autocast(**amp):
            e = model.encode(x, keep)
            t = task.shape[0]
            rep = {"tokens": e["tokens"].expand(t, -1, -1), "coords": e["coords"],
                   "cls": e["cls"].expand(t, -1) if e.get("cls") is not None else None,
                   "pooled": e["pooled"].expand(t, -1)}
            z = model.slots(rep, task, keep.expand(t, -1))                # (T, 1, W)
            out = model.heads(z, task)
        outs = []
        for i, kind in enumerate(kinds):
            if kind == "binary":
                outs.append(out["binary"][i, :1].float())
            elif kind == "regression":
                outs.append(out["regression"][i, :1].float())
            elif kind == "multiclass":
                outs.append(out["multiclass"][i, 0, :n_cls[i]].float().softmax(-1))
        return (z[:, 0].float().reshape(-1).cpu().numpy(),
                torch.cat(outs).cpu().numpy() if outs else np.zeros(0, np.float32))

    enc.slot_features = slot_features
    enc.slot_tasks = [reg[t].id for t in trained]
