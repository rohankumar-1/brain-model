"""3D LeJEPA loss.

Pooled terms (the 2D recipe): global-crop embeddings are pulled to their mean, plus one window pair,
with SigReg on the gathered batch. Dense terms: crops that cover the same coarse token agree there, and
SigReg is applied to sampled token embeddings (fine-patch crops agree through their parent cells).
``dense_weight=0`` gives the pooled recipe back. Predictive terms: blocks of each global crop are hidden; the
predictor regresses hidden token-head embeddings (``pred``, stop-grad target from the unmasked crop) and voxel
cell means (``pix``).
"""

from __future__ import annotations

import torch

from brain.objectives.dense import consensus, sample_slots
from brain.objectives.sigreg import SIGReg, gather_batch, prediction_loss


def lejepa_loss(
    out: dict,
    meta: dict,
    sigreg: SIGReg,
    *,
    lamb: float,
    win_weight: float,
    dense_weight: float = 0.0,
    tokens_per_view: int = 4,
    predict_weight: float = 0.0,
    pixel_weight: float = 0.0,
    gather_ddp: bool = True,
) -> dict[str, torch.Tensor]:
    """``out`` is ``ViewNet`` output (globals first in ``proj``); ``meta`` is ``batch_meta`` of the batch."""
    proj = gather_batch(out["proj"].float(), gather_ddp)
    inv, reg = prediction_loss(proj, n_global=meta["n_global"]), sigreg(proj)
    # no window pair with a learned window bank (``batch3d.prepare`` leaves it out; its weight must be 0 there)
    win = prediction_loss(out["win"].float()) if out["win"] is not None else proj.new_zeros(())
    total = (1.0 - lamb) * inv + lamb * reg + win_weight * win
    parts = {"inv": inv, "win": win, "sigreg": reg}
    dense = out.get("dense")
    if dense is not None and dense_weight > 0:
        d_inv, overlap = consensus(dense["emb"], dense["valid"], meta["pos"], meta["n_pos"])
        keep = [i for i, name in enumerate(meta["view_groups"]) if name != "win"]
        slots = sample_slots([dense["emb"][i] for i in keep], [dense["valid"][i] for i in keep], tokens_per_view)
        d_reg = sigreg(gather_batch(slots, gather_ddp))
        total = total + dense_weight * ((1.0 - lamb) * d_inv + lamb * d_reg)
        parts.update(dense_inv=d_inv, dense_sig=d_reg, overlap=overlap)
    pr = out.get("predict")
    if pr is not None:  # always in the graph, so a zero weight (the warmup of the ramp) keeps every parameter used
        m = pr["tgt"].float()
        p_lat = ((pr["latent"].float() - pr["target"].float()).square().mean(-1) * m).sum() / m.sum().clamp_min(1.0)
        total = total + predict_weight * p_lat
        parts["pred"] = p_lat
        if pr["pixel"] is not None:
            f = pr["fg"].float()
            p_pix = ((pr["pixel"].float() - pr["pixel_target"]).square().mean(-1) * f).sum() / f.sum().clamp_min(1.0)
            total = total + predict_weight * pixel_weight * p_pix
            parts["pix"] = p_pix
    parts["loss"] = total
    return parts
