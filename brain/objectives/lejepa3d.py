"""3D LeJEPA loss.

Pooled terms (the 2D recipe): global-crop embeddings are pulled to their mean, plus one window pair,
with SigReg on the gathered batch. Dense terms: crops that cover the same coarse token agree there, and
SigReg is applied to sampled token embeddings. Fine-patch crops also agree on the native fine grid
(``fine_inv``) so a lesion in one of the eight child tokens is not averaged away. ``dense_weight=0``
gives the pooled recipe back. Predictive terms: blocks of each global crop (and, with ``fine_predict_weight``,
each fine crop) are hidden; the predictor regresses hidden token-head embeddings (``pred`` / ``fine_pred``,
stop-grad target from the unmasked crop) and voxel cell means (``pix`` / ``fine_pix``).
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
    stopgrad_mean: bool = False,
    fine_weight: float = 0.0,
    fine_predict_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """``out`` is ``ViewNet`` output (globals first in ``proj``); ``meta`` is ``batch_meta`` of the batch."""
    proj = gather_batch(out["proj"].float(), gather_ddp)
    inv, reg = prediction_loss(proj, n_global=meta["n_global"]), sigreg(proj)
    win = prediction_loss(out["win"].float())
    total = (1.0 - lamb) * inv + lamb * reg + win_weight * win
    parts = {"inv": inv, "win": win, "sigreg": reg}
    dense = out.get("dense")
    if dense is not None and dense_weight > 0:
        d_inv, overlap = consensus(
            dense["emb"], dense["valid"], meta["pos"], meta["n_pos"], stopgrad_mean=stopgrad_mean)
        keep = [i for i, name in enumerate(meta["view_groups"]) if name != "win"]
        slots = sample_slots([dense["emb"][i] for i in keep], [dense["valid"][i] for i in keep], tokens_per_view)
        d_reg = sigreg(gather_batch(slots, gather_ddp))
        total = total + dense_weight * ((1.0 - lamb) * d_inv + lamb * d_reg)
        parts.update(dense_inv=d_inv, dense_sig=d_reg, overlap=overlap)
        fine = out.get("dense_fine")
        if fine is not None and fine_weight > 0 and "fine_pos" in meta:
            f_inv, f_overlap = consensus(
                fine["emb"], fine["valid"], meta["fine_pos"], meta["n_pos_fine"], stopgrad_mean=stopgrad_mean)
            total = total + dense_weight * fine_weight * (1.0 - lamb) * f_inv
            parts.update(fine_inv=f_inv, fine_overlap=f_overlap)
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
    pf = out.get("predict_fine")
    if pf is not None:
        m = pf["tgt"].float()
        f_lat = ((pf["latent"].float() - pf["target"].float()).square().mean(-1) * m).sum() / m.sum().clamp_min(1.0)
        total = total + predict_weight * fine_predict_weight * f_lat
        parts["fine_pred"] = f_lat
        if pf["pixel"] is not None:
            f = pf["fg"].float()
            f_pix = ((pf["pixel"].float() - pf["pixel_target"]).square().mean(-1) * f).sum() / f.sum().clamp_min(1.0)
            total = total + predict_weight * fine_predict_weight * pixel_weight * f_pix
            parts["fine_pix"] = f_pix
    parts["loss"] = total
    return parts
