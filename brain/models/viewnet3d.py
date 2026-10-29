"""One forward over every crop of a step, so DDP and SyncBatchNorm each see a single call."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.models.refine import choose_cells, detail_scores
from brain.objectives.dense import parent_mean
from brain.models.vit3d import BrainEncoder


def batch_meta(batch: dict) -> dict:
    """What the loss needs from a prepared batch besides the network output. Reads no GPU values."""
    groups = [g for g in batch["groups"] for _ in range(batch[f"{g}_view"].shape[0])]
    pos = [p for g in batch["groups"] for p in batch[f"{g}_pos"].long().unbind(0)]  # per view, (B, P)
    meta = {"n_global": batch["global_view"].shape[0], "n_pos": batch["n_pos"], "view_groups": groups, "pos": pos}
    if "fine_tok_pos" in batch:
        meta["fine_pos"] = list(batch["fine_tok_pos"].long().unbind(0))
        meta["n_pos_fine"] = int(batch["n_pos_fine"])
    return meta


def cell_means(view: torch.Tensor, patch_xyz, cells_xyz) -> torch.Tensor:
    """Mean voxel value of each cell of each patch: (R, 1, Z, Y, X) -> (R, tokens, cells), token order (Z, Y, X)."""
    (px, py, pz), (cx, cy, cz) = patch_xyz, cells_xyz
    x = F.avg_pool3d(view.float(), (pz // cz, py // cy, px // cx))
    r, _, gz, gy, gx = x.shape
    x = x.view(r, gz // cz, cz, gy // cy, cy, gx // cx, cx).permute(0, 1, 3, 5, 2, 4, 6)
    return x.reshape(r, (gz // cz) * (gy // cy) * (gx // cx), cz * cy * cx)


class ViewNet(nn.Module):
    """Encoder per crop group, then heads.

    The pooled projector runs once over the global (and optionally local) crops, like the 2D model, so its
    BatchNorm statistics are shared across crop types. The window pair gets its own pass. With a token
    head, every view also reports an embedding per coarse grid position it covers (fine crops average into
    the parent cell). Fine-patch tokens are also kept native for a second consensus / predict. With a
    predictor, hidden blocks of a crop are filled from the visible tokens: the target is that crop's own
    unmasked token-head embeddings (no gradient) and the cell means of every kept patch's voxels. With ``refine``
    (``{groups, frac_max, random_frac}``), a share of each crop's cells in those groups is encoded as 8 half-size
    tokens (``brain.models.refine``): drawn per row from U(0, frac_max) in training, frac_max in eval mode.
    """

    def __init__(self, enc: BrainEncoder, *, local_pooled: bool = False, dense: bool = True,
                 predict: bool = False, predict_fine: bool = False, cells_xyz: tuple[int, int, int] = (2, 2, 2),
                 refine: dict | None = None):
        super().__init__()
        self.enc = enc
        self.pooled_groups = ("global", "local") if local_pooled else ("global",)
        self.dense = bool(dense) and enc.token_head is not None
        self.predict = bool(predict) and self.dense and enc.predictor is not None
        self.predict_fine = bool(predict_fine) and self.predict
        self.cells_xyz = tuple(int(c) for c in cells_xyz)
        r = refine or {}
        self.refine_groups = tuple(r.get("groups", ("global",))) if float(r.get("frac_max", 0.0)) > 0 else ()
        self.refine_max, self.refine_random = float(r.get("frac_max", 0.0)), float(r.get("random_frac", 0.5))

    def _refine(self, view: torch.Tensor, keep: torch.Tensor, patch) -> torch.Tensor:
        frac = (torch.rand(keep.shape[0], device=keep.device) * self.refine_max if self.training
                else torch.full((keep.shape[0],), self.refine_max, device=keep.device))
        return choose_cells(detail_scores(view, patch), keep, frac, self.refine_random)

    def forward(self, v: dict) -> dict[str, torch.Tensor | dict]:
        b = v["modality"].shape[0]
        pooled: dict[str, torch.Tensor] = {}
        entries: dict[str, list[torch.Tensor]] = {"emb": [], "valid": []}
        fine_native: dict[str, list[torch.Tensor]] = {"emb": [], "valid": []}
        predicted = predicted_fine = None
        for g in v["groups"]:
            n = v[f"{g}_view"].shape[0]
            rows = lambda t: t.contiguous().reshape(n * b, *t.shape[2:])  # noqa: E731  (V, B, ...) -> (V*B, ...)
            keep = rows(v[f"{g}_fg"])
            meta = v["meta"].repeat(n) if "meta" in v else None              # (V*B,): row = view * B + volume
            ref = self._refine(rows(v[f"{g}_view"]), keep, v["patch"][g]) if g in self.refine_groups else None
            pooled[g], tokens = self.enc.encode_rows(
                rows(v[f"{g}_view"]), rows(v[f"{g}_coords"]), rows(v[f"{g}_patch_mm"]), keep,
                v["plans"][g], v["patch"][g], tokens=self.dense, meta=meta, refine=ref,
            )
            if self.dense:
                head = self.enc.token_head(tokens)
                emb, valid = parent_mean(head, keep, rows(v[f"{g}_parent"]).long(), v["n_par"][g])
                if self.predict and g == "global":
                    predicted = self._predict(
                        v, rows, head, keep, view_key="global_view", coords_key="global_coords",
                        patch_mm_key="global_patch_mm", ctx_key="ctx_keep", tgt_key="tgt_mask",
                        ctx_plan="ctx", full_plan="global", patch=v["patch"]["global"], meta=meta, refine=ref)
                if g == "fine":
                    fine_native["emb"] += list(head.view(n, b, *head.shape[1:]).unbind(0))
                    fine_native["valid"] += list(keep.view(n, b, -1).unbind(0))
                    if self.predict_fine and "fine_ctx_keep" in v:
                        predicted_fine = self._predict(
                            v, rows, head, keep, view_key="fine_view", coords_key="fine_coords",
                            patch_mm_key="fine_patch_mm", ctx_key="fine_ctx_keep", tgt_key="fine_tgt_mask",
                            ctx_plan="fine_ctx", full_plan="fine", patch=v["patch"]["fine"], meta=meta)
                entries["emb"] += list(emb.view(n, b, *emb.shape[1:]).unbind(0))
                entries["valid"] += list(valid.view(n, b, -1).unbind(0))
        proj = self.enc.projector(torch.cat([pooled[g] for g in self.pooled_groups])).view(-1, b, self.enc.proj_dim)
        win = self.enc.projector(pooled["win"]).view(2, b, self.enc.proj_dim)
        return {
            "proj": proj, "win": win, "dense": entries if self.dense else None,
            "dense_fine": fine_native if fine_native["emb"] else None,
            "predict": predicted, "predict_fine": predicted_fine,
        }

    def _predict(self, v: dict, rows, head: torch.Tensor, keep: torch.Tensor, *,
                 view_key: str, coords_key: str, patch_mm_key: str, ctx_key: str, tgt_key: str,
                 ctx_plan: str, full_plan: str, patch, meta: torch.Tensor | None = None,
                 refine: torch.Tensor | None = None) -> dict:
        view, coords = rows(v[view_key]), rows(v[coords_key])
        ctx = rows(v[ctx_key])
        _, ctx_tokens = self.enc.encode_rows(
            view, coords, rows(v[patch_mm_key]), ctx, v["plans"][ctx_plan], patch, tokens=True, meta=meta,
            refine=None if refine is None else refine & ctx,
        )
        latent, pixel = self.enc.predictor(ctx_tokens, ctx, keep, coords, v["plans"][full_plan])
        return {
            "latent": latent, "target": head.detach(), "tgt": rows(v[tgt_key]), "fg": keep, "pixel": pixel,
            "pixel_target": cell_means(view[:, :1], patch, self.cells_xyz) if pixel is not None else None,
        }
