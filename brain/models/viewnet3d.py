"""One forward over every crop of a step, so DDP and SyncBatchNorm each see a single call."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.common.windowing import canonical_window
from brain.objectives.dense import parent_mean
from brain.models.vit3d import BrainEncoder
from brain.models.window_bank import bank_of


def batch_meta(batch: dict) -> dict:
    """What the loss needs from a prepared batch besides the network output. Reads no GPU values."""
    groups = [g for g in batch["groups"] for _ in range(batch[f"{g}_view"].shape[0])]
    pos = [p for g in batch["groups"] for p in batch[f"{g}_pos"].long().unbind(0)]  # per view, (B, P)
    return {"n_global": batch["global_view"].shape[0], "n_pos": batch["n_pos"], "view_groups": groups, "pos": pos}


def cell_means(view: torch.Tensor, patch_xyz, cells_xyz) -> torch.Tensor:
    """Mean voxel value of each cell of each patch: (R, 1, Z, Y, X) -> (R, tokens, cells), token order (Z, Y, X)."""
    (px, py, pz), (cx, cy, cz) = patch_xyz, cells_xyz
    x = F.avg_pool3d(view.float(), (pz // cz, py // cy, px // cx))
    r, _, gz, gy, gx = x.shape
    x = x.view(r, gz // cz, cz, gy // cy, cy, gx // cx, cx).permute(0, 1, 3, 5, 2, 4, 6)
    return x.reshape(r, (gz // cz) * (gy // cy) * (gx // cx), cz * cy * cx)


class ViewNet(nn.Module):
    """Encoder per crop group, then heads.

    The pooled projector runs over the global crops; the window pair gets its own pass. With a token head, every
    view also reports an embedding per coarse grid position it covers (fine crops average into the parent cell).
    With a predictor, hidden blocks of each global crop are filled from the visible tokens: the target is that crop's
    own unmasked token-head embeddings (no gradient) and the cell means of every kept patch's voxels.

    With a learned window bank on the encoder (``encoder.window_bank``) the views arrive in native units, one channel
    (``batch3d.prepare``), and each goes through the bank with its volume's modality before the patch embedding, as in
    BrainQuery; the pixel target is then the canonical window of the native crop (as the multitask reconstruction
    target), and there is no window pair (``win`` is None).
    """

    def __init__(self, enc: BrainEncoder, *, dense: bool = True, predict: bool = False,
                 cells_xyz: tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        self.enc = enc
        self.dense = bool(dense) and enc.token_head is not None
        self.predict = bool(predict) and self.dense and enc.predictor is not None
        self.cells_xyz = tuple(int(c) for c in cells_xyz)

    def _input(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        """A view's encoder input: through the encoder's window bank when it has one (native-unit views)."""
        bank = bank_of(self.enc)
        return x if bank is None else bank(x, mod)

    def forward(self, v: dict) -> dict[str, torch.Tensor | dict]:
        b = v["modality"].shape[0]
        pooled: dict[str, torch.Tensor] = {}
        entries: dict[str, list[torch.Tensor]] = {"emb": [], "valid": []}
        predicted = None
        for g in v["groups"]:
            n = v[f"{g}_view"].shape[0]
            rows = lambda t: t.contiguous().reshape(n * b, *t.shape[2:])  # noqa: E731  (V, B, ...) -> (V*B, ...)
            keep = rows(v[f"{g}_fg"])
            meta = v["meta"].repeat(n) if "meta" in v else None              # (V*B,): row = view * B + volume
            pooled[g], tokens = self.enc.encode_rows(
                self._input(rows(v[f"{g}_view"]), v["modality"].repeat(n)), rows(v[f"{g}_coords"]),
                rows(v[f"{g}_patch_mm"]), keep,
                v["plans"][g], v["patch"][g], tokens=self.dense, meta=meta,
            )
            if self.dense:
                head = self.enc.token_head(tokens)
                emb, valid = parent_mean(head, keep, rows(v[f"{g}_parent"]).long(), v["n_par"][g])
                if self.predict and g == "global":
                    predicted = self._predict(v, rows, head, keep, meta)
                entries["emb"] += list(emb.view(n, b, *emb.shape[1:]).unbind(0))
                entries["valid"] += list(valid.view(n, b, -1).unbind(0))
        proj = self.enc.projector(pooled["global"]).view(-1, b, self.enc.proj_dim)
        win = self.enc.projector(pooled["win"]).view(2, b, self.enc.proj_dim) if "win" in pooled else None
        return {"proj": proj, "win": win, "dense": entries if self.dense else None, "predict": predicted}

    def _predict(self, v: dict, rows, head: torch.Tensor, keep: torch.Tensor, meta: torch.Tensor | None) -> dict:
        """The global crops' hidden blocks (``ctx_keep`` visible, ``tgt_mask`` predicted)."""
        view, coords, patch = rows(v["global_view"]), rows(v["global_coords"]), v["patch"]["global"]
        mod = v["modality"].repeat(v["global_view"].shape[0])
        ctx = rows(v["ctx_keep"])
        _, ctx_tokens = self.enc.encode_rows(
            self._input(view, mod), coords, rows(v["global_patch_mm"]), ctx, v["plans"]["ctx"], patch, tokens=True,
            meta=meta)
        latent, pixel = self.enc.predictor(ctx_tokens, ctx, keep, coords, v["plans"]["global"])
        return {
            "latent": latent, "target": head.detach(), "tgt": rows(v["tgt_mask"]), "fg": keep, "pixel": pixel,
            "pixel_target": cell_means(view[:, :1] if bank_of(self.enc) is None else canonical_window(view[:, :1], mod),
                                       patch, self.cells_xyz) if pixel is not None else None,
        }
