"""ViT wrapper with a mask token for iBOT. Downstream still loads the bare backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.vit import VisionTransformer

from braindino.heads import DINOHead, koleo_loss
from braindino.loss import dino_loss, ibot_loss


class BrainDINO(nn.Module):
    def __init__(self, backbone: VisionTransformer, proto_dim: int = 65536,
                 hidden: int = 2048, bottleneck: int = 256):
        super().__init__()
        self.backbone = backbone
        self.mask_token = nn.Parameter(torch.zeros(1, 1, backbone.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.dino_head = DINOHead(backbone.embed_dim, proto_dim, hidden, bottleneck)
        self.ibot_head = DINOHead(backbone.embed_dim, proto_dim, hidden, bottleneck)

    def tokens(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """CLS (B, C) and patch tokens (B, T, C). ``mask`` True = replace with the mask token."""
        vit = self.backbone
        patches = vit.patch_embed(x).flatten(2).transpose(1, 2)
        side = int(round(patches.shape[1] ** 0.5))
        if side * side != patches.shape[1]:
            raise ValueError(f"patch grid is not square: {patches.shape[1]}")
        if mask is not None:
            m = mask.unsqueeze(-1).to(dtype=patches.dtype)
            # Keep mask_token in the graph even when a row is unmasked (DDP).
            patches = patches * (1.0 - m) + self.mask_token.to(dtype=patches.dtype) * m
        pos = vit._pos_embed(side)
        b = patches.size(0)
        prefix = [vit.cls_token.expand(b, -1, -1) + pos[:, :1]]
        if vit.n_registers:
            prefix.append(vit.register_tokens.expand(b, -1, -1))
        z = torch.cat([*prefix, patches + pos[:, 1:]], dim=1)
        n_pre = 1 + vit.n_registers
        for block in vit.blocks:
            z = block(z)
        z = vit.norm(z)
        return z[:, 0], z[:, n_pre:]

    def forward(self, globals, locals, masks, teacher_dino, teacher_ibot, center, center_patch,
                student_temp: float, teacher_temp: float):
        """One DDP step: all crops in a single forward so the reducer sees one iteration.

        ``globals`` (G, B, C, H, W), ``locals`` (L, B, C, h, w) or None,
        ``masks`` (G, B, T). Teacher tensors are detached. Returns DINO, iBOT, KoLeo, CLS.
        """
        dino_s, cls_g, ibot_terms = [], [], []
        for v in range(globals.size(0)):
            cls, patches = self.tokens(globals[v], masks[v])
            dino_s.append(self.dino_head(cls))
            cls_g.append(cls)
            m = masks[v]
            if m.any():
                ibot_terms.append(ibot_loss(
                    self.ibot_head(patches[m]), teacher_ibot[v][m], None,
                    center_patch, student_temp, teacher_temp,
                ))
            else:
                ibot_terms.append(self.ibot_head(patches[:, 0]).sum() * 0)
        if locals is not None:
            for v in range(locals.size(0)):
                cls, _ = self.tokens(locals[v], None)
                dino_s.append(self.dino_head(cls))
        t_dino = [teacher_dino[i] for i in range(teacher_dino.size(0))]
        l_dino = dino_loss(dino_s, t_dino, center, student_temp, teacher_temp)
        l_ibot = torch.stack(ibot_terms).mean()
        l_koleo = koleo_loss(torch.cat(cls_g, 0))
        return l_dino, l_ibot, l_koleo, cls_g[0]

    def backbone_state(self) -> dict[str, torch.Tensor]:
        """Keys a frozen ``VisionTransformer`` can load."""
        return {k: v.detach().cpu() for k, v in self.backbone.state_dict().items()}
