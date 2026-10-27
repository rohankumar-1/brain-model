"""Task-conditioned multitask model on a 3D BrainEncoder.

    tokens = encoder(volume)                                  (B, N, D) + CLS
    slot_k = task_emb[task_k] + type_emb[type_k]              one query per requested task
    slots  = L x [cross-attend to tokens -> self-attend across slots -> MLP]
    output = typed head(slot_k)                               shared per output type, conditioned only by the slot

Heads are shared by every task of a type (binary logit, standardised regression value, survival hazards, the
segmentation decoder via FiLM; multiclass scores the slot against learned per-(task, class) vectors), so any number
of tasks of any types can be asked at once, and a new task is a new embedding row (few-shot: fit only that row).

``slot.embedding``: ``learned`` (a task embedding table) or ``text`` (a projection of frozen description
embeddings, ``brain.multitask.text``; an ablation). ``slot.mode``: ``slots`` (above) or ``shared`` (no per-task
attention: one pooled vector, concatenated with the task embedding, goes to the heads). ``slot.cross_task: false``
blocks attention between slots. ``slot.width`` narrows the slot before the heads (bottleneck ablation).

``task_tokens.enabled`` (OMNI-style, arXiv 2610.12574) replaces the slot layers: every trained task has a token in the
encoder's own sequence from the first block (isolated from the other task tokens, read by the patch tokens), and the
heads read its output. ``moe.blocks`` turns the encoder's top blocks' MLPs into mixtures of experts.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.data.coords import crop_bbox_mm, patch_size_mm, token_centers_mm
from brain.models.conv3d import Res, conv3, group_norm
from brain.models.vit3d import BrainEncoder
from brain.multitask.data import TYPE_ID
from brain.multitask.registry import Registry


def mlp(cin: int, hidden: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(cin, hidden), nn.GELU(), nn.Linear(hidden, cout))


class SlotLayer(nn.Module):
    def __init__(self, dim: int, heads: int, cross_task: bool = True):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = mlp(dim, 4 * dim, dim)
        self.cross_task = bool(cross_task)

    def forward(self, q, slot_pad, kv, kv_pad):
        h = self.n1(q)
        q = q + self.cross(h, kv, kv, key_padding_mask=kv_pad, need_weights=False)[0]
        h = self.n2(q)
        mask = None
        if not self.cross_task:                       # each slot sees only itself
            k = q.shape[1]
            mask = ~torch.eye(k, dtype=torch.bool, device=q.device)
        q = q + self.self_attn(h, h, h, key_padding_mask=None if mask is not None else slot_pad, attn_mask=mask,
                               need_weights=False)[0]
        return q + self.mlp(self.n3(q))


class SegDecoder(nn.Module):
    """Token grid (+ earlier-block taps) -> one logit map at ``out_stride`` per segmentation slot.

    The slot-independent parts (token projection, an image pyramid for detail the 16 mm tokens lack) run once per
    volume; the upsampling stages run once per segmentation slot, each stage FiLM-modulated by the slot
    (x * (1 + gamma) + beta, zero-initialised so training starts from the unconditioned decoder).
    """

    def __init__(self, in_dim: int, cond_dim: int, patch_zyx, widths=(128, 64, 32, 16)):
        super().__init__()
        pz, py, px = patch_zyx
        n_z, n_xy = int(math.log2(pz)) - 1, int(math.log2(py)) - 1  # upsamplings to stride 2 per axis
        if n_xy > len(widths) - 1 or n_z > n_xy:
            raise ValueError(f"patch {patch_zyx} needs {n_xy} upsamplings, decoder has {len(widths) - 1}")
        widths = tuple(widths[len(widths) - 1 - n_xy:])                # the finest n_xy + 1 widths
        n_up = n_xy
        self.steps = [(2 if i < n_z else 1, 2, 2) for i in range(n_up)]
        self.stem = nn.Sequential(nn.Conv3d(in_dim, widths[0], 1, bias=False), group_norm(widths[0]), nn.GELU(),
                                  Res(widths[0], widths[0]))
        # image pyramid, finest first: stride 2 then the decoder's steps in reverse
        pyr, cin = [], 1
        for i in range(n_up):
            stride = (2, 2, 2) if i == 0 else self.steps[n_up - i]
            pyr.append(nn.Sequential(conv3(cin, widths[n_up - i], stride), conv3(widths[n_up - i], widths[n_up - i])))
            cin = widths[n_up - i]
        self.pyramid = nn.ModuleList(pyr)
        self.ups = nn.ModuleList(
            nn.Sequential(nn.ConvTranspose3d(widths[i], widths[i + 1], s, stride=s, bias=False), group_norm(widths[i + 1]),
                          nn.GELU()) for i, s in enumerate(self.steps))
        self.fuse = nn.ModuleList(Res(2 * widths[i + 1], widths[i + 1]) for i in range(n_up))
        self.film = nn.ModuleList(nn.Linear(cond_dim, 2 * w) for w in widths)
        for f in self.film:
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)
        self.head = nn.Conv3d(widths[-1], 1, 1)
        self._register_load_state_dict_pre_hook(self._legacy_keys)

    @staticmethod
    def _legacy_keys(state, prefix, *_):
        """Checkpoints from before the shared ``conv3d.Res`` named its two convolutions ``a`` / ``b``."""
        for k in [k for k in state if k.startswith(prefix)]:
            new = re.sub(r"\.a\.(\d)", r".block.0.\1", re.sub(r"\.b\.(\d)", r".block.1.\1", k[len(prefix):]))
            if new != k[len(prefix):]:
                state[prefix + new] = state.pop(k)

    @staticmethod
    def _mod(x, film, cond):
        g, b = film(cond).chunk(2, dim=-1)
        return x * (1 + g.view(*g.shape, 1, 1, 1)) + b.view(*b.shape, 1, 1, 1)

    def forward(self, grid: torch.Tensor, image: torch.Tensor, rows: torch.Tensor, cond: torch.Tensor):
        """``grid`` (B, C, nz, ny, nx) token features, ``image`` (B, 1, Z, Y, X); ``rows`` (P,) the volume of each
        segmentation slot, ``cond`` (P, cond_dim) its slot. Returns (P, 1, Z/2, Y/2, X/2) logits."""
        x = self.stem(grid)
        feats, h = [], image
        for p in self.pyramid:
            h = p(h)
            feats.append(h)
        feats = feats[::-1]                                   # coarse to fine, one per decoder stage
        x = self._mod(x[rows], self.film[0], cond)
        for i, (up, fuse) in enumerate(zip(self.ups, self.fuse)):
            x = fuse(torch.cat([up(x), feats[i][rows]], dim=1))
            x = self._mod(x, self.film[i + 1], cond)
        return self.head(x)


class MultitaskModel(nn.Module):
    def __init__(self, backbone: BrainEncoder, reg: Registry, cfg: dict, text_emb: torch.Tensor | None = None):
        super().__init__()
        s = cfg.get("slot") or {}
        self.backbone = backbone
        self.reg = reg
        d_enc, dim = backbone.embed_dim, int(s.get("dim", 256))
        width = int(s.get("width") or dim)
        self.mode, self.embedding = str(s.get("mode", "slots")), str(s.get("embedding", "learned"))
        self.taps = tuple(int(t) for t in cfg.get("taps", (3, 7)) if int(t) < backbone.depth - 1)
        n_tasks, n_cls = len(reg), reg.max_classes()
        # ``task_tokens.enabled``: OMNI-style (arXiv 2610.12574) task tokens inside the encoder instead of slot layers
        # over its output: one token per trained task, always all in the sequence (``set_token_tasks``), isolated from
        # each other and read by the patch tokens; a task's answer is its token's output. Identity lives in the
        # encoder's width then.
        self.tokens = bool((cfg.get("task_tokens") or {}).get("enabled", False))
        if self.tokens:
            self.mode = "tokens"
            self.register_buffer("tt_write", torch.zeros(n_tasks, dtype=torch.bool))
        qdim = d_enc if self.tokens else dim
        # task identity
        if self.embedding == "text":
            if text_emb is None:
                raise ValueError("slot.embedding=text needs the description embeddings")
            self.register_buffer("text_emb", text_emb.float(), persistent=False)    # (n_tasks, E)
            self.task_proj = mlp(text_emb.shape[1], dim, qdim)
        else:
            self.task_emb = nn.Embedding(n_tasks, qdim)
            nn.init.normal_(self.task_emb.weight, std=0.02)
        self.type_emb = nn.Embedding(len(TYPE_ID), qdim)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        # ``moe``: mixture-of-experts MLPs in the encoder's last ``blocks`` blocks, experts copied from the trained MLP
        moe = cfg.get("moe") or {}
        if int(moe.get("blocks", 0)) and not backbone.moe_blocks:
            backbone.enable_moe(int(moe["blocks"]), int(moe.get("experts", 5)), int(moe.get("top_k", 2)))
        self.register_buffer("task_type", torch.tensor([TYPE_ID[t.type] for t in reg.tasks]), persistent=False)
        self.register_buffer("n_classes", torch.tensor([len(t.classes) for t in reg.tasks]), persistent=False)
        # tokens -> keys / values (with an absolute position code: RoPE leaves tokens position-relative)
        self.kv_norm = nn.LayerNorm(d_enc)
        self.kv = nn.Linear(d_enc, dim)
        self.pos = mlp(3, dim, dim)
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, dim))
        n_layers = int(s.get("layers", 3))
        if self.mode == "slots":
            self.layers = nn.ModuleList(SlotLayer(dim, int(s.get("heads", 8)), bool(s.get("cross_task", True)))
                                        for _ in range(n_layers))
        elif self.mode == "shared":
            self.shared = mlp(d_enc + dim, 2 * dim, dim)
        elif self.mode == "tokens":
            self.tt_proj = nn.Linear(d_enc, dim)                # a task token's output -> the heads' input
        else:
            raise ValueError(f"slot.mode {self.mode}")
        self.out_norm = nn.LayerNorm(dim)
        self.bottleneck = nn.Linear(dim, width) if width != dim else nn.Identity()
        # typed heads, shared across tasks
        self.binary = mlp(width, width, 1)
        self.regress = mlp(width, width, 1)
        self.cls_query = nn.Linear(width, dim)
        self.cls_emb = nn.Parameter(torch.randn(n_tasks, n_cls, dim) * 0.02)          # per (task, class)
        n_bins = max([len(t.survival_bins) for t in reg.tasks] + [1])
        self.survival = mlp(width, width, n_bins)
        # general embedding: one task-agnostic query over the same tokens, trained with LeJEPA (loss.general); its
        # slot is computed on its own so it never depends on which tasks a row asks
        gen = cfg.get("general") or {}
        self.has_general = bool(gen.get("enabled", False))
        if self.has_general and self.tokens:
            raise ValueError("model.general needs the slot layers (slot mode), not task_tokens")
        if self.has_general:
            self.general_q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.general_proj = mlp(width, width, int(gen.get("proj_dim", 128)))
        seg_cfg = cfg.get("seg") or {}
        patch_zyx = tuple(int(p) for p in reversed(backbone.patch_xyz))
        self.seg = SegDecoder(d_enc * (len(self.taps) + 1), width, patch_zyx,
                              tuple(seg_cfg.get("widths", (128, 64, 32, 16))))

    # -- encoder ---------------------------------------------------------------------------------------
    def _coords(self, shape_zyx, device):
        size = (shape_zyx[2], shape_zyx[1], shape_zyx[0])
        bbox = crop_bbox_mm((0, 0, 0), size, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)).to(device)
        return token_centers_mm(bbox, size, self.backbone.patch_xyz), patch_size_mm(bbox, size, self.backbone.patch_xyz)

    def set_token_tasks(self, ids) -> None:
        """The tasks whose tokens are always in the sequence and read by the patch tokens (the trained tasks)."""
        self.tt_write.zero_()
        self.tt_write[torch.as_tensor(list(ids), dtype=torch.long, device=self.tt_write.device)] = True

    def encode(self, x: torch.Tensor, keep: torch.Tensor, meta: torch.Tensor | None = None,
               task: torch.Tensor | None = None) -> dict:
        """Encoder outputs. With task tokens: every trained task's token, plus read-only tokens for the other tasks
        in ``task`` (held-out / few-shot), so an answer never depends on which other tasks were asked."""
        b = x.shape[0]
        coords, pmm = self._coords(x.shape[2:], x.device)
        tt = write = ids = None
        if self.tokens:
            asked = torch.zeros_like(self.tt_write)
            if task is not None and (task >= 0).any():
                asked[task[task >= 0].unique()] = True
            ids = torch.cat([self.tt_write.nonzero()[:, 0], (asked & ~self.tt_write).nonzero()[:, 0]])
            if ids.numel() == 0:
                raise ValueError("task tokens: no trained tasks set (set_token_tasks) and none asked")
            tt, write = self.task_queries(ids), self.tt_write[ids]
        out = self.backbone(x.permute(0, 1, 4, 3, 2), coords.unsqueeze(0).expand(b, -1, -1),
                            pmm.unsqueeze(0).expand(b, -1), keep=keep, taps=self.taps, meta=meta,
                            task_tokens=tt, task_write=write)
        out["coords"] = coords
        if self.tokens:
            out["tt_ids"] = ids
        return out

    def token_slots(self, enc: dict, task: torch.Tensor) -> torch.Tensor:
        """(B, K, width) from the task tokens' outputs; padded slots (task < 0) read the first token."""
        lut = torch.full((len(self.reg),), -1, dtype=torch.long, device=task.device)
        ids = enc["tt_ids"]
        lut[ids] = torch.arange(ids.numel(), device=task.device)
        pos = lut[task.clamp_min(0)]
        if bool(((pos < 0) & (task >= 0)).any()):
            raise ValueError("task tokens: a requested task has no token in this encoding (encode with task=...)")
        tok = enc["task_tokens"]                                            # (B, T, D)
        z = torch.gather(tok, 1, pos.clamp_min(0).unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
        return self.bottleneck(self.out_norm(self.tt_proj(z)))

    # -- slots -----------------------------------------------------------------------------------------
    def task_queries(self, task: torch.Tensor) -> torch.Tensor:
        t = task.clamp_min(0)
        base = self.task_proj(self.text_emb[t]) if self.embedding == "text" else self.task_emb(t)
        return base + self.type_emb(self.task_type[t])

    def _kv(self, enc: dict, keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if "_kv" not in enc:
            c = enc["coords"]
            cn = (c - c.mean(0)) / c.std(0).clamp_min(1e-6)
            kv = self.kv(self.kv_norm(enc["tokens"])) + self.pos(cn).unsqueeze(0)
            pad = ~keep
            if enc.get("cls") is not None:
                kv = torch.cat([self.kv(self.kv_norm(enc["cls"])).unsqueeze(1) + self.cls_pos, kv], 1)
                pad = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
            enc["_kv"] = (kv, pad)
        return enc["_kv"]

    def general(self, enc: dict, keep: torch.Tensor) -> torch.Tensor:
        """(B, width) task-agnostic embedding: the general query alone through the slot layers."""
        if self.mode == "shared":
            raise ValueError("the general embedding needs slot.mode=slots")
        kv, pad = self._kv(enc, keep)
        q = self.general_q.expand(kv.shape[0], -1, -1)
        none = torch.zeros(kv.shape[0], 1, dtype=torch.bool, device=kv.device)
        for layer in self.layers:
            q = layer(q, none, kv, pad)
        return self.bottleneck(self.out_norm(q))[:, 0]

    def slots(self, enc: dict, task: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        if self.tokens:
            return self.token_slots(enc, task)
        q = self.task_queries(task)                                         # (B, K, dim)
        if self.mode == "shared":
            pooled = enc["pooled"].unsqueeze(1).expand(-1, q.shape[1], -1)
            return self.bottleneck(self.out_norm(self.shared(torch.cat([pooled, q], -1))))
        kv, pad = self._kv(enc, keep)
        slot_pad = task < 0
        slot_pad = slot_pad & ~(slot_pad.all(1, keepdim=True) & (torch.arange(task.shape[1], device=task.device) == 0))
        for layer in self.layers:
            q = layer(q, slot_pad, kv, pad)
        return self.bottleneck(self.out_norm(q))

    def slots_alone(self, enc: dict, task: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """``slots`` with every task asked on its own (no slot attends to another), as at deployment; the scan is
        encoded once and its keys/values shared by the per-task queries."""
        if self.mode in ("shared", "tokens"):                 # already independent of the other asked tasks
            return self.slots(enc, task, keep)
        b, k = task.shape
        kv, pad = self._kv(enc, keep)
        one = {"_kv": (kv.repeat_interleave(k, 0), pad.repeat_interleave(k, 0))}
        return self.slots(one, task.reshape(b * k, 1), keep).reshape(b, k, -1)

    def heads(self, z: torch.Tensor, task: torch.Tensor) -> dict:
        t = task.clamp_min(0)
        logits = torch.einsum("bkd,bkcd->bkc", self.cls_query(z), self.cls_emb[t]) / math.sqrt(self.cls_emb.shape[-1])
        valid = torch.arange(logits.shape[-1], device=z.device) < self.n_classes[t].unsqueeze(-1)
        return {"binary": self.binary(z)[..., 0], "regression": self.regress(z)[..., 0],
                "multiclass": logits.masked_fill(~valid, float("-inf")), "survival": self.survival(z)}

    def forward(self, x: torch.Tensor, keep: torch.Tensor, task: torch.Tensor, seg_pair: torch.Tensor,
                encoder_grad: bool = True, alone: bool = False, meta: torch.Tensor | None = None) -> dict:
        with torch.set_grad_enabled(encoder_grad and torch.is_grad_enabled()):
            enc = self.encode(x, keep, meta, task)
        z = self.slots_alone(enc, task, keep) if alone else self.slots(enc, task, keep)
        out = self.heads(z, task)
        if self.has_general:
            out["general"] = self.general_proj(self.general(enc, keep))
        out["seg"] = None
        if len(seg_pair):
            b = x.shape[0]
            pz, py, px = reversed(self.backbone.patch_xyz)
            g = (x.shape[2] // pz, x.shape[3] // py, x.shape[4] // px)
            taps = [F.layer_norm(t, t.shape[-1:]) for t in enc["taps"]]       # pre-norm residual stream -> unit scale
            feats = torch.cat([*taps, enc["tokens"]], -1)                    # (B, N, D * (taps + 1))
            grid = feats.transpose(1, 2).reshape(b, feats.shape[-1], *g)
            out["seg"] = self.seg(grid, x, seg_pair[:, 0], z[seg_pair[:, 0], seg_pair[:, 1]])
        return out
