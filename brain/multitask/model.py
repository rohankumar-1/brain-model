"""BrainQuery (BQ): the project's model, a 3D BrainEncoder that answers task queries.

Named checkpoints (``configs/bench/models.yaml``): BQ-v3 / BQ-v4 are the self-supervised encoders alone; BQ-R7 ..
BQ-R14 (and R14-F) the task-token runs. The pieces:

    token_t = task_emb[t] + type_emb[type_t]                  one token per trained task, in the encoder's sequence
    tokens  = encoder([CLS | task tokens | patches])          task tokens see the patches and themselves, never each
                                                              other; patches read them (OMNI, arXiv 2610.12574)
    output  = typed head(proj(token_t))                       shared per output type, conditioned only by the token

Heads are shared by every task of a type (binary logit, standardised regression value, the segmentation decoder via
FiLM, whose ``dense_head`` answers the dense (self-supervised intensity) tasks; multiclass
scores the token against learned per-(task, class) vectors). Every trained
task's token is always in the sequence, so an answer never depends on which other tasks were asked; a held-out task
gets a read-only token (it reads the scan, nothing reads it), so few-shot fits only its embedding row.

``task_tokens.enabled: false`` is the slot readout (the from-scratch specialists in ``configs/single``, whose ResNet
encoders have no sequence for task tokens): slot queries cross-attend to the finished encoding through ``slot.layers``
slot layers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from brain.data.coords import crop_bbox_mm, patch_size_mm, token_centers_mm
from brain.models.conv3d import Res, conv3, group_norm
from brain.models.vit3d import BrainEncoder
from brain.models.window_bank import bank_input, bank_of
from brain.multitask.data import TYPE_ID
from brain.multitask.registry import Registry


def mlp(cin: int, hidden: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(cin, hidden), nn.GELU(), nn.Linear(hidden, cout))


class SlotLayer(nn.Module):
    """A slot query reads the encoding (cross-attention), then the other slots of its scan (self-attention)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = mlp(dim, 4 * dim, dim)

    def forward(self, q, slot_pad, kv, kv_pad):
        h = self.n1(q)
        q = q + self.cross(h, kv, kv, key_padding_mask=kv_pad, need_weights=False)[0]
        h = self.n2(q)
        q = q + self.self_attn(h, h, h, key_padding_mask=slot_pad, need_weights=False)[0]
        return q + self.mlp(self.n3(q))


def _mlp_heads(f, hw, hb, w, b):
    """Per-voxel task MLPs: ``f`` (C, Z, Y, X) shared features -> (K, Z, Y, X) logits through (K, C, H) / (K, H) weights."""
    g = F.gelu(torch.einsum("czyx,kch->khzyx", f, hw) + hb.view(*hb.shape, 1, 1, 1))
    return torch.einsum("khzyx,kh->kzyx", g, w) + b.view(-1, 1, 1, 1)


class SegDecoder(nn.Module):
    """Token grid (+ earlier-block taps) -> one logit map at stride 2 per segmentation slot.

    The slot-independent parts (token projection, an image pyramid for detail the 16 mm tokens lack) run once per
    volume. ``conditioned`` (the default): the upsampling stages run once per segmentation slot, each stage
    FiLM-modulated by the slot (x * (1 + gamma) + beta, zero-initialised so training starts from the unconditioned
    decoder); ``dense`` adds a 1x1 output for dense (intensity) slots, built only for a registry with dense tasks.

    ``conditioned=False`` (2026-11-04, ``model.seg.shared``): decoded once per volume, no FiLM, and a 1x1 output per
    registry task (``chan``: the slot's task index) read from the shared top features. The per-slot decoder runs every
    upsampling stage once per segmentation slot (a third of a ViT-B training step at 3 slots per scan); this one's cost
    does not grow with the slots, so a scan can be asked all its masks every step (``tasks.all_seg``). Gives up
    query-conditioned decoding of tasks the registry lacks. ``head_hidden`` > 0 (v2, 2026-11-05): each task reads the
    shared features through its own small per-voxel MLP (C -> head_hidden -> 1, GELU) instead of a 1x1 linear map;
    with 16 shared channels the linear readout starved the ~25 spatial tasks (decoder pilot: anatomy Dice 0.42 vs 0.63
    per slot, tumour collapsed), so v2 also widens the shared stages (``model.seg.widths``, e.g. 128 / 96 / 64 / 64).
    """

    def __init__(self, in_dim: int, cond_dim: int, patch_zyx, widths=(128, 64, 32, 16), in_chans: int = 1,
                 dense: bool = False, conditioned: bool = True, n_tasks: int = 1, head_hidden: int = 0):
        super().__init__()
        pz, py, px = patch_zyx
        n_z, n_xy = int(math.log2(pz)) - 1, int(math.log2(py)) - 1  # upsamplings to stride 2 per axis
        if n_xy > len(widths) - 1 or n_z > n_xy:
            raise ValueError(f"patch {patch_zyx} needs {n_xy} upsamplings, decoder has {len(widths) - 1}")
        widths = tuple(widths[len(widths) - 1 - n_xy:])                # the finest n_xy + 1 widths
        n_up = n_xy
        self.conditioned = bool(conditioned)
        self.steps = [(2 if i < n_z else 1, 2, 2) for i in range(n_up)]
        self.stem = nn.Sequential(nn.Conv3d(in_dim, widths[0], 1, bias=False), group_norm(widths[0]), nn.GELU(),
                                  Res(widths[0], widths[0]))
        # image pyramid, finest first: stride 2 then the decoder's steps in reverse
        pyr, cin = [], int(in_chans)                                  # the encoder's input channels
        for i in range(n_up):
            stride = (2, 2, 2) if i == 0 else self.steps[n_up - i]
            pyr.append(nn.Sequential(conv3(cin, widths[n_up - i], stride), conv3(widths[n_up - i], widths[n_up - i])))
            cin = widths[n_up - i]
        self.pyramid = nn.ModuleList(pyr)
        self.ups = nn.ModuleList(
            nn.Sequential(nn.ConvTranspose3d(widths[i], widths[i + 1], s, stride=s, bias=False), group_norm(widths[i + 1]),
                          nn.GELU()) for i, s in enumerate(self.steps))
        self.fuse = nn.ModuleList(Res(2 * widths[i + 1], widths[i + 1]) for i in range(n_up))
        if self.conditioned:
            self.film = nn.ModuleList(nn.Linear(cond_dim, 2 * w) for w in widths)
            for f in self.film:
                nn.init.zeros_(f.weight)
                nn.init.zeros_(f.bias)
            self.head = nn.Conv3d(widths[-1], 1, 1)
            self.dense_head = nn.Conv3d(widths[-1], 1, 1) if dense else None
            return
        c, h, t = int(widths[-1]), int(head_hidden), int(n_tasks)                     # the finest decoder width
        self.head_hidden = h
        if h:
            self.hid_w = nn.Parameter(torch.randn(t, c, h) * (2.0 / c) ** 0.5)
            self.hid_b = nn.Parameter(torch.zeros(t, h))
        self.out_w = nn.Parameter(torch.randn(t, h or c) * 0.02)
        self.out_b = nn.Parameter(torch.zeros(t))

    @staticmethod
    def _mod(x, film, cond):
        g, b = film(cond).chunk(2, dim=-1)
        return x * (1 + g.view(*g.shape, 1, 1, 1)) + b.view(*b.shape, 1, 1, 1)

    def _pyramid(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Image features, coarse to fine: one per decoder stage."""
        feats, h = [], image
        for p in self.pyramid:
            h = p(h)
            feats.append(h)
        return feats[::-1]

    def forward(self, grid: torch.Tensor, image: torch.Tensor, rows: torch.Tensor, cond: torch.Tensor,
                dense: torch.Tensor | None = None, chan: torch.Tensor | None = None):
        """``grid`` (B, C, nz, ny, nx) token features, ``image`` (B, in_chans, Z, Y, X); ``rows`` (P,) the volume of each
        spatial slot, ``cond`` (P, cond_dim) its slot, ``dense`` (P,) bool: the slot is a dense task (``dense_head``),
        ``chan`` (P,) its task index (unconditioned decoder). Returns (P, 1, Z/2, Y/2, X/2) logits / intensities."""
        if not self.conditioned:
            return self._forward_shared(grid, image, rows, chan)
        x = self.stem(grid)
        feats = self._pyramid(image)
        x = self._mod(x[rows], self.film[0], cond)
        for i, (up, fuse) in enumerate(zip(self.ups, self.fuse)):
            x = fuse(torch.cat([up(x), feats[i][rows]], dim=1))
            x = self._mod(x, self.film[i + 1], cond)
        if dense is None or self.dense_head is None or not bool(dense.any()):
            return self.head(x)
        return torch.where(dense.view(-1, 1, 1, 1, 1), self.dense_head(x), self.head(x))

    def _forward_shared(self, grid: torch.Tensor, image: torch.Tensor, rows: torch.Tensor,
                        chan: torch.Tensor | None) -> torch.Tensor:
        if chan is None:
            raise ValueError("the unconditioned decoder needs the task index of every slot (chan)")
        uniq, inv = torch.unique(rows, return_inverse=True)
        x = self.stem(grid[uniq])
        feats = self._pyramid(image[uniq])
        for i, (up, fuse) in enumerate(zip(self.ups, self.fuse)):
            x = fuse(torch.cat([up(x), feats[i]], dim=1))
        w, b = self.out_w[chan].to(x.dtype), self.out_b[chan].to(x.dtype)               # (P, C or H), (P,)
        out = x.new_empty((len(rows), 1, *x.shape[2:]))
        recompute = self.training and torch.is_grad_enabled()
        for u in range(len(uniq)):                  # per volume: its slots' heads, without copying x per slot
            sel = (inv == u).nonzero(as_tuple=True)[0]
            f = x[u]
            if self.head_hidden:
                # the (slots, hidden, Z, Y, X) activation is the decoder's largest (4 GB at 16 slots x 64 on a ViT-S
                # batch of 4: the v3 pilots ran out of memory): a few slots at a time, recomputed in the backward pass
                for part in sel.split(4):
                    hw, hb = self.hid_w[chan[part]].to(x.dtype), self.hid_b[chan[part]].to(x.dtype)
                    args = (f, hw, hb, w[part], b[part])
                    out[part, 0] = checkpoint(_mlp_heads, *args, use_reentrant=False) if recompute \
                        else _mlp_heads(*args)
            else:
                out[sel, 0] = torch.einsum("czyx,kc->kzyx", f, w[sel]) + b[sel].view(-1, 1, 1, 1)
        return out


class BrainQuery(nn.Module):
    def __init__(self, backbone: BrainEncoder, reg: Registry, cfg: dict):
        super().__init__()
        s = cfg.get("slot") or {}
        self.backbone = backbone
        self.reg = reg
        d_enc, dim = backbone.embed_dim, int(s.get("dim", 256))
        self.taps = tuple(int(t) for t in cfg.get("taps", (3, 7)) if int(t) < backbone.depth - 1)
        n_tasks, n_cls = len(reg), reg.max_classes()
        # ``task_tokens.enabled``: OMNI-style (arXiv 2610.12574) task tokens inside the encoder instead of slot layers
        # over its output: one token per trained task, always all in the sequence (``set_token_tasks``), isolated from
        # each other and read by the patch tokens; a task's answer is its token's output. Identity lives in the
        # encoder's width then.
        self.tokens = bool((cfg.get("task_tokens") or {}).get("enabled", False))
        # window bank (``window_bank.decoder_grad``, default false): the decoder's image pyramid reads the learned
        # windows but does not train them (the bank serves the encoder; the full-resolution input gradient through the
        # pyramid's first convolution cost 0.16 s of a 1.5 s ViT-S step at batch 8 on an L40S)
        self.bank_decoder_grad = bool((cfg.get("window_bank") or {}).get("decoder_grad", False))
        if self.tokens:
            self.register_buffer("tt_write", torch.zeros(n_tasks, dtype=torch.bool))
        qdim = d_enc if self.tokens else dim
        # task identity
        self.task_emb = nn.Embedding(n_tasks, qdim)
        nn.init.normal_(self.task_emb.weight, std=0.02)
        self.type_emb = nn.Embedding(len(TYPE_ID), qdim)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        self.register_buffer("task_type", torch.tensor([TYPE_ID[t.type] for t in reg.tasks]), persistent=False)
        self.register_buffer("n_classes", torch.tensor([len(t.classes) for t in reg.tasks]), persistent=False)
        # shared task tokens (registry ``token``, on with ``task_tokens.share``; earlier checkpoints keep one token per
        # task): members read the token of ``tok_of`` (the group's first task), each through its own offset
        # ``cls_emb[t, 0]`` (a row binary / segmentation tasks do not otherwise use)
        share = self.tokens and bool((cfg.get("task_tokens") or {}).get("share", False))
        tok_of = reg.token_of() if share else list(range(n_tasks))
        self.register_buffer("tok_of", torch.tensor(tok_of), persistent=False)
        self.register_buffer("tok_shared", torch.tensor([share and t.token is not None for t in reg.tasks]),
                             persistent=False)
        if self.tokens:
            self.tt_proj = nn.Linear(d_enc, dim)                # a task token's output -> the heads' input
        else:
            # slot readout: tokens -> keys / values with an absolute position code
            # (RoPE leaves tokens position-relative), then the slot layers
            self.kv_norm = nn.LayerNorm(d_enc)
            self.kv = nn.Linear(d_enc, dim)
            self.pos = mlp(3, dim, dim)
            self.cls_pos = nn.Parameter(torch.zeros(1, 1, dim))
            self.layers = nn.ModuleList(SlotLayer(dim, int(s.get("heads", 8))) for _ in range(int(s.get("layers", 3))))
        self.out_norm = nn.LayerNorm(dim)
        # typed heads, shared across tasks
        self.binary = mlp(dim, dim, 1)
        self.regress = mlp(dim, dim, 1)
        # ``regression.kind: hlgauss`` (Farebrother et al., ICML 2024): standardised targets as a distribution over
        # ``bins`` on ``range``, trained against a Gaussian (sigma = ``sigma_ratio`` bin widths) and read out as its
        # mean; the scalar head stays (unused) so scalar-trained checkpoints still load
        rg = cfg.get("regression") or {}
        self.regress_bins = None
        if rg.get("kind", "scalar") == "hlgauss":
            n, (lo, hi) = int(rg.get("bins", 80)), rg.get("range", [-5.0, 5.0])
            edges = torch.linspace(float(lo), float(hi), n + 1)
            self.register_buffer("reg_edges", edges, persistent=False)
            self.register_buffer("reg_centres", (edges[1:] + edges[:-1]) / 2, persistent=False)
            self.reg_sigma = float(rg.get("sigma_ratio", 0.75)) * float(edges[1] - edges[0])
            self.regress_bins = mlp(dim, dim, n)
        elif rg.get("kind", "scalar") != "scalar":
            raise ValueError(f"model.regression.kind {rg['kind']!r}")
        # ``token_map.enabled``: a logit per encoder patch token for a slot (its task's "is it here?"), from the final
        # patch tokens and the slot's answer vector: trained on mask tasks' token occupancy and on RSNA slice labels
        # (losses.token_occupancy_loss / slab_mil_loss), a coarse localisation target straight into the encoder
        tm = cfg.get("token_map") or {}
        self.tok_q = self.tok_k = self.tok_bias = None
        if tm.get("enabled"):
            dk = int(tm.get("dim", 64))
            self.tok_q, self.tok_k = nn.Linear(dim, dk), nn.Linear(d_enc, dk)
            self.tok_bias = nn.Parameter(torch.tensor(-3.0))          # prior ~5% of tokens on
        # multiple-instance binary readout (registry ``pool: mil``): the slot's token map (its per-location logits,
        # without the map's shared prior bias) soft-max pooled over the scan's foreground tokens,
        # (1 / r) log mean exp(r l), plus a per-task bias (zero at init, so a head starts at its token logit), added to
        # the task token's own logit; r per task, learned, from 8 (large: one location decides, a focal finding;
        # small: the mean, a diffuse one)
        # (``model.mil.enabled``; off in older configs, so their checkpoints load and answer unchanged)
        mil_on = bool((cfg.get("mil") or {}).get("enabled", False))
        self.register_buffer("mil", torch.tensor([mil_on and t.pool == "mil" for t in reg.tasks]), persistent=False)
        self.mil_logr = None
        if bool(self.mil.any()):
            if self.tok_q is None:
                raise ValueError("registry pool: mil needs model.token_map.enabled (the per-location logits)")
            self.mil_logr = nn.Parameter(torch.full((n_tasks,), math.log(8.0)))
            self.mil_bias = nn.Parameter(torch.zeros(n_tasks))
        self.cls_query = nn.Linear(dim, dim)
        self.cls_emb = nn.Parameter(torch.randn(n_tasks, n_cls, dim) * 0.02)          # per (task, class)
        if share:
            # members of a shared token read it through ``cls_emb[t, 0]`` added to the normalised answer (norm ~ sqrt
            # dim): at std 0.02 the six hemorrhage (or nine radiology) members start with near-identical answers and
            # token maps; std 0.3 gives each a distinct reading from step 0 (loaded checkpoints overwrite it)
            with torch.no_grad():
                shared = torch.tensor([t.token is not None for t in reg.tasks])
                self.cls_emb[shared, 0] = torch.randn(int(shared.sum()), dim) * 0.3
        self.survival = mlp(dim, dim, 1)        # no survival task is trained or asked: kept so checkpoints load
        seg_cfg = cfg.get("seg") or {}
        patch_zyx = tuple(int(p) for p in reversed(backbone.patch_xyz))
        # ``seg.shared``: decode once per volume, an output per task (SegDecoder unconditioned); else once per slot
        self.seg_shared = bool(seg_cfg.get("shared", False))
        in_chans = getattr(backbone, "in_chans", None) or backbone.patch_embed.proj.in_channels
        self.seg = SegDecoder(
            d_enc * (len(self.taps) + 1), dim, patch_zyx, tuple(seg_cfg.get("widths", (128, 64, 32, 16))),
            in_chans=in_chans,
            dense=any(t.type == "dense" for t in reg.tasks), conditioned=not self.seg_shared, n_tasks=n_tasks,
            head_hidden=int(seg_cfg.get("head_hidden", 0)))

    # -- encoder ---------------------------------------------------------------------------------------
    def _coords(self, shape_zyx, device):
        size = (shape_zyx[2], shape_zyx[1], shape_zyx[0])
        bbox = crop_bbox_mm((0, 0, 0), size, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)).to(device)
        return token_centers_mm(bbox, size, self.backbone.patch_xyz), patch_size_mm(bbox, size, self.backbone.patch_xyz)

    def set_token_tasks(self, ids) -> None:
        """The tasks whose tokens are always in the sequence and read by the patch tokens (the trained tasks)."""
        self.tt_write.zero_()
        self.tt_write[self.tok_of[torch.as_tensor(list(ids), dtype=torch.long, device=self.tt_write.device)]] = True

    def window_input(self, x: torch.Tensor, modality: torch.Tensor | None, hide: torch.Tensor | None = None,
                     study: dict | None = None) -> tuple[torch.Tensor, dict | None]:
        """The encoder's window bank (``backbone.window_bank``, ``brain.models.window_bank``) on native-unit input
        ``x`` (B, 1, Z, Y, X) and the study partners (``study["x"]`` by ``study["mod"]``), then the masked voxels
        ``hide`` (B, Z, Y, X) blanked (0 in every channel). Without a bank, or for input already windowed to K
        channels, ``x`` passes unchanged."""
        bank = bank_of(self.backbone)
        if bank is None:
            return x, study
        x = bank_input(bank, x, modality)
        if hide is not None:
            x = x.masked_fill(hide.unsqueeze(1), 0.0)
        if study is not None and len(study["row"]):
            study = {**study, "x": bank_input(bank, study["x"], study.get("mod"))}
        return x, study

    def encode(self, x: torch.Tensor, keep: torch.Tensor, meta: torch.Tensor | None = None,
               task: torch.Tensor | None = None, study: dict | None = None,
               modality: torch.Tensor | None = None) -> dict:
        """Encoder outputs. With task tokens: every trained task's token, plus read-only tokens for the other tasks
        in ``task`` (held-out / few-shot), so an answer never depends on which other tasks were asked.
        ``study`` (R14-F): the anchors' co-registered partner scans, ``x`` (P, C, Z, Y, X), ``row`` (P,) their
        anchor, ``seq`` (B,) / ``pseq`` (P,) sequence types; fused inside the encoder (``SequenceFusion``).
        With a window bank, native-unit ``x`` (B, 1, ...) needs ``modality`` (``window_input``)."""
        x, study = self.window_input(x, modality, study=study)
        b = x.shape[0]
        fuse = None
        if study is not None and len(study["row"]) and getattr(self.backbone, "fusion", None) is not None:
            row = study["row"]
            s_max = 1 + int(torch.bincount(row, minlength=b).max())
            index = torch.full((b, s_max), -1, dtype=torch.long, device=x.device)
            index[:, 0] = torch.arange(b, device=x.device)
            slot = torch.zeros(b, dtype=torch.long, device=x.device)
            for j in range(len(row)):                               # P is small (a few partners per batch)
                r = int(row[j])
                slot[r] += 1
                index[r, int(slot[r])] = b + j
            seq = torch.zeros_like(index)
            seq[:, 0] = study["seq"]
            part = index[:, 1:]
            sub = seq[:, 1:]
            sub[part >= 0] = study["pseq"][part[part >= 0] - b]
            fuse = {"n_out": b, "index": index, "seq": seq}
            x = torch.cat([x, study["x"].to(x.dtype)])
            keep = torch.cat([keep, keep[row]])
            # each partner's own META id (contrast): the anchor's would label a FLAIR partner as the anchor's T1
            pmeta = study.get("pmeta")
            if meta is not None:
                meta = torch.cat([meta, pmeta.to(meta) if pmeta is not None and len(pmeta) == len(row)
                                  else torch.zeros_like(meta[row])])
        n_rows = x.shape[0]
        coords, pmm = self._coords(x.shape[2:], x.device)
        tt = write = ids = None
        if self.tokens:
            asked = torch.zeros_like(self.tt_write)
            if task is not None and (task >= 0).any():
                asked[self.tok_of[task[task >= 0]].unique()] = True
            ids = torch.cat([self.tt_write.nonzero()[:, 0], (asked & ~self.tt_write).nonzero()[:, 0]])
            if ids.numel() == 0:
                raise ValueError("task tokens: no trained tasks set (set_token_tasks) and none asked")
            tt, write = self.task_queries(ids), self.tt_write[ids]
        out = self.backbone(x.permute(0, 1, 4, 3, 2), coords.unsqueeze(0).expand(n_rows, -1, -1),
                            pmm.unsqueeze(0).expand(n_rows, -1), keep=keep, taps=self.taps, meta=meta,
                            task_tokens=tt, task_write=write, **({"fuse": fuse} if fuse is not None else {}))
        out["coords"] = coords
        if self.tokens:
            out["tt_ids"] = ids
        return out

    def token_slots(self, enc: dict, task: torch.Tensor) -> torch.Tensor:
        """(B, K, width) from the task tokens' outputs; padded slots (task < 0) read the first token."""
        lut = torch.full((len(self.reg),), -1, dtype=torch.long, device=task.device)
        ids = enc["tt_ids"]
        lut[ids] = torch.arange(ids.numel(), device=task.device)
        pos = lut[self.tok_of[task.clamp_min(0)]]
        if bool(((pos < 0) & (task >= 0)).any()):
            raise ValueError("task tokens: a requested task has no token in this encoding (encode with task=...)")
        tok = enc["task_tokens"]                                            # (B, T, D)
        z = torch.gather(tok, 1, pos.clamp_min(0).unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
        return self.out_norm(self.tt_proj(z))

    # -- slots -----------------------------------------------------------------------------------------
    def task_queries(self, task: torch.Tensor) -> torch.Tensor:
        t = task.clamp_min(0)
        return self.task_emb(t) + self.type_emb(self.task_type[t])

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

    def slots(self, enc: dict, task: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        if self.tokens:
            return self.token_slots(enc, task)
        q = self.task_queries(task)                                         # (B, K, dim)
        kv, pad = self._kv(enc, keep)
        slot_pad = task < 0
        slot_pad = slot_pad & ~(slot_pad.all(1, keepdim=True) & (torch.arange(task.shape[1], device=task.device) == 0))
        for layer in self.layers:
            q = layer(q, slot_pad, kv, pad)
        return self.out_norm(q)

    def slots_alone(self, enc: dict, task: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """``slots`` with every task asked on its own (no slot attends to another), as at deployment; the scan is
        encoded once and its keys/values shared by the per-task queries."""
        if self.tokens:                                       # already independent of the other asked tasks
            return self.slots(enc, task, keep)
        b, k = task.shape
        kv, pad = self._kv(enc, keep)
        one = {"_kv": (kv.repeat_interleave(k, 0), pad.repeat_interleave(k, 0))}
        return self.slots(one, task.reshape(b * k, 1), keep).reshape(b, k, -1)

    def member(self, z: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """A shared token's output read as one member task: plus that task's offset (no-op for unshared tasks)."""
        t = task.clamp_min(0)
        if not bool(self.tok_shared[t].any()):
            return z
        return z + self.tok_shared[t].unsqueeze(-1).to(z.dtype) * self.cls_emb[t, 0].to(z.dtype)

    @torch.no_grad()
    def bins_from_scalar(self, width: float = 0.5) -> None:
        """Warm-start the HL-Gauss head from the scalar one: a Gaussian of std ``width`` around the scalar prediction
        r has logits (c_k r - c_k^2 / 2) / width^2, linear in r, so the bin head copies the scalar head's hidden layer
        and its output layer becomes c_k / width^2 times the scalar output (mean = r away from the range ends)."""
        c = self.reg_centres.float() / width ** 2
        self.regress_bins[0].load_state_dict(self.regress[0].state_dict())
        w, b = self.regress[2].weight.float(), self.regress[2].bias.float()       # (1, dim), (1,)
        self.regress_bins[2].weight.copy_(c[:, None] * w)
        self.regress_bins[2].bias.copy_(c * b - self.reg_centres.float() ** 2 / (2 * width ** 2))

    def heads(self, z: torch.Tensor, task: torch.Tensor) -> dict:
        t = task.clamp_min(0)
        logits = torch.einsum("bkd,bkcd->bkc", self.cls_query(z), self.cls_emb[t]) / math.sqrt(self.cls_emb.shape[-1])
        valid = torch.arange(logits.shape[-1], device=z.device) < self.n_classes[t].unsqueeze(-1)
        out = {"binary": self.binary(z)[..., 0], "multiclass": logits.masked_fill(~valid, float("-inf"))}
        if self.regress_bins is None:
            out["regression"] = self.regress(z)[..., 0]
        else:
            lg = self.regress_bins(z)
            out["regression"] = (lg.float().softmax(-1) * self.reg_centres).sum(-1)
            out["regression_hl"] = (lg, self.reg_edges, self.reg_sigma)
        return out

    def mil_logits(self, enc: dict, z: torch.Tensor, task: torch.Tensor, keep: torch.Tensor) -> torch.Tensor | None:
        """(B, K) pooled token-map logits of the ``pool: mil`` slots (0 elsewhere), to add to ``heads``' binary logit;
        None when no such slot is asked. ``keep`` (B, N): the foreground tokens pooled over."""
        if getattr(self, "mil_logr", None) is None:
            return None
        m = self.mil[task.clamp_min(0)] & (task >= 0)
        if not bool(m.any()):
            return None
        pairs = m.nonzero()
        t = task[pairs[:, 0], pairs[:, 1]]
        # the map's shared prior bias (tok_bias, tuned for token occupancy) left out: the pooled logit gets its own
        # bias, so 12 binary tasks do not pull on the prior every segmentation task's map uses
        lg = self.token_logits(enc, z, pairs).float() - self.tok_bias.float()      # (Q, N)
        kp = keep[pairs[:, 0]].bool()
        r = self.mil_logr[t].float().exp().clamp(0.05, 50.0).unsqueeze(-1)
        lse = torch.logsumexp((lg * r).masked_fill(~kp, float("-inf")), -1)
        pooled = (lse - kp.sum(-1).clamp_min(1).float().log()) / r[:, 0]
        pooled = torch.where(kp.any(-1), pooled, torch.zeros_like(pooled)) + self.mil_bias[t].float()   # no token: 0
        add = torch.zeros(task.shape, device=z.device, dtype=pooled.dtype)
        return add.index_put((pairs[:, 0], pairs[:, 1]), pooled)

    def token_logits(self, enc: dict, z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """(Q, N) token-map logits of the slots ``pairs`` (Q, 2) = (row, slot), in the encoder's (z, y, x) token order."""
        tokens = enc["tokens"]
        k = self.tok_k(F.layer_norm(tokens, tokens.shape[-1:]))                  # (B, N, dk)
        q = self.tok_q(z[pairs[:, 0], pairs[:, 1]])                                # (Q, dk)
        return torch.einsum("qd,qnd->qn", q, k[pairs[:, 0]]) / math.sqrt(q.shape[-1]) + self.tok_bias

    def forward(self, x: torch.Tensor, keep: torch.Tensor, task: torch.Tensor, seg_pair: torch.Tensor,
                encoder_grad: bool = True, alone: bool = False, meta: torch.Tensor | None = None,
                dense_pair: torch.Tensor | None = None, tok_pair: torch.Tensor | None = None,
                study: dict | None = None, modality: torch.Tensor | None = None,
                hide: torch.Tensor | None = None) -> dict:
        """Typed answers per slot; ``out["seg"]`` (segmentation slots, ``seg_pair`` order) and ``out["dense"]`` (dense
        slots, ``dense_pair`` order) come from one decoder pass; ``out["tok"]`` the token maps of ``tok_pair``.
        With a window bank ``x`` is native units (B, 1, ...) and ``modality`` (B,) is required; the bank's K channels
        feed the encoder and the decoder's image pyramid, and ``hide`` voxels are blanked after it (``window_input``)."""
        # task tokens live inside the encoder, so their gradient has to pass through it even when the encoder is frozen
        # (few-shot fitting: its weights have requires_grad off and get none)
        with torch.set_grad_enabled(torch.is_grad_enabled() and (encoder_grad or self.tokens)):
            x, study = self.window_input(x, modality, hide, study)
            enc = self.encode(x, keep, meta, task, study)
        z = self.slots_alone(enc, task, keep) if alone else self.slots(enc, task, keep)
        z = self.member(z, task)
        out = self.heads(z, task)
        mil = self.mil_logits(enc, z, task, keep)
        if mil is not None:
            out["binary"] = out["binary"] + mil.to(out["binary"].dtype)
        out["seg"] = out["dense"] = out["tok"] = None
        if tok_pair is not None and len(tok_pair) and self.tok_q is not None:
            out["tok"] = self.token_logits(enc, z, tok_pair)
        n_seg, n_dense = len(seg_pair), 0 if dense_pair is None else len(dense_pair)
        if n_seg or n_dense:
            b = x.shape[0]
            pz, py, px = reversed(self.backbone.patch_xyz)
            g = (x.shape[2] // pz, x.shape[3] // py, x.shape[4] // px)
            taps = [F.layer_norm(t, t.shape[-1:]) for t in enc["taps"]]       # pre-norm residual stream -> unit scale
            feats = torch.cat([*taps, enc["tokens"]], -1)                    # (B, N, D * (taps + 1))
            grid = feats.transpose(1, 2).reshape(b, feats.shape[-1], *g).contiguous()   # DDP grad layout
            pairs = torch.cat([seg_pair, dense_pair]) if n_dense else seg_pair
            is_dense = torch.arange(n_seg + n_dense, device=x.device) >= n_seg
            cond = z[pairs[:, 0], pairs[:, 1]]
            chan = {"chan": task[pairs[:, 0], pairs[:, 1]]} if self.seg_shared else {}
            img = x if self.bank_decoder_grad or bank_of(self.backbone) is None else x.detach()
            maps = self.seg(grid, img, pairs[:, 0], cond, dense=is_dense if n_dense else None, **chan)
            out["seg"] = maps[:n_seg] if n_seg else None
            out["dense"] = maps[n_seg:] if n_dense else None
        return out


class TokenTraining:
    """What a trainer does for task tokens, shared by the multitask and joint trainers.

    * registers the trained tasks as the always-present (written) tokens;
    * ``at_step``: the patch tokens' attention to the task tokens starts damped by ``write_bias`` logits and opens
      linearly over ``write_ramp_steps`` from ``start`` (a warm-started encoder: undamped, random task tokens shift its
      features ~40% before any training).
    A model without task tokens makes every call a no-op."""

    def __init__(self, model: BrainQuery, cfg_model: dict, trained: list[int]):
        self.model = model
        tt = cfg_model.get("task_tokens") or {}
        self.bias0, self.ramp = float(tt.get("write_bias", 10.0)), int(tt.get("write_ramp_steps", 2000))
        if model.tokens:
            model.set_token_tasks(trained)

    def at_step(self, step: int, start: int = 0) -> None:
        if self.model.tokens:
            self.model.backbone.task_write_bias = self.bias0 * max(0.0, 1.0 - (step - start) / max(self.ramp, 1))
