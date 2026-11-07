"""Task-conditioned multitask post-training of an SSL encoder (brain.multitask) on whole volumes.

    python -m brain.train multitask --config configs/multitask/tokens.yaml
    python -m brain.train multitask --synthetic --max-steps 4          # CPU smoke run on phantoms

The encoder starts from ``model.init_from`` (its architecture and token grid are copied into this run's config, so
frozen evaluation loads these checkpoints like any 3D encoder: ``model`` is the encoder) and trains at ``enc_lr_mult``
x the LR with layer-wise decay. The tasks enter as task tokens inside the encoder (``model.task_tokens``, the default);
checkpoints also hold ``multitask`` (task embeddings, heads, decoder) and ``tasks`` (registry order).
"""

from __future__ import annotations

import torch

from brain.common.optim import NO_DECAY_SUFFIXES, NO_DECAY_TABLES, POST_BLOCKS
from brain.common.paths import resolve
from brain.data.volumes import n_meta
from brain.multitask.data import SLAB_TASKS, TOKEN_MIL_TASKS, TYPE_ID, prepare, protected_tasks, sharp_tasks, task_loaders
from brain.multitask.evaluate import format_eval, run_eval
from brain.multitask.losses import Balancer, slab_mil_loss, slot_losses, token_occupancy, token_occupancy_loss
from brain.multitask.model import BrainQuery, TokenTraining
from brain.multitask.registry import load_registry
from brain.train import engine

# log column per task group: the raw (unbalanced) mean loss of the group's tasks, averaged over the steps of the
# logging window in which the group was asked (the balanced ``loss`` is weighted by the learned uncertainties)
GROUP_COLUMNS = {"demographics": "demo", "diagnosis": "dx", "hemorrhage": "ich", "tumor": "tumor",
                 "derived": "derived", "anatomy_seg": "aseg", "anatomy_volume": "avol", "lesion": "lesion",
                 "ssl_recon": "recon", "synthesis": "synth"}
# heads a resumed run may add (absent from its checkpoint): HL-Gauss bins, the token map
NEW_HEADS = ("regress_bins.", "tok_q.", "tok_k.", "tok_bias", "backbone.fusion.")
TYPE_COLUMNS = {"binary": "t_bin", "regression": "t_reg", "multiclass": "t_cls", "segmentation": "t_seg",
                "survival": "t_surv", "dense": "t_dense"}


def build_encoder(cfg: dict, device: torch.device, synthetic: bool):
    """The SSL encoder, trainable. Copies its model / views config into ``cfg`` (what frozen evaluation reads)."""
    from brain.downstream.embed3d import load_brain3d
    from brain.train.lejepa3d import build_encoder

    m = cfg["model"]
    if m.get("arch") == "resnet3d":          # from-scratch convolutional baseline (brain.models.resnet3d)
        from brain.models.resnet3d import ResNetEncoder

        r = m.get("resnet") or {}
        enc = ResNetEncoder(widths=r.get("widths", (64, 128, 240, 384)), blocks=r.get("blocks", (2, 2, 2, 2)),
                            in_chans=int(m.get("in_chans", 1)), patch_xyz=tuple(cfg["views"]["patch_xyz"]),
                            checkpoint_blocks=bool(m.get("checkpoint", True)) and not synthetic)
        return enc.to(device).train().requires_grad_(True)
    if synthetic or not m.get("init_from"):
        # the config records how the encoder was built (frozen evaluation rebuilds it from these, with other defaults)
        m.update(size=m.get("size") or "test", qk_norm=True, n_cls=int(m.get("n_cls", 1)), proj_dim=16, proj_hidden=32,
                 conv_stem=list(m.get("conv_stem") or []))
        enc = build_encoder(cfg, torch.device("cpu"))       # no loss.dense / predict: no token head, no predictor
    else:
        src = torch.load(resolve(m["init_from"]), map_location="cpu", weights_only=False)
        enc = load_brain3d(src, torch.device("cpu")).model
        scfg = src.get("config") or {}
        for k, v in (scfg.get("model") or {}).items():
            if k not in ("checkpoint",) and k not in m:
                m[k] = v
        cfg["views"] = {**(scfg.get("views") or {}), "gate": cfg["views"].get("gate")}
        cfg["views"]["patch_xyz"] = list(enc.patch_xyz)
        m["size"] = enc.size
        m["in_chans"] = int(enc.in_chans)                          # data.prepare windows the input to match
        # the SSL pool the encoder saw, for benchmark overlap records (records.ssl_exclude reads the run's config)
        cfg["data"]["ssl_exclude"] = list((scfg.get("data") or {}).get("ssl_exclude") or [])
        # the encoder keeps the numerics it was trained with (the defaults' True would not describe it)
        m["rope_fp32"], m["fp32_residual"] = bool(enc.rope.fp32), bool(enc.fp32_residual)
        if n_meta(m) and enc.meta_emb is None:                # metadata on an encoder trained without it
            enc.enable_meta(n_meta(m))
    enc.checkpoint_blocks = bool(m.get("checkpoint", True)) and not synthetic
    if hasattr(enc.patch_embed, "checkpoint"):                       # a conv stem follows the encoder's setting
        enc.patch_embed.checkpoint = enc.checkpoint_blocks
    fu = m.get("fusion") or {}
    if fu.get("after_block"):                 # study-level input (R14-F): sequence fusion inside the encoder
        enc.enable_fusion(int(fu["after_block"]), fu.get("heads"))
    return enc.to(device).train().requires_grad_(True)


def param_groups(model: BrainQuery, extra: list, wd: float, enc_mult: float, layer_decay: float) -> list[dict]:
    """Encoder blocks at ``enc_mult * layer_decay ** (depth - 1 - i)`` of the LR (embedding at ``** depth``, the final
    norm and heads at ``enc_mult``), the rest at 1. Biases, norms, learned tokens
    and the task tables get no weight decay."""
    depth = model.backbone.depth
    buckets: dict[tuple[float, float], list] = {}
    named = list(model.named_parameters()) + [(f"extra.{i}", p) for i, p in enumerate(extra)]
    for name, p in named:
        if not p.requires_grad:
            continue
        scale = 1.0
        if name.startswith("backbone."):
            if name.startswith("backbone.blocks."):
                scale = enc_mult * layer_decay ** (depth - 1 - int(name.split(".")[2]))
            elif name.startswith(tuple("backbone." + m for m in POST_BLOCKS)):
                scale = enc_mult                                    # final norm / heads: the deepest block's LR
            else:
                scale = enc_mult * layer_decay ** depth
        skip = p.ndim < 2 or name.endswith(NO_DECAY_SUFFIXES) or any(k in name for k in NO_DECAY_TABLES)
        buckets.setdefault((scale, 0.0 if skip else wd), []).append(p)
    return [{"params": ps, "lr": s, "weight_decay": d, "lr_scale": s} for (s, d), ps in buckets.items()]


def remap_tasks(state: dict, old_ids: list[str], new_ids: list[str], template: dict) -> dict:
    """A checkpoint's per-task tensors (first dimension = its task list: task embeddings, class vectors, token flags,
    balancer weights) re-indexed onto a changed registry by task id: kept tasks keep their rows, new ones take
    ``template``'s (the freshly built model's), dropped ones go. Everything else passes through."""
    pos = {t: i for i, t in enumerate(old_ids)}
    src = torch.tensor([pos.get(t, -1) for t in new_ids])
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v) and v.ndim and v.shape[0] == len(old_ids) and k in template \
                and template[k].shape[0] == len(new_ids) and template[k].shape[1:] == v.shape[1:]:
            t = template[k].detach().clone().to(v.dtype)
            have = src >= 0
            t[have] = v[src[have]]
            out[k] = t
        else:
            out[k] = v
    return out


class Trainer(engine.Trainer):
    unit = "vol"
    synthetic_ok = True

    def __init__(self, rt, cfg, args, ckpt, step):
        super().__init__(rt, cfg, args, ckpt, step)
        dev = rt.device
        o, d, tk = cfg["optim"], cfg["data"], cfg["tasks"]
        self.batch = int(o["batch_size"])
        self.batch_units = self.batch * rt.world
        self.clip = float(o.get("clip_grad", 1.0))
        self.stride = int(cfg["loss"].get("seg_stride", 2))
        self.amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
        self.reg = load_registry(d.get("registry"))
        # ``tasks.groups`` / ``tasks.ids`` restrict training (no-label control; single-task models), ``tasks.exclude`` drops tasks
        self.train_ids = self.reg.train_ids(tk)
        if not self.train_ids:
            raise SystemExit(f"tasks {tk} selects no trained task (groups: {self.reg.groups})")
        enc = self.build_encoder(cfg, dev, args.synthetic)
        self.patch = tuple(enc.patch_xyz)
        self.raw = BrainQuery(enc, self.reg, cfg["model"]).to(dev)
        self.tokens = TokenTraining(self.raw, cfg["model"], [self.reg.index[t] for t in self.train_ids])
        groups = self.reg.groups
        gw = tk.get("group_weights") or {}
        self.balancer = Balancer(len(self.reg), group_of=[groups.index(t.group) for t in self.reg.tasks],
                                 group_weight=[float(gw.get(g, 1.0)) for g in groups],
                                 clamp=cfg["loss"].get("uncertainty_clamp")).to(dev)
        self.module = self.raw
        groups = self.reg.groups
        self.group_of = torch.tensor([groups.index(t.group) for t in self.reg.tasks], device=dev)
        self.log_groups = [(i, GROUP_COLUMNS.get(g, g[:8])) for i, g in enumerate(groups) if g != "held_out"]
        self.g_sum = torch.zeros(len(groups), device=dev)
        self.g_cnt = torch.zeros(len(groups), device=dev)
        # the same raw losses by task type (binary, regression, ...), logged as t_* columns
        self.type_of = self.raw.task_type.long().to(dev)
        self.log_types = [(i, c) for name, c in TYPE_COLUMNS.items()
                          if bool((self.type_of == (i := TYPE_ID[name])).any())]
        self.t_sum = torch.zeros(len(TYPE_ID), device=dev)
        self.t_cnt = torch.zeros(len(TYPE_ID), device=dev)
        self.fov_unsafe, self.aug_unsafe = (t.to(dev) for t in protected_tasks(self.reg))
        self.sharp = sharp_tasks(self.reg, cfg)
        # tiny targets (registry ``sparse``): losses.sparse_seg_loss, with the token grid in output voxels
        self.sparse = torch.tensor([t.sparse for t in self.reg.tasks], device=dev) \
            if any(t.sparse for t in self.reg.tasks) else None
        px, py, pz = (int(v) for v in self.patch)
        self.cell_zyx = tuple(max(1, v // self.stride) for v in (pz, py, px))
        self.pos_weight = float(cfg["loss"].get("sparse_pos_weight", 20.0))
        # lesion tasks with true negatives (labels v11 empty masks): losses.negative_seg_loss
        neg = set(cfg["loss"].get("negative_tasks") or [])
        self.negatives = torch.tensor([t.id in neg for t in self.reg.tasks], device=dev) if neg else None
        # token maps: mask tasks' token occupancy (token_weight) and RSNA slice labels (slice_weight)
        self.tok_w = float(cfg["loss"].get("token_weight", 0.0))
        self.slice_w = float(cfg["loss"].get("slice_weight", 0.0))
        self.box_w = float(cfg["loss"].get("box_weight", 0.0))
        if (self.tok_w or self.slice_w or self.box_w) and self.raw.tok_q is None:
            raise SystemExit("loss.token_weight / slice_weight / box_weight need model.token_map.enabled")
        # token-MIL channel of each task (RSNA slice labels, then fastMRI+ boxes), and that channel's loss weight
        self.slab_sub = torch.full((len(self.reg),), -1, dtype=torch.long, device=dev)
        for k, t in enumerate(TOKEN_MIL_TASKS):
            if t in self.reg.index:
                self.slab_sub[self.reg.index[t]] = k
        self.mil_w = torch.tensor([self.slice_w] * len(SLAB_TASKS) + [self.box_w] * (len(TOKEN_MIL_TASKS) - len(SLAB_TASKS)),
                                  device=dev)
        self.aux_sum, self.aux_cnt = torch.zeros(2, device=dev), torch.zeros(2, device=dev)
        seed = int(cfg["seed"]) + rt.rank
        self.loader, self.val_loader, self.stats, n_train = task_loaders(
            cfg, self.reg, self.train_ids, batch=self.batch, workers=int(d.get("num_workers", 10)), seed=seed + int(step),
            main=rt.main, distributed=rt.world > 1, synthetic=args.synthetic,
            saved_stats=(ckpt or {}).get("label_stats"))
        self._banner_extra = f"train_volumes={n_train}"
        self.iter = iter(self.loader)
        self.epoch = 0
        self.opt = torch.optim.AdamW(
            param_groups(self.raw, list(self.balancer.parameters()), float(o["weight_decay"]),
                         float(o.get("enc_lr_mult", 0.25)), float(o.get("layer_decay", 0.8))),
            lr=float(o["lr"]), betas=tuple(o.get("betas", (0.9, 0.999))), fused=dev.type == "cuda")
        if ckpt is not None:
            old_ids, new_ids = list(ckpt.get("tasks") or []), [t.id for t in self.reg.tasks]
            retasked = bool(old_ids) and old_ids != new_ids
            if retasked:                 # registry changed since the checkpoint: re-index per-task rows by task id
                ckpt["multitask"] = remap_tasks(ckpt["multitask"], old_ids, new_ids, self.raw.state_dict())
                ckpt["balancer"] = remap_tasks(ckpt["balancer"], old_ids, new_ids, self.balancer.state_dict())
                if ckpt.get("ema"):
                    ckpt["ema"]["avg"] = remap_tasks(ckpt["ema"]["avg"], old_ids, new_ids, self.raw.state_dict())
                rt.log(f"[resume] registry changed: {len(old_ids)} -> {len(new_ids)} tasks "
                       f"(new: {sorted(set(new_ids) - set(old_ids))[:12]}); per-task rows re-indexed by id")
            res = self.raw.load_state_dict(ckpt["multitask"], strict=False)
            new = [k for k in res.missing_keys if k.startswith(NEW_HEADS)]
            if res.unexpected_keys or len(new) != len(res.missing_keys):
                raise SystemExit(f"[resume] state mismatch: missing={res.missing_keys[:4]} "
                                 f"unexpected={res.unexpected_keys[:4]}")
            self.balancer.load_state_dict(ckpt["balancer"])
            if self.raw.tokens:      # the saved write mask (a persistent buffer) is the parent run's task set
                self.raw.set_token_tasks([self.reg.index[t] for t in self.train_ids])
            if new or retasked:     # heads or tasks added on resume: the saved optimizer no longer fits
                if any(k.startswith("regress_bins.") for k in new):
                    self.raw.bins_from_scalar()          # HL-Gauss bins start from the scalar head's predictions
                rt.log(f"[resume] new heads {sorted({k.split('.')[0] for k in new})}; optimizer state reset")
            else:
                if "optimizer" in ckpt:              # weights-only snapshots (keep_full false) carry none
                    # load_state_dict restores the saved groups' hyperparameters too: keep this config's
                    fresh = [{k: g[k] for k in ("lr_scale", "weight_decay", "betas") if k in g}
                             for g in self.opt.param_groups]
                    self.opt.load_state_dict(ckpt["optimizer"])
                    for g, f in zip(self.opt.param_groups, fresh):
                        g.update(f)
        self.net = self.wrap(rt)
        n_enc = sum(p.numel() for p in enc.parameters())
        n_all = sum(p.numel() for p in self.raw.parameters())
        self._banner = (f"enc={enc.size} patch={self.patch} enc_params={n_enc:,} head_params={n_all - n_enc:,} "
                        f"tasks={len(self.train_ids)}/{len(self.reg)} batch={self.batch}x{rt.world} "
                        + getattr(self, "_banner_extra", ""))

    def build_encoder(self, cfg: dict, device: torch.device, synthetic: bool):
        """The encoder (overridden by brain.train.joint, whose encoder also carries the SSL projector)."""
        return build_encoder(cfg, device, synthetic)

    def wrap(self, rt):
        """The module one training forward runs through (DDP over ``self.raw`` on several GPUs)."""
        return torch.nn.parallel.DistributedDataParallel(self.raw, device_ids=[rt.local], find_unused_parameters=True) \
            if rt.world > 1 else self.raw

    def banner(self) -> str:
        return self._banner

    def _next(self) -> dict:
        try:
            return next(self.iter)
        except StopIteration:
            self.epoch += 1
            sampler = getattr(self.loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)
            self.iter = iter(self.loader)
            return next(self.iter)

    def step(self, step: int, lr: float) -> dict[str, torch.Tensor]:
        self.tokens.at_step(step)
        b = self.task_batch()
        self.opt.zero_grad(set_to_none=True)
        args, kw, pairs = self.task_inputs(b, step)
        with torch.autocast(**self.amp):
            out = self.net(*args, **kw)
        total, per_task, has = self.task_loss(out, b, pairs)
        self.apply(total, step)
        self.log_tasks(per_task, has, b)
        return {"loss": total.detach(), "k": (b["task"] >= 0).float().sum(1).mean()}

    # the step in parts, so brain.train.joint adds its SSL branch without a second copy of the task side
    def task_batch(self) -> dict:
        return prepare(self._next(), self.cfg, self.rt.device, self.patch, train=True, fov_unsafe=self.fov_unsafe,
                       aug_unsafe=self.aug_unsafe, sharp=self.sharp)

    def task_inputs(self, b: dict, step: int) -> tuple[tuple, dict, tuple]:
        """(positional args, keyword args) of BrainQuery's forward, and the token-map pairs ``task_loss`` needs."""
        seg_tok, slab_pair = self._token_pairs(b)
        tok_pair = torch.cat([seg_tok, slab_pair]) if len(seg_tok) or len(slab_pair) else None
        kw = dict(meta=b["meta"], dense_pair=b["dense_pair"], tok_pair=tok_pair,
                  study=b.get("study"))
        return (b["x"], b["keep"], b["task"], b["seg_pair"]), kw, (seg_tok, slab_pair)

    def task_loss(self, out: dict, b: dict, pairs: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(balanced total, per-task losses, has) of one task batch."""
        seg_tok, slab_pair = pairs
        loss, has = slot_losses(out, b["task"], b["y"], self.raw.task_type, b["seg"], b["seg_pair"], self.stride,
                                dense=b, sparse=self.sparse, cell_zyx=self.cell_zyx, pos_weight=self.pos_weight,
                                negatives=self.negatives)
        if out.get("tok") is not None:
            loss = self._token_losses(b, out["tok"], loss, len(seg_tok), slab_pair)
        total, per_task = self.balancer(loss, has, b["task"])
        return total, per_task, has

    def apply(self, total: torch.Tensor, step: int) -> None:
        """Backward, clip, and the update; a non-finite loss or gradient (one bad batch) skips the update instead of
        writing NaN into every weight. The optimizer's step is taken on every rank alike (the norm is of DDP-averaged
        gradients)."""
        total.backward()
        self.balancer.sync_grads()
        norm = torch.nn.utils.clip_grad_norm_(self.raw.parameters(), self.clip)
        ok = torch.isfinite(norm) & torch.isfinite(total.detach())
        if bool(ok):
            self.opt.step()
        else:
            self.opt.zero_grad(set_to_none=True)
            self.n_skipped = getattr(self, "n_skipped", 0) + 1
            if self.n_skipped <= 5 or self.n_skipped % 100 == 0:
                print(f"[nan] step {step}: non-finite loss / gradient, update skipped ({self.n_skipped} so far)",
                      flush=True)

    def log_tasks(self, per_task: torch.Tensor, has: torch.Tensor, b: dict) -> None:
        present = torch.zeros_like(per_task, dtype=torch.bool)
        present[b["task"][has]] = True
        for of, acc, cnt in ((self.group_of, self.g_sum, self.g_cnt), (self.type_of, self.t_sum, self.t_cnt)):
            g = of[present]
            g_loss = torch.zeros_like(acc).index_add(0, g, per_task[present])
            g_n = torch.zeros_like(acc).index_add(0, g, torch.ones_like(g, dtype=torch.float32))
            on = g_n > 0
            acc += torch.where(on, g_loss / g_n.clamp_min(1), torch.zeros_like(g_loss))
            cnt += on.float()

    def _token_pairs(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """The slots whose token maps train this step: every segmentation slot (token_weight) and the RSNA haemorrhage
        slots of rows with slice labels (slice_weight)."""
        none = torch.zeros(0, 2, dtype=torch.long, device=b["task"].device)
        seg = b["seg_pair"] if self.tok_w else none
        if not (self.slice_w or self.box_w) or b.get("slab") is None:
            return seg, none
        sub = torch.where(b["task"] >= 0, self.slab_sub[b["task"].clamp_min(0)], torch.full_like(b["task"], -1))
        labelled = (b["slab"] >= 0).any(-1)                                   # (B, channels)
        ok = (sub >= 0) & labelled.gather(1, sub.clamp_min(0)) & (self.mil_w[sub.clamp_min(0)] > 0)
        return seg, ok.nonzero()

    def _token_losses(self, b: dict, tok: torch.Tensor, loss: torch.Tensor, n_seg: int,
                      slab_pair: torch.Tensor) -> torch.Tensor:
        """Adds each token-map loss to its slot's loss (so the balancer weighs it with its task)."""
        if n_seg:
            pz, py, px = reversed(self.patch)
            occ, valid = token_occupancy(b["seg"], (pz, py, px))
            rows, slots = b["seg_pair"][:, 0], b["seg_pair"][:, 1]
            lt = token_occupancy_loss(tok[:n_seg], occ, valid & b["keep"][rows])
            loss = loss.index_put((rows, slots), self.tok_w * lt, accumulate=True)
            self.aux_sum[0] += lt.detach().mean()
            self.aux_cnt[0] += 1
        if len(slab_pair):
            r, k = slab_pair[:, 0], slab_pair[:, 1]
            sub = self.slab_sub[b["task"][r, k]]
            lab = b["slab"][r, sub].long()
            # boxes are not exhaustive: on a scan positive for the finding, tokens outside its boxes are unlabelled
            box_pos = (sub >= len(SLAB_TASKS)) & (b["y"][r, k] > 0.5)
            lab = torch.where(box_pos.unsqueeze(1) & (lab == 0), torch.full_like(lab, -1), lab)
            ls = slab_mil_loss(tok[n_seg:], lab, b["keep"][r])
            loss = loss.index_put((r, k), self.mil_w[sub] * ls, accumulate=True)
            self.aux_sum[1] += ls.detach().mean()
            self.aux_cnt[1] += 1
        return loss

    def log_extras(self) -> dict[str, str]:
        """Raw per-group, then per-type losses over the logging window (``-`` for one not asked in it); resets the
        window."""
        out = {}
        for acc, cnt, cols in ((self.g_sum, self.g_cnt, self.log_groups), (self.t_sum, self.t_cnt, self.log_types)):
            mean, n = (acc / cnt.clamp_min(1)).tolist(), cnt.tolist()
            acc.zero_()
            cnt.zero_()
            out.update({name: (f"{mean[i]:.4f}" if n[i] else "-") for i, name in cols})
        if self.tok_w or self.slice_w or self.box_w:   # raw token-map losses: occupancy, slice / box bags
            mean, n = (self.aux_sum / self.aux_cnt.clamp_min(1)).tolist(), self.aux_cnt.tolist()
            self.aux_sum.zero_()
            self.aux_cnt.zero_()
            out.update({c: (f"{mean[i]:.4f}" if n[i] else "-") for i, c in enumerate(("tokmap", "slabmil"))})
        return out

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        res = run_eval(self.raw, self.val_loader, self.cfg, self.reg, self.stats, self.rt.device, self.patch, self.amp)
        return {f"val_{k}": v for k, v in res.items()}

    def eval_lines(self, vals: dict[str, float]) -> list[str]:
        return format_eval(vals, self.reg)

    def state(self, model_only: bool) -> dict:
        s = {"model": self.raw.backbone.state_dict(), "multitask": self.raw.state_dict(),
             "balancer": self.balancer.state_dict(), "label_stats": self.stats, "tasks": [t.id for t in self.reg.tasks]}
        if not model_only:
            s["optimizer"] = self.opt.state_dict()
        return s
