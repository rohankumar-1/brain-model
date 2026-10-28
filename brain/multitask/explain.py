"""What a multitask checkpoint looks at, per task, and which tasks behave alike.

    python -m brain.multitask.explain heatmaps --ckpt CKPT --task hemorrhage_any --dataset bhsd --n 8
    python -m brain.multitask.explain tasks --ckpt CKPT --n 48

``heatmaps``: for one task on scans with a lesion mask, two token maps over the 16 x 16 x 8 mm grid, upsampled onto the
1 mm volume: the task slot's cross-attention (each layer, averaged over heads; the CLS key is reported separately)
and gradient x token (|sum_d dy/dtoken_d * token_d|: how much each token moves the task's output). With a mask, each
map's lesion enrichment = (share of the map's mass inside the lesion) / (lesion's share of the brain tokens): 1 means
no preference, > 1 means the map concentrates on the lesion. Writes ``<out>/heatmaps_<task>.png`` and ``.csv``.

``tasks``: every trained task asked on its own on the same scans (as the frozen suite's slot readout does) -> task x
task similarity of (a) last-layer attention maps (mean cosine over scans), (b) slot vectors (linear CKA across scans),
(c) the learned task embeddings (cosine), each hierarchically clustered, plus a PCA of the task embeddings coloured by
group. Writes ``<out>/tasks_*.png`` and the matrices as ``.csv``.

``tokens``: token features as colour. The encoder's tokens on a fixed set of scans (CT hemorrhage, CT, glioma FLAIR,
adult / elderly / infant T1-T2), projected onto their top three principal components (one basis per checkpoint, fitted
on the foreground tokens of all the scans, so colours compare across scans) and shown as RGB in three planes. Works on
any 3D checkpoint (SSL or multitask); pass several to compare. Writes ``<out>/tokens_<name>.png``.

``dynamics``: the learned task embeddings across a run's snapshots (``step_*.pt``), each aligned to the last by an
orthogonal Procrustes rotation and projected on the last one's top two PCs, plus the within- vs between-group cosine
ratio per step (does group structure emerge, and when). Writes ``<out>/dynamics.png`` and ``dynamics.gif``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from brain.downstream.tasks import DEFAULT_VOLUMES
from brain.multitask.data import gate_and_window
from brain.prep import store

ROOT = DEFAULT_VOLUMES.parent


# ------------------------------------------------------------------------------------------------ core
def encode(model, cfg, image_zyx: np.ndarray, modality: str, device) -> tuple[dict, torch.Tensor, torch.Tensor]:
    raw = torch.from_numpy(np.ascontiguousarray(image_zyx, np.float32))[None, None].to(device)
    mod = torch.tensor([0 if str(modality).lower().startswith("ct") else 1], device=device)
    gate = (cfg.get("views") or {}).get("gate") or {}
    x, keep = gate_and_window(raw, mod, tuple(model.backbone.patch_xyz), gate,
                              in_chans=model.backbone.patch_embed.proj.in_channels)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        enc = model.encode(x, keep)
    enc = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in enc.items()}
    return enc, keep, x


def slots_with_attention(model, enc: dict, task: torch.Tensor, keep: torch.Tensor):
    """``model.slots`` replayed with attention weights kept: returns the slot outputs (B, K, W) and per-layer
    cross-attention (L, B, K, n_kv) averaged over heads (key 0 is the CLS token when the encoder has one). Slot-readout
    checkpoints only: a task-token model has no slot layers to replay."""
    if model.tokens:
        raise SystemExit("explain reads slot attention: this checkpoint uses task tokens (no slot layers)")
    enc = dict(enc)
    enc.pop("_kv", None)
    kv, pad = model._kv(enc, keep)
    q = model.task_queries(task)
    slot_pad = task < 0
    attn = []
    for layer in model.layers:
        h = layer.n1(q)
        out, w = layer.cross(h, kv, kv, key_padding_mask=pad, need_weights=True, average_attn_weights=True)
        attn.append(w)
        q = q + out
        h = layer.n2(q)
        mask = None if layer.cross_task else ~torch.eye(q.shape[1], dtype=torch.bool, device=q.device)
        q = q + layer.self_attn(h, h, h, key_padding_mask=None if mask is not None else slot_pad, attn_mask=mask,
                                need_weights=False)[0]
        q = q + layer.mlp(layer.n3(q))
    return model.out_norm(q), torch.stack(attn)


def task_scalar(model, z: torch.Tensor, task: torch.Tensor, kind: str) -> torch.Tensor:
    """The task's scalar output per slot: binary logit, regression value, or the predicted class's logit."""
    out = model.heads(z, task)
    if kind == "binary":
        return out["binary"]
    if kind == "regression":
        return out["regression"]
    if kind == "multiclass":
        lg = out["multiclass"]
        return lg.gather(-1, lg.argmax(-1, keepdim=True))[..., 0]
    raise ValueError(f"no scalar output for {kind} tasks")


def grad_x_token(model, enc: dict, keep: torch.Tensor, t: int, kind: str) -> np.ndarray:
    """|sum_d dy/dtoken_d * token_d| per token for task ``t`` asked alone."""
    tokens = enc["tokens"].detach().clone().requires_grad_(True)
    e = {**enc, "tokens": tokens}
    e.pop("_kv", None)
    task = torch.tensor([[t]], device=tokens.device)
    with torch.enable_grad():
        z = model.slots(e, task, keep)
        y = task_scalar(model, z, task, kind).sum()
        (g,) = torch.autograd.grad(y, tokens)
    return (g * tokens).sum(-1).abs()[0].detach().cpu().numpy()


def token_grid(model, shape_zyx) -> tuple[int, int, int]:
    """Token grid (Z, Y, X) of a BrainQuery model or a bare 3D encoder."""
    pz, py, px = reversed(getattr(model, "backbone", model).patch_xyz)
    return shape_zyx[0] // pz, shape_zyx[1] // py, shape_zyx[2] // px


def to_volume(tok: np.ndarray, grid, shape_zyx) -> np.ndarray:
    t = torch.from_numpy(np.asarray(tok, np.float32).reshape(grid))[None, None]
    return F.interpolate(t, size=tuple(shape_zyx), mode="trilinear", align_corners=False)[0, 0].numpy()


def mask_tokens(mask_zyx: np.ndarray, grid) -> np.ndarray:
    """Fraction of each token's voxels inside ``mask``."""
    g = torch.from_numpy(mask_zyx.astype(np.float32))[None, None]
    return F.adaptive_avg_pool3d(g, grid)[0, 0].numpy().reshape(-1)


def enrichment(weights: np.ndarray, lesion_frac: np.ndarray, valid: np.ndarray) -> float:
    w, f = weights[valid], lesion_frac[valid]
    if w.sum() <= 0 or f.sum() <= 0:
        return float("nan")
    return float((w * f).sum() / w.sum() / (f.sum() / valid.sum()))


def encode_backbone(backbone, x: torch.Tensor, keep: torch.Tensor) -> dict:
    """``BrainQuery.encode`` for a bare 3D encoder (SSL checkpoints)."""
    from brain.data.coords import crop_bbox_mm, patch_size_mm, token_centers_mm

    size = (x.shape[4], x.shape[3], x.shape[2])
    bbox = crop_bbox_mm((0, 0, 0), size, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)).to(x.device)
    coords, pmm = token_centers_mm(bbox, size, backbone.patch_xyz), patch_size_mm(bbox, size, backbone.patch_xyz)
    b = x.shape[0]
    return backbone(x.permute(0, 1, 4, 3, 2), coords.unsqueeze(0).expand(b, -1, -1), pmm.unsqueeze(0).expand(b, -1),
                    keep=keep)


def load_backbone(path: Path, device):
    """The 3D encoder of any checkpoint (BrainQuery keeps it under ``model`` too) and its gate config."""
    from brain.downstream.embed3d import load_brain3d

    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc = load_brain3d(ck, device).model.to(device).eval().requires_grad_(False)
    return enc, ((ck.get("config") or {}).get("views") or {}).get("gate") or {}


# ------------------------------------------------------------------------------------------------ token colours
TOKEN_SCANS = (("bhsd", "CT hemorrhage"), ("rsna_ich", "CT"), ("brats2023_gli_t2f", "glioma FLAIR"),
               ("fomo_nki_t1w", "adult T1"), ("fomo_wayne_t1w", "older adult T1"), ("fomo_infant_t2w", "infant T2"))


def tokens(a) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
    picks = []
    for ds, label in TOKEN_SCANS:
        r = v[v["dataset"].eq(ds) & v["split"].isin(["test", "val"]) & v["brain_ok"].astype(str).str.lower().eq("true")]
        if len(r):
            picks.append((label, r.sample(1, random_state=a.seed).iloc[0]))
    imgs = [(label, r, np.asarray(store.read_image(ROOT / r.image_path), np.float32)) for label, r in picks]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ck in a.ckpt:
        name = Path(ck).parent.name + "_" + Path(ck).stem
        enc_net, gate = load_backbone(Path(ck), dev)
        feats = []
        for label, r, img in imgs:
            raw = torch.from_numpy(img)[None, None].to(dev)
            mod = torch.tensor([0 if str(r.modality).lower().startswith("ct") else 1], device=dev)
            x, keep = gate_and_window(raw, mod, tuple(enc_net.patch_xyz), gate,
                                      in_chans=enc_net.patch_embed.proj.in_channels)
            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                e = encode_backbone(enc_net, x, keep)
            feats.append((e["tokens"][0].float().cpu().numpy(), keep[0].cpu().numpy().astype(bool)))
        # one PCA basis for all scans, or (--per-modality) one for the CT scans and one for the MRI scans, so the
        # colours spend their three components on anatomy rather than on the CT / MRI difference
        is_ct = [str(r.modality).lower().startswith("ct") for _, r, _ in imgs]
        bases = [[i for i in range(len(imgs)) if is_ct[i] == m] for m in (True, False)] if a.per_modality \
            else [list(range(len(imgs)))]
        proj, lo, hi, var = [None] * len(imgs), [None] * len(imgs), [None] * len(imgs), []
        for idx in (b for b in bases if b):
            fg = np.concatenate([feats[i][0][feats[i][1]] for i in idx])
            mu = fg.mean(0)
            sub = fg[np.random.default_rng(0).choice(len(fg), min(len(fg), 40000), replace=False)] - mu
            _, sv, vt = np.linalg.svd(sub, full_matrices=False)
            pr = {i: (feats[i][0] - mu) @ vt[:3].T for i in idx}
            allp = np.concatenate([pr[i][feats[i][1]] for i in idx])
            l, h = np.percentile(allp, 2, axis=0), np.percentile(allp, 98, axis=0)
            for i in idx:
                proj[i], lo[i], hi[i] = pr[i], l, h
            var.append(sv[:3] ** 2 / (sv ** 2).sum())
        var = np.mean(var, axis=0)
        fig, axes = plt.subplots(len(imgs), 6, figsize=(18, 3.1 * len(imgs)), facecolor="black")
        for row, (label, r, img), p, l, h, (_, keep) in zip(axes, imgs, proj, lo, hi, feats):
            grid = token_grid(enc_net, img.shape)
            rgb = np.clip((p - l) / (h - l + 1e-6), 0, 1)
            rgb[~keep] = 0
            vol = torch.from_numpy(rgb.reshape(*grid, 3)).permute(3, 0, 1, 2)[None]
            vol = F.interpolate(vol, size=img.shape, mode="nearest")[0].permute(1, 2, 3, 0).numpy()   # (Z, Y, X, 3)
            ct = str(r.modality).lower().startswith("ct")
            lo_i, hi_i = (0, 80) if ct else np.percentile(img[img > img.min()], [1, 99.5])
            z, y, x = img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2
            planes = (("axial", img[z], vol[z]), ("coronal", img[:, y], vol[:, y]), ("sagittal", img[:, :, x], vol[:, :, x]))
            for j, (pl, im, col) in enumerate(planes):
                row[2 * j].imshow(im, cmap="gray", vmin=lo_i, vmax=hi_i, origin="lower")
                row[2 * j + 1].imshow(col, origin="lower")
                row[2 * j].set_title(f"{label} -- {pl}" if j == 0 else pl, color="white", fontsize=9)
                row[2 * j + 1].set_title("token PCA (RGB)", color="white", fontsize=9)
            for ax in row:
                ax.axis("off")
        basis = "one basis for CT, one for MRI" if a.per_modality else "one basis for all scans"
        fig.suptitle(f"{name}: encoder tokens, top-3 principal components as RGB (variance {var[0]:.0%} / {var[1]:.0%} / "
                     f"{var[2]:.0%}); {basis}; black = background tokens", color="white")
        fig.tight_layout()
        tag = "_permod" if a.per_modality else ""
        fig.savefig(out / f"tokens_{name}{tag}.png", dpi=70, facecolor="black")
        plt.close(fig)
        print("wrote", out / f"tokens_{name}{tag}.png", flush=True)


# ------------------------------------------------------------------------------------------------ dynamics
TASK_EMB_STEPS = "task_emb_steps.npz"


def task_emb_steps(run: Path) -> tuple[list[int], list[np.ndarray], list[str], str | None]:
    """The task embeddings at each ``step_*.pt`` snapshot of a run, or from ``task_emb_steps.npz`` (written by
    ``save_task_emb_steps``) once the snapshots are deleted: (steps, embeddings, task ids, registry path)."""
    import re

    if (run / TASK_EMB_STEPS).exists():
        z = np.load(run / TASK_EMB_STEPS)
        return z["steps"].tolist(), list(z["emb"]), z["tasks"].tolist(), str(z["registry"]) or None
    steps = sorted(run.glob("step_*.pt"), key=lambda q: int(re.findall(r"\d+", q.stem)[0]))
    if len(steps) < 2:
        raise SystemExit(f"need >= 2 step_*.pt snapshots or {TASK_EMB_STEPS} in {run}")
    embs, ids, cfg = [], None, None
    for q in steps:
        ck = torch.load(q, map_location="cpu", weights_only=False, mmap=True)
        ids = ids or list(ck["tasks"])
        cfg = cfg or ck["config"]
        embs.append(ck["multitask"]["task_emb.weight"].float().numpy())
    return [int(re.findall(r"\d+", q.stem)[0]) for q in steps], embs, ids, cfg["data"].get("registry")


def save_task_emb_steps(run: Path) -> Path:
    steps, embs, ids, registry = task_emb_steps(run)
    np.savez(run / TASK_EMB_STEPS, steps=np.array(steps), emb=np.stack(embs), tasks=np.array(ids),
             registry=np.array(registry or ""))
    return run / TASK_EMB_STEPS


def dynamics(a) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from brain.multitask.registry import load_registry

    step_n, embs, ids, registry = task_emb_steps(Path(a.run))
    reg = load_registry(registry).subset(ids)
    keep = [i for i, t in enumerate(reg.tasks) if t.trained]
    names = [reg.tasks[i].id for i in keep]
    groups = [reg.tasks[i].group for i in keep]
    E = [e[keep] - e[keep].mean(0) for e in embs]
    final = E[-1]
    aligned = []
    for e in E:                                             # orthogonal Procrustes onto the final embeddings
        u, _, vt = np.linalg.svd(e.T @ final)
        aligned.append(e @ u @ vt)
    _, sv, vt = np.linalg.svd(final, full_matrices=False)
    xy = [al @ vt[:2].T for al in aligned]
    same = np.array([[groups[i] == groups[j] and i != j for j in range(len(keep))] for i in range(len(keep))])
    diff = np.array([[groups[i] != groups[j] for j in range(len(keep))] for i in range(len(keep))])
    ratio = []
    for e in embs:
        en = e[keep] / np.linalg.norm(e[keep], axis=1, keepdims=True)
        s = en @ en.T
        ratio.append(s[same].mean() - s[diff].mean())
    gcol = {g: matplotlib.colormaps["tab10"](i % 10) for i, g in enumerate(dict.fromkeys(groups))}
    cols = [gcol[g] for g in groups]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lim = np.abs(np.concatenate(xy)).max() * 1.05
    n = len(xy)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4.2))
    for ax, p, st in zip(axes, xy, step_n):
        ax.scatter(p[:, 0], p[:, 1], c=cols, s=14)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_title(f"step {st}"); ax.set_xticks([]); ax.set_yticks([])
    axes[-1].plot(step_n, ratio, "o-")
    axes[-1].set_title("group separation\n(within - between cosine)")
    axes[-1].set_xlabel("step")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in gcol.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(gcol), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "dynamics.png", dpi=80)
    fig2, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(xy[0][:, 0], xy[0][:, 1], c=cols, s=30)
    labels = [ax.annotate(nm, xy[0][k], fontsize=6, alpha=0.7) for k, nm in enumerate(names)]
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    frames = 12 * (n - 1) + 1                              # 12 interpolated frames between snapshots

    def draw(f):
        i, t = min(f // 12, n - 2), (f % 12) / 12 if f < frames - 1 else 1.0
        p = (1 - t) * xy[i] + t * xy[i + 1]
        sc.set_offsets(p)
        for k, lb in enumerate(labels):
            lb.set_position(p[k])
        ax.set_title(f"task embeddings, step ~{int((1 - t) * step_n[i] + t * step_n[i + 1])}")
        return [sc, *labels]

    FuncAnimation(fig2, draw, frames=frames, blit=False).save(out / "dynamics.gif", writer=PillowWriter(fps=8))
    print(f"[dynamics] {n} snapshots {step_n}; group separation {np.round(ratio, 3).tolist()}; wrote {out}", flush=True)


# ------------------------------------------------------------------------------------------------ heatmaps
def heatmaps(a) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from brain.multitask.fewshot import load_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, reg, cfg = load_model(Path(a.ckpt), dev)
    t = reg.index[a.task]
    kind = reg[t].type
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
    rows = v[v["dataset"].eq(a.dataset) & v["split"].isin(a.splits.split(","))]
    mcol = a.mask_col
    if mcol:
        rows = rows[rows[mcol].fillna("").astype(str).str.len() > 0]
    rows = rows.sample(min(a.n, len(rows)), random_state=a.seed)
    recs, panels = [], []
    for r in rows.itertuples():
        img = np.asarray(store.read_image(ROOT / r.image_path), np.float32)
        enc, keep, _ = encode(model, cfg, img, r.modality, dev)
        grid = token_grid(model, img.shape)
        task = torch.tensor([[t]], device=dev)
        with torch.no_grad():
            z, attn = slots_with_attention(model, enc, task, keep)
            y = float(task_scalar(model, z, task, kind)[0, 0])
        has_cls = enc.get("cls") is not None
        att = attn[:, 0, 0].float().cpu().numpy()                        # (L, n_kv)
        cls_w = att[:, 0] if has_cls else np.zeros(len(att))
        att_tok = att[:, 1:] if has_cls else att
        gxt = grad_x_token(model, enc, keep, t, kind)
        valid = keep[0].cpu().numpy().astype(bool)
        rec = {"volume_id": r.volume_id, "output": y, "cls_attention_last": float(cls_w[-1])}
        lesion = None
        if mcol:
            m = store.read_image(ROOT / getattr(r, mcol), "mask") > 0
            lesion = m
            lf = mask_tokens(m, grid)
            rec["lesion_ml"] = float(m.sum() / 1000)
            for i in range(len(att_tok)):
                rec[f"enrich_attn_L{i}"] = enrichment(att_tok[i], lf, valid)
            rec["enrich_gradxtoken"] = enrichment(gxt, lf, valid)
        recs.append(rec)
        panels.append((r.volume_id, img, to_volume(att_tok.mean(0), grid, img.shape), to_volume(gxt, grid, img.shape),
                       lesion, rec))
        print(f"[heatmaps] {r.volume_id} " + " ".join(f"{k}={v:.3g}" for k, v in rec.items() if k != "volume_id"),
              flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(recs)
    df.to_csv(out / f"heatmaps_{a.task}.csv", index=False)
    print(df.drop(columns="volume_id").describe().round(3).to_string(), flush=True)

    ct = rows["modality"].astype(str).str.lower().str.startswith("ct").all()
    fig, axes = plt.subplots(len(panels), 4, figsize=(16, 4 * len(panels)), facecolor="black")
    axes = np.atleast_2d(axes)
    for row, (vid, img, att_v, gxt_v, lesion, rec) in zip(axes, panels):
        zs = np.nonzero(lesion.any(axis=(1, 2)))[0] if lesion is not None and lesion.any() else []
        z = int(np.argmax(lesion.sum(axis=(1, 2)))) if len(zs) else int(np.argmax(gxt_v.sum(axis=(1, 2))))
        base = np.clip((img[z] - 0) / 80, 0, 1) if ct else img[z]
        for ax in row:
            ax.imshow(base, cmap="gray", origin="lower")
            ax.axis("off")
        if lesion is not None:
            row[1].contour(lesion[z], [0.5], colors="#00e5a0", linewidths=0.8)
        row[0].set_title(f"{vid.split(':')[-1][:24]}\n{a.task} output {rec['output']:.2f}", color="white", fontsize=9)
        row[1].set_title("lesion mask" if lesion is not None else "", color="white", fontsize=9)
        for ax, vol, name in ((row[2], att_v, "attention (mean over layers)"), (row[3], gxt_v, "gradient x token")):
            sl = vol[z]
            ax.imshow(sl / (sl.max() + 1e-12), cmap="inferno", alpha=0.55, origin="lower")
            if lesion is not None:
                ax.contour(lesion[z], [0.5], colors="#00e5a0", linewidths=0.8)
            key = "enrich_gradxtoken" if "gradient" in name else f"enrich_attn_L{len(model.layers) - 1}"
            ax.set_title(f"{name}\nlesion enrichment {rec.get(key, float('nan')):.2f}", color="white", fontsize=9)
    fig.suptitle(f"{a.task}: where the task slot looks (attention) and what moves its answer (gradient x token)",
                 color="white")
    fig.tight_layout()
    fig.savefig(out / f"heatmaps_{a.task}.png", dpi=70, facecolor="black")
    print("wrote", out / f"heatmaps_{a.task}.png", flush=True)


# ------------------------------------------------------------------------------------------------ task similarity
def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(0), y - y.mean(0)
    num = np.linalg.norm(x.T @ y) ** 2
    den = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(num / den) if den > 0 else float("nan")


def tasks(a) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    from brain.multitask.fewshot import load_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, reg, cfg = load_model(Path(a.ckpt), dev)
    trained = [reg.index[t.id] for t in reg.trained]
    names = [reg[t].id for t in trained]
    groups = [reg[t].group for t in trained]
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
    pool = v[v["pretrain"].astype(bool) & v["split"].eq("test")]
    # a mix of modalities and cohorts: n scans, at most n // 6 per dataset family
    pool = pool.groupby("dataset_family", group_keys=False).apply(lambda d: d.sample(min(len(d), max(1, a.n // 6)),
                                                                                    random_state=a.seed))
    rows = pool.sample(min(a.n, len(pool)), random_state=a.seed)
    task = torch.tensor(trained, device=dev).unsqueeze(1)                 # (T, 1): one task per query
    att_maps, slot_vecs = [], []
    for i, r in enumerate(rows.itertuples()):
        img = np.asarray(store.read_image(ROOT / r.image_path), np.float32)
        enc, keep, _ = encode(model, cfg, img, r.modality, dev)
        T = task.shape[0]
        rep = {"tokens": enc["tokens"].expand(T, -1, -1), "coords": enc["coords"],
               "cls": enc["cls"].expand(T, -1) if enc.get("cls") is not None else None,
               "pooled": enc["pooled"].expand(T, -1)}
        with torch.no_grad():
            z, attn = slots_with_attention(model, rep, task, keep.expand(T, -1))
        last = attn[-1, :, 0].float()                                     # (T, n_kv), last layer
        if enc.get("cls") is not None:
            last = last[:, 1:]
        last = last / last.sum(-1, keepdim=True).clamp_min(1e-12)
        att_maps.append(last.cpu().numpy())
        slot_vecs.append(z[:, 0].float().cpu().numpy())
        if (i + 1) % 8 == 0:
            print(f"[tasks] {i + 1}/{len(rows)}", flush=True)
    A = np.stack(att_maps)                                                # (S, T, N)
    Z = np.stack(slot_vecs)                                               # (S, T, W)
    T = len(names)
    An = A / np.linalg.norm(A, axis=-1, keepdims=True).clip(1e-12)
    att_sim = np.einsum("stn,sun->tu", An, An) / A.shape[0]
    cka = np.array([[linear_cka(Z[:, i], Z[:, j]) for j in range(T)] for i in range(T)])
    E = model.task_emb.weight[trained].float().cpu().numpy()
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    emb_sim = En @ En.T
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gcol = {g: matplotlib.colormaps["tab10"](i % 10) for i, g in enumerate(dict.fromkeys(groups))}

    def clustered(sim, title, fname):
        d = np.clip(1 - sim, 0, None)
        np.fill_diagonal(d, 0)
        order = leaves_list(linkage(squareform((d + d.T) / 2, checks=False), "average"))
        pd.DataFrame(sim, index=names, columns=names).to_csv(out / f"{fname}.csv")
        fig, ax = plt.subplots(figsize=(14, 13))
        ax.imshow(sim[np.ix_(order, order)], cmap="viridis")
        ax.set_xticks(range(T), [names[k] for k in order], rotation=90, fontsize=6)
        ax.set_yticks(range(T), [names[k] for k in order], fontsize=6)
        for tick, k in zip(ax.get_yticklabels(), order):
            tick.set_color(gcol[groups[k]])
        for tick, k in zip(ax.get_xticklabels(), order):
            tick.set_color(gcol[groups[k]])
        ax.set_title(title)
        fig.colorbar(ax.images[0], ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(out / f"{fname}.png", dpi=90)
        plt.close(fig)

    clustered(att_sim, f"Attention-map similarity (last layer, mean cosine over {len(rows)} test scans)", "tasks_attention")
    clustered(cka, f"Slot-vector similarity (linear CKA over {len(rows)} test scans)", "tasks_cka")
    clustered(emb_sim, "Learned task-embedding similarity (cosine)", "tasks_embedding")
    pc = np.linalg.svd(E - E.mean(0), full_matrices=False)
    xy = (E - E.mean(0)) @ pc[2][:2].T
    fig, ax = plt.subplots(figsize=(11, 9))
    for g in gcol:
        idx = [k for k in range(T) if groups[k] == g]
        ax.scatter(xy[idx, 0], xy[idx, 1], color=gcol[g], label=g, s=40)
    for k in range(T):
        ax.annotate(names[k], xy[k], fontsize=6, alpha=0.8)
    var = pc[1] ** 2 / (pc[1] ** 2).sum()
    ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax.legend(fontsize=8)
    ax.set_title("Task embeddings, PCA")
    fig.tight_layout()
    fig.savefig(out / "tasks_embedding_pca.png", dpi=90)
    # within- vs between-group similarity, per view: is there group structure at all?
    same = np.array([[groups[i] == groups[j] and i != j for j in range(T)] for i in range(T)])
    diff = np.array([[groups[i] != groups[j] for j in range(T)] for i in range(T)])
    for name, s in (("attention", att_sim), ("cka", cka), ("embedding", emb_sim)):
        print(f"[tasks] {name:9s} mean similarity within group {s[same].mean():.3f} | between groups {s[diff].mean():.3f}",
              flush=True)
    print("wrote", out, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("heatmaps")
    h.add_argument("--ckpt", required=True)
    h.add_argument("--task", required=True)
    h.add_argument("--dataset", required=True, help="volumes.csv dataset, e.g. bhsd, hemseg, brats2023_gli_t1c")
    h.add_argument("--mask-col", default="mask_ich", help="volumes.csv mask column for the enrichment ('' = none)")
    h.add_argument("--splits", default="test,val")
    h.add_argument("--n", type=int, default=8)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--out", default="runs/diagnostics/explain")
    t = sub.add_parser("tasks")
    t.add_argument("--ckpt", required=True)
    t.add_argument("--n", type=int, default=48)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="runs/diagnostics/explain")
    k = sub.add_parser("tokens")
    k.add_argument("--ckpt", nargs="+", required=True, help="one or more 3D checkpoints (SSL or multitask)")
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--per-modality", action="store_true", help="separate PCA bases for CT and MRI scans")
    k.add_argument("--out", default="runs/diagnostics/explain")
    d = sub.add_parser("dynamics")
    d.add_argument("--run", required=True, help="a run directory with step_*.pt snapshots")
    d.add_argument("--out", default="runs/diagnostics/explain")
    a = ap.parse_args()
    {"heatmaps": heatmaps, "tasks": tasks, "tokens": tokens, "dynamics": dynamics}[a.cmd](a)


if __name__ == "__main__":
    main()
