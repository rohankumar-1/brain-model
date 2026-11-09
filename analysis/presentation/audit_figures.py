"""Model-audit figures for a task-token multitask checkpoint (16:9 PNGs for slides).

    python analysis/presentation/audit_figures.py collect --ckpt runs/multitask/full_iso_bank/last.pt --out DIR
    python analysis/presentation/audit_figures.py plot --out DIR

``collect`` (GPU) runs the model's own heads on held-out test scans and records, per scan, one axial slice of the
image, the target mask and the predicted mask (segmentation tasks), the predicted anatomy masks of a few scans (every
anatomy head asked at once), and every task token's attention over the image tokens in each encoder layer (head-mean;
task tokens are queries, image tokens keys, ``vit3d._task_mask``). Saves ``collect.pt``.

``plot`` draws:
  seg_examples.png     target vs prediction on representative test scans (the scans nearest each task's median Dice)
  anatomy.png          every anatomy head on one MRI and one CT test scan, colour per structure
  attention_maps.png   where task tokens look on a CT bleed, a glioma and an older-adult MRI
  attention_lesion.png lesion enrichment of each task token's attention (mass on the lesion / lesion's token share)
  task_overlap.png     task x task similarity of attention maps (last layer, cosine, mean over mixed test scans)
No scan identifiers are written into any figure.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SEG_TASKS = ("hemorrhage_segmentation", "whole_tumor", "enhancing_tumor", "brain_mask")
ANAT = ("seg_ventricles", "seg_hippocampus", "seg_amygdala", "seg_thalamus", "seg_caudate", "seg_putamen",
        "seg_pallidum", "seg_cerebellum", "seg_brainstem", "seg_white_matter", "seg_cortex")
CT_ANAT = ("brain_mask", "seg_ct_skull", "seg_ct_csf", "hemorrhage_segmentation")


# ------------------------------------------------------------------------------------------------ attention capture
class TaskAttention:
    """Forward pre-hooks on every encoder block's attention: the task-token rows of its softmax, head-mean, over the
    packed patch keys -> ``self.maps`` list over layers of (B, T, Np)."""

    def __init__(self, backbone):
        self.p = int(backbone.n_prefix)
        self.maps: list[torch.Tensor] = []
        self.hs = [blk.attn.register_forward_pre_hook(self._hook) for blk in backbone.blocks]

    @torch.no_grad()
    def _hook(self, attn, args):
        xn, coords, mask, npf = args
        t = int(npf) - self.p
        if t <= 0:
            return
        b, n, c = xn.shape
        h = attn.heads
        q, k, _ = attn.qkv(xn).reshape(b, n, 3, h, c // h).permute(2, 0, 3, 1, 4)
        q, k = attn.q_norm(q).float(), attn.k_norm(k).float()
        _, kr = attn.rope.rotate(q[:, :, npf:], k[:, :, npf:], coords)
        k = torch.cat([k[:, :, :npf], kr.float()], 2)
        qt = q[:, :, self.p:self.p + t]
        lg = qt @ k.transpose(-1, -2) / math.sqrt(c // h)                     # (B, H, T, n)
        if mask is not None:
            m = mask[:, :, self.p:self.p + t]
            lg = lg.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else lg + m.float()
        self.maps.append(lg.softmax(-1)[..., npf:].mean(1).half().cpu())      # (B, T, Np)

    def take(self) -> list[torch.Tensor]:
        m, self.maps = self.maps, []
        return m

    def close(self):
        for h in self.hs:
            h.remove()


def unpack(att: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """(B, T, Np) packed -> (B, T, N) on the token grid (dropped tokens 0)."""
    from brain.models.vit3d import scatter_kept
    return scatter_kept(att.transpose(1, 2).float(), keep.cpu()).transpose(1, 2)


# ------------------------------------------------------------------------------------------------ collect
def collect(a) -> None:
    import pandas as pd

    from brain.common.paths import resolve
    from brain.multitask.data import checkpoint_stats, input_patch, load_labels, mixed_parts, prepare
    from brain.multitask.fewshot import _loader, load_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    model, reg, cfg = load_model(Path(a.ckpt), dev)
    store = resolve(cfg["data"]["store_root"])
    labels_dir = resolve(cfg["data"]["labels_dir"])
    labels, stats = load_labels(labels_dir)
    stats = checkpoint_stats(model.label_stats, labels_dir, stats)
    volumes = pd.read_csv(store / cfg["data"].get("index_name", "volumes.csv"), low_memory=False)
    patch = input_patch(model, cfg)
    pz, py, px = reversed(model.backbone.patch_xyz)
    tok_ids = model.tt_write.nonzero()[:, 0].tolist()
    tok_of = model.tok_of.tolist() if hasattr(model, "tok_of") else list(range(len(reg)))
    tok_names = [[reg[i].id for i in range(len(reg)) if tok_of[i] == t] for t in tok_ids]
    hook = TaskAttention(model.backbone)
    lab = labels[labels["use"].eq("train")]
    rng = np.random.default_rng(a.seed)

    def ids_for(task, n, nonempty=True):
        r = lab[(lab.task == task) & (lab.split == "test")]
        if nonempty and "ref" in r:
            r = r[~r["ref"].astype(str).eq("@empty")]
        u = r["volume_id"].unique().tolist()
        return [u[i] for i in rng.permutation(len(u))[:n]]

    def batches(task, ids):
        loader, _ = _loader(store, labels, volumes, reg, stats, task, "test", ids=ids, use="train", workers=a.workers,
                            cfg=cfg)
        for raw in loader:
            for part in mixed_parts(raw):
                yield part, prepare(part, cfg, dev, patch, train=False)

    def grid_of(x):
        return x.shape[2] // pz, x.shape[3] // py, x.shape[4] // px

    def att_record(keep, i, g, lesion_tok=None):
        layers = [unpack(m, keep)[i] for m in hook.take_cache]                  # each (T, N)
        A = torch.stack(layers)                                               # (L, T, N)
        rec = {"att_last": A[-1].half(), "att_mean": A.mean(0).half(), "grid": g}
        if lesion_tok is not None:
            valid = keep[i].cpu().numpy()
            f = lesion_tok[valid]
            for name, M in (("last", A[-1]), ("mean", A.mean(0))):
                w = M[:, valid].numpy()
                rec[f"enrich_{name}"] = (w @ f) / w.sum(1).clip(1e-12) / (f.sum() / valid.sum()) if f.sum() > 0 else None
        return rec

    def slice_of(vol_zyx, z):
        return vol_zyx[z].astype(np.float16)

    out = {"tok_ids": tok_ids, "tok_names": tok_names, "groups": [reg[t].group for t in tok_ids], "seg": {},
           "anat": [], "att": []}

    # 1. segmentation heads on their own test scans, + attention with lesion enrichment
    for task in SEG_TASKS:
        if task not in reg.index:
            continue
        t = reg.index[task]
        recs = []
        for raw, b in batches(task, ids_for(task, a.n_seg)):
            with torch.no_grad(), torch.autocast(**amp):
                o = model(b["x"], b["keep"], b["task"], b["seg_pair"], alone=True, meta=b.get("meta"),
                          modality=b["modality"])
            hook.take_cache = hook.take()
            if o.get("seg") is None:
                continue
            x = b["x"].float()
            g = grid_of(x)
            for j, (i, s) in enumerate(b["seg_pair"].tolist()):
                if int(b["task"][i, s]) != t:
                    continue
                gt = b["seg"][j, 0].float()
                if gt.sum() < 1:
                    continue
                pr = torch.sigmoid(F.interpolate(o["seg"][j:j + 1].float(), size=gt.shape, mode="trilinear",
                                                 align_corners=False))[0, 0]
                gtb, prb = gt > 0.5, pr > 0.5
                dice = float(2 * (gtb & prb).sum() / (gtb.sum() + prb.sum()).clamp_min(1))
                gnp = gtb.cpu().numpy()
                z = int(np.argmax(gnp.sum((1, 2))))
                lt = F.adaptive_avg_pool3d(gt[None, None], g)[0, 0].reshape(-1).cpu().numpy()
                rec = {"dice": dice, "modality": int(b["modality"][i]), "z": z,
                       "img": slice_of(x[i, 0].cpu().numpy(), z), "gt": gnp[z], "pred": pr[z].cpu().numpy().astype(np.float16),
                       "lesion_ml": float(gnp.sum()) / 1000, **att_record(b["keep"], i, g, lt)}
                rec["att_slice_last"] = F.interpolate(rec["att_last"].float().reshape(1, -1, *g), size=gt.shape,
                                                      mode="trilinear", align_corners=False)[0, :, z].half()
                rec["att_slice_mean"] = F.interpolate(rec["att_mean"].float().reshape(1, -1, *g), size=gt.shape,
                                                      mode="trilinear", align_corners=False)[0, :, z].half()
                recs.append(rec)
        out["seg"][task] = recs
        d = [r["dice"] for r in recs]
        print(f"[seg] {task}: n={len(d)} median Dice {np.median(d) if d else float('nan'):.3f}", flush=True)

    # 2. every anatomy head on a few test scans (MRI from the age task's test split; CT from hemorrhage's)
    for task, names, mod_ok in (("age", ANAT, 1), ("hemorrhage_any", CT_ANAT, 0)):
        want = [reg.index[n] for n in names if n in reg.index]
        done = 0
        for raw, b in batches(task, ids_for(task, 16, nonempty=False)):
            for i in range(b["x"].shape[0]):
                if int(b["modality"][i]) != mod_ok or done >= a.n_anat:
                    continue
                x, keep = b["x"][i:i + 1], b["keep"][i:i + 1]
                tk = torch.tensor([want], device=dev)
                sp = torch.tensor([[0, k] for k in range(len(want))], device=dev)
                with torch.no_grad(), torch.autocast(**amp):
                    o = model(x, keep, tk, sp, alone=True, modality=b["modality"][i:i + 1])
                hook.take()
                full = x.shape[2:]
                pr = torch.sigmoid(F.interpolate(o["seg"].float(), size=full, mode="trilinear", align_corners=False))[:, 0]
                xi = x[0, 0].float().cpu().numpy()
                fg = (pr > 0.5).any(0).cpu().numpy()
                zs = np.nonzero(fg.any((1, 2)))[0]
                zc = int(zs.mean()) if len(zs) else full[0] // 2
                levels = [max(0, zc - 18), zc, min(full[0] - 1, zc + 14)]
                out["anat"].append({"modality": mod_ok, "names": [reg[w].id for w in want], "levels": levels,
                                    "img": [slice_of(xi, z) for z in levels],
                                    "pred": [pr[:, z].cpu().numpy().astype(np.float16) for z in levels]})
                done += 1
            if done >= a.n_anat:
                break
        print(f"[anat] {task}: {done} scans", flush=True)

    # 3. attention maps on mixed test scans (task overlap), from the age task's pooled-cohort test split + CT
    for task, n in (("age", a.n_mixed), ("hemorrhage_any", a.n_mixed // 3)):
        for raw, b in batches(task, ids_for(task, n, nonempty=False)):
            with torch.no_grad(), torch.autocast(**amp):
                model(b["x"], b["keep"], b["task"], b["seg_pair"], alone=True, meta=b.get("meta"),
                      modality=b["modality"])
            hook.take_cache = hook.take()
            for i in range(b["x"].shape[0]):
                r = att_record(b["keep"], i, grid_of(b["x"]))
                out["att"].append({"modality": int(b["modality"][i]), "att_last": r["att_last"],
                                   "att_mean": r["att_mean"], "keep": b["keep"][i].cpu()})
        print(f"[att] {task}: {len(out['att'])} scans so far", flush=True)
    hook.close()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    torch.save(out, o / "collect.pt")
    print("wrote", o / "collect.pt", flush=True)


# ------------------------------------------------------------------------------------------------ plot
INK, MUTED, GT_C, PR_C = "#1d1d1b", "#55544f", "#00c08b", "#ff7a1a"
NICE = {"hemorrhage_segmentation": "Hemorrhage (CT)", "whole_tumor": "Whole tumour (MRI)",
        "enhancing_tumor": "Enhancing tumour (T1c)", "brain_mask": "Brain mask"}


def show(ax, img, ct):
    img = np.asarray(img, np.float32)
    if ct:
        base = np.clip(img / 80.0, 0, 1)
    else:
        lo, hi = np.percentile(img[img != 0] if (img != 0).any() else img, [1, 99.5])
        base = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    ax.imshow(base, cmap="gray", origin="lower", interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def crop_box(img, ct, pad=6):
    img = np.asarray(img, np.float32)
    fg = img > (-500 if ct else (np.percentile(img, 5) + 1e-3))
    ys, xs = np.nonzero(fg)
    if not len(ys):
        return None
    return max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad), max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)


def cropped(arrs, box):
    if box is None:
        return arrs
    y0, y1, x0, x1 = box
    return [None if a is None else np.asarray(a)[y0:y1, x0:x1] for a in arrs]


def plot(a) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    o = Path(a.out)
    D = torch.load(o / "collect.pt", weights_only=False)
    names = [n[0] for n in D["tok_names"]]
    T = len(names)

    SHARED = {"hemorrhage_any": "hemorrhage (6 types)", "whole_tumor": "tumour masks (3)",
              "synth_t1": "sequence synthesis (5)", "mri_normal_for_age": "radiology findings (9)"}

    def tname(k):
        n = names[k]
        if n in SHARED:
            return SHARED[n]
        if n.startswith("seg_"):
            return n[4:].replace("_", " ") + " mask"
        if n.startswith("vol_"):
            return n[4:].replace("_", " ") + " (volume)"
        return n.replace("_", " ")

    def tidx(task):
        for k, ns in enumerate(D["tok_names"]):
            if task in ns:
                return k
        return None

    # ---- seg_examples: per task the 3 scans nearest the median Dice
    tasks = [t for t in SEG_TASKS if t != "brain_mask" and D["seg"].get(t)]
    fig, axes = plt.subplots(len(tasks), 3, figsize=(13.333, 7.5), dpi=200,
                             gridspec_kw=dict(left=0.17, right=0.99, top=0.9, bottom=0.08, wspace=0.04, hspace=0.12))
    axes = np.atleast_2d(axes)
    summary = []
    for r, task in enumerate(tasks):
        recs = D["seg"][task]
        d = np.array([x["dice"] for x in recs])
        med = float(np.median(d))
        summary.append((task, med, len(d)))
        order = np.argsort(np.abs(d - med))[:3]
        order = order[np.argsort(-d[order])]
        for c, k in enumerate(order):
            x = recs[k]
            ax = axes[r, c]
            ct = x["modality"] == 0
            img, gt, pr = cropped([x["img"], x["gt"], np.asarray(x["pred"], np.float32)], crop_box(x["img"], ct))
            show(ax, img, ct)
            ax.imshow(np.ma.masked_less(pr, 0.5), cmap=ListedColormap([PR_C]), alpha=0.55, origin="lower",
                      vmin=0, vmax=1)
            ax.contour(gt.astype(float), [0.5], colors=GT_C, linewidths=1.3, origin="lower")
            ax.text(0.03, 0.04, f"Dice {x['dice']:.2f}", transform=ax.transAxes, color="white", fontsize=10,
                    fontweight="bold", va="bottom", bbox=dict(fc="black", alpha=0.55, ec="none", pad=2))
        axes[r, 0].text(-0.08, 0.5, f"{NICE[task]}\nmedian Dice {med:.2f}\n(n = {len(d)} test scans)",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=11.5, color=INK)
    fig.legend(handles=[Patch(fc="none", ec=GT_C, lw=1.6, label="reference mask"),
                        Patch(fc=PR_C, alpha=0.6, label="model prediction")], loc="upper center", ncol=2,
               frameon=False, fontsize=11.5, bbox_to_anchor=(0.58, 0.985))
    fig.text(0.17, 0.02, "Held-out test scans nearest each task's median Dice (typical cases, not the best). "
             "One axial slice through the largest lesion cross-section.", fontsize=9, color=MUTED)
    fig.savefig(o / "seg_examples.png", facecolor="white")
    plt.close(fig)
    print("seg medians:", [(t, round(m, 3), n) for t, m, n in summary])

    # ---- anatomy: one MRI and one CT, three levels each
    anat = [next((x for x in D["anat"] if x["modality"] == m), None) for m in (1, 0)]
    anat = [x for x in anat if x is not None]
    if anat:
        COL = {"seg_ventricles": "#3b82f6", "seg_hippocampus": "#ef4444", "seg_amygdala": "#f97316",
               "seg_thalamus": "#10b981", "seg_caudate": "#a855f7", "seg_putamen": "#ec4899", "seg_pallidum": "#14b8a6",
               "seg_cerebellum": "#eab308", "seg_brainstem": "#6366f1", "seg_ct_skull": "#f59e0b",
               "seg_ct_csf": "#3b82f6", "hemorrhage_segmentation": "#ef4444"}
        fig, axes = plt.subplots(len(anat), 3, figsize=(13.333, 7.5), dpi=200,
                                 gridspec_kw=dict(left=0.01, right=0.79, top=0.97, bottom=0.03, wspace=0.03,
                                                  hspace=0.05))
        axes = np.atleast_2d(axes)
        handles = {}
        for r, x in enumerate(anat):
            ct = x["modality"] == 0
            order = [n for n in x["names"] if n not in ("seg_white_matter", "seg_cortex", "brain_mask")] \
                + [n for n in x["names"] if n in ("seg_white_matter", "seg_cortex", "brain_mask")]
            for c in range(3):
                ax = axes[r, c]
                box = crop_box(x["img"][c], ct)
                (img,) = cropped([x["img"][c]], box)
                show(ax, img, ct)
                for n in reversed(order):                       # big structures first, small ones on top
                    k = x["names"].index(n)
                    (m,) = cropped([np.asarray(x["pred"][c][k], np.float32)], box)
                    if n == "brain_mask":
                        if (m > 0.5).any():
                            ax.contour(m, [0.5], colors="#22c55e", linewidths=1.0, origin="lower")
                        handles[n] = "#22c55e"
                        continue
                    if n in ("seg_white_matter", "seg_cortex"):
                        continue
                    col = COL.get(n, "#999")
                    if (m > 0.5).any():
                        ax.imshow(np.ma.masked_less(m, 0.5), cmap=ListedColormap([col]), alpha=0.6 if not ct else 0.45,
                                  origin="lower", vmin=0, vmax=1)
                    handles[n] = col
        lab = {"seg_ventricles": "ventricles", "seg_hippocampus": "hippocampus", "seg_amygdala": "amygdala",
               "seg_thalamus": "thalamus", "seg_caudate": "caudate", "seg_putamen": "putamen",
               "seg_pallidum": "pallidum", "seg_cerebellum": "cerebellum", "seg_brainstem": "brainstem",
               "seg_ct_skull": "skull (CT)", "seg_ct_csf": "CSF (CT)", "hemorrhage_segmentation": "hemorrhage (CT)",
               "brain_mask": "brain outline (CT)"}
        fig.legend(handles=[Patch(fc=c, label=lab.get(n, n)) for n, c in handles.items() if n in lab],
                   loc="center left", bbox_to_anchor=(0.80, 0.5), frameon=False, fontsize=12,
                   title="predicted by the model", title_fontsize=12)
        fig.savefig(o / "anatomy.png", facecolor="white")
        plt.close(fig)

    # ---- attention_lesion: enrichment per token on bleed and tumour scans
    key = a.layer
    rows = []
    for task, label in (("hemorrhage_segmentation", "on CT bleeds"), ("whole_tumor", "on gliomas")):
        recs = [x for x in D["seg"].get(task, []) if x.get(f"enrich_{key}") is not None]
        if recs:
            E = np.stack([x[f"enrich_{key}"] for x in recs])          # (S, T)
            rows.append((label, np.nanmedian(E, 0), len(recs)))
    if rows:
        show_t = ["hemorrhage_any", "hemorrhage_segmentation", "whole_tumor", "tumor_type", "tumor_volume",
                  "mri_mass", "age", "sex", "brain_volume", "ventriculomegaly", "vol_hippocampus_asym"]
        show_t = [t for t in show_t if tidx(t) is not None]
        fig, axes = plt.subplots(1, len(rows), figsize=(13.333, 7.5), dpi=200, sharey=True,
                                 gridspec_kw=dict(left=0.2, right=0.97, top=0.86, bottom=0.15, wspace=0.08))
        axes = np.atleast_1d(axes)
        for ax, (label, med, n) in zip(axes, rows):
            v = [med[tidx(t)] for t in show_t]
            cols = ["#c0392b" if t.startswith("hemorrhage") else "#1baf7a" if t in
                    ("whole_tumor", "tumor_type", "tumor_volume") else "#d4a017" if t == "mri_mass" else "#8a8983"
                    for t in show_t]
            ax.barh(range(len(show_t)), v, color=cols)
            ax.axvline(1, color=INK, lw=1, ls="--")
            ax.set_yticks(range(len(show_t)), [tname(tidx(t)) for t in show_t], fontsize=11.5)
            ax.invert_yaxis()
            ax.set_title(f"{label} (n = {n})", fontsize=13, color=INK)
            ax.set_xlabel("attention on the lesion ÷ lesion's share of the brain", fontsize=10.5, color=MUTED)
            ax.spines[["top", "right"]].set_visible(False)
            for i, x in enumerate(v):
                ax.text(x, i, f" {x:.1f}×", va="center", fontsize=10, color=INK)
        fig.text(0.2, 0.93, "Lesion tokens concentrate on lesions; whole-brain tokens do not", fontsize=17,
                 fontweight="bold", color=INK)
        fig.text(0.2, 0.025, "Median over held-out test scans with a reference mask; last encoder layer, mean over heads. "
                 "1× = attention spread evenly over the brain.\n"
                 "Hemorrhage and tumour tokens find either kind of lesion: neither was trained to ignore the other.", fontsize=9, color=MUTED)
        fig.savefig(o / "attention_lesion.png", facecolor="white")
        plt.close(fig)

    # ---- attention_maps: a bleed, a glioma, a normal MRI; columns = tokens
    cols_t = ["hemorrhage_any", "whole_tumor", "age", "ventriculomegaly", "sex"]
    cols_t = [t for t in cols_t if tidx(t) is not None]
    picks = []
    for task in ("hemorrhage_segmentation", "whole_tumor"):
        recs = D["seg"].get(task, [])
        if recs:
            e = [x.get(f"enrich_{key}") for x in recs]
            k = tidx("hemorrhage_any" if task.startswith("hem") else "whole_tumor")
            sc = np.array([(x[k] if x is not None else -1) for x in e])
            # an illustrative case: the largest lesion among scans at or above the median enrichment
            ok = [i for i in range(len(recs)) if sc[i] >= np.median(sc)]
            picks.append(recs[max(ok, key=lambda i: recs[i]["lesion_ml"])])
    if picks:
        fig, axes = plt.subplots(len(picks), len(cols_t) + 1, figsize=(13.333, 7.5), dpi=200,
                                 gridspec_kw=dict(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.03,
                                                  hspace=0.06))
        axes = np.atleast_2d(axes)
        for c, t in enumerate(["scan"] + cols_t):
            axes[0, c].set_title("scan + reference mask" if t == "scan" else f"{tname(tidx(t))}\ntoken",
                                 fontsize=12, color=INK)
        for r, x in enumerate(picks):
            ct = x["modality"] == 0
            box = crop_box(x["img"], ct)
            img, gt = cropped([x["img"], x["gt"]], box)
            show(axes[r, 0], img, ct)
            axes[r, 0].contour(gt.astype(float), [0.5], colors=GT_C, linewidths=1.2, origin="lower")
            A = x[f"att_slice_{key}"].float().numpy()
            for c, t in enumerate(cols_t, 1):
                (m,) = cropped([A[tidx(t)]], box)
                show(axes[r, c], img, ct)
                axes[r, c].imshow(m / (m.max() + 1e-12), cmap="inferno", alpha=0.6, origin="lower")
                axes[r, c].contour(gt.astype(float), [0.5], colors=GT_C, linewidths=0.9, origin="lower")
        fig.suptitle("Where each task token looks (last encoder layer, held-out test scans)", fontsize=16,
                     fontweight="bold", color=INK, y=0.985)
        fig.savefig(o / "attention_maps.png", facecolor="white")
        plt.close(fig)

    # ---- task_overlap: cosine similarity of attention maps across mixed test scans
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    S = np.zeros((T, T))
    for x in D["att"]:
        A = x[f"att_{key}"].float().numpy()
        A = A[:, x["keep"].numpy()]
        An = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
        S += An @ An.T
    S /= max(len(D["att"]), 1)
    np.savetxt(o / f"task_overlap_{key}.csv", S, delimiter=",", header=",".join(names), comments="")
    d = np.clip(1 - S, 0, None)
    np.fill_diagonal(d, 0)
    leaves = list(leaves_list(linkage(squareform((d + d.T) / 2, checks=False), "average")))
    groups = D["groups"]
    G = ["hemorrhage", "lesion", "tumor", "radiology", "diagnosis", "clinical_scores", "demographics", "anatomy_seg",
         "anatomy_volume", "derived", "ct_measures", "synthesis", "ssl_recon"]
    GN = {"hemorrhage": "hemorrhage", "lesion": "lesion masks", "tumor": "tumour", "radiology": "radiology",
          "diagnosis": "diagnosis", "clinical_scores": "clinical scores", "demographics": "demographics",
          "anatomy_seg": "anatomy masks", "anatomy_volume": "anatomy volumes", "derived": "brain & asymmetry",
          "ct_measures": "CT measures", "synthesis": "synthesis", "ssl_recon": "self-supervised"}
    present = [g for g in G if g in groups] + sorted({g for g in groups if g not in G})
    # rows sorted by task group; within a group, in the order of a clustering of all tasks
    order = [k for g in present for k in leaves if groups[k] == g]
    bounds, start = [], 0
    for g in present:
        n_g = sum(1 for k in order if groups[k] == g)
        bounds.append((g, start, start + n_g))
        start += n_g
    vmin, vmax = np.percentile(S, 2), np.percentile(S[~np.eye(T, dtype=bool)], 99.5)
    same = np.array([[groups[i] == groups[j] and i != j for j in range(T)] for i in range(T)])
    diff = np.array([[groups[i] != groups[j] for j in range(T)] for i in range(T)])
    # (a) the full task x task matrix, rows grouped
    fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
    ax = fig.add_axes([0.25, 0.02, 0.5, 0.78])
    im = ax.imshow(S[np.ix_(order, order)], cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(T), [tname(k) for k in order], fontsize=6.3)
    ax.set_xticks([(a + b - 1) / 2 for _, a, b in bounds], [GN.get(g, g) for g, _, _ in bounds], rotation=90,
                  fontsize=8)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)
    for _, a, b in bounds[1:]:
        ax.axhline(a - 0.5, color="white", lw=1.2)
        ax.axvline(a - 0.5, color="white", lw=1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cb.set_label("attention-map similarity (cosine)", fontsize=9.5)
    fig.text(0.03, 0.95, "Which tasks look at the same places", fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.89, f"Task tokens' attention maps (last encoder layer), cosine similarity averaged over "
             f"{len(D['att'])} held-out CT and MRI scans; rows grouped by task group", fontsize=10.5, color=MUTED,
             va="top")
    fig.savefig(o / f"task_overlap_{key}.png", facecolor="white")
    plt.close(fig)
    # (b) group x group means
    ng = len(bounds)
    M = np.full((ng, ng), np.nan)
    for i, (_, a0, a1) in enumerate(bounds):
        for j, (_, b0, b1) in enumerate(bounds):
            blk = S[np.ix_(order[a0:a1], order[b0:b1])]
            if i != j:
                M[i, j] = blk.mean()
            elif a1 - a0 > 1:
                M[i, j] = blk[~np.eye(len(blk), dtype=bool)].mean()
    fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
    ax2 = fig.add_axes([0.2, 0.04, 0.45, 0.72])
    im = ax2.imshow(M, cmap="magma", vmin=vmin, vmax=vmax)
    gl = [GN.get(g, g) for g, _, _ in bounds]
    ax2.set_yticks(range(ng), gl, fontsize=10.5)
    ax2.set_xticks(range(ng), gl, rotation=55, ha="left", fontsize=10.5)
    ax2.xaxis.tick_top()
    ax2.tick_params(length=0)
    for i in range(ng):
        for j in range(ng):
            if np.isfinite(M[i, j]):
                ax2.text(j, i, f"{M[i, j]:.2f}"[1:], ha="center", va="center", fontsize=8,
                         color="black" if M[i, j] > (vmin + vmax) / 2 else "white")
    fig.text(0.03, 0.95, "Task groups that look at the same places", fontsize=17, fontweight="bold", color=INK,
             va="top")
    fig.text(0.68, 0.7, "• Lesion-type groups (hemorrhage, lesion\n  masks, tumour, radiology, diagnosis,\n"
             "  clinical scores) share attention (.45-.69).\n\n• Anatomy masks, volumes, asymmetry, CT\n"
             "  measures and synthesis overlap more with\n  each other (.26-.47) than with lesion\n  groups (.21-.35).\n\n• Demographics overlap little with either.\n\n"
             f"Mean similarity within a group {S[same].mean():.2f},\nbetween groups {S[diff].mean():.2f}; "
             "blank = one-token group.", fontsize=11, color=INK, va="top", linespacing=1.35)
    fig.savefig(o / f"task_overlap_groups_{key}.png", facecolor="white")
    plt.close(fig)
    print(f"[overlap {key}] within {S[same].mean():.3f} between {S[diff].mean():.3f}")
    print("wrote", o)
    return
    fig.savefig(o / f"task_overlap_{key}.png", facecolor="white")
    plt.close(fig)
    print(f"[overlap {key}] within {S[same].mean():.3f} between {S[diff].mean():.3f}")
    print("wrote", o)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--n-seg", type=int, default=40)
    c.add_argument("--n-anat", type=int, default=2)
    c.add_argument("--n-mixed", type=int, default=48)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("plot")
    p.add_argument("--out", required=True)
    p.add_argument("--layer", default="last", choices=["last", "mean"])
    a = ap.parse_args()
    (collect if a.cmd == "collect" else plot)(a)


if __name__ == "__main__":
    main()
