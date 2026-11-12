"""Why epidural hemorrhage (EDH) is hard: every RSNA val / test EDH-positive scan plus matched negatives (other bleeds,
no bleed) through a checkpoint's six hemorrhage heads, with the EDH token map, the bleed mask and the RSNA slice labels
(where the readers marked EDH) kept for the slice the model looked at and the slice the readers marked.

    python analysis/edh_cases.py collect --ckpt runs/multitask/r17/last.pt --out runs/diagnostics/edh_r17
    python analysis/edh_cases.py plot --out runs/diagnostics/edh_r17 --fig presentation/.../edh

Figures show no scan ids (cases are numbered)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HEM = ["hemorrhage_epidural", "hemorrhage_any", "hemorrhage_subdural", "hemorrhage_subarachnoid",
       "hemorrhage_intraparenchymal", "hemorrhage_intraventricular"]
SHORT = {"hemorrhage_epidural": "EDH", "hemorrhage_any": "any", "hemorrhage_subdural": "SDH",
         "hemorrhage_subarachnoid": "SAH", "hemorrhage_intraparenchymal": "IPH", "hemorrhage_intraventricular": "IVH"}


def collect(a) -> None:
    import pandas as pd
    import torch.nn.functional as F

    from brain.common.paths import resolve
    from brain.multitask.data import (SLAB_TASKS, MultitaskDataset, checkpoint_stats, load_labels, load_slices,
                                      make_loader, prepare)
    from brain.multitask.fewshot import load_model

    dev = torch.device("cuda")
    amp = dict(device_type="cuda", dtype=torch.bfloat16)
    model, reg, cfg = load_model(Path(a.ckpt), dev)
    model.eval()
    d = cfg["data"]
    store = resolve(d["store_root"])
    labels_dir = resolve(d["labels_dir"])
    labels, stats = load_labels(labels_dir)
    stats = checkpoint_stats(model.label_stats, labels_dir, stats)
    volumes = pd.read_csv(store / d.get("index_name", "volumes.csv"), low_memory=False)
    slices = load_slices(resolve(d["rsna_slices"]))

    lab = labels[labels["use"].eq("train") & labels["task"].isin(HEM) & labels["split"].isin(["val", "test"])]
    wide = lab.pivot_table(index=["volume_id", "split"], columns="task", values="value", aggfunc="first").reset_index()
    wide = wide.dropna(subset=["hemorrhage_epidural", "hemorrhage_any"])
    rng = np.random.default_rng(0)
    pos = wide[wide.hemorrhage_epidural == 1]
    oth = wide[(wide.hemorrhage_epidural == 0) & (wide.hemorrhage_any == 1)]
    non = wide[wide.hemorrhage_any == 0]
    pick = pd.concat([pos, oth.iloc[rng.permutation(len(oth))[:a.n_neg]], non.iloc[rng.permutation(len(non))[:a.n_neg]]])
    print(f"[edh] positives {len(pos)}  other-bleed {len(oth)} (taking {min(a.n_neg, len(oth))})  "
          f"no-bleed {len(non)} (taking {min(a.n_neg, len(non))})", flush=True)

    tids = [reg.index[t] for t in HEM]
    seg_t = reg.index["hemorrhage_segmentation"]
    k_edh = SLAB_TASKS.index("hemorrhage_epidural")
    k_any = SLAB_TASKS.index("hemorrhage_any")
    patch = model.backbone.patch_xyz
    rows, cases = [], {}
    for split in ("val", "test"):
        ids = pick.loc[pick.split == split, "volume_id"].tolist()
        ds = MultitaskDataset(store, labels, volumes, reg, stats, split=split, tasks=["hemorrhage_epidural"],
                              train=False, use="train", volume_ids=ids, max_seg=1, slices=slices)
        loader = make_loader(ds, batch_size=2, shuffle=False, seed=0, workers=a.workers, drop_last=False)
        for raw in loader:
            b = prepare(raw, cfg, dev, patch, train=False)
            B = b["x"].shape[0]
            task = torch.tensor([tids + [seg_t]] * B, device=dev)
            seg_pair = torch.tensor([[i, len(tids)] for i in range(B)], device=dev)
            tok_pair = torch.tensor([[i, j] for i in range(B) for j in (0, 2)], device=dev)    # EDH, SDH maps
            with torch.no_grad(), torch.autocast(**amp):
                o = model(b["x"], b["keep"], task, seg_pair, alone=True, meta=b.get("meta"), modality=b["modality"],
                          tok_pair=tok_pair)
            pz, py, px = reversed(patch)
            g = (b["x"].shape[2] // pz, b["x"].shape[3] // py, b["x"].shape[4] // px)
            prob = torch.sigmoid(o["binary"][:, :len(tids)].float()).cpu().numpy()
            tok = o["tok"].float().reshape(B, 2, *g)
            keep = b["keep"].reshape(B, *g)
            slab = b.get("slab")
            for i in range(B):
                vid = str(raw["volume_id"][i])
                y = wide.set_index("volume_id").loc[vid]
                rec = {"volume_id": vid, "split": split,
                       **{f"y_{SHORT[t]}": float(y[t]) if pd.notna(y[t]) else np.nan for t in HEM},
                       **{f"p_{SHORT[t]}": float(prob[i, j]) for j, t in enumerate(HEM)}}
                kp = keep[i]
                edh = tok[i, 0].masked_fill(~kp, -1e4)
                zm = int(edh.amax((1, 2)).argmax())                                   # token slab the model looks at
                rec["edh_tok_max"] = float(edh.max())
                lab_z = None
                if slab is not None and slab.shape[-1] == kp.numel():
                    s = slab[i, k_edh].reshape(g)
                    sa = slab[i, k_any].reshape(g)
                    if bool((s >= 0).any()):
                        zpos = ((s == 1) & kp).any((1, 2)).nonzero()[:, 0].cpu().numpy()
                        rec["edh_slabs"] = int(len(zpos))
                        rec["any_slabs"] = int(((sa == 1) & kp).any((1, 2)).sum())
                        if len(zpos):
                            lab_z = int(zpos[len(zpos) // 2])
                            on = (s == 1) & kp
                            rec["edh_tok_on"] = float(edh[on].max())                # map's best on a marked slab
                            rec["edh_tok_off"] = float(edh[(s == 0) & kp].max()) if bool(((s == 0) & kp).any()) else np.nan
                rows.append(rec)
                # images: the model's slab and the readers' slab, raw HU (native slices), bleed mask, EDH / SDH maps
                img = b["x"][i, 0].float()                              # window bank: raw HU
                seg = torch.sigmoid(F.interpolate(o["seg"][i:i + 1].float(), size=img.shape, mode="trilinear",
                                                  align_corners=False))[0, 0]
                zz = {"model": zm * pz + pz // 2}
                if lab_z is not None:
                    zz["label"] = lab_z * pz + pz // 2
                rec_img = {}
                for k, z in zz.items():
                    def up(m, z=z):                                    # token slab -> image slice
                        return F.interpolate(m[z // pz][None, None].float(), size=img.shape[1:], mode="nearest")[0, 0]
                    rec_img[k] = {"z": z, "img": img[z].half().cpu(), "seg": seg[z].half().cpu(),
                                  "edh": up(tok[i, 0].masked_fill(~kp, float("nan"))).half().cpu(),
                                  "sdh": up(tok[i, 1].masked_fill(~kp, float("nan"))).half().cpu(),
                                  "lab": up(slab[i, k_edh].reshape(g).float()).half().cpu()
                                  if lab_z is not None else None}
                cases[vid] = rec_img
            print(f"[edh] {split}: {len(rows)} scans", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "edh_scores.csv", index=False)
    torch.save(cases, out / "edh_cases.pt")
    print("wrote", out, flush=True)


# --------------------------------------------------------------------------------------------------------- plot
def plot(a) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    INK, MUTED = "#1d1d1b", "#55544f"
    o = Path(a.out)
    fig_dir = Path(a.fig)
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(o / "edh_scores.csv")
    cases = torch.load(o / "edh_cases.pt", weights_only=False)
    pos = d[d.y_EDH == 1]
    oth = d[(d.y_EDH == 0) & (d.y_any == 1)]
    non = d[d.y_any == 0]
    sdh_only = oth[(oth.y_SDH == 1) & (oth[["y_SAH", "y_IPH", "y_IVH"]].fillna(0).sum(1) == 0)]
    edh_only = pos[pos[["y_SDH", "y_SAH", "y_IPH", "y_IVH"]].fillna(0).sum(1) == 0]

    def auc(p, n, col="p_EDH"):
        y = np.r_[np.ones(len(p)), np.zeros(len(n))]
        return roc_auc_score(y, np.r_[p[col], n[col]]) if len(p) and len(n) else float("nan")

    lines = [f"positives {len(pos)} (EDH only {len(edh_only)}, co-labelled {len(pos) - len(edh_only)}); "
             f"other-bleed negatives {len(oth)}, no-bleed {len(non)}",
             f"EDH vs no bleed         {auc(pos, non):.3f}",
             f"EDH vs other bleeds     {auc(pos, oth):.3f}",
             f"EDH-only vs SDH-only    {auc(edh_only, sdh_only):.3f}  ({len(edh_only)} vs {len(sdh_only)})",
             f"'any' head, EDH vs other bleeds {auc(pos, oth, 'p_any'):.3f} (a bleed-size prior alone)",
             f"Spearman p_EDH vs p_any over bleeds: {pd.concat([pos, oth])[['p_EDH', 'p_any']].corr('spearman').iloc[0, 1]:.2f}"]
    co = pos[["y_SDH", "y_SAH", "y_IPH", "y_IVH"]].fillna(0).mean()
    lines.append("EDH positives co-labelled: " + ", ".join(f"{k[2:]} {v:.0%}" for k, v in co.items()))
    if "edh_slabs" in pos:
        q = pos.dropna(subset=["edh_slabs"])
        lo = q.p_EDH <= q.p_EDH.median()
        lines.append(f"marked EDH slices: median {q.edh_slabs.median():.0f} "
                     f"(missed half {q[lo].edh_slabs.median():.0f}, found half {q[~lo].edh_slabs.median():.0f})")
        if "edh_tok_on" in q:
            r = q.dropna(subset=["edh_tok_on", "edh_tok_off"])
            lines.append(f"EDH map peaks on a marked slab in {(r.edh_tok_on >= r.edh_tok_off).mean():.0%} of "
                         f"positives ({len(r)})")
    (fig_dir / "edh_stats.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- score distributions
    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=200)
    bins = np.logspace(np.log10(max(d.p_EDH.min(), 1e-5)), 0, 40)
    for g, lab, c in ((non, "no bleed", "#8a8983"), (sdh_only, "SDH only (no EDH)", "#6250d6"),
                      (oth.drop(sdh_only.index), "other bleeds (no EDH)", "#eb6834"), (pos, "EDH", "#c0392b")):
        ax.hist(g.p_EDH, bins=bins, histtype="step", lw=2.2, color=c, density=True, label=f"{lab} (n={len(g)})")
    ax.set_xscale("log")
    ax.set_xlabel("EDH head probability", fontsize=12)
    ax.set_ylabel("density", fontsize=12)
    ax.legend(frameon=False, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("EDH head: separates bleed from no bleed, not epidural from other bleeds", fontsize=16,
                 fontweight="bold", x=0.06, ha="left")
    fig.text(0.06, 0.9, "   ".join(lines[1:4]), fontsize=10.5, color=MUTED)
    fig.savefig(fig_dir / "edh_scores.png", facecolor="white")
    plt.close(fig)

    # ---- case grids
    def panel(ax, c, key, title):
        r = c.get(key) or c["model"]
        img = r["img"].float().numpy()
        ax.imshow(np.clip((img - 0) / 120, 0, 1), cmap="gray", interpolation="bilinear")       # blood window L60 W120
        e = r["edh"].float().numpy()
        if np.isfinite(e).any():
            m = np.where(np.isfinite(e), e, np.nan)
            top = np.nanpercentile(m, 99.5)
            ax.contour(np.nan_to_num(m, nan=-1e4) >= top, levels=[0.5], colors=["#ff3b30"], linewidths=1.2)
        if r.get("lab") is not None and key == "label":
            ax.contour(r["lab"].float().numpy() > 0.5, levels=[0.5], colors=["#00c08b"], linewidths=0.6,
                       linestyles="dotted")
        ax.contour(r["seg"].float().numpy() > 0.5, levels=[0.5], colors=["#ffd60a"], linewidths=0.9)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.axis("off")

    def grid(df, name, title, key):
        df = df[df.volume_id.isin(cases)].head(a.n_show)
        if not len(df):
            return
        n = len(df)
        cols = min(n, 4)
        rws = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rws, cols, figsize=(13.333, 7.5 * rws / 2), dpi=200, squeeze=False,
                                 gridspec_kw=dict(left=0.02, right=0.98, top=0.86, bottom=0.03, wspace=0.04,
                                                  hspace=0.18))
        for k, (ax, r) in enumerate(zip(axes.flat, df.itertuples())):
            co = "+".join(s for s in ("SDH", "SAH", "IPH", "IVH") if getattr(r, f"y_{s}") == 1) or "none"
            panel(ax, cases[r.volume_id], key,
                  f"case {k + 1}: p(EDH) {r.p_EDH:.3f}  p(SDH) {r.p_SDH:.2f}  p(any) {r.p_any:.2f}\n"
                  f"label EDH {int(r.y_EDH)}, also {co}")
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.suptitle(title, fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.97)
        fig.text(0.02, 0.905, "Blood window (L 60, W 120). Red: top 0.5% of the EDH token map; yellow: predicted "
                 "bleed mask; green dotted: tokens on slices the readers marked EDH.", fontsize=9.5, color=MUTED)
        fig.savefig(fig_dir / f"{name}.png", facecolor="white")
        plt.close(fig)

    grid(pos.sort_values("p_EDH", ascending=False), "edh_found", "Epidural cases the model ranks highest "
         "(readers' EDH slice)", "label")
    grid(pos.sort_values("p_EDH"), "edh_missed", "Epidural cases the model ranks lowest (readers' EDH slice)", "label")
    grid(oth.sort_values("p_EDH", ascending=False), "edh_false_alarm",
         "Highest EDH scores on scans without EDH (slice the EDH map peaks on)", "model")
    print("wrote", fig_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--n-neg", type=int, default=300)
    c.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("plot")
    p.add_argument("--out", required=True)
    p.add_argument("--fig", required=True)
    p.add_argument("--n-show", type=int, default=8)
    a = ap.parse_args()
    (collect if a.cmd == "collect" else plot)(a)


if __name__ == "__main__":
    main()
