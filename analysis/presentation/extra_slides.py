"""Three result slides (16:9 PNGs) from existing evaluation outputs:
  cq500_fewshot.png   CQ500 AUC vs labelled CQ500 positives: RSNA heads read zero-shot vs readouts fitted on CQ500 labels
  cq500_roc.png       ROC of the RSNA heads read zero-shot on CQ500, per finding
  regression.png      predicted vs true age and brain volume, own heads on held-out test scans
    python analysis/presentation/extra_slides.py OUT_DIR"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, BLUE, ORANGE, GREY = "#1d1d1b", "#55544f", "#1c5cab", "#eb6834", "#8a8983"
HI = ROOT / "runs/diagnostics/head_init_cq500"
MODEL = "BQ-FISO"
FIND = [("ICH", "any bleed", .942), ("IPH", "intraparenchymal", .954), ("IVH", "intraventricular", .931),
        ("SDH", "subdural", .952), ("SAH", "subarachnoid", .957), ("EDH", "epidural", .973)]

# ---- few-shot curves -----------------------------------------------------------------------------------------
k = pd.read_csv(HI / f"{MODEL}_kcurve.csv")
fig, axes = plt.subplots(2, 3, figsize=(13.333, 7.5), dpi=200, sharex=True,
                         gridspec_kw=dict(left=0.07, right=0.98, top=0.8, bottom=0.1, hspace=0.32, wspace=0.18))
for ax, (t, name, lit) in zip(axes.flat, FIND):
    d = k[k.target == t]
    zs = d[d.arm == "zero_shot"].auc.iloc[0]
    for arm, col, lab in (("scratch_last", GREY, "readout fitted on CQ500 labels"),
                          ("init_last", ORANGE, "RSNA head, fine-tuned on CQ500 labels")):
        r = d[d.arm == arm].sort_values("P")
        ax.plot(r.P, r.auc, "o-", color=col, lw=2, ms=5, label=lab)
    ax.axhline(zs, color=BLUE, lw=2.2, label="RSNA head, zero CQ500 labels")
    ax.axhline(lit, color=INK, lw=1, ls=":", label="supervised on ~290k CTs (Chilamkurthy 2018)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_title(name, fontsize=12.5, color=INK)
    ax.set_ylim(0.6, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel("labelled CQ500 positives used for fitting", fontsize=10.5, color=MUTED)
for ax in axes[:, 0]:
    ax.set_ylabel("AUC on CQ500", fontsize=10.5, color=MUTED)
h, lab = axes[0, 0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=4, frameon=False, fontsize=10)
fig.text(0.07, 0.955, "Reading a new hospital's CTs needs no labels from it", fontsize=18, fontweight="bold",
         color=INK)
fig.text(0.07, 0.02, "CQ500 (471 scans, an Indian hospital network; never trained on). Fitted arms: L2 logistic readout, "
         "75 random draws per point, scored on the remaining scans. New run.", fontsize=9, color=MUTED)
fig.savefig(OUT / "cq500_fewshot.png", facecolor="white")
plt.close(fig)
print("few-shot: zero-shot vs 8 positives",
      {t: (round(k[(k.target == t) & (k.arm == "zero_shot")].auc.iloc[0], 3),
           round(k[(k.target == t) & (k.arm == "scratch_last") & (k.P == 8)].auc.iloc[0], 3)) for t, _, _ in FIND})

# ---- ROC -----------------------------------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from brain.downstream.tasks import DEFAULT_VOLUMES, load_task, load_volumes  # noqa: E402

d = np.load(HI / f"{MODEL}.npz")
row = {v: i for i, v in enumerate(d["vids"])}
heads = list(d["heads"])
bv = load_volumes(DEFAULT_VOLUMES)
tasks = {t: load_task(t, bv) for t in ("cq500", "cq500_edh")}
HEAD = {"EDH": "hemorrhage_epidural", "ICH": "hemorrhage_any", "IPH": "hemorrhage_intraparenchymal",
        "IVH": "hemorrhage_intraventricular", "SAH": "hemorrhage_subarachnoid", "SDH": "hemorrhage_subdural"}
fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
ax = fig.add_axes([0.08, 0.1, 0.42, 0.75])
cols = ["#1c5cab", "#c0392b", "#1baf7a", "#6250d6", "#eb6834", "#d4a017"]
rows = []
for (t, name, lit), c in zip(FIND, cols):
    task = tasks["cq500_edh" if t == "EDH" else "cq500"]
    y = np.asarray(task.y, float)
    y = y.reshape(len(y), -1)[:, 0] if t == "EDH" else y[:, task.label_names.index(t)]
    vids = [str(v) for v in task.volume_ids]
    keep = np.array([v in row for v in vids])
    p = d["logits"][[row[v] for v, h in zip(vids, keep) if h], heads.index(HEAD[t])]
    y = y[keep]
    fpr, tpr, _ = roc_curve(y, p)
    a = roc_auc_score(y, p)
    rows.append((name, a, lit, int(y.sum())))
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{name}  {a:.3f}")
ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--")
ax.set_xlabel("false-positive rate", fontsize=12, color=INK)
ax.set_ylabel("true-positive rate", fontsize=12, color=INK)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=11, loc="lower right", title="finding  AUC", title_fontsize=11)
fig.text(0.08, 0.93, "CQ500 read by the RSNA heads, zero-shot", fontsize=18, fontweight="bold", color=INK)
tx = 0.58
fig.text(tx, 0.8, "finding", fontsize=11.5, fontweight="bold", color=INK)
fig.text(tx + 0.17, 0.8, "ours,\nzero-shot", fontsize=11.5, fontweight="bold", color=BLUE)
fig.text(tx + 0.27, 0.8, "supervised\n(Chilamkurthy 2018)*", fontsize=11.5, fontweight="bold", color=INK)
for i, (name, a, lit, npos) in enumerate(rows):
    y0 = 0.72 - i * 0.065
    fig.text(tx, y0, f"{name} ({npos})", fontsize=11.5, color=INK)
    fig.text(tx + 0.17, y0, f"{a:.3f}", fontsize=12, color=BLUE, fontweight="bold")
    fig.text(tx + 0.27, y0, f"{lit:.3f}", fontsize=12, color=INK)
fig.text(tx, 0.26, "* Chilamkurthy et al. 2018: trained on ~290k labelled head CTs\n  from Indian hospitals; "
         "CQ500 was its external test.\n  Ours: trained on RSNA 2019 (a different population),\n  no CQ500 labels, no "
         "tuning on CQ500.\n  (n) = positive scans of 471.", fontsize=10, color=MUTED, va="top", linespacing=1.4)
fig.savefig(OUT / "cq500_roc.png", facecolor="white")
plt.close(fig)
print("roc:", [(n, round(a, 3)) for n, a, _, _ in rows])

# ---- regression scatters -------------------------------------------------------------------------------------
ck = torch.load(ROOT / "runs/multitask/full_iso_bank/last.pt", map_location="cpu", weights_only=False, mmap=True)
stats = ck.get("label_stats") or {}
pr = pd.read_parquet(ROOT / "runs/diagnostics/te_full_iso/test_eval_preds.parquet")
pr = pr[pr.model == "full_iso_bank/last"]
fig, axes = plt.subplots(1, 2, figsize=(13.333, 7.5), dpi=200,
                         gridspec_kw=dict(left=0.07, right=0.97, top=0.8, bottom=0.12, wspace=0.22))
for ax, (t, name, unit, scale) in zip(axes, (("age", "Age", "years", 1.0), ("brain_volume", "Normalised brain volume", "ml", None))):
    r = pr[pr.task == t]
    s = stats.get(t, {})
    mu, sd = float(s.get("mean", 0.0)), float(s.get("std", 1.0))
    y, p = r.y.to_numpy() * sd + mu, r.p.to_numpy() * sd + mu
    if t == "brain_volume" and np.nanmedian(y) < 10:          # stored in litres
        y, p, unit = y * 1000, p * 1000, "ml"
    ss = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = np.abs(y - p).mean()
    ax.scatter(y, p, s=7, alpha=0.35, color=BLUE, lw=0)
    lo, hi = np.percentile(np.r_[y, p], [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"true {name.lower()} ({unit})", fontsize=12, color=INK)
    ax.set_ylabel(f"predicted ({unit})", fontsize=12, color=INK)
    ax.set_title(f"{name}\nR² {ss:.2f}, mean abs. error {mae:.1f} {unit}  (n = {len(y)})", fontsize=12.5,
                 color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    print(t, f"R2 {ss:.3f} MAE {mae:.2f} {unit} n={len(y)} range {y.min():.1f}-{y.max():.1f}")
fig.text(0.07, 0.93, "One model, one forward pass: age and brain volume on held-out scans", fontsize=18,
         fontweight="bold", color=INK)
fig.text(0.07, 0.015, "Own heads of the new run on held-out test scans pooled over cohorts (CT and MRI, newborns to "
         "elderly).\nNormalised brain volume: brain-mask volume after registration to a standard head size (an "
         "atrophy measure).", fontsize=9, color=MUTED)
fig.savefig(OUT / "regression.png", facecolor="white")
plt.close(fig)
print("wrote", OUT)
