"""Per-group training loss (from the trainer's log table) and validation skill (metrics.json, EMA weights) of a
multitask run, as small multiples; optional reference run drawn as a dashed line (its final EMA skill).

    python analysis/train_curves.py --log logs/train_28600203.out --run runs/multitask/r17 \
        --ref runs/multitask/full_iso_bank --out runs/diagnostics/r17_curves"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

INK, MUTED, GRID, BLUE, REF = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
# log columns that are per-group training losses (the rest: optimiser / bookkeeping, or per-output-type losses)
SKIP = {"loss", "gnorm", "clip", "skip", "k", "lr", "vol/s", "peak", "time", "ng_rows", "ng_slots"}
NAME = {"demo": "demographics", "dx": "diagnosis", "ich": "hemorrhage", "lesion": "lesion", "derived": "derived",
        "ct_measu": "CT measures", "avol": "anatomy volumes", "tumor": "tumour", "recon": "reconstruction (SSL)",
        "synth": "synthesis", "radiolog": "radiology findings", "aseg": "anatomy masks", "tokmap": "token maps",
        "slabmil": "RSNA slice MIL", "ng_loss": "native-CT rows", "t_bin": "all binary", "t_reg": "all regression",
        "t_cls": "all multiclass", "t_seg": "all segmentation", "t_dense": "all dense", "clinical": "clinical",
        "aneurysm": "aneurysm", "vessels": "vessels"}


def read_log(path: Path) -> pd.DataFrame:
    head, rows = None, {}
    for line in path.read_text(errors="replace").splitlines():
        if line.lstrip().startswith("loss ") and "gnorm" in line:
            head = line.split()
        elif line.startswith("[step") and head:
            m = re.match(r"\[step\s+(\d+)/\d+\]\s+(.*)", line)
            if not m:
                continue
            vals = m.group(2).split()
            if len(vals) != len(head):
                continue
            rows[int(m.group(1))] = [np.nan if v == "-" else _num(v) for v in vals]   # a requeued segment wins
    return pd.DataFrame.from_dict(rows, orient="index", columns=head).sort_index()


def _num(v: str) -> float:
    try:
        return float(v.replace("GiB", ""))
    except ValueError:
        return np.nan


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--smooth-steps", type=int, default=2000)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = read_log(a.log)
    every = int(np.median(np.diff(d.index))) if len(d) > 2 else 50
    win = max(1, a.smooth_steps // every)
    cols = [c for c in d.columns if c not in SKIP and d[c].notna().sum() > 20 and not c.startswith("t_")]

    def grid(n):
        nc = 4
        return int(np.ceil(n / nc)), nc

    # ---- training loss per group
    nr, nc = grid(len(cols))
    fig, axes = plt.subplots(nr, nc, figsize=(13.333, 2.1 * nr + 1.2), dpi=170, squeeze=False,
                             gridspec_kw=dict(hspace=0.55, wspace=0.28, left=0.05, right=0.98, top=0.9, bottom=0.06))
    rows = []
    for ax, c in zip(axes.flat, cols):
        s = d[c].dropna()
        sm = s.rolling(win, min_periods=max(1, win // 3)).mean()
        ax.plot(s.index / 1000, s.values, color=GRID, lw=0.6)
        ax.plot(sm.index / 1000, sm.values, color=BLUE, lw=2)
        lo, hi = np.nanpercentile(s.values, [1, 99])
        ax.set_ylim(max(0, lo - 0.05 * (hi - lo)), hi + 0.05 * (hi - lo))
        ax.set_title(NAME.get(c, c), fontsize=10, color=INK, loc="left")
        ax.tick_params(labelsize=7.5, colors=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        last = d.index.max()
        cur = s[s.index > last - 5000].mean()
        prev = s[(s.index > last - 25000) & (s.index <= last - 20000)].mean()
        rows.append({"group": NAME.get(c, c), "train_loss_now": cur, "train_loss_20k_ago": prev,
                     "change_per_10k_pct": 100 * (cur - prev) / prev / 2 if prev else np.nan})
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    fig.suptitle(f"Training loss per task group (grey: every {every} steps; blue: {a.smooth_steps}-step mean); "
                 "x = thousand steps", fontsize=12, color=INK, x=0.05, ha="left")
    fig.savefig(a.out / "train_loss_by_group.png", facecolor="white")
    plt.close(fig)

    # ---- validation skill per group (EMA)
    h = json.loads((a.run / "metrics.json").read_text())["history"]
    ref = None
    if a.ref is not None:
        rh = json.loads((a.ref / "metrics.json").read_text())["history"]
        ref = rh[-1].get("ema") or rh[-1]
    groups = sorted({k for e in h for k in (e.get("ema") or {}) if k.startswith("val_group_")})
    nr, nc = grid(len(groups) + 1)
    fig, axes = plt.subplots(nr, nc, figsize=(13.333, 2.1 * nr + 1.2), dpi=170, squeeze=False,
                             gridspec_kw=dict(hspace=0.55, wspace=0.28, left=0.05, right=0.98, top=0.88, bottom=0.06))
    for ax, k in zip(axes.flat, ["val_score"] + groups):
        st = np.array([e["step"] for e in h if k in (e.get("ema") or {})]) / 1000
        v = np.array([e["ema"][k] for e in h if k in (e.get("ema") or {})])
        ax.plot(st, v, color=BLUE, lw=2)
        if ref is not None and k in ref:
            ax.axhline(ref[k], color=REF, lw=1.4, ls="--")
        ax.set_title("overall score" if k == "val_score" else k.replace("val_group_", "").replace("_", " "),
                     fontsize=10, color=INK, loc="left")
        ax.tick_params(labelsize=7.5, colors=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        if len(v) > 25:
            rows.append({"group": "VAL " + k.replace("val_group_", ""), "val_now": v[-3:].mean(),
                         "val_20k_ago": v[-23:-20].mean(), "ref_final": ref.get(k) if ref else np.nan})
    for ax in list(axes.flat)[len(groups) + 1:]:
        ax.axis("off")
    fig.suptitle("Validation skill per task group, EMA weights (blue: this run; orange dashed: "
                 f"{a.ref.name if a.ref else ''} final); x = thousand steps", fontsize=12, color=INK, x=0.05, ha="left")
    fig.savefig(a.out / "val_skill_by_group.png", facecolor="white")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(a.out / "summary.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
