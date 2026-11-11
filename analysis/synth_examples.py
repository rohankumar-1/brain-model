"""Synthesis examples: validation scans, each synthesis task, the input scan, the real partner contrast (target) and
each checkpoint's prediction, axial and coronal through the centre. No volume ids in the figure.

    python analysis/synth_examples.py --ckpts runs/multitask/r17_cool/last_ema.pt runs/multitask/full_iso_bank/last.pt \
        --names R17 full_iso_bank --out runs/diagnostics/synth_examples"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASKS = ("synth_t1", "synth_t1c", "synth_t2", "synth_flair", "synth_ct")


def collect(ck, ids_by_task, workers):
    """{task: [(input, target, weight, prediction) per case]} at the decoder stride, float numpy (Z, Y, X)."""
    import pandas as pd

    from brain.common.paths import resolve
    from brain.multitask.data import (MultitaskDataset, checkpoint_stats, load_labels, make_loader, mixed_parts,
                                      ngeo_setup, prepare)
    from brain.multitask.fewshot import load_model

    dev = torch.device("cuda")
    model, reg, cfg = load_model(Path(ck), dev)
    model.eval()
    d = cfg["data"]
    store = resolve(d["store_root"])
    labels_dir = resolve(d["labels_dir"])
    labels, stats = load_labels(labels_dir)
    stats = checkpoint_stats(model.label_stats, labels_dir, stats)
    volumes = pd.read_csv(store / d.get("index_name", "volumes.csv"), low_memory=False)
    out = {}
    for task, ids in ids_by_task.items():
        if ids is None:      # first checkpoint picks the cases; later ones reuse them
            g = labels[(labels.task == task) & (labels.split == "val") & labels["use"].eq("train")]
            u = np.sort(g.volume_id.astype(str).unique())
            ids = u[np.random.default_rng(1).permutation(len(u))[:ids_by_task.n]].tolist() if len(u) else []
            ids_by_task[task] = ids
        if not ids:
            continue
        t = reg.index[task]
        ngeo_kw, cf = ngeo_setup(cfg, store)
        ds = MultitaskDataset(store, labels, volumes, reg, stats, split="val", tasks=[t], train=False, use="train",
                              volume_ids=ids, max_seg=0, **ngeo_kw)
        loader = make_loader(ds, batch_size=2, shuffle=False, seed=0, workers=workers, drop_last=False, collate_fn=cf)
        rows = []
        for raw in (p for full in loader for p in mixed_parts(full)):
            if "affine" in raw:
                continue
            b = prepare(raw, cfg, dev, model.backbone.patch_xyz, train=False)
            if not len(b["dense_pair"]):
                continue
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o = model(b["x"], b["keep"], b["task"], b["seg_pair"], alone=True, meta=b.get("meta"),
                          dense_pair=b["dense_pair"], study=b.get("study"), modality=b["modality"])
            s = b["x"].shape[2] // b["dense_tgt"].shape[2]
            xin = F.avg_pool3d(b["x"][:, :1].float(), s) if s > 1 else b["x"][:, :1].float()
            for k, (i, _) in enumerate(b["dense_pair"].tolist()):
                rows.append(tuple(a.cpu().numpy() for a in (xin[i, 0], b["dense_tgt"][k, 0], b["dense_w"][k, 0],
                                                            o["dense"][k, 0].float())))
        out[task] = rows
        print(f"[synth] {Path(ck).parent.name} {task}: {len(rows)} cases", flush=True)
    return out


class Ids(dict):
    n = 2


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    ids = Ids({t: None for t in TASKS})
    ids.n = a.n
    res = [collect(ck, ids, a.workers) for ck in a.ckpts]
    for rr, name in zip(res[1:], a.names[1:]):          # same cases, same loader order in every checkpoint
        for t in TASKS:
            if len(rr.get(t, [])) != len(res[0].get(t, [])):
                raise SystemExit(f"{name} {t}: {len(rr.get(t, []))} cases, first checkpoint {len(res[0].get(t, []))}")
    cases = [(t, i) for t in TASKS for i in range(len(res[0].get(t, [])))]
    cols = ["input", "target"] + a.names + [f"|error| {a.names[0]}"]
    a.out.mkdir(parents=True, exist_ok=True)
    for view in ("axial", "coronal"):
        fig, axes = plt.subplots(len(cases), len(cols), figsize=(2.3 * len(cols), 2.4 * len(cases)), dpi=130,
                                 squeeze=False, gridspec_kw=dict(wspace=0.04, hspace=0.18))
        for r, (task, i) in enumerate(cases):
            xin, tgt, w, _ = res[0][task][i]
            preds = [rr[task][i][3] for rr in res]
            mz = int(np.argmax((w > 0).sum((1, 2)))) if view == "axial" else None
            my = int(np.argmax((w > 0).sum((0, 2)))) if view == "coronal" else None
            sl = (lambda v: v[mz][::-1]) if view == "axial" else (lambda v: v[:, my][::-1])
            lo, hi = np.percentile(tgt[w > 0], [1, 99]) if (w > 0).any() else (0, 1)
            ilo, ihi = np.percentile(xin[xin != 0], [1, 99]) if (xin != 0).any() else (0, 1)
            mae = [float((np.abs(p - tgt) * w).sum() / max(w.sum(), 1)) for p in preds]
            imgs = [(sl(xin), ilo, ihi, "gray"), (sl(tgt), lo, hi, "gray")] + \
                [(sl(p), lo, hi, "gray") for p in preds] + [(sl(np.abs(preds[0] - tgt) * (w > 0)), 0, hi - lo, "magma")]
            for c, (im, vlo, vhi, cm) in enumerate(imgs):
                ax = axes[r, c]
                ax.imshow(im, cmap=cm, vmin=vlo, vmax=vhi, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(cols[c], fontsize=10)
                if 2 <= c < 2 + len(preds):
                    ax.text(0.02, 0.02, f"MAE {mae[c - 2]:.3f}", transform=ax.transAxes, color="yellow", fontsize=7)
            axes[r, 0].set_ylabel(task.replace("synth_", "→ ").upper(), fontsize=10)
        fig.savefig(a.out / f"synth_{view}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
