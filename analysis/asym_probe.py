"""Is left-right information in the encoder? For a lateralised target (default: vol_ventricle_asym), probe the frozen
patch tokens split at the grid's left-right midline (scans are template-aligned): features [mean left, mean right,
|left - right|], 64 PCs each, ridge; plus the whole-brain mean (side-blind) and the task's own token output. A high
split-probe R^2 with a low head R^2 means the head / its token failed to read information the encoder has.

    python analysis/asym_probe.py --ckpts runs/multitask/r17/last.pt runs/multitask/full_iso_bank/last.pt \
        --out runs/diagnostics/asym_probe"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def feats(ck, task_id, n, workers):
    import pandas as pd

    from brain.common.paths import resolve
    from brain.multitask.data import MultitaskDataset, checkpoint_stats, load_labels, make_loader, prepare
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
    mri = set(volumes.loc[volumes["modality"].astype(str).eq("mri"), "volume_id"].astype(str))
    t = reg.index[task_id]
    out = {}
    for split in ("train", "val"):
        g = labels[(labels.task == task_id) & (labels.split == split) & labels["use"].eq("train")
                   & labels.volume_id.astype(str).isin(mri)]
        u = g.volume_id.unique()
        ids = u[np.random.default_rng(0).permutation(len(u))[:n]].tolist()
        ds = MultitaskDataset(store, labels, volumes, reg, stats, split=split, tasks=[task_id], train=False,
                              use="train", volume_ids=ids, max_seg=0)
        loader = make_loader(ds, batch_size=4, shuffle=False, seed=0, workers=workers, drop_last=False)
        rows = []
        for raw in loader:
            patch = model.backbone.patch_xyz
            b = prepare(raw, cfg, dev, patch, train=False)
            B = b["x"].shape[0]
            task = torch.full((B, 1), t, device=dev)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                enc = model.encode(b["x"], b["keep"], b.get("meta"), task, modality=b["modality"])
                z = model.token_slots(enc, task)[:, 0].float()
                o = model.heads(model.member(model.token_slots(enc, task), task), task)["regression"][:, 0].float()
            pz, py, px = reversed(patch)
            gz, gy, gx = b["x"].shape[2] // pz, b["x"].shape[3] // py, b["x"].shape[4] // px
            tok = enc["tokens"].float().reshape(B, gz, gy, gx, -1)
            k = b["keep"].reshape(B, gz, gy, gx, 1).float()
            h = gx // 2
            left = (tok[:, :, :, :h] * k[:, :, :, :h]).sum((1, 2, 3)) / k[:, :, :, :h].sum((1, 2, 3)).clamp_min(1)
            right = (tok[:, :, :, gx - h:] * k[:, :, :, gx - h:]).sum((1, 2, 3)) / k[:, :, :, gx - h:].sum((1, 2, 3)).clamp_min(1)
            whole = (tok * k).sum((1, 2, 3)) / k.sum((1, 2, 3)).clamp_min(1)
            for i in range(B):
                rows.append({"y": float(b["y"][i, 0]), "head": float(o[i]), "z": z[i].cpu().numpy(),
                             "left": left[i].cpu().numpy(), "right": right[i].cpu().numpy(),
                             "whole": whole[i].cpu().numpy()})
        out[split] = rows
        print(f"[asym] {Path(ck).parent.name} {split}: {len(rows)} scans", flush=True)
    return out


def probe(tr, te, key):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def X(rows):
        if key == "split":
            return np.stack([np.r_[r["left"], r["right"], np.abs(r["left"] - r["right"])] for r in rows])
        return np.stack([r[key] for r in rows])
    ytr, yte = np.array([r["y"] for r in tr]), np.array([r["y"] for r in te])
    m = make_pipeline(StandardScaler(), PCA(min(128, len(tr) - 1), random_state=0), RidgeCV(np.logspace(-2, 5, 15)))
    m.fit(X(tr), ytr)
    return r2_score(yte, m.predict(X(te)))


def main() -> None:
    from sklearn.metrics import r2_score

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--task", default="vol_ventricle_asym")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = []
    for ck in a.ckpts:
        f = feats(ck, a.task, a.n, a.workers)
        tr, te = f["train"], f["val"]
        head = r2_score([r["y"] for r in te], [r["head"] for r in te])
        res = {k: probe(tr, te, k) for k in ("z", "whole", "split")}
        lines.append(f"{ck}: {a.task} R^2 on val (n={len(te)}): own head {head:.3f} | probe on its token output "
                     f"{res['z']:.3f} | whole-brain mean patches {res['whole']:.3f} | left/right split patches "
                     f"{res['split']:.3f}")
        print(lines[-1], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{a.task}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
