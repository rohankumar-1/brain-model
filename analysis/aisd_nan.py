"""AISD frozen-seg NaN reproduction (2026-11-05): seg3d.train_decoder on an in-memory cache of 24 volumes."""
import numpy as np, pandas as pd, torch
from pathlib import Path
from brain.downstream import seg3d as S
from brain.downstream.tasks import load_task, DEFAULT_VOLUMES
from brain.downstream.embed3d import windows_for
dev = torch.device("cuda")
t = load_task("aisd", pd.read_csv(DEFAULT_VOLUMES))
enc = S.load_frozen_encoder(Path("runs/multitask/r14/last.pt"), dev)
c, w = windows_for("canonical", "ct")[0]
tr = [str(v) for v in t.volume_ids if t.splits[str(v)] == "train"][:16]
va = [str(v) for v in t.volume_ids if t.splits[str(v)] == "val"][:8]
ids = tr + va
F, A, M = [], [], []
for vid in ids:
    v, m = S._read(t, vid, DEFAULT_VOLUMES.parent, True)
    x = torch.from_numpy(np.ascontiguousarray(v[0])).to(dev)
    F.append(enc.embed_tokens(x, c, w, modality="ct")[0].half().cpu().numpy())
    if vid in tr:
        A.append(np.stack([enc.embed_tokens(S.aug_volume(x, k), c, w, modality="ct")[0].half().cpu().numpy() for k in (1, 2, 3)]))
    M.append(m)
cache = S.Cache(ids, np.stack(F), np.stack(M), np.stack(A), {i: i for i in range(len(tr))})
split = lambda s: np.arange(len(tr)) if s == "train" else np.arange(len(tr), len(ids)) if s == "val" else np.array([], int)
grid = S.grid_for(S.PATCH)
torch.autograd.set_detect_anomaly(False)
for lr in (3e-3, 1e-3):
    dec = S.TokenDecoder3D(F[0].shape[-1], S.n_classes_for(t.regions), grid, dropout=0.1).to(dev)
    h = S.train_decoder(dec, cache, t, split, steps=300, batch_size=8, lr=lr, seed=0, eval_every=50, device=dev,
                        patience=100, crop=(8, 8, 8))
    print("lr", lr, [(r["step"], round(r["train_loss"], 3)) for r in h])
