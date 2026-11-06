"""Gradient conflict between task groups on the shared parameters of a multitask / joint checkpoint.

For each training batch (the checkpoint's own training mixture, no augmentation, model in eval mode) the unweighted mean
loss of each task group present is back-propagated separately to (a) the encoder and (b) the shared slot module
(kv projection + slot layers). Gradients are summed per group over two disjoint halves of the batches, A and B.

* ``rel[g]`` = cos(A_g, B_g): how reproducible group g's gradient direction is (1 = pure signal, 0 = pure noise).
* ``cos[g, h]`` = mean(cos(A_g, B_h), cos(B_g, A_h)) for g != h: agreement between groups, across halves so that
  shared scans do not inflate it. ``dis[g, h]`` = cos / sqrt(rel_g rel_h): the same, corrected for noise.
  Negative = the groups pull the shared weights in opposite directions (conflict); ~0 = independent; positive = shared.
* ``norm[g]``: the mean per-batch gradient norm, i.e. which groups dominate the shared update before balancing.

    python runs/diagnostics/grad_conflict.py --ckpt runs/multitask/v7nosmall_s42/best.pt --batches 160 --batch 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brain.multitask.data import checkpoint_stats, prepare, task_loaders  # noqa: E402
from brain.multitask.fewshot import load_model  # noqa: E402
from brain.multitask.losses import slot_losses  # noqa: E402
from brain.common.paths import resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=160)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/diagnostics/grad_conflict")
    a = ap.parse_args()
    dev = torch.device("cuda")
    amp = dict(device_type="cuda", dtype=torch.bfloat16)
    model, reg, cfg = load_model(Path(a.ckpt), dev)
    stride = int(cfg["loss"].get("seg_stride", 2))
    patch = tuple(model.backbone.patch_xyz)
    enc_p = [p for n, p in model.backbone.named_parameters() if not n.startswith("predictor.")]
    slot_p = list(model.kv.parameters()) + list(model.kv_norm.parameters()) + list(model.layers.parameters())
    for p in enc_p + slot_p:
        p.requires_grad_(True)
    sets = {"encoder": enc_p, "slots": slot_p}

    train_ids = reg.train_ids(cfg["tasks"])                 # the trainer's selection (tasks.exclude etc.)
    loader, _, stats, n = task_loaders(cfg, reg, train_ids, batch=a.batch, workers=a.workers, seed=1234, main=False,
                                       distributed=False)
    model.label_stats = checkpoint_stats(model.label_stats, resolve(cfg["data"]["labels_dir"]), stats)
    groups = [g for g in reg.groups if g != "held_out"]
    gi = {g: i for i, g in enumerate(groups)}
    group_of = torch.tensor([gi.get(t.group, -1) for t in reg.tasks], device=dev)
    flat = {k: sum(p.numel() for p in v) for k, v in sets.items()}
    acc = {k: torch.zeros(2, len(groups), flat[k], device=dev) for k in sets}      # (half, group, params)
    cnt = np.zeros((2, len(groups)))
    norms = {k: [[] for _ in groups] for k in sets}
    print(f"[grad] {a.ckpt}: {len(train_ids)} tasks, {len(groups)} groups, {n} train scans; params "
          f"encoder {flat['encoder']:,} slots {flat['slots']:,}", flush=True)

    it = iter(loader)
    halves = np.random.default_rng(0).permutation(np.arange(a.batches) % 2)
    for step in range(a.batches):
        raw = next(it)
        b = prepare(raw, cfg, dev, patch, train=False)
        with torch.autocast(**amp):
            out = model(b["x"], b["keep"], b["task"], b["seg_pair"], meta=b.get("meta"))
        loss, has = slot_losses(out, b["task"], b["y"], model.task_type, b["seg"], b["seg_pair"], stride)
        g_slot = group_of[b["task"].clamp_min(0)]
        present = sorted({int(g) for g in g_slot[has].tolist() if g >= 0})
        half = int(halves[step])                           # random halves: alternating batches can be correlated
        for j, g in enumerate(present):
            sel = has & (g_slot == g)
            lg = loss[sel].mean()
            grads = torch.autograd.grad(lg, enc_p + slot_p, retain_graph=j < len(present) - 1, allow_unused=True)
            ge = torch.cat([(x if x is not None else torch.zeros_like(p)).flatten().float()
                            for x, p in zip(grads[:len(enc_p)], enc_p)])
            gs = torch.cat([(x if x is not None else torch.zeros_like(p)).flatten().float()
                            for x, p in zip(grads[len(enc_p):], slot_p)])
            for k, v in (("encoder", ge), ("slots", gs)):
                acc[k][half, g] += v
                norms[k][g].append(float(v.norm()))
            cnt[half, g] += 1
        if (step + 1) % 20 == 0:
            print(f"[grad] {step + 1}/{a.batches} batches; group batches {cnt.sum(0).astype(int).tolist()}", flush=True)

    res = {"ckpt": a.ckpt, "groups": groups, "batches": a.batches, "batch": a.batch, "group_batches": cnt.tolist()}
    ok = (cnt >= 4).all(0)
    for k in sets:
        A, B = acc[k][0], acc[k][1]
        nA, nB = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12), B / B.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cross = (nA @ nB.T).cpu().numpy()                     # cos(A_g, B_h)
        rel = np.diag(cross).copy()
        cos = (cross + cross.T) / 2
        dis = cos / np.sqrt(np.outer(np.clip(rel, 1e-3, None), np.clip(rel, 1e-3, None)))
        cos[~ok, :] = cos[:, ~ok] = np.nan
        dis[~ok, :] = dis[:, ~ok] = np.nan
        res[k] = {"rel": rel.tolist(), "cos": cos.tolist(), "dis": dis.tolist(),
                  "norm": [float(np.mean(v)) if v else float("nan") for v in norms[k]]}
        np.set_printoptions(precision=2, suppress=True, linewidth=200)
        print(f"\n== {k}: reliability (cos of a group's two halves)")
        for g, r, nm, c in zip(groups, rel, res[k]["norm"], cnt.sum(0)):
            print(f"  {g:18s} rel {r:+.2f}  mean grad norm {nm:.3g}  batches {int(c)}")
        print(f"== {k}: between-group cosine (raw, across halves)\n" + "\n".join(
            f"  {g[:12]:12s} " + " ".join(f"{x:+.2f}" for x in row) for g, row in zip(groups, cos)))
        print(f"== {k}: noise-corrected\n" + "\n".join(
            f"  {g[:12]:12s} " + " ".join(f"{x:+.2f}" for x in row) for g, row in zip(groups, dis)), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{Path(a.ckpt).parent.name}_{Path(a.ckpt).stem}.json"
    f.write_text(json.dumps(res, indent=1))
    print("wrote", f, flush=True)


if __name__ == "__main__":
    main()
