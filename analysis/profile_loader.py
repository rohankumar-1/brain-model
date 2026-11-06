"""Where a training item's CPU time goes (2026-11-04): cProfile over ``MultitaskDataset.__getitem__`` for random items of
a run config's training set, single process (what each dataloader worker does).

    python runs/diagnostics/profile_loader.py [--config configs/multitask/r14.yaml] [--n 150]
"""
import argparse
import cProfile
import pstats
import time

import numpy as np
import pandas as pd
import torch
import yaml

from brain.multitask.data import MultitaskDataset, load_labels
from brain.multitask.registry import load_registry
from brain.train.defaults import _merge, default_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multitask/r14.yaml")
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    cfg = _merge(default_config("multitask"), yaml.safe_load(open(a.config)))
    d, tk, aug = cfg["data"], cfg["tasks"], cfg.get("aug") or {}
    reg = load_registry(d.get("registry"))
    labels, stats = load_labels(d["labels_dir"])
    vols = pd.read_csv(f"{d['store_root']}/{d.get('index_name', 'volumes.csv')}", low_memory=False)
    ds = MultitaskDataset(d["store_root"], labels, vols, reg, stats, split="train", train=True,
                          tasks=reg.train_ids(tk), k_max=int(tk["k_max"]), p_all=float(tk["p_all"]),
                          rare_beta=float(tk["rare_beta"]), max_seg=int(tk["max_seg"]),
                          group_weights=tk.get("group_weights"), mask_p=float(tk.get("mask_p", 0.0)),
                          skull_strip=float((aug.get("skull_strip") or {}).get("p", 0.0)),
                          task_aware=bool(aug.get("task_aware", False)),
                          strip_match=list((aug.get("skull_strip") or {}).get("match") or []),
                          synth_contrast=aug.get("synth_contrast"))
    idx = np.random.default_rng(0).choice(len(ds), a.n, replace=False)
    torch.manual_seed(0)
    times = []
    prof = cProfile.Profile()
    for i in idx:
        t0 = time.perf_counter()
        prof.enable()
        ds[int(i)]
        prof.disable()
        times.append(time.perf_counter() - t0)
    t = np.asarray(times)
    print(f"[loader] {a.n} items: mean {t.mean():.3f} s, median {np.median(t):.3f} s, p90 {np.percentile(t, 90):.3f} s")
    pstats.Stats(prof).sort_stats("cumulative").print_stats(28)


if __name__ == "__main__":
    main()
