"""A multitask checkpoint's own trained heads applied unchanged to external datasets: does what a head learned on one
cohort transfer to another without refitting? (The frozen benchmark instead fits a new readout on each dataset.)

    python -m brain.multitask.transfer_heads --ckpts BQ-R6 BQ-12k-tokens --out runs/bench/transfer

``PAIRS``: a benchmark task (and the label within it), the trained registry task whose head answers it, and which
scans are scored: ``all`` for tasks the benchmark cross-validates (every scan is a test scan there), ``test`` for
fixed-split ones, so the scans match the frozen-readout scores. Each head is also scored on its own in-domain test
split; per-scan predictions go to ``transfer_preds.csv``. Regression heads are mapped back to the label's units with the checkpoint's training statistics. Scores have
a bootstrap 95% interval (``test_eval.bootstrap``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brain.common.paths import resolve

PAIRS = [
    ("cq500", "ICH", "hemorrhage_any", "all"),
    ("cq500", "IPH", "hemorrhage_intraparenchymal", "all"),
    ("cq500", "IVH", "hemorrhage_intraventricular", "all"),
    ("cq500", "SDH", "hemorrhage_subdural", "all"),
    ("cq500", "SAH", "hemorrhage_subarachnoid", "all"),
    ("cq500_edh", None, "hemorrhage_epidural", "all"),
    ("ixi_age", None, "age", "test"),
    ("pixar_age", None, "age", "test"),
    ("ms3seg_age", None, "age", "all"),
]
KIND = {"hemorrhage": "binary", "age": "regression"}


def blur_to(mm: float):
    """``evaluate.prepare`` with every scan first blurred along its acquisition axis to an effective ``mm`` slice
    thickness (``augment.through_plane``, the training augmentation; scans already that thick are unchanged): a test of
    whether a head trained on thick slices fails on thin ones for the thinness."""
    import brain.multitask.evaluate as ev
    from brain.data.augment import through_plane

    orig = ev.prepare

    def prep(batch, cfg, device, patch_xyz, train, **kw):
        img = batch["image"].to(device).float().unsqueeze(1)
        th = batch["thick"].to(device) if "thick" in batch else None
        img = through_plane(img, {"p": 1.0, "fwhm_vox": (mm, mm)}, None if th is None else (th[:, 0], th[:, 1]))
        return orig({**batch, "image": img[:, 0]}, cfg, device, patch_xyz, train, **kw)

    ev.prepare = prep


def _heads(model, cfg, reg, store, labels, volumes, stats, task: str, split: str, ids: list[str], dev, amp, workers):
    """{volume_id: head output} for ``task``'s head on ``ids`` (rows of ``labels``, split ``split``)."""
    from brain.multitask.evaluate import run_eval
    from brain.multitask.fewshot import _loader

    loader, _ = _loader(store, labels, volumes, reg, stats, task, split, ids=ids, use="train", workers=workers)
    _, pr = run_eval(model, loader, cfg, reg, stats, dev, tuple(model.backbone.patch_xyz), amp, return_preds=True)
    return {v: (y, p) for y, p, v in pr.get(reg.index[task], [])}


def main() -> None:
    from brain.bench.spec import load
    from brain.downstream.models import resolve_model
    from brain.downstream.tasks import DEFAULT_VOLUMES, load_task, load_volumes
    from brain.multitask.data import checkpoint_stats, load_labels
    from brain.multitask.fewshot import load_model
    from brain.multitask.test_eval import bootstrap, pick, score

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True, help="multitask checkpoints or models.yaml names")
    ap.add_argument("--max-test", type=int, default=400, help="in-domain test scans per head")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/bench/transfer")
    ap.add_argument("--blur-to-mm", type=float, default=None,
                    help="blur every scan through-plane to this slice thickness first (thin-vs-thick test)")
    a = ap.parse_args()
    if a.blur_to_mm:
        blur_to(a.blur_to_mm)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    labels_dir = resolve(load().raw["labels_dir"])
    labels, base_stats = load_labels(labels_dir)
    bench_vols = load_volumes(DEFAULT_VOLUMES)
    rows, scans = [], []
    for name in a.ckpts:
        ref = resolve_model(name)
        model, reg, cfg = load_model(Path(ref.ckpt), dev)
        stats = checkpoint_stats(model.label_stats, labels_dir, base_stats)
        store = resolve(cfg["data"]["store_root"])
        volumes = pd.read_csv(store / cfg["data"].get("index_name", "volumes.csv"), low_memory=False)
        for trained in dict.fromkeys(t for *_, t, _ in PAIRS):
            kind = KIND[trained.split("_")[0]]
            own = labels[(labels.task == trained) & (labels.split == "test") & (labels.use == "train")]
            out = _heads(model, cfg, reg, store, labels, volumes, stats, trained, "test",
                         pick(own.volume_id.unique().tolist(), a.max_test), dev, amp, a.workers)
            y, p = np.array([v[0] for v in out.values()]), np.array([v[1] for v in out.values()])
            lo, hi = bootstrap(kind, y, p)
            rows.append({"model": ref.name, "head": trained, "where": "in-domain", "benchmark": "", "label": "",
                         "n": len(y), "score": score(kind, y, p), "lo": lo, "hi": hi})
            print(f"[transfer] {ref.name} {trained} in-domain {rows[-1]['score']:.3f} (n={len(y)})", flush=True)
        for bench, label, trained, scope in PAIRS:
            kind = KIND[trained.split("_")[0]]
            task = load_task(bench, bench_vols)
            y_all = np.asarray(task.y, dtype=float)
            y_all = y_all[:, task.label_names.index(label)] if label else y_all.reshape(len(y_all), -1)[:, 0]
            truth = dict(zip(map(str, task.volume_ids), y_all))
            ids = [v for v in truth if scope == "all" or task.splits.get(v) == "test"]
            # the external scans as rows of the trained task (their label value is unused: scored against ``truth``)
            fake = pd.DataFrame({"volume_id": ids, "task": trained, "value": 0.0, "ref": "", "split": "test",
                                 "use": "train"})
            out = _heads(model, cfg, reg, store, fake, volumes, stats, trained, "test", ids, dev, amp, a.workers)
            vids = [v for v in ids if v in out]
            p = np.array([out[v][1] for v in vids], dtype=float)
            if kind == "regression":                       # standardised head output -> label units
                p = p * float(stats[trained]["std"]) + float(stats[trained]["mean"])
            y = np.array([truth[v] for v in vids], dtype=float)
            scans += [{"model": ref.name, "head": trained, "benchmark": bench, "label": label or "", "volume_id": v,
                       "y": yy, "p": pp} for v, yy, pp in zip(vids, y, p)]
            lo, hi = bootstrap(kind, y, p)
            rows.append({"model": ref.name, "head": trained, "where": "external", "benchmark": bench,
                         "label": label or "", "n": len(y), "score": score(kind, y, p), "lo": lo, "hi": hi})
            print(f"[transfer] {ref.name} {trained} -> {bench}{':' + label if label else ''} "
                  f"{rows[-1]['score']:.3f} (n={len(y)})", flush=True)
    outdir = resolve(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "transfer_heads.csv", index=False)
    pd.DataFrame(scans).to_csv(outdir / "transfer_preds.csv", index=False)     # per scan, for stratified analyses
    print(f"[transfer] -> {outdir / 'transfer_heads.csv'}")


if __name__ == "__main__":
    main()
