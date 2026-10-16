#!/usr/bin/env python3
"""Run the frozen-encoder downstream suite on one labeled task.

  python -m downstream.run_suite --task cq500 --ckpt runs/vit_tiny_v2/last.pt
  python -m downstream.run_suite --task brats2023_gli --ckpt runs/vit_tiny_v2/last.pt

Classification and regression: cached slice embeddings (class + mean patch + max
patch) feed pooled or slice-attention heads. Focal tasks use 64 axial slices;
eval windows default to ``multi`` (CT brain/subdural/bone, MRI tissue/wide).
Segmentation: a UNETR-style decoder with skip connections from early ViT blocks
(``downstream.seg_train``).

Results land in runs/downstream/<task>/<ckpt_tag>/<run>/.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from downstream.embeddings import UNIFORM_SLICES, load_or_build_embeddings, resolve_path
from downstream.heads import DEFAULT_HEADS, build_heads, make_logreg_pool, make_ridge
from downstream.token_probes import make_token_heads
from downstream.token_readout import token_grid_zyx, token_readout_views
from downstream.metrics import HEADLINE, binary_metrics, bootstrap_ci, multilabel_metrics, regression_metrics
from downstream.results import eval_banner, finalize_run, new_run_dir, print_head_table, save_head_result, write_json
from downstream.tasks import (
    FOCAL_SLICE_TASKS, FOCAL_SLICES, SEGMENTATION_TASKS, TASK_NAMES, TaskBundle,
    load_task, load_volumes, split_xy, DEFAULT_VOLUMES,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=list(TASK_NAMES), required=True)
    p.add_argument("--ckpt", type=Path, default=None, help="ViT (or legacy VAE) checkpoint")
    p.add_argument("--slices-root", type=Path, default=DEFAULT_VOLUMES.parent)
    p.add_argument("--out-root", type=Path, default=ROOT / "runs" / "downstream")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "runs" / "downstream" / "_cache")
    p.add_argument("--max-slices", type=int, default=None,
                   help="uniform axial slices per volume (default 64 on focal tasks, 32 otherwise)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--force-embed", action="store_true")
    p.add_argument("--heads", type=str, default=DEFAULT_HEADS, help="comma-separated head names")
    p.add_argument("--msd-target", type=str, default="enhancing_high")
    p.add_argument("--cq500-threshold", type=float, default=0.5)
    p.add_argument("--seg-steps", type=int, default=15000)
    p.add_argument("--seg-batch", type=int, default=32)
    p.add_argument("--seg-lr", type=float, default=1e-3)
    p.add_argument("--seg-sequences", choices=["all", "first"], default="all",
                   help="all: encode t1n/t1c/t2w/t2f separately and fuse tokens; first: t1n only")
    p.add_argument("--seg-scale", type=int, choices=[1, 2], default=1, help="upsample input (2 = 512 px)")
    p.add_argument("--seg-last-only", action="store_true",
                   help="segmentation decoder reads only the final block (like the 3D token decoder)")
    p.add_argument("--cache-only", action="store_true", help="build the segmentation pixel cache and exit")
    p.add_argument("--volumes-csv", type=Path, default=None, help="volume table (default slices_zarr/volumes.csv)")
    p.add_argument("--window", choices=["canonical", "multi", "shifted"], default="multi",
                   help="asinh window(s) for frozen-head embeddings; segmentation stays canonical")
    p.add_argument("--token-heads", action=argparse.BooleanOptionalAction, default=True,
                   help="3D tokens: also fit the 3D-conv probe (tok_conv) on the full-width grid")
    p.add_argument("--readout", choices=["axial", "raw"], default="axial",
                   help="axial: PCA, then per-plane and 4x4x4 probes on a cubic token grid; raw: pool every token")
    return p.parse_args()


def score(task: TaskBundle, y_true: np.ndarray, y_score: np.ndarray) -> dict:
    if task.task_type == "multilabel":
        return multilabel_metrics(y_true, y_score, label_names=task.label_names)
    if task.task_type == "regression":
        return regression_metrics(y_true, y_score)
    return binary_metrics(y_true, y_score)


def require_stores(task: TaskBundle, slices_root: Path) -> None:
    kinds = [("image", task.image_paths)] + ([("mask", task.mask_paths)] if task.task_type == "segmentation" else [])
    missing = {k: [v for v in task.volume_ids if not resolve_path(slices_root, paths.get(v, "")).is_file()]
               for k, paths in kinds}
    if any(missing.values()):
        raise SystemExit(f"{task.name} is not ready: " + "; ".join(
            f"{len(v)} {k}s missing (e.g. {v[0]})" for k, v in missing.items() if v))
    print(f"[data] {len(task.volume_ids)} volumes on disk")


def write_seg_manifest(task: TaskBundle, out_root: Path) -> None:
    out = out_root / task.name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"volume_id": task.volume_ids, "split": [task.splits[v] for v in task.volume_ids],
                  "image_path": [task.image_paths.get(v, "") for v in task.volume_ids],
                  "mask_path": [task.mask_paths.get(v, "") for v in task.volume_ids]}).to_csv(out / "manifest.csv", index=False)
    write_json(out / "regions.json", {k: list(v) for k, v in task.regions.items()})


def fit_and_score(head, task: TaskBundle, packs: dict) -> tuple[dict, np.ndarray]:
    tr, va, te = packs["train"], packs["val"], packs["test"]
    head.fit(tr["mus"], tr["y"], tr["mask"], va["mus"], va["y"], va["mask"],
             z=tr.get("z"), z_val=va.get("z"))
    scores = head.predict_scores(te["mus"], te["mask"], z=te.get("z"))
    metrics = score(task, te["y"], scores)
    key = HEADLINE[task.task_type]
    y = te["y"]
    lo, hi = bootstrap_ci(lambda i: score(task, y[i], scores[i]).get(key, float("nan")), len(y))
    metrics[f"{key}_ci"] = [lo, hi]
    if task.cohort:
        cohorts = np.asarray(te["cohort"])
        metrics["by_cohort"] = {
            c: {**score(task, y[cohorts == c], scores[cohorts == c]), "n": int((cohorts == c).sum())}
            for c in sorted(set(cohorts))
        }
    metrics.update(head=head.name, n_train=len(tr["volume_ids"]), n_val=len(va["volume_ids"]), n_test=len(te["volume_ids"]))
    if getattr(head, "chosen_", None) is not None:
        metrics["chosen_param"] = float(head.chosen_)
    return metrics, scores


def ensure_val(packs: dict, frac: float = 0.15, seed: int = 0) -> dict:
    """Tasks without a val split hold out a fraction of train for early stopping."""
    if len(packs["val"]["volume_ids"]):
        return packs
    tr = packs["train"]
    order = np.random.default_rng(seed).permutation(len(tr["volume_ids"]))
    n = max(1, int(frac * len(order)))
    take = lambda idx: {k: ([v[i] for i in idx] if isinstance(v, list) else v[idx]) for k, v in tr.items()}
    return {**packs, "val": take(order[:n]), "train": take(order[n:])}


def _rename(head, name: str, pool: str | None = None):
    head.name = name
    if pool is not None:
        head.pool = pool
    return head


def readout_head_sets(task: TaskBundle, views: dict, seed: int) -> list[tuple[str, dict, list]]:
    """Heads for each spatial view. Linear probes see the flattened layout."""
    reg = task.task_type == "regression"
    multi = task.task_type == "multilabel"
    n_out = len(task.label_names) if multi else 1
    sets = []

    def linear(name: str, pool: str):
        if reg:
            return _rename(make_ridge(), name, pool)
        return _rename(make_logreg_pool(multi, seed), name, pool)

    plane_heads = [linear("plane_ridge" if reg else "plane_logreg", "flat")]
    grid_heads = [linear("grid_ridge" if reg else "grid_logreg", "mean")]
    global_heads = [linear("pca_global_ridge" if reg else "pca_global_logreg", "mean")]
    sets.append(("plane", views["plane"], plane_heads))
    sets.append(("grid", views["grid"], grid_heads))
    sets.append(("global", views["global"], global_heads))
    return sets


def headline(task: TaskBundle, m: dict) -> str:
    key = HEADLINE[task.task_type]
    lo, hi = m[f"{key}_ci"]
    return f"{key}={m[key]:.4f} [{lo:.4f}, {hi:.4f}]"


def main() -> None:
    args = parse_args()
    ckpt = args.ckpt if args.ckpt is None or args.ckpt.is_absolute() else (ROOT / args.ckpt).resolve()
    if ckpt is not None and not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    volumes = load_volumes(args.volumes_csv)
    msd = args.slices_root / "labels" / "msd_volume_labels.csv"
    task = load_task(args.task, volumes, threshold=args.cq500_threshold,
                     msd_labels=msd if msd.exists() else None, msd_target=args.msd_target)
    eval_banner(task.name, f"{task.task_type}  ·  {len(task.volume_ids)} volumes")
    print(f"[task] {task.name} type={task.task_type} n_volumes={len(task.volume_ids)}")
    max_slices = args.max_slices
    if max_slices is None:
        max_slices = FOCAL_SLICES if task.name in FOCAL_SLICE_TASKS else UNIFORM_SLICES
    if ckpt is not None or args.cache_only:
        require_stores(task, args.slices_root)

    if task.name in SEGMENTATION_TASKS or task.task_type == "segmentation":
        if ckpt is not None:
            from downstream.embeddings import peek_is_brain3d

            if peek_is_brain3d(ckpt):
                raise SystemExit(
                    "3D BrainModel segmentation is not wired yet (needs a 3D decoder on token maps). "
                    "Frozen clf/reg tasks work. See docs/brainmodel_downstream.md"
                )
        write_seg_manifest(task, args.out_root)
        if args.window != "canonical":
            print("[seg] decoder uses the canonical asinh window; --window applies to frozen heads only")
        if args.cache_only:
            from downstream.seg_train import build_cache

            build_cache(task, args.slices_root, args.cache_dir, max_slices, args.seg_sequences == "all")
            return
        if ckpt is None:
            print("[seg] manifest only. Pass --ckpt to train the decoder.")
            return
        from downstream.seg_train import run_latent_segmentation

        run_latent_segmentation(task, ckpt=ckpt, slices_root=args.slices_root, out_root=args.out_root,
                                cache_dir=args.cache_dir, max_slices=max_slices, seed=args.seed,
                                run_name=args.run_name, steps=args.seg_steps, batch_size=args.seg_batch,
                                lr=args.seg_lr, all_sequences=args.seg_sequences == "all", scale=args.seg_scale,
                                window="canonical", last_only=args.seg_last_only)
        return
    if ckpt is None:
        raise SystemExit("--ckpt is required for classification and regression tasks")

    emb = load_or_build_embeddings(ckpt=ckpt, slices_root=args.slices_root, image_paths=task.image_paths,
                                   volume_ids=task.volume_ids, cache_dir=args.cache_dir, task=task.name,
                                   max_slices=max_slices, force=args.force_embed,
                                   window=args.window, modalities=task.modalities)
    print(f"[embed] {emb['mus'].shape} ({emb['representation']}) cache={emb['cache_path']}")
    packs = ensure_val(split_xy(task, emb), seed=args.seed)
    print("[split] " + " ".join(f"{s}={len(p['volume_ids'])}" for s, p in packs.items()))

    multilabel = task.task_type == "multilabel"
    grid = token_grid_zyx(emb["mus"].shape[1]) if emb["representation"] == "tokens" else None
    use_readout = args.readout == "axial" and grid is not None
    if use_readout:
        views = token_readout_views(packs, seed=args.seed, grid_zyx=grid)
        print(f"[readout] grid={grid} pca={views['pca_dim']} explained={views['explained_variance']:.3f} "
              f"plane={views['plane']['train']['mus'].shape} bins={views['grid']['train']['mus'].shape}")
        head_sets = readout_head_sets(task, views, args.seed)
        if args.token_heads:
            n_out = len(task.label_names) if multilabel else 1
            head_sets.append(("tokens", packs, make_token_heads(
                emb["mus"].shape[-1], n_out, multilabel, args.seed, task.task_type == "regression",
                grid_zyx=grid)))
    else:
        heads = build_heads(latent_dim=emb["latent_dim"], max_slices=emb["max_slices"],
                            n_out=len(task.label_names) if multilabel else 1, multilabel=multilabel,
                            seed=args.seed, which=[h.strip() for h in args.heads.split(",") if h.strip()],
                            task_type=task.task_type, spatial=emb["representation"] == "tokens")
        head_sets = [("raw", packs, heads)]
    run_dir = new_run_dir(args.out_root, task.name, ckpt, args.run_name)
    all_metrics = {}
    head_names = []
    scored = []
    for view_name, view_packs, heads in head_sets:
        for head in heads:
            print(f"\n[head] {head.name} ({view_name}) …")
            metrics, scores = fit_and_score(head, task, view_packs)
            save_head_result(run_dir, head.name, metrics, y_true=view_packs["test"]["y"], y_score=scores,
                             volume_ids=view_packs["test"]["volume_ids"])
            all_metrics[head.name] = metrics
            head_names.append(head.name)
            scored.append((head.name, view_name, metrics))
            extra = f"  param={metrics['chosen_param']:g}" if "chosen_param" in metrics else ""
            print("  " + headline(task, metrics) + extra)
    print_head_table(task.name, task.task_type, scored)
    config = {"task": task.name, "task_type": task.task_type, "label_names": task.label_names,
              "regions": {k: list(v) for k, v in task.regions.items()}, "ckpt": str(ckpt),
              "representation": emb["representation"], "slices_root": str(args.slices_root),
              "max_slices": max_slices, "latent_dim": emb["latent_dim"], "heads": head_names,
              "readout": "axial" if use_readout else "raw",
              "seed": args.seed, "window": args.window, "time_utc": datetime.now(timezone.utc).isoformat(),
              "embed_cache": str(emb["cache_path"]), **{f"n_{s}": len(p["volume_ids"]) for s, p in packs.items()}}
    print(f"[ok] summary → {finalize_run(run_dir, config, all_metrics)}")


if __name__ == "__main__":
    main()
