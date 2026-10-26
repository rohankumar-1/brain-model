#!/usr/bin/env python3
"""Run the frozen-encoder brain.downstream suite on one labeled task.

  python -m brain.downstream.run_suite --task cq500 --ckpt runs/vit_tiny_v2/last.pt
  python -m brain.downstream.run_suite --task brats2023_gli --ckpt runs/vit_tiny_v2/last.pt

Classification and regression: cached slice embeddings (class + mean patch + max
patch) feed pooled or slice-attention heads. Focal tasks use 64 axial slices;
eval windows default to ``multi`` (CT brain/subdural/bone, MRI tissue/wide).
Segmentation: a UNETR-style decoder with skip connections from early ViT blocks
(``brain.downstream.seg_train``).

Results land in runs/downstream/<task>/<ckpt_tag>/<run>/ (predictions, config) and as one experiment record per
head (``brain.downstream.records``; query with ``python -m brain.downstream.db report``). Caches live under
runs/downstream/_cache/v<store format> so a re-prepped store never reuses stale embeddings.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import REPO_ROOT as ROOT
from brain.prep.store import FORMAT_VERSION
from brain.downstream.embeddings import UNIFORM_SLICES, load_or_build_embeddings, resolve_path
from brain.downstream.heads import DEFAULT_HEADS, build_heads, make_logreg_pool, make_ridge
from brain.downstream.token_probes import make_token_heads
from brain.downstream.token_readout import token_grid_zyx, token_readout_views
from brain.downstream import cv
from brain.downstream.metrics import HEADLINE, binary_metrics, bootstrap_ci, multilabel_metrics, regression_metrics
from brain.downstream.results import eval_banner, finalize_run, new_run_dir, print_head_table, save_head_result, write_json
from brain.downstream.tasks import (
    FOCAL_SLICE_TASKS, FOCAL_SLICES, SEGMENTATION_TASKS, TASK_NAMES, TaskBundle,
    load_task, load_volumes, split_xy, DEFAULT_VOLUMES,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=list(TASK_NAMES), required=True)
    p.add_argument("--ckpt", type=Path, default=None, help="2D ViT or 3D BrainEncoder checkpoint")
    p.add_argument("--slices-root", type=Path, default=DEFAULT_VOLUMES.parent)
    p.add_argument("--out-root", type=Path, default=ROOT / "runs" / "downstream")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "runs" / "downstream" / "_cache" / f"v{FORMAT_VERSION}")
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
    p.add_argument("--cv", choices=["auto", "on", "off"], default="auto",
                   help="repeated subject-grouped k-fold CV instead of the fixed split (auto: small held-out tasks)")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=3)
    p.add_argument("--cv-max-subjects", type=int, default=200, help="auto mode: CV below this many subjects")
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
    if len(va["volume_ids"]):                        # val numbers let a report pick heads without peeking at test
        metrics["val"] = score(task, va["y"], head.predict_scores(va["mus"], va["mask"], z=va.get("z")))
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
    from brain.downstream.records import now_utc, pretrain_overlap

    started = now_utc()
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
            from brain.downstream.embeddings import peek_is_brain3d

            if peek_is_brain3d(ckpt):
                raise SystemExit(
                    "3D BrainModel segmentation is not wired yet (needs a 3D decoder on token maps). "
                    "Frozen clf/reg tasks work. See docs/brainmodel_downstream.md"
                )
        write_seg_manifest(task, args.out_root)
        if args.window != "canonical":
            print("[seg] decoder uses the canonical asinh window; --window applies to frozen heads only")
        if args.cache_only:
            from brain.downstream.seg_train import build_cache

            build_cache(task, args.slices_root, args.cache_dir, max_slices, args.seg_sequences == "all")
            return
        if ckpt is None:
            print("[seg] manifest only. Pass --ckpt to train the decoder.")
            return
        from brain.downstream.seg_train import run_latent_segmentation

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
    multilabel = task.task_type == "multilabel"
    grid = token_grid_zyx(emb["mus"].shape[1]) if emb["representation"] == "tokens" else None
    use_readout = args.readout == "axial" and grid is not None

    def head_sets_for(task_s: TaskBundle, packs_s: dict) -> list[tuple[str, dict, list]]:
        """Fresh heads for one split (PCA readouts are refit on this split's training volumes)."""
        if use_readout:
            views = token_readout_views(packs_s, seed=args.seed, grid_zyx=grid)
            print(f"[readout] grid={grid} pca={views['pca_dim']} explained={views['explained_variance']:.3f} "
                  f"plane={views['plane']['train']['mus'].shape} bins={views['grid']['train']['mus'].shape}")
            sets = readout_head_sets(task_s, views, args.seed)
            if args.token_heads:
                n_out = len(task_s.label_names) if multilabel else 1
                sets.append(("tokens", packs_s, make_token_heads(
                    emb["mus"].shape[-1], n_out, multilabel, args.seed, task_s.task_type == "regression",
                    grid_zyx=grid)))
        else:
            sets = [("raw", packs_s, build_heads(
                latent_dim=emb["latent_dim"], max_slices=emb["max_slices"],
                n_out=len(task_s.label_names) if multilabel else 1, multilabel=multilabel, seed=args.seed,
                which=[h.strip() for h in args.heads.split(",") if h.strip()], task_type=task_s.task_type,
                spatial=emb["representation"] == "tokens"))]
        # Prefix-token models: a linear head on each global embedding the model has (CLS; the registers; both),
        # next to the patch-token heads, so a CLS / register model is read out the way it summarises a volume.
        n_vol = len(emb["volume_ids"])
        globals_ = {}
        if emb.get("cls") is not None:
            globals_["cls"] = np.asarray(emb["cls"], dtype=np.float32)
        if emb.get("registers") is not None:
            globals_["reg"] = np.asarray(emb["registers"], dtype=np.float32)
            if "cls" in globals_:
                globals_["clsreg"] = np.concatenate([globals_["cls"], globals_["reg"]], axis=1)
        # Multitask checkpoints: the trained tasks' slots (one query per task) and the model's own task outputs
        # (brain.multitask.slots), read out like the other global embeddings.
        if emb.get("slots") is not None:
            globals_["slots"] = np.asarray(emb["slots"], dtype=np.float32)
            globals_["taskout"] = np.asarray(emb["taskout"], dtype=np.float32)
        if emb.get("general") is not None:
            globals_["general"] = np.asarray(emb["general"], dtype=np.float32)
        reg = task_s.task_type == "regression"
        for view_name, vec in globals_.items():
            g_emb = {**emb, "mus": vec[:, None, :], "mask": np.ones((n_vol, 1), dtype=bool)}
            g_emb.pop("z", None)
            g_packs = ensure_val(split_xy(task_s, g_emb), seed=args.seed)
            head = _rename(make_ridge(), f"{view_name}_ridge", "mean") if reg else \
                _rename(make_logreg_pool(multilabel, args.seed), f"{view_name}_logreg", "mean")
            sets.append((view_name, g_packs, [head]))
        return sets

    overlap = pretrain_overlap(task.volume_ids, volumes)
    cv_on = cv.use_cv(args.cv, task, volumes, overlap, args.cv_max_subjects)
    run_dir = new_run_dir(args.out_root, task.name, ckpt, args.run_name)
    all_metrics = {}
    head_names = []
    scored = []
    if not cv_on:
        packs = ensure_val(split_xy(task, emb), seed=args.seed)
        print("[split] " + " ".join(f"{s}={len(p['volume_ids'])}" for s, p in packs.items()))
        for view_name, view_packs, heads in head_sets_for(task, packs):
            for head in heads:
                print(f"\n[head] {head.name} ({view_name}) …")
                metrics, scores = fit_and_score(head, task, view_packs)
                metrics["view"] = view_name
                save_head_result(run_dir, head.name, metrics, y_true=view_packs["test"]["y"], y_score=scores,
                                 volume_ids=view_packs["test"]["volume_ids"])
                all_metrics[head.name] = metrics
                head_names.append(head.name)
                scored.append((head.name, view_name, metrics))
                extra = f"  param={metrics['chosen_param']:g}" if "chosen_param" in metrics else ""
                print("  " + headline(task, metrics) + extra)
        split_sizes = {f"n_{s}": len(p["volume_ids"]) for s, p in packs.items()}
        cv_info = None
    else:
        folds = cv.make_folds(task, volumes, args.cv_folds, args.cv_repeats, args.seed)
        n_folds = len({f for _, f, _ in folds})
        print(f"[cv] {len(np.unique(cv.subjects(task, volumes)))} subjects, {n_folds} folds x "
              f"{args.cv_repeats} repeats (pretrain overlap {overlap})")
        preds: dict[str, list] = {}
        vals: dict[str, list] = {}
        views_of: dict[str, str] = {}
        sizes: dict[str, list] = {}
        for r, f, splits in folds:
            task_f = replace(task, splits=splits)
            packs_f = split_xy(task_f, emb)
            for view_name, view_packs, heads in head_sets_for(task_f, packs_f):
                for head in heads:
                    m, scores = fit_and_score(head, task_f, view_packs)
                    preds.setdefault(head.name, []).append((r, view_packs["test"]["volume_ids"], scores))
                    va = view_packs["val"]
                    if len(va["volume_ids"]):          # val predictions, pooled over folds to choose heads stably
                        vals.setdefault(head.name, []).append(
                            (va["y"], head.predict_scores(va["mus"], va["mask"], z=va.get("z"))))
                    sizes.setdefault(head.name, []).append((m["n_train"], m["n_val"]))
                    views_of[head.name] = view_name
            print(f"[cv] repeat {r + 1} fold {f + 1}/{n_folds} done")
        key = HEADLINE[task.task_type]
        for name, fold_preds in preds.items():
            reps = cv.pooled(task, fold_preds)
            per_rep = {r: score(task, y, s) for r, (_, y, s) in reps.items()}
            ids0, y0, s0 = reps[min(reps)]
            metrics = dict(per_rep[min(per_rep)])
            metrics[key] = float(np.nanmean([m.get(key, float("nan")) for m in per_rep.values()]))
            metrics[f"{key}_repeats"] = [float(m.get(key, float("nan"))) for m in per_rep.values()]
            lo, hi = bootstrap_ci(lambda i: score(task, y0[i], s0[i]).get(key, float("nan")), len(y0))
            metrics[f"{key}_ci"] = [lo, hi]
            # val: one score over every fold's inner-validation predictions pooled (a ~15% inner set per fold is a
            # handful of subjects; averaging per-fold R2/AUC over those is too noisy to pick a head on)
            if vals.get(name):
                metrics["val"] = score(task, np.concatenate([y for y, _ in vals[name]]),
                                       np.concatenate([p for _, p in vals[name]]))
            n_tr = [a for a, _ in sizes[name]]
            metrics.update(head=name, view=views_of[name], n_train=int(np.mean(n_tr)),
                           n_val=int(np.mean([b for _, b in sizes[name]])), n_test=len(ids0),
                           cv={"folds": n_folds, "repeats": args.cv_repeats})
            save_head_result(run_dir, name, metrics, y_true=y0, y_score=s0, volume_ids=ids0)
            all_metrics[name] = metrics
            head_names.append(name)
            scored.append((name, views_of[name], metrics))
            print(f"  {name}: " + headline(task, metrics) + f"  (repeats {[round(x, 3) for x in metrics[f'{key}_repeats']]})")
        split_sizes = {"n_train": int(np.mean([m["n_train"] for m in all_metrics.values()])),
                       "n_test": len(task.volume_ids)}
        cv_info = {"folds": n_folds, "repeats": args.cv_repeats, "val_frac": cv.VAL_FRAC, "grouped_by": "dedup_key"}
    print_head_table(task.name, task.task_type, scored)
    config = {"task": task.name, "task_type": task.task_type, "label_names": task.label_names,
              "regions": {k: list(v) for k, v in task.regions.items()}, "ckpt": str(ckpt),
              "representation": emb["representation"], "slices_root": str(args.slices_root),
              "max_slices": max_slices, "latent_dim": emb["latent_dim"], "heads": head_names,
              "readout": "axial" if use_readout else "raw",
              "seed": args.seed, "window": args.window, "time_utc": datetime.now(timezone.utc).isoformat(),
              "started_utc": started, "volumes_csv": str(args.volumes_csv or DEFAULT_VOLUMES),
              "token_heads": args.token_heads, "msd_target": args.msd_target if task.name == "msd" else None,
              "pretrain_overlap": overlap,
              "embed_cache": str(emb["cache_path"]), **split_sizes, "cv": cv_info}
    print(f"[ok] summary → {finalize_run(run_dir, config, all_metrics)}")
    drop_big_cache(Path(emb["cache_path"]))


def drop_big_cache(path: Path) -> None:
    """Delete a task's token cache once its heads are done when the tokens are memory-mapped (> BIG_CACHE_BYTES; RSNA
    is ~40 GB per model): seven models' worth filled the scratch quota on 2026-10-25. BRAIN_KEEP_BIG_CACHE=1 keeps it."""
    import os

    side = path.with_suffix(".mus.npy")
    if not side.exists() or os.environ.get("BRAIN_KEEP_BIG_CACHE") == "1":
        return
    gb = side.stat().st_size / 2**30
    side.unlink()
    path.unlink(missing_ok=True)
    print(f"[cache] dropped the {gb:.0f} GB token cache {side} (BRAIN_KEEP_BIG_CACHE=1 keeps it)", flush=True)


if __name__ == "__main__":
    main()
