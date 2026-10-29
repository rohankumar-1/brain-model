"""Replace classical brain masks with SynthStrip without re-running affine registration.

Walks existing format-6 image zips, replaces their ``brain`` array, and (when present) patches
``brain_method`` / ``brain_frac`` / ``brain_ok`` / ``brain_dice_tpl`` in ``volumes.csv``. Dataset-supplied masks
(``brain_method=source``) are left alone. MRI intensities stay as stored (their z-scores used the classical
parenchyma mask) unless ``--restat`` re-standardises them on the new mask; the raw volume is gone, so that is an
affine correction of the stored z with the old clip limits kept saturated.

QC: when ``BRAIN_TEMPLATE`` / ``BRAIN_TEMPLATE_BRAIN`` are set and the volume was registered, ``brain_ok`` also requires
Dice >= ``brainmask.TEMPLATE_DICE_MIN`` with the template brain inside the scan's foreground.

    python -m brain.prep.remask --root STORE --shard 0/4 --gpu [--force] [--restat]
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path

import numpy as np

from brain.prep import brainmask, intensity, store

MASK_REV = brainmask.MASK_REV
MASK_TAILS =(".brain.zip", ".seg.zip", ".ICH.zip", ".lesion.zip", ".lvo.zip", ".cow.zip")
SKIP_DIRS = {"manifests_v4_backup", "fg_masks", "labels", "manifests_before_shared_grid"}


def _plain(v):
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer, int, bool, np.bool_)):
        return v.item() if hasattr(v, "item") else v
    return v if isinstance(v, (str, type(None))) else str(v)


def is_image_zip(path: Path) -> bool:
    name = path.name
    return name.endswith(".zip") and ".partial." not in name and not any(name.endswith(t) for t in MASK_TAILS)


def parse_folder_specs(raw: str) -> list[str]:
    """Split ``a,b`` or ``a:b``; globs are expanded later against the store root."""
    return [x.strip() for x in raw.replace(":", ",").split(",") if x.strip()]


def _dirs_for_specs(root: Path, specs: list[str]) -> list[Path]:
    dirs, seen = [], set()
    for spec in specs:
        if any(c in spec for c in "*?["):
            matches = sorted(p for p in root.glob(spec) if p.is_dir())
        else:
            p = root / spec
            matches = [p] if p.is_dir() else []
        for p in matches:
            if p.name in SKIP_DIRS or p in seen:
                continue
            seen.add(p)
            dirs.append(p)
    dirs.sort(key=lambda p: p.name)
    return dirs


def image_zips(root: Path, folders: list[str] | None = None, exclude: list[str] | None = None) -> list[Path]:
    root = Path(root)
    dirs = _dirs_for_specs(root, folders) if folders else sorted(
        p for p in root.iterdir() if p.is_dir() and p.name not in SKIP_DIRS)
    if exclude:
        drop = {p.name for p in _dirs_for_specs(root, exclude)} | set(exclude)
        dirs = [d for d in dirs if d.name not in drop]
    out = []
    for folder in dirs:
        out.extend(p for p in sorted(folder.glob("*.zip")) if is_image_zip(p))
    return out


def brain_zip_for(image: Path) -> Path:
    return image.with_name(image.stem + ".brain.zip")


def _squeeze(vol: np.ndarray) -> np.ndarray:
    if vol.ndim == 4:
        vol = vol[:, 0] if vol.shape[1] == 1 else vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"expected (Z,Y,X), got {tuple(vol.shape)}")
    return np.ascontiguousarray(vol, dtype=np.float32)


@functools.lru_cache(maxsize=4)
def template_prior(shape_zyx: tuple[int, int, int]) -> np.ndarray | None:
    """Template brain on the registered store grid of this shape, or None when no template is configured."""
    from brain.prep import registration

    tpath, bpath = os.environ.get("BRAIN_TEMPLATE"), os.environ.get("BRAIN_TEMPLATE_BRAIN")
    if not tpath or not bpath or not Path(tpath).is_file() or not Path(bpath).is_file():
        return None
    tpl = registration.load_template(tpath, bpath)
    return registration.template_brain_on_grid(tpl, registration.output_grid(tpl, tuple(shape_zyx[::-1])))


def remask_one(image: Path, *, force: bool = False, include_source: bool = False, restat: bool = False) -> dict:
    plan = remask_plan(image, force=force, include_source=include_source)
    if plan["status"] != "need":
        return plan
    t0 = time.perf_counter()
    try:
        mask = brainmask.synthstrip(plan["vol"], plan["modality"])
    except Exception as e:
        return {"image": plan["image"], "status": "error", "error": str(e)[:400],
                "prev_method": plan.get("prev_method")}
    return remask_finish(plan, mask, time.perf_counter() - t0, restat=restat)


def remask_plan(image: Path, *, force: bool = False, include_source: bool = False) -> dict:
    """Skip/error row, or ``status='need'`` plus volume for a batched GPU call."""
    rel = image.as_posix()
    base = {"image": rel, "status": "ok"}
    try:
        attrs = store.read_attrs(image, "image")
    except Exception as e:
        return {**base, "status": "error", "error": f"attrs: {e}"}
    if attrs.get("format_version") != store.FORMAT_VERSION:
        return {**base, "status": "skip_format"}
    method = str(attrs.get("brain_method") or "")
    base["prev_method"] = method or None
    if method == "source" and not include_source:
        return {**base, "status": "skip_source"}
    if not force:
        try:
            battrs = store.read_attrs(image, "brain")
            if battrs.get("brain_method") == "synthstrip" and battrs.get("brain_rev") == MASK_REV:
                return {**base, "status": "skip_done", "brain_frac": battrs.get("brain_frac")}
        except Exception:                              # no brain array yet
            pass
    try:
        vol = _squeeze(store.read(image, "image"))
    except Exception as e:
        return {**base, "status": "error", "error": f"read: {e}"}
    modality = str(attrs.get("modality") or ("ct" if attrs.get("kind") == "ct_hu" else "mri")).lower()
    if modality.startswith("ct"):
        modality = "ct"
    return {**base, "status": "need", "vol": vol, "modality": modality, "attrs": attrs,
            "stripped": bool(attrs.get("stripped_input"))}


def remask_finish(plan: dict, mask: np.ndarray, dt: float, restat: bool = False) -> dict:
    """Rewrite the image zip with the new ``brain`` array (and, with ``restat``, re-standardised MRI z-scores)."""
    vol, attrs = plan["vol"], plan.get("attrs") or {}
    prior = template_prior(tuple(vol.shape)) if attrs.get("reg_ok") else None
    fg = intensity.foreground(vol, plan["modality"]) if prior is not None else None
    info = {
        "brain_method": "synthstrip",
        "brain_rev": MASK_REV,
        **brainmask.qc(mask, prior, fg),
        "stripped_input": bool(plan.get("stripped")),
    }
    image = Path(plan["image"])
    arrays, array_attrs = store.read_all(image)
    image_attrs = {k: v for k, v in array_attrs["image"].items() if k != "format_version"}
    image_attrs |= {k: info[k] for k in store.BRAIN_ATTRS if k in info}
    extra = {}
    if restat and plan["modality"] == "mri" and info["brain_ok"]:
        res = intensity.mri_restat(vol, mask)
        if res is not None:
            z, m, s = res
            arrays["image"] = z[:, None]
            med, mad = image_attrs.get("raw_median"), image_attrs.get("raw_mad")
            image_attrs |= {"stats_from": "brain_synthstrip", "restat_m": m, "restat_s": s}
            if med is not None and mad is not None:  # z = (raw - med) / mad, so z' = (raw - (med + mad m)) / (mad s)
                image_attrs |= {"raw_median": float(med) + float(mad) * m, "raw_mad": float(mad) * s}
            extra = {"restat": True, "restat_m": m, "restat_s": s}
    out = {"image": (arrays["image"], image_attrs), "brain": (mask.astype(np.uint8), info)}
    out |= {n: (a, {k: v for k, v in array_attrs[n].items() if k != "format_version"})
            for n, a in arrays.items() if n not in out}
    store.write_arrays(image, out)
    row = {k: v for k, v in plan.items() if k not in {"vol", "modality", "stripped", "_k", "attrs"}}
    return {**row, "status": "ok", **info, **extra, "dt": round(dt, 2)}


def load_ok_rows(root: Path) -> list[dict]:
    rows = []
    for p in sorted(Path(root).glob("remask_*of*.jsonl")):
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def patch_volumes_csv(root: Path, rows: list[dict] | None = None) -> int:
    """Apply remask jsonl rows onto volumes.csv. Safe to run once after all shards finish."""
    path = Path(root) / "volumes.csv"
    rows = rows if rows is not None else load_ok_rows(root)
    if not path.is_file() or not rows:
        return 0
    import pandas as pd

    by_name = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        p = Path(r["image"])
        by_name[(p.parent.name, p.stem)] = r
    if not by_name:
        return 0
    df = pd.read_csv(path, low_memory=False)
    n = 0
    for i, rec in df.iterrows():
        key = (str(rec.get("folder", "")), str(rec.get("case_id", "")))
        if key not in by_name:
            continue
        info = by_name[key]
        df.at[i, "brain_method"] = info.get("brain_method")
        df.at[i, "brain_frac"] = info.get("brain_frac")
        df.at[i, "brain_ok"] = info.get("brain_ok")
        if info.get("brain_dice_tpl") is not None:
            df.at[i, "brain_dice_tpl"] = info.get("brain_dice_tpl")
        n += 1
    if n:
        tmp = path.with_name(path.name + f".partial.{os.getpid()}")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--gpu", action="store_true", help="pass -g to SynthStrip (needs CUDA)")
    ap.add_argument("--force", action="store_true", help="redo volumes already at the current MASK_REV")
    ap.add_argument("--include-source", action="store_true", help="overwrite dataset-supplied brain masks")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--folders", default="",
                    help="only these store folders (comma or colon; globs ok). Default: all")
    ap.add_argument("--exclude-folders", default="",
                    help="store folders to skip (comma or colon; globs ok)")
    ap.add_argument("--tag", default="",
                    help="manifest prefix so parallel jobs do not share remask_0ofN.jsonl")
    ap.add_argument("--batch", type=int, default=int(os.environ.get("SYNTHSTRIP_BATCH", 8)),
                    help="volumes per GPU forward (padded). OOM splits the batch.")
    ap.add_argument("--restat", action="store_true",
                    help="MRI: rewrite the image z-scores re-standardised on the new mask (only when brain_ok)")
    ap.add_argument("--patch-index", action="store_true",
                    help="only fold remask_*.jsonl into volumes.csv (run once after shards)")
    a = ap.parse_args()
    if a.patch_index:
        n = patch_volumes_csv(a.root)
        print(f"[remask] patched {n} volumes.csv rows", flush=True)
        return
    if a.gpu:
        os.environ["SYNTHSTRIP_GPU"] = "1"
    i, n = map(int, a.shard.split("/"))
    folder_list = parse_folder_specs(a.folders)
    exclude_list = parse_folder_specs(a.exclude_folders)
    images = image_zips(a.root, folder_list or None, exclude_list or None)[i::n]
    if a.limit:
        images = images[: a.limit]
    tag = f"{a.tag}_" if a.tag else ""
    manifest = a.root / f"remask_{tag}{i}of{n}.jsonl"
    batch_n = max(1, a.batch)
    print(f"[remask] shard {i}/{n}: {len(images)} images, gpu={a.gpu} batch={batch_n} "
          f"folders={folder_list or 'ALL'} exclude={exclude_list or '-'} tag={a.tag or '-'} "
          f"synthstrip={os.environ.get('BRAIN_SYNTHSTRIP')}", flush=True)
    t0 = time.time()
    done = []
    pending: list[dict] = []
    sess = None

    def emit(k: int, row: dict) -> None:
        row = {key: _plain(val) for key, val in row.items() if key not in {"vol", "_k", "attrs"}}
        out.write(json.dumps(row) + "\n")
        out.flush()
        done.append(row)
        if k % 20 == 0 or row["status"] not in ("ok", "skip_done", "skip_source", "skip_format"):
            name = Path(row["image"]).name
            folder = Path(row["image"]).parent.name
            print(f"[remask] {k}/{len(images)} {folder}/{name} {row['status']} "
                  f"({(time.time() - t0) / max(k, 1):.1f} s/vol wall)", flush=True)

    def flush() -> None:
        nonlocal sess
        if not pending:
            return
        if sess is None:
            from brain.prep.synthstrip_runtime import session as ss_session
            sess = ss_session(gpu=a.gpu)
        t_inf = time.perf_counter()
        masks = sess.run_batch([p["vol"] for p in pending], [p["modality"] for p in pending])
        dt = (time.perf_counter() - t_inf) / len(pending)
        for plan, mask in zip(pending, masks):
            emit(int(plan.get("_k") or len(done) + 1), remask_finish(plan, mask, dt, restat=a.restat))
        pending.clear()

    from concurrent.futures import ThreadPoolExecutor

    io_workers = min(8, max(2, batch_n))
    k = 0
    with ThreadPoolExecutor(io_workers) as ex, open(manifest, "a") as out:
        it = iter(images)
        while True:
            chunk = []
            for _ in range(max(batch_n * 4, 32)):
                try:
                    chunk.append(next(it))
                except StopIteration:
                    break
            if not chunk:
                break
            plans = list(ex.map(
                lambda p: remask_plan(p, force=a.force, include_source=a.include_source), chunk))
            for plan in plans:
                k += 1
                if plan["status"] != "need":
                    emit(k, plan)
                    continue
                plan["_k"] = k
                pending.append(plan)
                if len(pending) >= batch_n:
                    flush()
        flush()
    n_ok = sum(r["status"] == "ok" for r in done)
    print(f"[remask] wrote {n_ok} synthstrip masks ({sum(r['status']=='skip_done' for r in done)} already done)",
          flush=True)


if __name__ == "__main__":
    main()
