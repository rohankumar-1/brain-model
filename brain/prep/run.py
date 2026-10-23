"""Preprocess datasets into the store: sharded, resumable, flagged volumes excluded and listed.

    python -m brain.prep.run --root STORE --datasets cq500,oasis,brats,rsna --shard 0/8 --workers 16 \\
        --template $BRAIN_TEMPLATE            # affine registration + automatic brain mask are the defaults
    python -m brain.prep.run ... --registration none --brain-mask none   # head-centering only (store format 4 behaviour)
    python -m brain.prep.run --root STORE --summary      # per-dataset timing and status counts
    python -m brain.prep.run --root STORE --purge --yes  # delete store files this pipeline did not write

Existing current-format zips are skipped; changing registration or brain-mask settings needs ``--force`` or a new root.

``--gpu-mask`` (needs CUDA): workers read/register/resample; the main process runs SynthStrip in GPU batches; workers
then normalise and write. Without it, ``--brain-mask auto`` runs SynthStrip on the CPU inside each worker (slow).
``--max-scratch-pct`` stops dispatching new cases when ``~/orcd/.quota`` reports scratch above that fill (it refreshes
about every 30 min); the run exits 3 and resumes from its manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from brain.prep import pipeline, sources, store

STAGES = ("read", "n4", "center", "register", "resample", "brainmask", "intensity")


def _plain(v):
    v = v.item() if hasattr(v, "item") else v
    return None if isinstance(v, float) and not math.isfinite(v) else v


def read_manifests(root: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_json(f, lines=True) for f in sorted(root.glob("manifest_*.jsonl"))], ignore_index=True)
    return df.drop_duplicates(["dataset", "case_id"], keep="last", ignore_index=True)  # a reprocessed case counts once


def discover(names: list[str]) -> list[sources.Case]:
    d = sources.DATASETS
    found = {"cq500": lambda: sources.cq500(d / "seg_cq500" / "volumes", d / "cq500"),
             "oasis": sources.oasis, "brats": sources.brats, "rsna": sources.rsna,
             "abide": sources.abide1, "ixi": sources.ixi, "isles": sources.isles22, "atlas": sources.atlas,
             "pixar": sources.pixar, "piop1": sources.piop1, "piop2": sources.piop2, "id1000": sources.id1000,
             "wmh": sources.wmh, "isles24": sources.isles24, "instance": sources.instance, "bhsd": sources.bhsd, "hemseg": sources.hemseg, "rsna_ica": sources.rsna_ica,
             "ms3seg": sources.ms3seg, "valdo": sources.valdo,
             "aisd": sources.aisd, "brainmetshare": sources.brainmetshare, "fomo": sources.fomo,
             "msd": sources.msd}  # SinoCT dropped: ~36 mm slabs that cannot be placed in a whole-head template
    return sorted((c for n in names for c in found[n]()), key=lambda c: (c.dataset, c.case_id))


def work(args) -> dict:
    """Process one case. Flagged or failed volumes are not stored and any old file is removed."""
    case, root, z_anchor, opts = args
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    row = {"dataset": case.dataset, "case_id": case.case_id}
    t0 = time.perf_counter()
    try:
        return _store(root, case, pipeline.process(case, z_anchor=z_anchor, opts=opts), row, t0)
    except Exception:
        return _failed(root, case, row, t0)


def _store(root: Path, case, res, row: dict, t0: float) -> dict:
    row |= res.info | {f"t_{k}": v for k, v in res.timing.items()}
    if res.info["qc_flag"]:
        store.remove_case(root, case)
        row["status"] = "flagged"
    else:
        store.write_case(root, case, res)
        row["status"] = "ok"
    row["seconds"] = time.perf_counter() - t0
    return row


def _failed(root: Path, case, row: dict, t0: float) -> dict:
    """An error (often transient: I/O, a bad GPU) leaves any existing file alone; the case is retried next run."""
    row |= {"status": "error", "error": traceback.format_exc(limit=2)[-400:], "seconds": time.perf_counter() - t0}
    return row


def gpu_self_test() -> str | None:
    """None if CUDA runs a real kernel here, else why not (e.g. a GPU newer than this PyTorch build)."""
    import torch

    if not torch.cuda.is_available():
        return "CUDA is not available"
    try:
        x = torch.randn(1, 1, 8, 8, 8, device="cuda")
        y = torch.nn.functional.conv3d(x, torch.randn(1, 1, 3, 3, 3, device="cuda"), padding=1)
        torch.cuda.synchronize()
        return None if torch.isfinite(y).all() else "non-finite result"
    except Exception as e:                              # "no kernel image is available for execution on the device"
        return f"{torch.cuda.get_device_name(0)}: {str(e).splitlines()[0][:160]}"


ERROR_BREAKER = 20  # this many errors before the first success stops the run (a broken node, not bad cases)


def work_prepare(args):
    """GPU-mask mode, stage 1 in a worker: (Partial or None, error row or None, start time)."""
    case, root, z_anchor, opts = args
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    t0 = time.perf_counter()
    try:
        return pipeline.prepare(case, z_anchor=z_anchor, opts=opts), None, t0
    except Exception:
        return None, _failed(root, case, {"dataset": case.dataset, "case_id": case.case_id}, t0), t0


def work_finish(args) -> dict:
    """GPU-mask mode, stage 2 in a worker: mask (precomputed or classical) -> intensities -> write."""
    part, synth, method, root, opts, t0 = args
    case = part.case
    row = {"dataset": case.dataset, "case_id": case.case_id}
    try:
        return _store(root, case, pipeline.finish(part, opts, synth=synth, brain_method=method), row, t0)
    except Exception:
        return _failed(root, case, row, t0)


def attrs_satisfy(attrs: dict, required: list[str]) -> bool:
    """``key`` must be present; ``key=value`` must also match (string comparison)."""
    for r in required:
        key, _, want = r.partition("=")
        if key not in attrs or (want and str(attrs[key]) != want):
            return False
    return True


def scratch_used_pct(path: Path = Path.home() / "orcd" / ".quota") -> float | None:
    """Scratch fill (%) from the ORCD quota cache, or None when it cannot be read."""
    try:
        for line in path.read_text().splitlines():
            if line.strip().upper().startswith("SCRATCH"):
                nums = re.findall(r"[0-9.]+", line)
                return float(nums[2])
    except (OSError, IndexError, ValueError):
        return None
    return None


def run_gpu(pool, todo: list, root: Path, z_anchor: str, opts, out, batch: int, guard) -> bool:
    """Bounded two-stage loop. Returns False when the disk guard stopped dispatch early."""
    from brain.prep.synthstrip_runtime import session

    sess = None
    q: queue.Queue = queue.Queue()
    it = iter(todo)
    inflight, pending, k, stopped = 0, [], 0, False
    n_ok = n_err = 0
    width = 2 * pool._processes
    t_start = time.time()

    def err(e):
        q.put(("crash", repr(e)))

    def submit_prepare() -> bool:
        nonlocal inflight
        c = next(it, None)
        if c is None:
            return False
        pool.apply_async(work_prepare, ((c, root, z_anchor, opts),), callback=lambda r: q.put(("prep", r)),
                         error_callback=err)
        inflight += 1
        return True

    def submit_finish(part, synth, method, t0):
        nonlocal inflight
        pool.apply_async(work_finish, ((part, synth, method, root, opts, t0),), callback=lambda r: q.put(("fin", r)),
                         error_callback=err)
        inflight += 1

    def flush():
        nonlocal sess
        if not pending:
            return
        try:
            if sess is None:
                sess = session(gpu=True)
            masks = sess.run_batch([p.raw for p, _ in pending], [p.case.modality for p, _ in pending])
            for (p, t0), m in zip(pending, masks):
                submit_finish(p, m, None, t0)
            import torch

            torch.cuda.empty_cache()                    # workers registering on the same GPU need the memory
        except Exception as e:                          # GPU trouble: classical masks for this batch
            print(f"[run] synthstrip batch failed ({e!r:.200}); classical for {len(pending)}", flush=True)
            for p, t0 in pending:
                submit_finish(p, None, "classical", t0)
        pending.clear()

    while inflight < width and submit_prepare():
        pass
    while inflight or pending:
        try:
            kind, r = q.get(timeout=10)
        except queue.Empty:
            flush()
            continue
        inflight -= 1
        if kind == "crash":
            print(f"[run] worker crash: {r}", flush=True)
        elif kind == "prep":
            part, row, t0 = r
            if row is not None:
                kind, r = "fin", row
            elif part.needs_synthstrip and opts.brain_mask in ("auto", "synthstrip"):
                pending.append((part, t0))
                if len(pending) >= batch:
                    flush()
            else:
                submit_finish(part, None, None, t0)
        if kind == "fin":
            k += 1
            n_err = n_err + 1 if r["status"] == "error" else n_err
            n_ok = n_ok + 1 if r["status"] != "error" else n_ok
            if n_ok == 0 and n_err >= ERROR_BREAKER:
                print(f"[run] {n_err} errors and no success: stopping (last: {r.get('error', '')[-200:]!r})", flush=True)
                stopped = True
            out.write(json.dumps({**{key: _plain(v) for key, v in r.items()}, "written_at": time.time()}) + "\n")
            out.flush()
            if k % 50 == 0 or r["status"] == "error":
                print(f"[run] {k}/{len(todo)} {r['dataset']}/{r['case_id']} {r['status']} "
                      f"({(time.time() - t_start) / k:.1f} s/case wall)", flush=True)
            if k % 100 == 0 and guard():
                stopped = True
        if not stopped:
            while inflight + len(pending) < width and submit_prepare():
                pass
        elif not inflight:
            flush()
    return not stopped


def summary(root: Path, names: list[str]) -> None:
    df = read_manifests(root)
    t = [f"t_{s}" for s in STAGES if f"t_{s}" in df]
    g = df.groupby("dataset").agg(n=("case_id", "size"), ok=("status", lambda s: (s == "ok").sum()),
                                  flagged=("status", lambda s: (s == "flagged").sum()),
                                  error=("status", lambda s: (s == "error").sum()),
                                  sec_mean=("seconds", "mean"), sec_p95=("seconds", lambda s: s.quantile(0.95)),
                                  **{c: (c, "mean") for c in t})
    print(g.round(2).to_string())
    total = pd.Series([c.folder for c in discover(names)]).value_counts()
    per = df.assign(folder=df.dataset.map(lambda d: next((d[: -len(c) - 1] for c in sources.BRATS_CONTRASTS
                                                          if d.endswith("_" + c)), d))).groupby("folder").seconds.mean()
    est = (total * per.reindex(total.index)).dropna()
    print(f"\nfull run estimate (single-core seconds summed): {est.sum() / 3600:.0f} core-hours")
    print((est / 3600).round(1).rename("core-hours").to_string())
    for w in (16, 32, 64, 128):
        print(f"  wall time on {w:3d} cores (ideal scaling): {est.sum() / 3600 / w:.1f} h")
    flagged = df[df.status != "ok"].reindex(columns=["dataset", "case_id", "status", "qc_reason", "error"])
    flagged.to_csv(root / "excluded.csv", index=False)
    print(f"excluded volumes listed in {root / 'excluded.csv'} ({len(flagged)})")


def purge(root: Path, names: list[str], yes: bool) -> None:
    keep = set()
    done = read_manifests(root)
    ok = set(done.loc[done.status == "ok", "case_id"])
    for c in discover(names):
        if c.case_id in ok:
            image, masks = store.paths(root, c)
            keep |= {image, *masks.values()}
    stale = [p for p in root.glob("*/*.zip*") if p not in keep]
    print(f"{len(keep)} files kept, {len(stale)} stale")
    if yes:
        for p in stale:
            p.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--datasets", default="cq500,oasis,brats,rsna")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit-per-dataset", type=int, default=0, help="first N cases of each dataset (benchmarks)")
    ap.add_argument("--z-anchor", choices=["auto", "volume", "brain"], default="auto")
    ap.add_argument("--match", default="", help="only cases whose dataset tag contains this substring")
    ap.add_argument("--cases", type=Path, default=None, help="file of case_ids (one per line): only these cases")
    ap.add_argument("--force", action="store_true")
    pipeline.add_cli_options(ap)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest jsonl (default <root>/manifest_{shard}.jsonl)")
    ap.add_argument("--gpu-mask", action="store_true", help="SynthStrip in GPU batches in the main process")
    ap.add_argument("--mask-batch", type=int, default=8, help="volumes per SynthStrip GPU forward")
    ap.add_argument("--max-scratch-pct", type=float, default=95.0,
                    help="stop dispatching new cases above this scratch fill (from ~/orcd/.quota)")
    ap.add_argument("--require-attr", action="append", default=[],
                    help="treat a current-format zip as stale unless its image attrs have this key, or key=value "
                         "(repeatable); e.g. --require-attr reg_affine redoes volumes written before the transform "
                         "was stored, --require-attr reg_backend=torch redoes everything not registered on the GPU")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--purge", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    names = a.datasets.split(",")
    if a.summary:
        return summary(a.root, names)
    if a.purge:
        return purge(a.root, names, a.yes)

    opts = pipeline.options_from_args(a)
    i, n = map(int, a.shard.split("/"))
    cases = [c for c in discover(names) if a.match in c.dataset]
    if a.cases:
        wanted = {ln.strip() for ln in a.cases.read_text().splitlines() if ln.strip()}
        cases = [c for c in cases if c.case_id in wanted]
    if a.limit_per_dataset:  # evenly spaced sample from each store folder (benchmarks)
        groups: dict[str, list] = {}
        for c in cases:
            groups.setdefault(c.folder, []).append(c)
        cases = [c for g in groups.values() for c in g[:: max(1, len(g) // a.limit_per_dataset)][: a.limit_per_dataset]]
    cases = cases[i::n]
    manifest = Path(a.manifest) if a.manifest else a.root / f"manifest_{i}of{n}.jsonl"
    if not manifest.is_absolute():
        manifest = a.root / manifest
    a.root.mkdir(parents=True, exist_ok=True)
    prev = pd.read_json(manifest, lines=True) if manifest.exists() and not a.force else None
    # ids repeat across BraTS cohorts. Errors are retried (often transient I/O); flagged volumes are deterministic.
    # With --require-attr an "ok" manifest row is not proof enough: the zip's attributes decide (checked below).
    done_status = ["flagged"] if a.require_attr else ["ok", "flagged"]
    done = set(zip(prev.dataset[prev.status.isin(done_status)], prev.case_id[prev.status.isin(done_status)])) \
        if prev is not None else set()
    todo = [c for c in cases if (c.dataset, c.case_id) not in done]
    if not a.force:
        keep, skipped = [], []
        for c in todo:
            image = store.paths(a.root, c)[0]
            if store.is_current(image, "image", (pipeline.SIZE[2], 1, pipeline.SIZE[1], pipeline.SIZE[0]), np.float16) \
                    and attrs_satisfy(store.read_attrs(image, "image"), a.require_attr):
                skipped.append(c)
            else:
                keep.append(c)
        todo = keep
        if skipped:
            print(f"[run] skipping {len(skipped)} cases with existing zips", flush=True)
            with open(manifest, "a") as out:
                for c in skipped:
                    out.write(json.dumps({"dataset": c.dataset, "case_id": c.case_id,
                                          "status": "ok", "skipped_existing": True}) + "\n")
    print(f"[run] shard {i}/{n}: {len(cases)} cases, {len(done)} already done, {len(todo)} to do", flush=True)

    def guard() -> bool:
        pct = scratch_used_pct()
        if pct is not None and pct >= a.max_scratch_pct:
            print(f"[run] scratch at {pct:.1f}% >= {a.max_scratch_pct}%: no new cases", flush=True)
            return True
        return False

    if guard():
        sys.exit(3)
    if a.gpu_mask or a.reg_backend == "torch":
        why = gpu_self_test()
        if why is not None:
            print(f"[run] GPU unusable ({why}); exiting before touching any file", flush=True)
            sys.exit(6)
    t0 = time.time()
    ctx = mp.get_context("spawn")                     # the parent may hold CUDA; never fork it
    with ctx.Pool(a.workers, maxtasksperchild=100) as pool, open(manifest, "a") as out:
        if a.gpu_mask:
            complete = run_gpu(pool, todo, a.root, a.z_anchor, opts, out, a.mask_batch, guard)
        else:
            complete = True
            for k, row in enumerate(pool.imap_unordered(work, [(c, a.root, a.z_anchor, opts) for c in todo]), 1):
                out.write(json.dumps({**{k: _plain(v) for k, v in row.items()}, "written_at": time.time()}) + "\n")
                out.flush()
                if k % 50 == 0 or row["status"] == "error":
                    print(f"[run] {k}/{len(todo)} {row['dataset']}/{row['case_id']} {row['status']} "
                          f"({(time.time() - t0) / k:.1f} s/case wall)", flush=True)
                if k % 100 == 0 and guard():
                    complete = False
                    pool.terminate()
                    break
    if not complete:
        sys.exit(3)


if __name__ == "__main__":
    main()
