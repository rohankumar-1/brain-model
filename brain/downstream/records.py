"""Experiment records: one self-contained JSON per evaluated (model, task, head) or (model, probe).

A record holds everything needed to compare and reproduce a number later:

* ``model``    checkpoint path, content fingerprint, training step, run name, training config and the data provenance
               the checkpoint was trained on (git commit, store format, grid, index hash; see ``provenance``).
* ``data``     the evaluation store (format, grid, index hash), split sizes, and how much of the task's data the encoder
               already saw in self-supervised pretraining (``pretrain_overlap``: fraction of task volumes in the SSL set).
* ``protocol`` every argument of the run (head, readout, window, seeds, steps, ...), plus the command line.
* ``metrics``  ``{"test": {...}, "val": {...}}`` as the evaluators return them (CIs, per-label, per-cohort, per-class).
* ``env``      git commit/dirty, host, Slurm job, GPU, library versions, start/end time.
* ``artifacts`` paths of predictions, decoders and the run directory (on scratch; records live under the repo).

Records are written atomically under ``$BRAIN_RESULTS/records`` (default ``<repo>/results/records``, in the backed-up
home filesystem) so concurrent Slurm jobs never contend for a shared file. ``brain.downstream.db`` folds them into a
SQLite database for querying; the JSON files stay the source of truth.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from brain.common.paths import REPO_ROOT

RECORD_VERSION = 1
RESULTS_ROOT = Path(os.environ.get("BRAIN_RESULTS", REPO_ROOT / "results"))
# environment settings that change what an evaluation computes (recorded with every result; the encoding ones are also
# part of the embedding cache name, ``embeddings.cache_tag``)
ENV_FLAGS = ("BRAIN_FINE_POOL", "BRAIN_FINE_GRID", "BRAIN_EVAL_TILE", "BRAIN_REFINE", "BRAIN_EVAL_TAPS",
             "BRAIN_EVAL_TAPS_ONLY", "BRAIN_RSNA_MAX_TRAIN", "BRAIN_RSNA_MAX_EVAL", "BRAIN_EMBED_BATCH",
             "BRAIN_STRIPPED", "OMP_NUM_THREADS")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jsonable(o: Any):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, Path):
        return str(o)
    return o


# ----------------------------------------------------------------------------- context

@functools.lru_cache(maxsize=1)
def git_state() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(["git", "-C", str(REPO_ROOT), *cmd], capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD") or None, "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def env_state() -> dict:
    gpu = None
    try:
        import torch

        torch_v = torch.__version__
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        torch_v = None
    return {"host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "gpu": gpu,
            "python": platform.python_version(), "torch": torch_v,
            "flags": {k: os.environ[k] for k in ENV_FLAGS if k in os.environ}}


def file_sha1(path: Path, chunk: int = 1 << 20) -> str | None:
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


@functools.lru_cache(maxsize=8)
def _cached_sha1(path: str, size: int, mtime: float) -> str | None:
    return file_sha1(Path(path))


def index_sha1(path: Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    st = Path(path).stat()
    return _cached_sha1(str(path), st.st_size, st.st_mtime)


def fingerprint(path: Path, edge: int = 1 << 20) -> str:
    """Cheap content fingerprint of a large file: size plus the first and last MiB."""
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(edge))
        if size > edge:
            f.seek(max(edge, size - edge))
            h.update(f.read(edge))
    return h.hexdigest()


def provenance(store_root: Path | None, index: Path | None = None) -> dict:
    """What a training run or an evaluation read: code version, store format and grid, index hash."""
    from brain.common.grid import GRID_XYZ
    from brain.prep.store import FORMAT_VERSION

    index = index if index is not None else (Path(store_root) / "volumes.csv" if store_root else None)
    return {"git": git_state(), "store_root": str(store_root) if store_root else None,
            "store_format": FORMAT_VERSION, "grid_xyz": list(GRID_XYZ),
            "index": str(index) if index else None, "index_sha1": index_sha1(index) if index else None}


@functools.lru_cache(maxsize=16)
def _model_info(path: str, size: int, mtime: float) -> dict:
    p = Path(path)
    info: dict = {"ckpt": path, "fingerprint": fingerprint(p), "size": size,
                  "mtime": datetime.fromtimestamp(mtime, timezone.utc).isoformat(timespec="seconds"),
                  "run_name": p.parent.name, "file": p.name}
    try:
        import torch

        ck = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
        info.update(step=ck.get("step"), best_step=ck.get("best_step"), best_metric=ck.get("best_metric"),
                    best_val=ck.get("best_val"), train_config=ck.get("config"), provenance=ck.get("provenance"))
        cfg = ck.get("config") or {}
        info["method"] = (ck.get("provenance") or {}).get("method") or cfg.get("method")
        info["size_name"] = (cfg.get("model") or {}).get("size")
    except Exception as e:                              # a record is still useful without the metadata
        info["load_error"] = repr(e)[:200]
    return jsonable(info)


def model_info(ckpt: Path | str | None, name: str | None = None, variant: str | None = None) -> dict:
    """``model_id`` = ``<run>/<file>@<fingerprint[:10]>[+variant]``; controls without a checkpoint use ``name``."""
    if ckpt is None:
        return {"model_id": name or "unknown", "ckpt": None, "control": True, "variant": variant}
    p = Path(ckpt).resolve()
    st = p.stat()
    info = dict(_model_info(str(p), st.st_size, st.st_mtime))
    mid = f"{info['run_name']}/{p.stem}@{info['fingerprint'][:10]}"
    info.update(model_id=mid + (f"+{variant}" if variant else ""), control=False, variant=variant)
    return info


def pretrain_overlap(volume_ids, volumes) -> float | None:
    """Fraction of the task's volumes the encoder could have seen in SSL (``pretrain``, ``ssl_keep`` and the train
    split: SSL reads only ``split == train``)."""
    if volumes is None or "pretrain" not in getattr(volumes, "columns", ()):
        return None
    df = volumes.set_index("volume_id") if "volume_id" in volumes.columns else volumes
    rows = df.reindex([v for v in volume_ids if v in df.index])
    if rows.empty:
        return None
    seen = rows["pretrain"].astype(bool)
    if "ssl_keep" in rows.columns:
        seen &= rows["ssl_keep"].astype(bool)
    if "split" in rows.columns:
        seen &= rows["split"].astype(str).eq("train")
    return float(seen.mean())


# ----------------------------------------------------------------------------- writing

def record_dir() -> Path:
    return RESULTS_ROOT / "records"


def write_record(rec: dict) -> Path:
    """Atomic write of one record; returns its path."""
    rec = jsonable({"record_version": RECORD_VERSION, "eval_id": rec.get("eval_id") or uuid.uuid4().hex, **rec})
    month = (rec.get("env", {}).get("finished_utc") or now_utc())[:7]
    out = record_dir() / month / f"{rec['eval_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=False, allow_nan=False) + "\n")
    os.replace(tmp, out)
    return out


def make_record(*, suite: str, kind: str, task: str, task_type: str, head: str, model: dict, data: dict,
                protocol: dict, metrics: dict, started_utc: str | None = None, view: str | None = None,
                artifacts: dict | None = None, notes: str | None = None) -> dict:
    env = env_state() | {"git": git_state(), "argv": sys.argv, "started_utc": started_utc,
                         "finished_utc": now_utc()}
    return {"eval_id": uuid.uuid4().hex, "suite": suite, "kind": kind, "task": task, "task_type": task_type,
            "head": head, "view": view, "model": model, "data": data, "protocol": protocol, "metrics": metrics,
            "env": env, "artifacts": artifacts or {}, "notes": notes}


def split_metrics(m: dict) -> dict:
    """Evaluator dicts carry test metrics at the top level and val metrics under ``val``; history is an artifact."""
    m = dict(m)
    val = m.pop("val", None)
    m.pop("history", None)
    return {"test": m, **({"val": val} if val else {})}


def record_run(run_dir: Path, config: dict, all_metrics: dict[str, dict], *, suite: str) -> list[Path]:
    """One record per head of a finished ``brain.downstream`` run (called by ``results.finalize_run``)."""
    store_root = config.get("slices_root") or config.get("store_root")
    index = config.get("volumes_csv") or (Path(store_root) / "volumes.csv" if store_root else None)
    data = provenance(Path(store_root) if store_root else None, Path(index) if index else None) | {
        k: config.get(k) for k in ("n_train", "n_val", "n_test", "pretrain_overlap")}
    ckpt = config.get("ckpt")
    model = config.get("model_info") or (model_info(ckpt) if ckpt else model_info(None, name=config.get("model_name")))
    protocol = {k: v for k, v in config.items() if k not in ("time_utc", "started_utc", "n_train", "n_val", "n_test",
                                                              "pretrain_overlap", "model_info")}
    out = []
    for head, m in all_metrics.items():
        art = {"run_dir": str(run_dir)}
        for cand in (run_dir / "predictions" / f"{head}.npz", run_dir / "predictions" / f"{head}_volumes.csv",
                     run_dir / "seg_decoder.pt", run_dir / "heads" / f"{head}.json"):
            if cand.exists():
                art[cand.stem if cand.parent.name != "heads" else "metrics_json"] = str(cand)
        rec = make_record(suite=suite, kind="task", task=config["task"], task_type=config.get("task_type", ""),
                          head=head, view=m.get("view") or config.get("readout"), model=model, data=data,
                          protocol=protocol | {"head": head}, metrics=split_metrics(m),
                          started_utc=config.get("started_utc"), artifacts=art)
        out.append(write_record(rec))
    return out
