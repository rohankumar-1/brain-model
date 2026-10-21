"""Re-encode the JPEG-LS volume archives on pool as JPEG-XL lossless (``archive.MAGIC_JXL``), verified exact.

Work is split into chunks of members (~1 GB of input) so many short jobs can share it: a worker claims a chunk with a
lock file, decodes every JPEG-LS member, encodes it as JPEG-XL, decodes that again and requires the array to be
identical, and writes the chunk's members to ``<archive>.jxl.d/chunk_NNNNN.zip`` (atomic rename). Members that are not
JPEG-LS volumes, or whose JPEG-XL encode does not round-trip, are carried over unchanged.

``merge`` rebuilds the archive in the original member order from the chunks (other members copied from the original),
re-opens it, checks names, CRCs and member counts, and only then replaces the original and drops the chunks.

    python -m brain.prep.transcode status ARCHIVE...
    python -m brain.prep.transcode work ARCHIVE... --workers 24 --deadline 780     # exit 10: work left
    python -m brain.prep.transcode ready ARCHIVE...     # prints archives whose merge this caller should submit
    python -m brain.prep.transcode merge ARCHIVE
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from brain.prep import archive

CHUNK_BYTES = int(os.environ.get("TRANSCODE_CHUNK_BYTES", 1 << 30))
MIN_FREE_GB = float(os.environ.get("TRANSCODE_MIN_FREE_GB", 40))   # never claim work below this much free pool
STALE_LOCK_S = 20 * 60           # a quicktest job is killed after 15 min; its lock is stale after that
MORE_WORK = 10
WAITING = 11                     # the active archive is fully staged and merging: nothing to claim for a while


def free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def is_transcoded(zpath: Path) -> bool:
    """The archive's first volume member is already JPEG-XL (merged earlier) and nothing is staged."""
    if staging(zpath).exists():
        return False
    with zipfile.ZipFile(zpath) as zf:
        first = next((i.filename for i in zf.infolist() if i.filename.endswith(".jls")), None)
        return first is None or zf.open(first).read(8) == archive.MAGIC_JXL


def staging(zpath: Path) -> Path:
    return zpath.with_name(zpath.name + ".jxl.d")


def plan(zpath: Path) -> list[list[str]]:
    """JPEG-LS members in archive order, grouped into chunks of about ``CHUNK_BYTES`` compressed input."""
    chunks, cur, size = [], [], 0
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".jls"):
                continue
            cur.append(info.filename)
            size += info.compress_size
            if size >= CHUNK_BYTES:
                chunks.append(cur)
                cur, size = [], 0
    if cur:
        chunks.append(cur)
    return chunks


def load_plan(zpath: Path) -> list[list[str]]:
    d = staging(zpath)
    p = d / "plan.json"
    if p.exists():
        return json.loads(p.read_text())
    d.mkdir(exist_ok=True)
    chunks = plan(zpath)
    tmp = p.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(chunks))
    os.replace(tmp, p)
    return json.loads(p.read_text())


def chunk_path(zpath: Path, k: int) -> Path:
    return staging(zpath) / f"chunk_{k:05d}.zip"


def done_chunks(zpath: Path, n: int) -> set[int]:
    return {k for k in range(n) if chunk_path(zpath, k).exists()}


def claim(zpath: Path, k: int) -> bool:
    lock = chunk_path(zpath, k).with_suffix(".lock")
    try:
        if lock.exists() and time.time() - lock.stat().st_mtime > STALE_LOCK_S:
            lock.unlink()
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except (FileExistsError, FileNotFoundError):
        return False
    os.write(fd, f"{socket.gethostname()} {os.environ.get('SLURM_JOB_ID', os.getpid())}\n".encode())
    os.close(fd)
    return True


def _transcode(args: tuple[str, str]) -> tuple[str, bytes | None, str]:
    zpath, name = args
    with zipfile.ZipFile(zpath) as zf:
        blob = zf.read(name)
    if blob[:8] != archive.MAGIC_JLS:
        return name, None, "not-jls"
    vol = archive.decode_volume(blob)
    try:
        new = archive.encode_volume(vol, "jxl")
        if not np.array_equal(archive.decode_volume(new), vol):
            return name, None, "mismatch"
    except Exception as e:                                       # noqa: BLE001 - keep the original member
        return name, None, f"error:{type(e).__name__}"
    if len(new) >= len(blob):
        return name, None, "larger"
    return name, new, "ok"


def work_chunk(zpath: Path, k: int, members: list[str], pool: ProcessPoolExecutor) -> dict:
    out = chunk_path(zpath, k)
    tmp = out.with_suffix(f".partial{os.getpid()}")
    stats = {"ok": 0, "kept": 0, "in": 0, "out": 0}
    with zipfile.ZipFile(zpath) as src, zipfile.ZipFile(tmp, "w", allowZip64=True) as dst:
        for name, new, why in pool.map(_transcode, [(str(zpath), m) for m in members]):
            info = src.getinfo(name)
            stats["in"] += info.compress_size
            if new is None:
                data = src.read(name)
                dst.writestr(name, data, compress_type=info.compress_type)
                stats["kept"] += 1
                stats["out"] += info.compress_size
                if why not in ("larger",):
                    print(f"[transcode] keep {name}: {why}", flush=True)
            else:
                dst.writestr(name, new, compress_type=zipfile.ZIP_STORED)
                stats["ok"] += 1
                stats["out"] += len(new)
    with zipfile.ZipFile(tmp) as chk:
        if chk.testzip() is not None or len(chk.namelist()) != len(members):
            tmp.unlink()
            raise RuntimeError(f"{out}: chunk failed its own check")
    os.replace(tmp, out)
    out.with_suffix(".lock").unlink(missing_ok=True)
    return stats


def work(zips: list[Path], workers: int, deadline: float) -> int:
    """Stage chunks of the first archive (in list order) that is not transcoded yet. One archive at a time: chunks of
    the next archive are only staged after this one merged, so pool never holds more than one extra copy."""
    t0, longest = time.time(), 0.0
    active = next((z for z in zips if z.exists() and not is_transcoded(z)), None)
    if active is None:
        return 0
    chunks = load_plan(active)
    if not status(active)["todo"]:
        print(f"[transcode] {active.name} is fully staged; waiting for its merge", flush=True)
        return WAITING
    with ProcessPoolExecutor(workers) as pool:
        for k, members in enumerate(chunks):
            if chunk_path(active, k).exists():
                continue
            if time.time() - t0 + 1.3 * max(longest, 60.0) > deadline:
                print(f"[transcode] stopping before the deadline ({time.time() - t0:.0f}s used)", flush=True)
                return MORE_WORK
            if free_gb(active.parent) < MIN_FREE_GB:
                print(f"[transcode] only {free_gb(active.parent):.0f} GB free on pool: not claiming", flush=True)
                return MORE_WORK
            if not claim(active, k):
                continue
            t = time.time()
            s = work_chunk(active, k, members, pool)
            dt = time.time() - t
            longest = max(longest, dt)
            print(f"[transcode] {active.name} chunk {k + 1}/{len(chunks)}: {s['ok']} jxl, {s['kept']} kept, "
                  f"{s['in'] / 1e9:.2f} -> {s['out'] / 1e9:.2f} GB in {dt:.0f}s", flush=True)
    return MORE_WORK      # this archive is staged (or others hold its last chunks); its merge, then the next archive


def status(zpath: Path) -> dict:
    d = staging(zpath)
    if not (d / "plan.json").exists():
        return {"archive": zpath.name, "chunks": None, "done": 0, "todo": True, "merged": False}
    chunks = json.loads((d / "plan.json").read_text())
    done = done_chunks(zpath, len(chunks))
    return {"archive": zpath.name, "chunks": len(chunks), "done": len(done), "todo": len(done) < len(chunks),
            "merged": False}


def claim_merge(zpath: Path) -> bool:
    """True once per archive, when every chunk is done: the caller submits the merge."""
    st = status(zpath)
    if st["chunks"] is None or st["todo"]:
        return False
    try:
        os.close(os.open(staging(zpath) / "merge.claimed", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    return True


def merge(zpath: Path) -> None:
    """Rebuild ``zpath`` from its chunks. On failure the claim and the partial are removed so a later worker resubmits
    the merge; chunks already copied are gone and get re-staged from the (untouched) original."""
    try:
        _merge(zpath)
    except BaseException:
        (staging(zpath) / "merge.claimed").unlink(missing_ok=True)
        tmp = zpath.with_name(zpath.name + ".jxl.partial")
        chunks_left = staging(zpath).exists() and any(staging(zpath).glob("chunk_*.zip"))
        if tmp.exists() and chunks_left:
            tmp.unlink()
        raise


def _merge(zpath: Path) -> None:
    d = staging(zpath)
    chunks = json.loads((d / "plan.json").read_text())
    tmp = zpath.with_name(zpath.name + ".jxl.partial")
    if tmp.exists() and not any(chunk_path(zpath, k).exists() for k in range(len(chunks))):
        raise SystemExit(f"{tmp} exists and its chunks are gone: inspect it and rename it over {zpath.name} by hand")
    tmp.unlink(missing_ok=True)
    missing = [k for k in range(len(chunks)) if not chunk_path(zpath, k).exists()]
    if missing:
        raise SystemExit(f"{zpath.name}: {len(missing)} chunks not done")
    where = {m: k for k, ms in enumerate(chunks) for m in ms}
    staged = sum(chunk_path(zpath, k).stat().st_size for k in range(len(chunks))) / 1e9
    if free_gb(zpath.parent) < 5:
        raise SystemExit(f"{zpath.name}: pool is full ({free_gb(zpath.parent):.0f} GB free)")
    print(f"[transcode] merging {zpath.name}: {staged:.1f} GB of chunks, {free_gb(zpath.parent):.0f} GB free; "
          "each chunk is deleted once copied (the original archive stays until the swap)", flush=True)
    opened: dict[int, zipfile.ZipFile] = {}
    n_jxl = 0
    with zipfile.ZipFile(zpath) as src, zipfile.ZipFile(tmp, "w", allowZip64=True) as dst:
        for info in src.infolist():
            if info.filename in where:
                k = where[info.filename]
                if k not in opened:
                    for j, z in opened.items():          # chunks are visited in order: the previous one is done
                        z.close()
                        chunk_path(zpath, j).unlink()
                    opened = {k: zipfile.ZipFile(chunk_path(zpath, k))}
                cinfo = opened[k].getinfo(info.filename)
                data = opened[k].read(info.filename)
                n_jxl += data[:8] == archive.MAGIC_JXL
                dst.writestr(info.filename, data, compress_type=cinfo.compress_type)
            else:
                dst.writestr(info, src.read(info.filename), compress_type=info.compress_type)
        names = [i.filename for i in src.infolist()]
    for j, z in opened.items():
        z.close()
        chunk_path(zpath, j).unlink()
    with zipfile.ZipFile(tmp) as chk:
        if [i.filename for i in chk.infolist()] != names:
            raise SystemExit(f"{tmp}: member list differs from the original")
        bad = chk.testzip()
        if bad is not None:
            raise SystemExit(f"{tmp}: CRC error in {bad}")
    old, new = zpath.stat().st_size, tmp.stat().st_size
    os.replace(tmp, zpath)                    # if this fails, tmp is a complete verified archive: retry the rename
    shutil.rmtree(d)
    print(f"[transcode] merged {zpath.name}: {n_jxl} JPEG-XL members, {old / 1e9:.1f} -> {new / 1e9:.1f} GB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["status", "work", "ready", "merge"])
    ap.add_argument("archives", nargs="+", type=Path)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--deadline", type=float, default=780.0, help="seconds this worker may run")
    a = ap.parse_args()
    if a.action == "status":
        for z in a.archives:
            print(status(z) if z.exists() else {"archive": z.name, "missing": True})
    elif a.action == "work":
        try:
            sys.exit(work(a.archives, a.workers, a.deadline))
        except OSError as e:          # e.g. ESTALE: a merge swapped the archive this worker had open; retry later
            if e.errno not in (116,):
                raise
            print(f"[transcode] {e}; will retry", flush=True)
            sys.exit(MORE_WORK)
    elif a.action == "ready":
        for z in a.archives:
            if z.exists() and claim_merge(z):
                print(z)
    else:
        for z in a.archives:
            merge(z)


if __name__ == "__main__":
    main()
