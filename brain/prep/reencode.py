"""Rewrite format-6 image zips (float16 ``image``) as format 7 (int16 + ``scale``, see ``brain.prep.store``), in place.

Per zip: read the float16 image, encode it (CT: air clamp below -500 HU, 1 HU steps; MRI: round(z * 1000)), write a
temp zip next to it holding the new ``image`` plus every other entry of the old zip copied byte for byte (``brain``
array, its attrs, anything else), verify, then ``os.replace``. Verification (any failure leaves the original untouched):
- CT: voxels below -500 HU are -1024, the rest within half a step of the old value (1 HU steps: <= 0.5 HU; the stored
  float16 HU are interpolated, not integers, so most voxels move by a fraction of a HU);
- MRI: max |decoded - old| <= step / 2 (0.0005 z), plus float32 rounding;
- every non-image zip entry byte-identical, image attrs identical apart from ``format_version`` / ``dtype_rev`` /
  ``scale``, same shape and chunks; the values checked are ``store.decode`` of the new array (what every reader returns).

Resumable: zips already at ``dtype_rev`` 7 are skipped, as are paths the shard's manifest already lists as done.
Sharded by striding over the sorted path list (``--shard i/n``). One manifest line per zip (old / new bytes, max error,
seconds) in ``<out>/manifest_<i>of<n>.jsonl``. Before each write it checks free space (``os.statvfs``) and the ORCD
quota cache plus this shard's own net bytes, and stops (exit 3) below the margins; ENOSPC / EDQUOT also stop it cleanly.
SIGTERM / SIGINT finish the current zip and exit (exit 4) so a requeue resumes.

    python -m brain.prep.reencode --root STORE --shard 0/16            # paths from STORE/volumes.csv
    python -m brain.prep.reencode --root PILOT --paths list.txt --out runs/diagnostics/reencode/pilot_out
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import signal
import stat
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from brain.prep import store

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "runs" / "diagnostics" / "reencode"
QUOTA = Path.home() / "orcd" / ".quota"
MRI_EXTRA = 1e-6          # float32 decode rounding on top of step / 2 (|z| <= 20)
CHANGED_ATTRS = {"format_version", "dtype_rev", "scale", "air_below"}


class Stop(Exception):
    """Out of space (or about to be): stop the shard cleanly."""


def tmp_for(path: Path) -> Path:
    """Deterministic temp name (not ``*.zip``), so a retry after a kill overwrites the leftover."""
    return path.with_name(path.name + ".reencode-tmp")


def quota_gb(path: Path = QUOTA) -> tuple[float, float] | None:
    """(used, limit) GB of SCRATCH from the ORCD quota cache (refreshed ~30 min), or None."""
    try:
        for line in path.read_text().splitlines():
            if line.strip().upper().startswith("SCRATCH"):
                nums = re.findall(r"[0-9.]+", line)
                return float(nums[0]), float(nums[1])
    except (OSError, IndexError, ValueError):
        return None
    return None


class SpaceGuard:
    """Free-space check before every write: the filesystem (statvfs; on this NFS mount it reports the whole export,
    not the user quota) and the quota cache, corrected by the bytes this process has added (usually negative) since
    the cache was read. Writing one temp zip of ``need`` bytes must leave both margins."""

    def __init__(self, min_free_gb: float, quota_margin_gb: float, quota_file: Path = QUOTA, reread_s: float = 300):
        self.min_free, self.margin = min_free_gb * 1e9, quota_margin_gb * 1e9
        self.quota_file, self.reread_s = quota_file, reread_s
        self.delta = 0.0           # net bytes this process added since the quota cache was last read
        self._q, self._qm, self._t = None, None, -1e18

    def _quota(self):
        if time.monotonic() - self._t > self.reread_s:
            mtime = self.quota_file.stat().st_mtime if self.quota_file.exists() else None
            if self._q is None or mtime != self._qm:
                self._q, self._qm, self.delta = quota_gb(self.quota_file), mtime, 0.0
            self._t = time.monotonic()
        return self._q

    def check(self, folder: Path, need: int) -> None:
        s = os.statvfs(folder)
        free = s.f_bavail * s.f_frsize
        if free - need < self.min_free:
            raise Stop(f"filesystem free {free / 1e9:.1f} GB < margin {self.min_free / 1e9:.0f} GB")
        q = self._quota()
        if q is not None:
            used, limit = q[0] * 1e9 + self.delta, q[1] * 1e9
            if limit - used - need < self.margin:
                raise Stop(f"scratch quota: ~{used / 1e9:.1f} of {limit / 1e9:.0f} GB used (cache + this shard), "
                           f"margin {self.margin / 1e9:.0f} GB")


def _attrs_plain(attrs: dict) -> dict:
    return json.loads(json.dumps({k: store.plain(v) for k, v in attrs.items()}))


def _open_image(path: Path):
    import zarr
    from zarr.storage import ZipStore

    zs = ZipStore(path, mode="r")
    return zs, zarr.open_group(zs, mode="r")["image"]


def needs_reencode(path: Path) -> bool:
    zs, a = _open_image(path)
    try:
        return int(a.attrs.get("dtype_rev") or 0) < store.DTYPE_REV
    finally:
        zs.close()


def reencode_zip(path: Path, guard: SpaceGuard | None = None, clevel: int | None = None,
                 mri_step: float | None = None, ct_step: float | None = None, keep_tmp: bool = False,
                 air_below: float | None = store.CT_AIR_BELOW) -> dict:
    """Re-encode one image zip in place. Returns the manifest record (status ok / skip; raises on failure, leaving
    the original untouched)."""
    path = Path(path)
    t0 = time.perf_counter()
    old_bytes = path.stat().st_size
    zs, a = _open_image(path)
    try:
        attrs = dict(a.attrs)
        if int(attrs.get("dtype_rev") or 0) >= store.DTYPE_REV:
            return {"status": "skip", "reason": "already format 7", "old_bytes": old_bytes}
        kind = attrs.get("kind")
        if kind not in store.IMAGE_KINDS:
            raise ValueError(f"image has no known kind ({kind!r})")
        if attrs.get("format_version") not in store.READ_VERSIONS:
            raise ValueError(f"unexpected format_version {attrs.get('format_version')!r}")
        old = np.asarray(a[:])
        chunks, shape, old_dtype = tuple(a.chunks), tuple(a.shape), str(a.dtype)
    finally:
        zs.close()
    t1 = time.perf_counter()
    step = {"ct_hu": ct_step, "mri_z": mri_step}[kind]
    enc, extra = store.encode_image(old, kind, step, air_below)
    new_attrs = {**attrs, "format_version": store.FORMAT_VERSION, **extra}

    tmp = tmp_for(path)
    if guard is not None:
        guard.check(path.parent, old_bytes)
    try:
        t2 = time.perf_counter()
        store.write_zip(tmp, {"image": (enc, new_attrs)}, clevel=clevel)
        with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "a") as zout:
            have = set(zout.namelist())
            for info in zin.infolist():
                if info.filename.startswith("image/") or info.filename in have:
                    continue
                zout.writestr(info, zin.read(info))                # same name, bytes, date and (stored) method
        t3 = time.perf_counter()
        rec = verify(path, tmp, old, attrs, chunks, shape, kind, extra["scale"], extra.get("air_below"))
        os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))           # keep the original permissions (DUA data: 600)
        new_bytes = tmp.stat().st_size
        rec = {"status": "ok", "kind": kind, "old_bytes": old_bytes, "new_bytes": new_bytes, "old_dtype": old_dtype,
               **rec, "t_read": round(t1 - t0, 3), "t_encode": round(t2 - t1, 3), "t_write": round(t3 - t2, 3),
               "t_verify": round(time.perf_counter() - t3, 3)}
        if keep_tmp:
            return rec | {"status": "verified"}
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if guard is not None:
        guard.delta += new_bytes - old_bytes
    return rec | {"t_s": round(time.perf_counter() - t0, 3)}


def verify(path: Path, tmp: Path, old: np.ndarray, attrs: dict, chunks, shape, kind: str, scale: float,
           air_below: float | None = None) -> dict:
    """Checks of the temp zip against the original (see the module docstring). Raises AssertionError on a mismatch."""
    with zipfile.ZipFile(path) as zo, zipfile.ZipFile(tmp) as zn:
        names_o, names_n = set(zo.namelist()), set(zn.namelist())
        lost = {n for n in names_o - names_n}
        extra = {n for n in names_n - names_o if not n.startswith("image/")}
        assert not lost - {n for n in names_o if n.startswith("image/c/")}, f"entries lost: {sorted(lost)[:5]}"
        assert not extra, f"unexpected entries: {sorted(extra)[:5]}"
        for n in names_o:
            if n.startswith("image/"):
                continue
            if n == "zarr.json":
                assert json.loads(zo.read(n)) == json.loads(zn.read(n)), "root group metadata differs"
            else:
                assert zo.read(n) == zn.read(n), f"entry {n} differs"
    zs, a = _open_image(tmp)
    try:
        new = np.asarray(a[:])
        nattrs = dict(a.attrs)
        assert tuple(a.chunks) == chunks and tuple(a.shape) == shape, "image shape / chunks differ"
    finally:
        zs.close()
    assert new.dtype == np.int16, new.dtype
    assert nattrs["format_version"] == store.FORMAT_VERSION and nattrs["dtype_rev"] == store.DTYPE_REV
    assert nattrs["scale"] == scale and nattrs["kind"] == kind
    keep = lambda d: {k: v for k, v in _attrs_plain(d).items() if k not in CHANGED_ATTRS}  # noqa: E731
    assert keep(nattrs) == keep(attrs), "image attrs differ"
    dec = store.decode(new, nattrs)                    # what read / read_image / read_many return (float32)
    assert dec.dtype == np.float32
    ref = old.astype(np.float32)
    err = np.abs(dec - ref)
    if kind == "ct_hu":
        air = ref < air_below if air_below is not None else np.zeros(ref.shape, bool)
        assert np.all(dec[air] == store.CT_AIR), "air clamp"
        err[air] = 0
        mx = float(err.max(initial=0.0))
        assert mx <= scale / 2 + 1e-6, f"CT max error {mx} HU"
        n_body = air.size - int(air.sum())
        return {"max_err": round(mx, 6), "n_air_clamped": int((air & (ref != store.CT_AIR)).sum()),
                "frac_changed": round(float(np.count_nonzero(err)) / max(n_body, 1), 4)}
    mx = float(err.max(initial=0.0))
    assert mx <= scale / 2 + MRI_EXTRA, f"MRI max error {mx}"
    return {"max_err": round(mx, 7)}


def load_paths(root: Path, index: Path | None, paths_file: Path | None) -> list[str]:
    if paths_file is not None:
        rels = [ln.strip() for ln in open(paths_file) if ln.strip() and not ln.startswith("#")]
    else:
        import pandas as pd

        rels = pd.read_csv(index or root / "volumes.csv", usecols=["image_path"])["image_path"].dropna().astype(str)
    return sorted(set(rels))


def done_paths(manifest: Path) -> set[str]:
    out = set()
    if manifest.exists():
        for line in open(manifest):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") in ("ok", "skip", "missing"):
                out.add(r["path"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="store root (image paths are relative to it)")
    ap.add_argument("--index", type=Path, default=None, help="index CSV with image_path (default ROOT/volumes.csv)")
    ap.add_argument("--paths", type=Path, default=None, help="text file of image paths relative to --root instead")
    ap.add_argument("--shard", default="0/1", help="i/n: this task takes every n-th path starting at i")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="manifest folder")
    ap.add_argument("--clevel", type=int, default=None, help=f"Blosc zstd level (default {store.CLEVEL})")
    ap.add_argument("--ct-step", type=float, default=None, help="CT HU per integer step (default 1; 0.125 keeps "
                    "every float16 HU >= 128 exact at ~+1.5 MB per CT)")
    ap.add_argument("--ct-air-below", default=str(store.CT_AIR_BELOW),
                    help="CT HU below this are stored as -1024 ('none': no clamp, keeps the asinh windows' air range)")
    ap.add_argument("--mri-step", type=float, default=None, help="MRI z per integer step (default 0.001)")
    ap.add_argument("--min-free-gb", type=float, default=200.0, help="statvfs free space margin")
    ap.add_argument("--quota-margin-gb", type=float, default=15.0, help="scratch quota headroom (cache + own bytes)")
    ap.add_argument("--max-mb-s", type=float, default=0.0, help="cap on this task's read+write MB/s (0: none)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many rewritten zips (0: all)")
    ap.add_argument("--dry-run", action="store_true", help="encode, write the temp zip and verify, but keep the original")
    a = ap.parse_args(argv)

    i, n = (int(x) for x in a.shard.split("/"))
    air_below = None if str(a.ct_air_below).lower() == "none" else float(a.ct_air_below)
    rels = load_paths(a.root, a.index, a.paths)[i::n]
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = a.out / f"manifest_{i}of{n}.jsonl"
    done = done_paths(manifest)
    todo = [r for r in rels if r not in done]
    print(f"[reencode] shard {i}/{n}: {len(rels)} paths, {len(rels) - len(todo)} done before, {len(todo)} to check; "
          f"root={a.root} clevel={a.clevel or store.CLEVEL}", flush=True)

    stopping = {"sig": None}

    def on_signal(sig, _frame):
        stopping["sig"] = sig
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    guard = SpaceGuard(a.min_free_gb, a.quota_margin_gb)
    t_start, moved, n_ok, n_err, saved = time.monotonic(), 0, 0, 0, 0
    code = 0
    with open(manifest, "a") as out:
        def log(rec):
            out.write(json.dumps(rec) + "\n")
            out.flush()

        for k, rel in enumerate(todo):
            if stopping["sig"] is not None:
                log({"path": None, "status": "stopped", "reason": f"signal {stopping['sig']}"})
                code = 4
                break
            if a.limit and n_ok >= a.limit:
                break
            path = a.root / rel
            if not path.is_file():
                log({"path": rel, "status": "missing"})
                continue
            try:
                rec = reencode_zip(path, guard, a.clevel, a.mri_step, a.ct_step, keep_tmp=a.dry_run, air_below=air_below)
            except Stop as e:
                log({"path": rel, "status": "stopped", "reason": str(e)})
                print(f"[reencode] STOP: {e}", flush=True)
                code = 3
                break
            except OSError as e:
                if e.errno in (errno.ENOSPC, errno.EDQUOT):
                    log({"path": rel, "status": "stopped", "reason": f"{errno.errorcode[e.errno]}: {e}"})
                    print(f"[reencode] STOP: out of space ({e})", flush=True)
                    code = 3
                    break
                log({"path": rel, "status": "error", "error": f"{type(e).__name__}: {e}"})
                n_err += 1
                continue
            except Exception as e:                                  # noqa: BLE001 - logged; original untouched
                log({"path": rel, "status": "error", "error": f"{type(e).__name__}: {e}"})
                n_err += 1
                continue
            if a.dry_run and rec["status"] == "verified":
                tmp_for(path).unlink(missing_ok=True)
                rec["status"] = "dry_run"
            log({"path": rel, **rec})
            if rec["status"] in ("ok", "dry_run"):
                n_ok += 1
                moved += rec["old_bytes"] * 2 + rec["new_bytes"] * 2  # read old twice (encode, verify), write + verify
                saved += rec["old_bytes"] - rec["new_bytes"]
            if a.max_mb_s > 0:                                      # throttle: average MB/s of this task
                ahead = moved / (a.max_mb_s * 1e6) - (time.monotonic() - t_start)
                if ahead > 0:
                    time.sleep(ahead)
            if (k + 1) % 500 == 0:
                el = time.monotonic() - t_start
                print(f"[reencode] {k + 1}/{len(todo)} checked, {n_ok} rewritten, {n_err} errors, "
                      f"saved {saved / 1e9:.1f} GB, {n_ok / max(el, 1e-9):.2f} zips/s", flush=True)
    el = time.monotonic() - t_start
    print(f"[reencode] done: {n_ok} rewritten, {n_err} errors, saved {saved / 1e9:.2f} GB in {el / 60:.1f} min "
          f"(exit {code or (5 if n_err else 0)})", flush=True)
    return code or (5 if n_err else 0)


if __name__ == "__main__":
    sys.exit(main())
