"""Remove everything in the store that the current format does not use (dry run unless ``--yes``).

* zips whose arrays are not ``store.FORMAT_VERSION`` (stale cases a re-prep did not rewrite: dropped datasets,
  cases that failed or were flagged), format-5 ``<case>.brain.zip`` files, interrupted ``.partial.*`` writes;
* manifests of earlier store versions, remask logs and old index backups; folders of retired steps (``DEAD_DIRS``).

Index outputs (``volumes.csv``, ``labels/``, slice tables) and the current manifests are kept.

    python -m brain.prep.store_cleanup --root STORE            # report
    python -m brain.prep.store_cleanup --root STORE --yes      # delete
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from brain.prep import store

DEAD_DIRS = ("fg_masks", "manifests_before_shared_grid", "manifests_v4_backup", "sinoct")
KEEP_DIRS = ("labels",)


def _stale(path: Path) -> bool:
    name = path.name
    if ".partial." in name or name.endswith(".brain.zip"):
        return True
    for array in ("image", "mask"):
        try:
            return store.read_attrs(path, array).get("format_version") != store.FORMAT_VERSION
        except Exception:
            continue
    return True                                        # unreadable: not something the current store can use


def plan(root: Path, workers: int = 8) -> list[Path]:
    out = [p for p in root.glob("manifest_*.jsonl") if not p.name.startswith(f"manifest_v{store.FORMAT_VERSION}_")]
    out += sorted(root.glob("remask_*.jsonl"))
    out += sorted(root.glob("*.bak_before_*"))          # index backups from earlier incremental v5 builds
    out += [root / d for d in DEAD_DIRS if (root / d).exists()]
    folders = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name not in DEAD_DIRS + KEEP_DIRS]
    zips = [p for d in folders for p in d.iterdir() if p.is_file() and ".zip" in p.name]
    with ThreadPoolExecutor(workers) as ex:
        out += [p for p, s in zip(zips, ex.map(_stale, zips)) if s]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    paths = plan(a.root, a.workers)
    files = [p for p in paths if p.is_file()]
    by_folder: dict[str, int] = {}
    for p in files:
        key = p.parent.name if p.parent != a.root else "(top level)"
        by_folder[key] = by_folder.get(key, 0) + 1
    gb = sum(p.stat().st_size for p in files) / 1e9
    print(f"[store-cleanup] {len(files)} files ({gb:.1f} GB) + dirs {[p.name for p in paths if p.is_dir()]}")
    for k, n in sorted(by_folder.items(), key=lambda t: -t[1]):
        print(f"  {k:28s} {n}")
    if a.yes:
        for p in paths:
            shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
        print("[store-cleanup] deleted")


if __name__ == "__main__":
    main()
