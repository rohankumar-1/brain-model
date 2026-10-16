"""Download BrainMetShare (Stanford AIMI) into its own pool folder.

156 skull-stripped brain-metastasis MRI studies (4 co-registered sequences;
105 train with segs, 51 test). Lives only under ``$BRAIN_DATASETS/brainmetshare``.

    python -m data.download_brainmetshare
    python -m data.download_brainmetshare --prep
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from data.prep.sources import DATASETS

DATASET = "brainmetshare:1emj:v1_0"
URL = "https://stanford.redivis.com/datasets/1emj-bjxt3p6s0"
OUT = DATASETS / "brainmetshare"
STORE = OUT / "processed"
TABLES = (
    ("train_nifti:cbv1", "nifti/train"),
    ("test_nifti:ezk7", "nifti/test"),
    ("train_png:nyfw", "png/train"),
    ("test_png:ax8y", "png/test"),
)


def dataset():
    import redivis
    return redivis.user("aimi").dataset(DATASET)


def write_metadata(dest: Path = OUT) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    ds = dataset()
    props = ds.get().properties
    keep = {k: props.get(k) for k in (
        "name", "qualifiedReference", "url", "description", "accessLevel",
        "tableCount", "numBytes", "totalNumBytes", "totalRawFileBytes",
        "methodologyMarkdown", "usageMarkdown",
    )}
    keep["version"] = (props.get("version") or props.get("currentVersion") or {})
    (dest / "dataset.json").write_text(json.dumps(keep, indent=2, default=str) + "\n")
    md = dest / "README.md"
    md.write_text(
        f"# BrainMetShare\n\n{props.get('description') or ''}\n\n"
        f"Source: {URL}\n\n{props.get('methodologyMarkdown') or ''}\n"
    )
    meta = ds.table("ai_brain_mets_primary_cancer:eze9").to_pandas_dataframe()
    cols = [c for c in meta.columns if not str(c).startswith("_") or meta[c].notna().any()]
    out = dest / "primary_cancer.csv"
    meta[cols].to_csv(out, index=False)
    print(f"[brainmetshare] metadata {len(meta)} patients cols={list(meta[cols].columns)} -> {out}", flush=True)
    return out


def iter_files(node):
    for c in node.list():
        if "Directory" in type(c).__name__:
            yield from iter_files(c)
        else:
            yield c


def _save_file(f, dest: Path) -> tuple[str, int, str]:
    rel = Path(str(getattr(f, "path", f.name)).lstrip("/"))
    out = dest / rel
    size = int(getattr(f, "size", 0) or 0)
    if out.exists() and out.stat().st_size > 0 and (not size or out.stat().st_size == size):
        return str(rel), out.stat().st_size, "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(6):
        try:
            f.download(str(out), overwrite=True, progress=False)
            return str(rel), out.stat().st_size, "ok"
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed {rel}: {last}") from last


def download_table(qualified: str, rel: str, dest: Path, concurrency: int) -> Path:
    out = dest / rel
    out.mkdir(parents=True, exist_ok=True)
    print(f"[brainmetshare] listing {qualified}", flush=True)
    files = list(iter_files(dataset().table(qualified).to_directory()))
    print(f"[brainmetshare] downloading {qualified} n={len(files)} -> {out}", flush=True)
    n_ok = n_skip = n_bytes = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = [ex.submit(_save_file, f, out) for f in files]
        for i, fut in enumerate(as_completed(futs), 1):
            _, sz, st = fut.result()
            n_bytes += sz
            n_ok += st == "ok"
            n_skip += st == "skip"
            if i % 25 == 0 or i == len(futs):
                print(f"[brainmetshare] {rel} {i}/{len(futs)} wrote={n_ok} skip={n_skip} "
                      f"{n_bytes / 1e9:.2f} GB", flush=True)
    print(f"[brainmetshare] {rel} files={n_ok + n_skip} {n_bytes / 1e9:.2f} GB", flush=True)
    return out


def write_manifest(dest: Path) -> Path:
    rows = []
    for p in sorted(dest.rglob("*")):
        if not p.is_file() or p.name in {"file_manifest.csv", "dataset.json"}:
            continue
        rel = p.relative_to(dest)
        rows.append({"path": str(rel), "bytes": p.stat().st_size, "suffix": p.suffix})
    df = pd.DataFrame(rows)
    path = dest / "file_manifest.csv"
    df.to_csv(path, index=False)
    print(f"[brainmetshare] manifest {len(df)} files {df['bytes'].sum() / 1e9:.2f} GB -> {path}", flush=True)
    return path


def fetch(dest: Path = OUT, concurrency: int = 4) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    write_metadata(dest)
    for qualified, rel in TABLES:
        download_table(qualified, rel, dest, concurrency)
    write_manifest(dest)
    return dest


def prep(store: Path = STORE, workers: int = 8) -> None:
    cmd = [sys.executable, "-m", "data.prep.run", "--root", str(store), "--datasets", "brainmetshare",
           "--workers", str(workers), "--z-anchor", "auto"]
    print(f"[brainmetshare] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    fetch(a.out, a.concurrency)
    from data.prep import sources
    cases = sources.brainmetshare(a.out / "nifti")
    n_mask = sum(1 for c in cases if c.masks)
    print(f"[brainmetshare] {len(cases)} volumes, {n_mask} with segs under {a.out / 'nifti'}", flush=True)
    if a.prep:
        prep(STORE, a.workers)


if __name__ == "__main__":
    main()
