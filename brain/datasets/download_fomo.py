"""Download selected FOMO260K folders onto scratch.

    python -m brain.datasets.download_fomo              # mapping + holdouts + pull + extract
    python -m brain.datasets.download_fomo --mapping    # tables and hold-out lists only
    python -m brain.datasets.download_fomo --extract    # unzip structural NIfTIs already on disk

Scratch is a 1 TB quota. Each session zip is extracted to ``extracted/`` (structural
NIfTIs only) and then deleted. Pass ``--keep-zips`` to leave archives on disk.
"""

from __future__ import annotations

import argparse
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from brain.prep.fomo import (
    FOMO_REPO,
    FOMO_ROOT,
    PULL_FOLDERS,
    SKIP_FOLDERS,
    extracted_path,
    is_structural,
    load_mappings,
    write_holdout_lists,
)


def _hub():
    from huggingface_hub import hf_hub_download, list_repo_files
    return hf_hub_download, list_repo_files


def fetch_tables(root: Path = FOMO_ROOT) -> list[Path]:
    hf_hub_download, _ = _hub()
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for name in ("mapping.tsv", "participants.tsv", "mri_info.tsv", "README.md"):
        dest = root / name
        if dest.exists() and dest.stat().st_size > 0:
            out.append(dest)
            print(f"[fomo] table {name} (have)", flush=True)
            continue
        p = hf_hub_download(FOMO_REPO, name, repo_type="dataset", local_dir=str(root))
        out.append(Path(p))
        print(f"[fomo] table {name} -> {p}", flush=True)
    for pt in (*PULL_FOLDERS, "PT030_OpenNeuro"):
        dest = root / pt / "mapping.tsv"
        if dest.exists() and dest.stat().st_size > 0:
            out.append(dest)
            print(f"[fomo] table {pt}/mapping.tsv (have)", flush=True)
            continue
        try:
            p = hf_hub_download(FOMO_REPO, f"{pt}/mapping.tsv", repo_type="dataset", local_dir=str(root))
            out.append(Path(p))
            print(f"[fomo] table {pt}/mapping.tsv", flush=True)
        except Exception as e:
            print(f"[fomo] skip {pt}/mapping.tsv ({e})", flush=True)
    return out


def fetch_folders(folders: tuple[str, ...] = PULL_FOLDERS, root: Path = FOMO_ROOT,
                  workers: int = 8, keep_zips: bool = False) -> Path:
    """Per-file download. ``snapshot_download`` + glob patterns listed an empty set on hub 0.36."""
    hf_hub_download, list_repo_files = _hub()
    root.mkdir(parents=True, exist_ok=True)
    keep = [f for f in folders if f not in SKIP_FOLDERS]
    print(f"[fomo] list {FOMO_REPO}", flush=True)
    listed = list_repo_files(FOMO_REPO, repo_type="dataset")
    want = [f for f in listed if any(f == pt or f.startswith(pt + "/") for pt in keep)]
    print(f"[fomo] download {len(want)} files folders={keep} -> {root}", flush=True)

    def one(rel: str) -> tuple[str, str]:
        dest = root / rel
        if already_extracted(root, rel):
            return rel, "skip"
        status = "skip"
        if not (dest.exists() and dest.stat().st_size > 0):
            dest.parent.mkdir(parents=True, exist_ok=True)
            delay = 10
            last = ""
            status = ""
            for attempt in range(8):
                try:
                    hf_hub_download(
                        FOMO_REPO, rel, repo_type="dataset", local_dir=str(root),
                        etag_timeout=60,
                    )
                    status = "ok"
                    break
                except Exception as e:
                    last = str(e).split("\n", 1)[0][:180]
                    if dest.exists() and dest.stat().st_size > 0:
                        status = "ok"
                        break
                    wait = delay if ("429" in last or "Too Many Requests" in last) else min(delay, 20)
                    time.sleep(wait)
                    delay = min(delay * 2, 180)
            if not status:
                return rel, f"err:{last}"
        if rel.endswith(".zip") and dest.exists() and dest.stat().st_size > 0:
            n_ok, _ = ingest_zip(dest, root, delete=not keep_zips)
            if n_ok < 0:
                return rel, "badzip"
        return rel, status

    n_ok = n_skip = n_err = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, rel) for rel in want]
        for i, fut in enumerate(as_completed(futs), 1):
            rel, status = fut.result()
            n_ok += status == "ok"
            n_skip += status == "skip"
            n_err += status.startswith("err")
            if i % 100 == 0 or status not in ("ok", "skip"):
                print(f"[fomo] {i}/{len(want)} {status} {rel}", flush=True)
    print(f"[fomo] files ok={n_ok} skip={n_skip} err={n_err} / {len(want)}", flush=True)
    return root


def _extract_zip(zip_path: Path, dest_dir: Path) -> tuple[str, int, int]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        return str(zip_path), -1, 0
    with zf:
        for info in zf.infolist():
            if info.is_dir() or not is_structural(info.filename):
                continue
            out = dest_dir / Path(info.filename).name
            if out.exists() and out.stat().st_size == info.file_size:
                n_skip += 1
                continue
            with zf.open(info) as src, open(out, "wb") as dst:
                dst.write(src.read())
            n_ok += 1
    return str(zip_path), n_ok, n_skip


EXTRACTED_MARK = ".extracted"


def session_extract_dir(root: Path, zip_path: Path) -> Path:
    return extracted_path(root, zip_path.parents[1].name, zip_path.parent.name, zip_path.stem, "_").parent


def already_extracted(root: Path, rel: str) -> bool:
    """True when this session zip was ingested and deleted (marker file on the extract dir)."""
    if not rel.endswith(".zip"):
        return False
    zp = Path(rel)
    if len(zp.parts) < 3:
        return False
    return (extracted_path(root, zp.parts[0], zp.parts[1], zp.stem, "_").parent / EXTRACTED_MARK).exists()


def ingest_zip(zip_path: Path, root: Path, delete: bool = True) -> tuple[int, int]:
    """Copy structural NIfTIs out of one session zip, then drop the archive."""
    dest = session_extract_dir(root, zip_path)
    _, n_ok, n_skip = _extract_zip(zip_path, dest)
    if n_ok < 0:
        return -1, 0
    dest.mkdir(parents=True, exist_ok=True)
    (dest / EXTRACTED_MARK).touch()
    if delete and zip_path.exists():
        zip_path.unlink()
    return n_ok, n_skip


def extract_structural(root: Path = FOMO_ROOT, folders: tuple[str, ...] = PULL_FOLDERS,
                       workers: int = 4, delete_zips: bool = True) -> int:
    zips = []
    for pt in folders:
        zips.extend(sorted((root / pt).glob("sub-*/ses-*.zip")))
    print(f"[fomo] extract structural from {len(zips)} session zips delete_zips={delete_zips}", flush=True)
    n_files = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(ingest_zip, zp, root, delete_zips) for zp in zips]
        for i, fut in enumerate(as_completed(futs), 1):
            n_ok, n_skip = fut.result()
            if n_ok < 0:
                print(f"[fomo] bad zip ({i}/{len(futs)})", flush=True)
                continue
            n_files += max(n_ok, 0) + n_skip
            if i % 200 == 0:
                print(f"[fomo] extracted {i}/{len(futs)} zips files={n_files}", flush=True)
    print(f"[fomo] extracted structural files={n_files} -> {root / 'extracted'}", flush=True)
    return n_files


def fetch(root: Path = FOMO_ROOT, workers: int = 2, mapping_only: bool = False,
          extract_only: bool = False, keep_zips: bool = False) -> Path:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    if not extract_only:
        fetch_tables(root)
        write_holdout_lists(load_mappings(root), root)
        if mapping_only:
            return root
    extract_workers = max(2, min(workers, 8))
    extract_structural(root, PULL_FOLDERS, workers=extract_workers, delete_zips=not keep_zips)
    if not extract_only:
        fetch_folders(PULL_FOLDERS, root, workers=workers, keep_zips=keep_zips)
        extract_structural(root, PULL_FOLDERS, workers=extract_workers, delete_zips=not keep_zips)
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=FOMO_ROOT)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mapping", action="store_true", help="tables + hold-out lists only")
    ap.add_argument("--extract", action="store_true", help="extract zips already on disk")
    ap.add_argument("--keep-zips", action="store_true", help="leave session archives after extracting NIfTIs")
    a = ap.parse_args()
    fetch(a.out, workers=a.workers, mapping_only=a.mapping, extract_only=a.extract, keep_zips=a.keep_zips)
    print("[fomo] done", flush=True)


if __name__ == "__main__":
    main()
