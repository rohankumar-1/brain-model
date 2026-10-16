"""Download INSTANCE 2022 (MICCAI ICH NCCT) into the pool and preprocess it there.

The official share is a OneDrive zip (``train_2.zip``). Raw files and the processed
Zarr store both live under ``$BRAIN_DATASETS/instance2022`` — nothing is written to scratch.

    python -m data.download_instance --prep
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

from data.prep.sources import DATASETS

SHARE = "https://1drv.ms/u/s!ApYSTFRpDKWAgUrdBzCCipLwl0jF?e=Njt3RP"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
OUT = DATASETS / "instance2022"
STORE = OUT / "processed"


def _headers(url: str, extra: list[str] | None = None) -> str:
    cmd = ["curl", "-sI", "-A", UA, "--max-redirs", "0", *(extra or []), url]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _header(blob: str, name: str) -> str:
    m = re.search(rf"(?im)^{re.escape(name)}:\s*(.+)$", blob)
    return m.group(1).strip() if m else ""


def resolve_share(share: str = SHARE) -> tuple[str, str]:
    """Turn a 1drv.ms link into the SharePoint ``download.aspx`` URL that actually streams the zip."""
    loc = _header(_headers(share), "Location")
    blob = loc or share
    cid = re.search(r"/personal/([0-9a-fA-F]+)/", blob, re.I)
    token = re.search(r"(s![A-Za-z0-9_-]+)", blob)
    if cid is None or token is None:
        raise SystemExit(f"could not parse OneDrive share redirect:\n{blob[:500]}")
    url = (f"https://my.microsoftpersonalcontent.com/personal/{cid.group(1).lower()}"
           f"/_layouts/15/download.aspx?share={token.group(1)}")
    return url, blob


def filename_for(url: str, redirect: str) -> str:
    disp = _header(_headers(url, ["-L", "--max-redirs", "0"]), "Content-Disposition")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disp, re.I)
    if m:
        return Path(unquote(m.group(1))).name
    m = re.search(r"spopath=([^&]+)", redirect, re.I)
    if m:
        return Path(unquote(m.group(1))).name
    return "train_2.zip"


def download(share: str = SHARE, dest: Path = OUT) -> Path:
    url, redirect = resolve_share(share)
    out = dest / filename_for(url, redirect)
    if out.exists() and out.stat().st_size > 0:
        print(f"[instance] already have {out} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)
        return out
    dest.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    print(f"[instance] downloading {out.name} <- {url}", flush=True)
    subprocess.run(
        ["curl", "-fL", "-A", UA, "-C", "-", "--retry", "8", "--retry-delay", "5",
         "-o", str(part), url],
        check=True,
    )
    part.rename(out)
    print(f"[instance] wrote {out} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)
    return out


def extract(archive: Path, dest: Path = OUT / "extracted") -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.rglob("*.nii.gz")):
        print(f"[instance] extract already has NIfTI under {dest}", flush=True)
        return dest
    print(f"[instance] unzip {archive.name} -> {dest}", flush=True)
    subprocess.run(["unzip", "-n", "-q", str(archive), "-d", str(dest)], check=True)
    n = len(list(dest.rglob("*.nii.gz")))
    print(f"[instance] {n} NIfTI files", flush=True)
    return dest


def fetch(share: str = SHARE) -> Path:
    """Download and extract into the pool. Returns the extract root."""
    return extract(download(share))


def prep(store: Path = STORE, workers: int = 16) -> None:
    cmd = [sys.executable, "-m", "data.prep.run", "--root", str(store), "--datasets", "instance",
           "--workers", str(workers), "--z-anchor", "auto"]
    print(f"[instance] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--share", default=SHARE)
    ap.add_argument("--prep", action="store_true", help="run v4 preprocess into pool after extract")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    root = fetch(a.share)
    from data.prep import sources
    cases = sources.instance(root)
    n_mask = sum(1 for c in cases if c.masks)
    print(f"[instance] {len(cases)} volumes, {n_mask} with ICH masks", flush=True)
    if not cases:
        raise SystemExit(f"no NCCT volumes found under {root}")
    if a.prep:
        prep(STORE, a.workers)


if __name__ == "__main__":
    main()
