"""Download INSTANCE 2022 (MICCAI ICH NCCT) into the pool (``$BRAIN_DATASETS/instance2022``).

The official share is a OneDrive zip (``train_2.zip``). Prep into the main store with
``brain.prep.run --datasets instance``.

    python -m brain.datasets.download_instance
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

from brain.prep.sources import DATASETS

SHARE = "https://1drv.ms/u/s!ApYSTFRpDKWAgUrdBzCCipLwl0jF?e=Njt3RP"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
OUT = DATASETS / "instance2022"


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--share", default=SHARE)
    a = ap.parse_args()
    root = fetch(a.share)
    from brain.prep import sources
    cases = sources.instance(root)
    n_mask = sum(1 for c in cases if c.masks)
    print(f"[instance] {len(cases)} volumes, {n_mask} with ICH masks", flush=True)
    if not cases:
        raise SystemExit(f"no NCCT volumes found under {root}")


if __name__ == "__main__":
    main()
