#!/usr/bin/env python3
"""Download OASIS-1 and OASIS-2 (raw MRI + clinical/demographic tables) → pool/datasets.

Sources: https://sites.wustl.edu/oasisbrains/home/oasis-1/
         https://sites.wustl.edu/oasisbrains/home/oasis-2/

Pulls:
  * OASIS-1 clinical: fact sheet PDF, demographics/clinical xlsx, reliability xlsx
  * OASIS-1 MRI: oasis_cross-sectional_disc{1..12}.tar.gz (~1.5 GB each)
  * OASIS-2 clinical: longitudinal demographics xlsx
  * OASIS-2 MRI: OAS2_RAW_PART{1,2}.tar.gz

Skips FreeSurfer derivatives (not needed for MRI + age/CDR downstream).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

DEFAULT_OUT = Path("/home/roku/orcd/pool/datasets/oasis")

CLINICAL = [
    (
        "https://sites.wustl.edu/oasisbrains/files/2024/03/oasis_cross-sectional_facts-bcc7a002dfb104f4.pdf",
        "oasis1/oasis_cross-sectional_facts.pdf",
    ),
    (
        "https://sites.wustl.edu/oasisbrains/files/2024/04/oasis_cross-sectional-5708aa0a98d82080.xlsx",
        "oasis1/oasis_cross-sectional.xlsx",
    ),
    (
        "https://sites.wustl.edu/oasisbrains/files/2024/04/oasis_cross-sectional-reliability-063c8642b909ee76.xlsx",
        "oasis1/oasis_cross-sectional_reliability.xlsx",
    ),
    (
        "https://sites.wustl.edu/oasisbrains/files/2024/03/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx",
        "oasis2/oasis_longitudinal_demographics.xlsx",
    ),
]

MRI = [
    *[
        (
            f"https://download.nrg.wustl.edu/data/oasis_cross-sectional_disc{i}.tar.gz",
            f"oasis1/oasis_cross-sectional_disc{i}.tar.gz",
        )
        for i in range(1, 13)
    ],
    (
        "https://download.nrg.wustl.edu/data/OAS2_RAW_PART1.tar.gz",
        "oasis2/OAS2_RAW_PART1.tar.gz",
    ),
    (
        "https://download.nrg.wustl.edu/data/OAS2_RAW_PART2.tar.gz",
        "oasis2/OAS2_RAW_PART2.tar.gz",
    ),
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path, min_bytes: int = 1000) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"[skip] {dest} ({dest.stat().st_size/1e9:.2f} GB)")
        return

    # resume via Range if .part exists
    headers = {}
    mode = "wb"
    existing = 0
    if part.exists():
        existing = part.stat().st_size
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"[resume] {dest.name} from {existing/1e9:.2f} GB")
    else:
        print(f"[get] {url}")
        print(f"      → {dest}")

    headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    )
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, part.open(mode) as out:
            # if server ignored Range and returned 200 full body, restart
            if existing and getattr(resp, "status", 200) == 200 and "Content-Range" not in resp.headers:
                out.close()
                part.unlink(missing_ok=True)
                print("[warn] server ignored Range; restarting full download")
                return download(url, dest, min_bytes=min_bytes)
            total = resp.headers.get("Content-Length")
            total_i = int(total) + existing if total else None
            n = existing
            last_print = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                n += len(chunk)
                if n - last_print >= 50 << 20:
                    if total_i:
                        print(f"  {n/1e9:.2f}/{total_i/1e9:.2f} GB ({100*n/total_i:.1f}%)")
                    else:
                        print(f"  {n/1e9:.2f} GB")
                    last_print = n
    except Exception as e:
        # if Range not supported (416), wipe and retry once
        if existing and "416" in str(e):
            part.unlink(missing_ok=True)
            return download(url, dest, min_bytes=min_bytes)
        raise

    if part.stat().st_size < min_bytes:
        raise RuntimeError(f"download too small: {part} ({part.stat().st_size} B)")
    part.replace(dest)
    print(f"[ok] {dest} ({dest.stat().st_size/1e9:.2f} GB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--clinical-only", action="store_true")
    p.add_argument("--mri-only", action="store_true")
    args = p.parse_args()

    items = []
    if not args.mri_only:
        items.extend(CLINICAL)
    if not args.clinical_only:
        items.extend(MRI)

    ok, fail = [], []
    for url, rel in items:
        dest = args.out_dir / rel
        try:
            download(url, dest)
            ok.append(rel)
        except Exception as e:
            print(f"[FAIL] {rel}: {e}")
            fail.append((rel, str(e)))

    print(f"\n[done] ok={len(ok)} fail={len(fail)}")
    for r in ok:
        print(f"  OK   {r}")
    for r, e in fail:
        print(f"  FAIL {r}: {e}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
