#!/usr/bin/env python3
"""Download BraTS challenge training data (labeled MRI) via Synapse → pool/datasets.

Dedup policy
------------
* BraTS **2021 adult glioma ≈ BraTS 2023 GLI training** (same subjects; ET label
  4→3 in 2023). We download **2023 GLI only** and treat it as the 2021+2023
  glioma set.
* **2023 MEN** (pre-op meningioma) — distinct pathology from glioma.
* **2023/2024 MET** — Synapse folders currently only expose READMEs (no zip).
* **2024 GLI** (post-treatment glioma) and **2024 MEN-RT** — new cohorts, but
  require accepting "BraTS 2024 Post-Challenge Data Access" in the Synapse UI
  (API token lacks authorize/modify scope to self-sign).
* Skip PED / SSA / Path.

Every training case includes **t1n, t1c, t2w, t2f (FLAIR)** + ``*-seg.nii.gz``.

Usage::

  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  python data/download_brats_synapse.py
  python data/download_brats_synapse.py --include-2024   # after UI access
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from brain.common.paths import DATASETS

DEFAULT_OUT = DATASETS / "brats"

# (syn_id, relative_path, notes)
CORE_DOWNLOADS = [
    (
        "syn51514132",
        "2023_GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip",
        "2023 GLI train (=2021 glioma cohort); labeled; t1n/t1c/t2w/t2f+seg",
    ),
    (
        "syn51514055",
        "2023_MEN/ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData.zip",
        "2023 MEN train; labeled; t1n/t1c/t2w/t2f+seg",
    ),
    (
        "syn51718026",
        "2023_MEN/BraTS-MEN-TRAIN-FIX-V4.zip",
        "2023 MEN label/file fixes overlay",
    ),
    (
        "syn51615438",
        "2023_GLI/BraTS2023_2017_GLI_Mapping.xlsx",
        "name mapping across BraTS years",
    ),
]

DOWNLOADS_2024 = [
    (
        "syn60086071",
        "2024_GLI/BraTS2024-BraTS-GLI-TrainingData.zip",
        "2024 post-treatment GLI train; labeled; mpMRI+seg",
    ),
    (
        "syn64314352",
        "2024_GLI/BraTS2024-BraTS-GLI-AdditionalTrainingData.zip",
        "2024 GLI additional train",
    ),
    (
        "syn60085033",
        "2024_MEN_RT/BraTS2024-MEN-RT-TrainingData.zip",
        "2024 meningioma RT train; labeled",
    ),
]


def _strip_proxy() -> None:
    for k in list(os.environ):
        if "proxy" in k.lower():
            del os.environ[k]


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_one(syn, syn_id: str, dest: Path, expected_md5: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if expected_md5:
            got = md5_file(dest)
            if got == expected_md5:
                print(f"[skip] {dest.name} (md5 ok)")
                return dest
            print(f"[redo] {dest.name} md5 mismatch {got} != {expected_md5}")
        else:
            print(f"[skip] {dest.name} exists ({dest.stat().st_size/1e9:.2f} GB)")
            return dest
    print(f"[get] {syn_id} → {dest}")
    entity = syn.get(syn_id, downloadLocation=str(dest.parent), ifcollision="overwrite.local")
    src = Path(entity.path)
    if src.resolve() != dest.resolve():
        src.replace(dest)
    print(f"[ok] {dest} ({dest.stat().st_size/1e9:.2f} GB)")
    return dest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--include-2024", action="store_true")
    p.add_argument("--only-2024", action="store_true")
    return p.parse_args()


def main() -> None:
    _strip_proxy()
    args = parse_args()
    import synapseclient

    syn = synapseclient.Synapse()
    syn.login()
    print(f"[out] {args.out_dir}")

    items = []
    if not args.only_2024:
        items.extend(CORE_DOWNLOADS)
    if args.include_2024 or args.only_2024:
        items.extend(DOWNLOADS_2024)

    # Prefetch expected md5s
    meta = {}
    for syn_id, rel, notes in items:
        try:
            h = syn.restGET(f"/entity/{syn_id}/filehandles")["list"][0]
            meta[syn_id] = h
            print(
                f"[plan] {rel}  {h['contentSize']/1e9:.2f} GB  "
                f"md5={h['contentMd5']}  # {notes}"
            )
        except Exception as e:
            print(f"[blocked] {syn_id} ({rel}): {e}")
            print(
                "  → Open Synapse in browser, accept "
                "'BraTS 2024 Post-Challenge Data Access', then re-run "
                "with --include-2024"
            )

    ok, fail = [], []
    for syn_id, rel, notes in items:
        if syn_id not in meta:
            fail.append((syn_id, rel, "no access"))
            continue
        dest = args.out_dir / rel
        try:
            download_one(syn, syn_id, dest, expected_md5=meta[syn_id]["contentMd5"])
            # verify md5
            got = md5_file(dest)
            exp = meta[syn_id]["contentMd5"]
            if got != exp:
                raise RuntimeError(f"md5 fail {got} != {exp}")
            ok.append(rel)
        except Exception as e:
            print(f"[FAIL] {rel}: {e}")
            fail.append((syn_id, rel, str(e)))

    print(f"\n[done] ok={len(ok)} fail={len(fail)}")
    for r in ok:
        print(f"  OK   {r}")
    for syn_id, rel, err in fail:
        print(f"  FAIL {rel}: {err}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
