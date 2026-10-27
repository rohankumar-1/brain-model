"""Raw datasets on pool as large, bit-exact archives; staging them back for a re-prep.

Pool holds, per dataset folder, only compressed files: original downloads that are already archives (RSNA zip, MSD tar,
FOMO/ABIDE JPEG-LS zips) and ``<name>.raw.zip`` holding everything else. NIfTI volumes go in as JPEG-LS (the DICOM
lossless codec; ``brain.prep.archive``): integer data exactly, float data quantised to 16 bits, plus the original
NIfTI header so qform/sform, units and intent come back unchanged. Every encode is decoded and compared on the spot;
a volume that does not round-trip (e.g. int32 beyond 16 bits) is stored byte-for-byte instead. Other files (labels,
CSVs, PNGs, Analyze pairs) are stored or deflated byte-for-byte. Known intermediates (aborted packing runs,
per-dataset processed stores, extracted copies of an archive kept beside them) are deleted, not archived.

Packing never deletes a file before the zip is closed, re-opened, CRC-checked member by member and matched to the
source by name (and, for byte-for-byte members, size). An extracted copy is only dropped after the archive kept
beside it is shown to hold every one of its files at the same size.

    python -m brain.prep.rawpack plan                       # what would happen, per dataset (no writes)
    python -m brain.prep.rawpack pack --datasets ixi,atlas  # pack, verify, then delete the packed files
    python -m brain.prep.rawpack stage --dest $TMPDIR/datasets --datasets ixi,msd_braintumor
        # -> export BRAIN_DATASETS=$TMPDIR/datasets before brain.prep.run
"""

from __future__ import annotations

import argparse
import base64
import json
import multiprocessing
import os
import shutil
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from brain.common.paths import DATASETS

STORED_SUFFIXES = (".gz", ".zip", ".jls", ".png", ".jpg", ".jpeg", ".xlsx", ".bz2", ".xz", ".zst", ".tgz", ".parquet")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
NIFTI_SUFFIXES = (".nii.gz", ".nii")
JLS, META = ".jls", ".jls.json"


@dataclass(frozen=True)
class Policy:
    """Per-dataset rules. ``drop``: intermediates deleted outright. ``covered``: extracted copy -> the kept archive
    (relative to the dataset folder) that must contain all of its files; the copy is deleted once that holds."""

    drop: tuple[str, ...] = ()
    covered: dict = field(default_factory=dict)
    skip: bool = False                               # already only archives (+ small metadata files)


POLICIES = {
    "abide1": Policy(skip=True),
    "fomo260k": Policy(skip=True),
    "rsna": Policy(skip=True),
    "brats": Policy(drop=("extracted.jls.zip.d", "extracted.jls.zip.partial")),
    "oasis": Policy(drop=("extracted.jls.zip.d", "extracted.jls.zip.partial")),
    "aisd": Policy(drop=("processed",)),
    "instance2022": Policy(drop=("processed",)),
    "brainmetshare": Policy(drop=("processed",)),
    "msd_braintumor": Policy(drop=("Task01_BrainTumour/derived",),
                             covered={"Task01_BrainTumour": "Task01_BrainTumour.tar"}),
    "bhsd": Policy(covered={"extracted/label_192": "label_192.zip"}),
}


def raw_zip(folder: Path) -> Path:
    return folder / f"{folder.name}.raw.zip"


def _is_kept_archive(folder: Path, p: Path) -> bool:
    """A top-level original archive (left as is), never a member of a tree."""
    return p.parent == folder and p.name.endswith(ARCHIVE_SUFFIXES) and not p.name.endswith(".partial")


def _files(root: Path) -> list[Path]:
    out = []
    for dirpath, _, names in os.walk(root):
        out.extend(Path(dirpath) / n for n in names)
    return sorted(out)


def _archive_members(path: Path) -> dict[str, int]:
    """Regular-file members -> size, for a zip or tar."""
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            return {i.filename.rstrip("/"): i.file_size for i in zf.infolist() if not i.is_dir()}
    with tarfile.open(path) as tf:
        return {m.name.lstrip("./"): m.size for m in tf.getmembers() if m.isfile()}


def covered_by(tree: Path, archive: Path) -> tuple[bool, str]:
    """Every file under ``tree`` is in ``archive`` at the same size. Member paths may be prefixed by the tree's name
    or be relative to it; macOS ``._`` resource forks are ignored."""
    members = _archive_members(archive)
    missing = 0
    for p in _files(tree):
        if p.name.startswith("._") or p.name == ".DS_Store":
            continue
        rel = p.relative_to(tree).as_posix()
        size = p.stat().st_size
        if members.get(f"{tree.name}/{rel}") != size and members.get(rel) != size:
            missing += 1
    return missing == 0, f"{missing} files of {tree} not in {archive.name}"


def _plan(folder: Path) -> dict:
    pol = POLICIES.get(folder.name, Policy())
    drops = [folder / d for d in pol.drop if (folder / d).exists()]
    covered = {folder / t: folder / a for t, a in pol.covered.items() if (folder / t).exists() and (folder / a).exists()}
    skip_trees = set(drops) | set(covered)
    files, nbytes = [], 0
    if not pol.skip:
        for p in _files(folder):
            if p.name.startswith(raw_zip(folder).name) or _is_kept_archive(folder, p) \
                    or any(p == d or d in p.parents for d in skip_trees):          # incl. our .partial / .unverified
                continue
            files.append(p)
            nbytes += p.stat().st_size
    return {"folder": folder, "skip": pol.skip, "drops": drops, "covered": covered, "files": files, "bytes": nbytes}


def pool_free_gb(quota: Path = Path.home() / "orcd" / ".quota") -> float | None:
    try:
        for line in quota.read_text().splitlines():
            if line.strip().upper().startswith("POOL"):
                cells = [c.strip() for c in line.split("|")]
                return float(cells[2]) - float(cells[1])
    except (OSError, IndexError, ValueError):
        return None
    return None


def _rm(p: Path) -> None:
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    else:
        p.unlink(missing_ok=True)


def _prune_empty_dirs(folder: Path) -> None:
    for dirpath, dirnames, filenames in sorted(os.walk(folder), key=lambda t: -len(t[0])):
        d = Path(dirpath)
        if d != folder and not any(d.iterdir()):
            d.rmdir()


def is_nifti(p: Path) -> bool:
    return p.name.lower().endswith(NIFTI_SUFFIXES)


def _tolerance(meta: dict) -> float:
    if meta.get("kind") == "uint16_quant":       # half a quantisation step + float32 rounding in restore_array
        lo, hi = float(meta["quant_lo"]), float(meta["quant_hi"])
        return 0.5001 * (hi - lo) / 65535.0 + 8 * float(np.finfo(np.float32).eps) * max(abs(lo), abs(hi), 1.0)
    return 0.011 if meta.get("was_float_int") else 0.0


def encode_nifti(path: str) -> tuple[bytes, dict] | None:
    """JPEG-LS blob + metadata (incl. the raw NIfTI header) for one file, or None when it does not round-trip."""
    import nibabel as nib

    from brain.prep import archive

    try:
        _, blob, meta = archive._encode_one(path)
        img = nib.load(path)
        raw = img.dataobj
        src = np.asanyarray(raw.get_unscaled() if hasattr(raw, "get_unscaled") else raw)
        back = archive.restore_array(archive.decode_volume(blob), meta)
        src = np.nan_to_num(np.squeeze(src).astype(np.float64))
        if back.shape != src.shape or float(np.max(np.abs(back.astype(np.float64) - src), initial=0.0)) > _tolerance(meta):
            return None
        meta["nifti_header"] = base64.b64encode(img.header.binaryblock).decode()
        meta["src_shape"] = [int(s) for s in img.shape]
        return blob, meta
    except Exception:
        return None


def restore_nifti(blob: bytes, meta: dict, dest: Path) -> None:
    """Write the NIfTI a JPEG-LS member came from (original header; storage values; slope/inter re-applied)."""
    import nibabel as nib

    from brain.prep import archive

    vol = archive.restore_array(archive.decode_volume(blob), meta)
    vol = vol.reshape(meta.get("src_shape") or vol.shape)
    block = base64.b64decode(meta["nifti_header"])
    klass, hklass = (nib.Nifti1Image, nib.Nifti1Header) if len(block) == 348 else (nib.Nifti2Image, nib.Nifti2Header)
    hdr = hklass(binaryblock=block)
    orig = np.dtype(meta.get("orig_dtype", vol.dtype))
    if meta.get("kind") != "uint16_quant" and not meta.get("was_float_int"):
        vol = vol.astype(orig)
    img = klass(vol, None, header=hdr)
    img.set_data_dtype(orig)
    slope, inter = meta.get("slope"), meta.get("inter")
    if slope is not None or inter is not None:
        img.header.set_slope_inter(float(slope if slope is not None else 1.0), float(inter or 0.0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.partial{''.join(dest.suffixes)}"
    nib.save(img, str(tmp))
    os.replace(tmp, dest)


def _expected(folder: Path, files: list[Path], jls: set[str]) -> dict[str, int | None]:
    """Zip member -> size (None: JPEG-LS members, whose size differs from the source by design)."""
    out: dict[str, int | None] = {}
    for p in files:
        rel = p.relative_to(folder).as_posix()
        if rel in jls:
            out[rel + JLS] = None
            out[rel + META] = None
        else:
            out[rel] = p.stat().st_size
    return out


def write_zip(folder: Path, files: list[Path], dest: Path, workers: int = 8) -> set[str]:
    """Write ``dest``; returns the relative paths stored as JPEG-LS (the rest are byte-for-byte)."""
    tmp = dest.with_name(dest.name + ".partial")
    tmp.unlink(missing_ok=True)
    nifti = [p for p in files if is_nifti(p)]
    jls: set[str] = set()
    def encoded(ex):
        # bounded window: Executor.map submits everything and buffers finished blobs behind a slow one (OOM on big sets)
        window = 4 * max(1, workers)
        for start in range(0, len(nifti), window):
            batch = nifti[start:start + window]
            yield from zip(batch, ex.map(encode_nifti, [str(p) for p in batch]))

    # forkserver: forking this (multi-threaded: numpy, zipfile) process can deadlock a worker
    ctx = multiprocessing.get_context("forkserver")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf, ProcessPoolExecutor(max(1, workers), mp_context=ctx) as ex:
        for p, enc in encoded(ex):
            rel = p.relative_to(folder).as_posix()
            if enc is None:
                zf.write(p, rel, compress_type=zipfile.ZIP_STORED)
                continue
            blob, meta = enc
            zf.writestr(rel + JLS, blob, compress_type=zipfile.ZIP_STORED)
            zf.writestr(rel + META, json.dumps(meta), compress_type=zipfile.ZIP_DEFLATED)
            jls.add(rel)
        for p in files:
            if is_nifti(p):
                continue
            stored = p.name.lower().endswith(STORED_SUFFIXES)
            zf.write(p, p.relative_to(folder).as_posix(),
                     compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,
                     compresslevel=None if stored else 6)
    os.replace(tmp, dest)
    print(f"[rawpack] {dest.name}: {len(jls)} JPEG-LS volumes, {len(files) - len(jls)} byte-for-byte files", flush=True)
    return jls


def verify_zip(folder: Path, files: list[Path], dest: Path, exact: bool = True) -> tuple[bool, str]:
    """Every source file has its member(s) (byte-for-byte ones at the same size), nothing else is in the zip
    (``exact=False``: the zip may hold files an interrupted run already deleted), and every CRC checks out."""
    with zipfile.ZipFile(dest) as zf:
        have = {i.filename: i.file_size for i in zf.infolist() if not i.is_dir()}
        jls = {n[: -len(JLS)] for n in have if n.endswith(JLS)}
        want = _expected(folder, files, jls)
        bad = [n for n, size in want.items() if n not in have or (size is not None and have[n] != size)]
        if exact:
            bad += [n for n in have if n not in want]
        if bad:
            return False, f"name/size mismatch on {len(bad)} entries (e.g. {bad[0]})"
        crc = zf.testzip()
        if crc is not None:
            return False, f"CRC error in {crc}"
    return True, f"{len(files)} files verified ({len(jls)} as JPEG-LS)"


def pack(folder: Path, apply: bool, reserve_gb: float = 20.0, workers: int = 8) -> None:
    plan = _plan(folder)
    tag = f"[rawpack] {folder.name}"
    if plan["skip"]:
        print(f"{tag}: already archives only", flush=True)
        return
    for d in plan["drops"]:
        print(f"{tag}: drop intermediate {d.relative_to(folder)}", flush=True)
        if apply:
            _rm(d)
    for tree, archive in plan["covered"].items():
        ok, msg = covered_by(tree, archive)
        print(f"{tag}: {tree.relative_to(folder)} covered by {archive.name}: {ok} ({msg if not ok else 'all files'})",
              flush=True)
        if ok and apply:
            _rm(tree)
        elif not ok:                                   # not a pure copy: archive it with everything else
            extra = _files(tree)
            plan["files"] = sorted(plan["files"] + extra)
            plan["bytes"] += sum(p.stat().st_size for p in extra)
    if not plan["files"]:
        print(f"{tag}: nothing left to pack", flush=True)
        return
    gb = plan["bytes"] / 1e9
    dest = raw_zip(folder)
    print(f"{tag}: {len(plan['files'])} files, {gb:.1f} GB -> {dest.name}", flush=True)
    if not apply:
        return
    free = pool_free_gb()
    if not dest.exists() and free is not None and free - reserve_gb < gb:
        print(f"{tag}: SKIP, pool has {free:.0f} GB free (needs {gb:.0f} + {reserve_gb:.0f} GB)", flush=True)
        return
    resumed = dest.exists()
    if not resumed:
        write_zip(folder, plan["files"], dest, workers)
    else:                                              # an interrupted run: the zip must hold what is still here
        print(f"{tag}: {dest.name} exists; verifying it against the remaining files", flush=True)
    ok, msg = verify_zip(folder, plan["files"], dest, exact=not resumed)
    print(f"{tag}: verify {ok} ({msg})", flush=True)
    if not ok:
        dest.rename(dest.with_name(dest.name + ".unverified"))
        raise SystemExit(f"{tag}: verification failed; sources kept")
    for p in plan["files"]:
        p.unlink()
    _prune_empty_dirs(folder)
    print(f"{tag}: packed and removed {len(plan['files'])} source files", flush=True)


def stage(folder: Path, dest: Path) -> None:
    """Rebuild a dataset folder for prep: unpack ``<name>.raw.zip`` (JPEG-LS members back to NIfTI under their
    original names), untar kept tars that cover a tree, and link the other kept archives (RSNA's zip, JPEG-LS zips)."""
    out = dest / folder.name
    out.mkdir(parents=True, exist_ok=True)
    pol = POLICIES.get(folder.name, Policy())
    for p in sorted(folder.iterdir()):
        if p == raw_zip(folder):
            with zipfile.ZipFile(p) as zf:
                for name in zf.namelist():
                    if name.endswith(META) or name.endswith("/"):
                        continue
                    if name.endswith(JLS):
                        restore_nifti(zf.read(name), json.loads(zf.read(name[: -len(JLS)] + META)),
                                      out / name[: -len(JLS)])
                    else:
                        zf.extract(name, out)
        elif p.is_file() and p.name.endswith((".tar", ".tar.gz", ".tgz")) and p.name in pol.covered.values():
            with tarfile.open(p) as tf:
                tf.extractall(out, filter="data")
        elif not (out / p.name).exists():
            (out / p.name).symlink_to(p)
    print(f"[rawpack] staged {folder.name} -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "pack", "stage"])
    ap.add_argument("--root", type=Path, default=DATASETS)
    ap.add_argument("--datasets", default="", help="comma list of dataset folders (default: all)")
    ap.add_argument("--dest", type=Path, default=None, help="stage: where to rebuild the dataset folders")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    names = [n for n in a.datasets.split(",") if n] or sorted(p.name for p in a.root.iterdir() if p.is_dir())
    folders = [a.root / n for n in names]
    if a.action == "stage":
        if a.dest is None:
            raise SystemExit("stage needs --dest")
        for f in folders:
            stage(f, a.dest)
        return
    # smallest first: each finished dataset frees pool space for the next
    sized = sorted(((_plan(f)["bytes"], f) for f in folders), key=lambda t: t[0])
    for _, f in sized:
        pack(f, apply=a.action == "pack", workers=a.workers)


if __name__ == "__main__":
    main()
