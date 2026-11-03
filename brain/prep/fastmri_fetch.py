"""Streaming fetch of the fastMRI+ annotated brain scans (2026-11-01): the fastMRI brain release is ~1.5 TB of
multi-coil k-space in tar.xz batches (no index, so a member can only be reached by reading through the archive);
fastMRI+ (github.com/microsoft/fastmri-plus) annotates 1,001 of its scans (``Annotations/brain_file_list.csv``:
451 AXFLAIR, 340 AXT1, 210 AXT1POST). Each batch streams from its signed URL straight through ``xz -d`` and is never
stored: unwanted members are read past, a wanted ``.h5`` lands on node-local temp, is reduced to its coil-combined
image (``reconstruction_rss``, or a root-sum-of-squares of the inverse FFT when only k-space is there, as in
``test_full``) with the ISMRMRD header and the file's attributes, and is appended to a per-batch zip on pool. The
k-space is dropped.

Data use agreement (NYU fastMRI): the work dir, the links file (``work/fastmri/links.txt``, ``<archive> <url>`` per
line, mode 600) and the pool dir are owner-only; URLs never reach a log (errors are printed without the query).

    python -m brain.prep.fastmri_fetch --archive brain_multicoil_val_batch_0.tar.xz [--limit-gb 2]
      -> pool/datasets/fastmri_brain/fastmri_plus_<batch>.zip   (<name>.h5: image, header, attrs)
         work/fastmri/manifest_<batch>.csv (kept scans), work/fastmri/members_<batch>.txt (every member seen)
"""

from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import tarfile
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

from brain.common.paths import DATASETS, SCRATCH

WORK = SCRATCH / "work" / "fastmri"
LINKS = WORK / "links.txt"
WANTED = WORK / "fastmri-plus" / "Annotations" / "brain_file_list.csv"
POOL = DATASETS / "fastmri_brain"
CHUNK = 8 << 20


def _clean(e: Exception) -> str:
    """An error message with any signed query string cut off."""
    return re.sub(r"\?[^\s'\"]*", "?<query>", str(e))[:300]


class ResumableStream:
    """Bytes of a URL, reconnecting with a Range request from the last byte read when the connection drops."""

    def __init__(self, url: str, limit: int | None = None, retries: int = 50):
        self.url, self.pos, self.limit, self.retries = url, 0, limit, retries
        self.resp = None

    def _open(self):
        import requests

        headers = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        r = requests.get(self.url, headers=headers, stream=True, timeout=(30, 120))
        if r.status_code not in (200, 206):
            raise IOError(f"HTTP {r.status_code}")
        self.resp = r

    def chunks(self):
        tries = 0
        while self.limit is None or self.pos < self.limit:
            try:
                if self.resp is None:
                    self._open()
                for c in self.resp.iter_content(CHUNK):
                    if not c:
                        continue
                    if self.limit is not None:
                        c = c[: self.limit - self.pos]
                    self.pos += len(c)
                    tries = 0
                    yield c
                    if self.limit is not None and self.pos >= self.limit:
                        return
                return
            except Exception as e:  # noqa: BLE001  dropped connection: resume from self.pos
                tries += 1
                self.resp = None
                if tries > self.retries:
                    raise IOError(f"giving up at byte {self.pos}: {_clean(e)}") from None
                print(f"[fastmri] reconnect at {self.pos / 1e9:.2f} GB ({_clean(e)})", flush=True)
                time.sleep(min(60, 2 ** tries))


def links() -> dict[str, str]:
    out = {}
    for line in LINKS.read_text().splitlines():
        if line.strip():
            name, url = line.split(None, 1)
            out[name] = url.strip()
    return out


def wanted() -> set[str]:
    return {line.strip() for line in WANTED.read_text().splitlines() if line.strip().startswith("file_brain")}


def rss_from_kspace(kspace: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """(S, C, H, W) complex k-space -> (S, h, w) root-sum-of-squares image, centre-cropped (fastMRI's convention)."""
    img = []
    for k in kspace:
        x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))
        img.append(np.sqrt((np.abs(x) ** 2).sum(0)))
    img = np.stack(img).astype(np.float32)
    h, w = img.shape[-2:]
    oh, ow = min(out_hw[0], h), min(out_hw[1], w)
    t, l = (h - oh) // 2, (w - ow) // 2
    return img[:, t:t + oh, l:l + ow]


def _recon_hw(header: str) -> tuple[int, int]:
    m = re.search(r"<reconSpace>.*?<matrixSize>\s*<x>(\d+)</x>\s*<y>(\d+)</y>", header, re.S)
    return (int(m.group(2)), int(m.group(1))) if m else (320, 320)


def reduce_h5(src: Path) -> tuple[bytes, dict]:
    """A fastMRI .h5 -> a small .h5 (image float32 gzip, ``ismrmrd_header``, the original attributes) and a summary."""
    import h5py

    with h5py.File(src, "r") as f:
        header = f["ismrmrd_header"][()]
        header = header.decode() if isinstance(header, bytes) else str(header)
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        if "reconstruction_rss" in f:
            img, how = f["reconstruction_rss"][()].astype(np.float32), "rss"
        else:
            img, how = rss_from_kspace(f["kspace"][()], _recon_hw(header)), "rss_from_kspace"
    buf = io.BytesIO()
    with h5py.File(buf, "w") as g:
        g.create_dataset("image", data=img, compression="gzip", compression_opts=4, shuffle=True)
        g.create_dataset("ismrmrd_header", data=header)
        for k, v in attrs.items():
            g.attrs[k] = v
        g.attrs["image_source"] = how
    field = re.search(r"<systemFieldStrength_T>([\d.]+)<", header)
    model = re.search(r"<systemModel>([^<]+)<", header)
    fov = re.search(r"<reconSpace>.*?<fieldOfView_mm>\s*<x>([\d.]+)</x>\s*<y>([\d.]+)</y>\s*<z>([\d.]+)</z>", header, re.S)
    return buf.getvalue(), {"n_slices": img.shape[0], "h": img.shape[1], "w": img.shape[2], "source": how,
                            "acquisition": attrs.get("acquisition", ""), "field_T": field.group(1) if field else "",
                            "model": model.group(1) if model else "",
                            "fov_mm": "x".join(fov.groups()) if fov else ""}


def fetch(archive: str, limit_gb: float | None = None) -> None:
    import pandas as pd

    os.umask(0o077)
    url = links()[archive]
    want = wanted()
    batch = archive.removesuffix(".tar.xz").removeprefix("brain_multicoil_")
    POOL.mkdir(parents=True, exist_ok=True)
    out_zip = POOL / f"fastmri_plus_{batch}.zip"
    man_path, mem_path = WORK / f"manifest_{batch}.csv", WORK / f"members_{batch}.txt"
    done = set()
    if out_zip.exists():
        with zipfile.ZipFile(out_zip) as z:
            done = {Path(n).stem for n in z.namelist()}
    stream = ResumableStream(url, limit=int(limit_gb * 1e9) if limit_gb else None)
    xz = subprocess.Popen(["xz", "-dc", "-T4"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=CHUNK)
    err: list[str] = []

    def feed():
        try:
            for c in stream.chunks():
                xz.stdin.write(c)
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001
            err.append(_clean(e))
        finally:
            try:
                xz.stdin.close()
            except BrokenPipeError:
                pass

    threading.Thread(target=feed, daemon=True).start()
    rows, seen, t0 = [], 0, time.time()
    tmp_root = os.environ.get("TMPDIR", "/tmp")
    try:
        with tarfile.open(fileobj=xz.stdout, mode="r|") as tar, open(mem_path, "w") as mem, \
                zipfile.ZipFile(out_zip, "a", compression=zipfile.ZIP_STORED) as z:
            for m in tar:
                if not m.isfile():
                    continue
                seen += 1
                name = Path(m.name).stem
                mem.write(m.name + "\n")
                if name not in want or name in done:
                    continue
                with tempfile.TemporaryDirectory(dir=tmp_root) as d:
                    p = Path(d) / "scan.h5"
                    with tar.extractfile(m) as src, open(p, "wb") as dst:
                        while c := src.read(CHUNK):
                            dst.write(c)
                    try:
                        blob, info = reduce_h5(p)
                    except Exception as e:  # noqa: BLE001
                        print(f"[fastmri] {name}: {_clean(e)}", flush=True)
                        continue
                z.writestr(f"{name}.h5", blob)
                done.add(name)
                rows.append({"file": name, "archive": archive, "member": m.name, **info})
                print(f"[fastmri] kept {name} ({len(rows)} this batch; {stream.pos / 1e9:.1f} GB read, "
                      f"{stream.pos / 1e6 / max(time.time() - t0, 1):.0f} MB/s)", flush=True)
                if len(rows) % 10 == 0:
                    pd.DataFrame(rows).to_csv(man_path, index=False)
    except tarfile.ReadError as e:
        if not limit_gb:                      # a --limit-gb pilot ends mid-member by design
            raise IOError(f"tar stream: {_clean(e)}") from None
    finally:
        xz.kill()
    pd.DataFrame(rows).to_csv(man_path, index=False)
    if err:
        raise IOError(f"download: {err[0]}")
    print(f"[fastmri] {archive}: {seen} members, kept {len(rows)}, {stream.pos / 1e9:.1f} GB in "
          f"{(time.time() - t0) / 60:.0f} min; {len(done)} of {len(want)} wanted now in {out_zip.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", required=True)
    ap.add_argument("--limit-gb", type=float, default=None, help="pilot: stop after this many compressed GB")
    a = ap.parse_args()
    fetch(a.archive, a.limit_gb)


if __name__ == "__main__":
    main()
