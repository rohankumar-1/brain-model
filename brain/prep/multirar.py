"""Extract a multi-volume RAR5 archive (``x.part1.rar``, ``x.part2.rar`` ...) through libarchive.

``bsdtar`` opens one volume at a time, so members that span two volumes come out truncated, and the p7zip on the
cluster predates RAR5. libarchive's RAR5 reader follows volumes when all of them are opened together
(``archive_read_open_filenames``), which this module does through ctypes.

    python -m brain.prep.multirar --dest DIR x.part1.rar x.part2.rar x.part3.rar
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
from pathlib import Path

ARCHIVE_OK, ARCHIVE_EOF, ARCHIVE_WARN = 0, 1, -20
LIBARCHIVE = os.environ.get("LIBARCHIVE", "/home/software/anaconda3/2023.07/lib/libarchive.so")


def _lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(LIBARCHIVE if Path(LIBARCHIVE).exists() else ctypes.util.find_library("archive"))
    lib.archive_read_new.restype = ctypes.c_void_p
    lib.archive_read_support_format_all.argtypes = [ctypes.c_void_p]
    lib.archive_read_support_filter_all.argtypes = [ctypes.c_void_p]
    lib.archive_read_open_filenames.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]
    lib.archive_read_next_header.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.archive_read_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.archive_read_data.restype = ctypes.c_ssize_t
    lib.archive_read_free.argtypes = [ctypes.c_void_p]
    lib.archive_error_string.argtypes = [ctypes.c_void_p]
    lib.archive_error_string.restype = ctypes.c_char_p
    lib.archive_entry_pathname.argtypes = [ctypes.c_void_p]
    lib.archive_entry_pathname.restype = ctypes.c_char_p
    lib.archive_entry_filetype.argtypes = [ctypes.c_void_p]
    lib.archive_entry_filetype.restype = ctypes.c_uint
    lib.archive_entry_size.argtypes = [ctypes.c_void_p]
    lib.archive_entry_size.restype = ctypes.c_int64
    return lib


def extract(volumes: list[Path], dest: Path, select=None) -> list[Path]:
    """Extract every regular file (or those whose member name passes ``select``) under ``dest``; returns the paths.
    Each file's size is checked against its header, so a truncated member raises instead of passing silently."""
    lib = _lib()
    a = lib.archive_read_new()
    lib.archive_read_support_format_all(a)
    lib.archive_read_support_filter_all(a)
    names = (ctypes.c_char_p * (len(volumes) + 1))(*[str(v).encode() for v in volumes], None)
    if lib.archive_read_open_filenames(a, names, 1 << 20) != ARCHIVE_OK:
        raise RuntimeError(f"open {volumes[0]}: {lib.archive_error_string(a)}")
    out, entry, buf = [], ctypes.c_void_p(), ctypes.create_string_buffer(1 << 20)
    dest = Path(dest).resolve()
    try:
        while True:
            rc = lib.archive_read_next_header(a, ctypes.byref(entry))
            if rc == ARCHIVE_EOF:
                break
            if rc not in (ARCHIVE_OK, ARCHIVE_WARN):
                raise RuntimeError(f"header: {lib.archive_error_string(a)}")
            name = lib.archive_entry_pathname(entry).decode()
            if lib.archive_entry_filetype(entry) != 0o100000 or (select is not None and not select(name)):
                continue
            target = (dest / name).resolve()
            if dest not in target.parents:                       # no member may escape the destination
                raise RuntimeError(f"unsafe member path: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            want, got = lib.archive_entry_size(entry), 0
            with open(target, "wb") as fh:
                while True:
                    n = lib.archive_read_data(a, buf, len(buf))
                    if n < 0:
                        raise RuntimeError(f"{name}: {lib.archive_error_string(a)}")
                    if n == 0:
                        break
                    fh.write(buf.raw[:n])
                    got += n
            if got != want:
                raise RuntimeError(f"{name}: {got} of {want} bytes")
            out.append(target)
    finally:
        lib.archive_read_free(a)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volumes", nargs="+", type=Path)
    ap.add_argument("--dest", type=Path, required=True)
    a = ap.parse_args()
    files = extract(sorted(a.volumes), a.dest)
    print(f"[multirar] {len(files)} files -> {a.dest}")


if __name__ == "__main__":
    main()
