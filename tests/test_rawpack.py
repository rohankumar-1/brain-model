"""rawpack: bit-exact pack -> verify -> delete, covered extracted copies, and staging back for prep."""

from __future__ import annotations

import gzip
import tarfile
import zipfile
from pathlib import Path

from brain.prep import rawpack


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_pack_verifies_then_removes_sources_and_stage_restores_bytes(tmp_path, monkeypatch):
    ds = tmp_path / "pool" / "toy"
    (ds / "sub-01").mkdir(parents=True)
    (ds / "sub-01" / "T1.nii.gz").write_bytes(gzip.compress(b"\x00\x01" * 500))
    (ds / "labels.csv").write_text("id,y\n1,0\n")
    (ds / "orig.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)           # a kept original archive
    want = _tree(ds)
    monkeypatch.setattr(rawpack, "pool_free_gb", lambda: 1e6)
    rawpack.pack(ds, apply=True)
    assert sorted(p.name for p in ds.iterdir()) == ["orig.zip", "toy.raw.zip"]
    with zipfile.ZipFile(ds / "toy.raw.zip") as zf:
        info = {i.filename: i.compress_type for i in zf.infolist()}
    assert info == {"labels.csv": zipfile.ZIP_DEFLATED, "sub-01/T1.nii.gz": zipfile.ZIP_STORED}
    out = tmp_path / "stage"
    rawpack.stage(ds, out)
    got = _tree(out / "toy")
    assert {k: v for k, v in got.items() if k != "toy.raw.zip"} == want   # bytes identical, kept zip linked


def test_extracted_copy_is_dropped_only_when_the_kept_archive_covers_it(tmp_path, monkeypatch):
    ds = tmp_path / "pool" / "msd_braintumor"
    tree = ds / "Task01_BrainTumour"
    (tree / "imagesTr").mkdir(parents=True)
    (tree / "imagesTr" / "a.nii.gz").write_bytes(b"x" * 10)
    (tree / "derived").mkdir()
    (tree / "derived" / "t1.nii.gz").write_bytes(b"d")                     # our own derived file: intermediate
    with tarfile.open(ds / "Task01_BrainTumour.tar", "w") as tf:
        tf.add(tree / "imagesTr" / "a.nii.gz", arcname="Task01_BrainTumour/imagesTr/a.nii.gz")
    monkeypatch.setattr(rawpack, "pool_free_gb", lambda: 1e6)
    (tree / "imagesTr" / "b.nii.gz").write_bytes(b"y")                     # not in the tar: not a pure copy
    rawpack.pack(ds, apply=True)                                          # -> archived, not dropped
    assert not (tree / "derived").exists() and not tree.exists()
    with zipfile.ZipFile(ds / "msd_braintumor.raw.zip") as zf:
        assert sorted(zf.namelist()) == ["Task01_BrainTumour/imagesTr/a.nii.gz", "Task01_BrainTumour/imagesTr/b.nii.gz"]
    (ds / "msd_braintumor.raw.zip").unlink()
    (tree / "imagesTr").mkdir(parents=True)
    (tree / "imagesTr" / "a.nii.gz").write_bytes(b"x" * 10)
    rawpack.pack(ds, apply=True)                                          # a pure copy of the tar -> dropped
    assert sorted(p.name for p in ds.iterdir()) == ["Task01_BrainTumour.tar"]
    rawpack.stage(ds, tmp_path / "stage")
    assert (tmp_path / "stage" / "msd_braintumor" / "Task01_BrainTumour" / "imagesTr" / "a.nii.gz").read_bytes() == b"x" * 10


def test_verify_catches_a_changed_file(tmp_path):
    ds = tmp_path / "toy"
    ds.mkdir()
    f = ds / "a.txt"
    f.write_text("hello")
    rawpack.write_zip(ds, [f], ds / "toy.raw.zip")
    f.write_text("hello, longer")
    ok, msg = rawpack.verify_zip(ds, [f], ds / "toy.raw.zip")
    assert not ok and "mismatch" in msg


def test_an_interrupted_pack_resumes_from_its_zip(tmp_path, monkeypatch):
    ds = tmp_path / "toy"
    ds.mkdir()
    files = []
    for i in range(4):
        f = ds / f"v{i}.nii.gz"
        f.write_bytes(gzip.compress(bytes([i]) * 100))
        files.append(f)
    rawpack.write_zip(ds, files, rawpack.raw_zip(ds))       # zip written, then killed while deleting sources
    files[0].unlink()
    files[1].unlink()
    monkeypatch.setattr(rawpack, "pool_free_gb", lambda: 1e6)
    rawpack.pack(ds, apply=True)
    assert sorted(p.name for p in ds.iterdir()) == ["toy.raw.zip"]
    with zipfile.ZipFile(ds / "toy.raw.zip") as zf:
        assert len(zf.namelist()) == 4


def test_nifti_goes_in_as_jpegls_and_stages_back_with_its_header(tmp_path, monkeypatch):
    import nibabel as nib
    import numpy as np

    ds = tmp_path / "pool" / "toy"
    (ds / "sub-01").mkdir(parents=True)
    rng = np.random.default_rng(0)
    aff = np.array([[-0.9, 0.05, 0, 90], [0.04, 0.9, 0, -120], [0, 0, 1.2, -70], [0, 0, 0, 1]])
    t1 = nib.Nifti1Image(rng.integers(-1024, 3000, (40, 48, 20)).astype(np.int16), aff)
    t1.header.set_slope_inter(0.5, 10.0)
    t1.header.set_qform(aff, code=1)
    t1.header["descrip"] = b"scanner T1"
    flair = nib.Nifti1Image(rng.normal(300, 80, (40, 48, 20)).astype(np.float32), aff)
    seg = nib.Nifti1Image((rng.random((40, 48, 20)) > 0.9).astype(np.uint8), aff)
    big = nib.Nifti1Image(rng.integers(0, 200_000, (8, 8, 4)).astype(np.int32), aff)   # beyond 16 bits
    for name, img in (("T1.nii.gz", t1), ("FLAIR.nii.gz", flair), ("seg.nii.gz", seg), ("big.nii", big)):
        nib.save(img, str(ds / "sub-01" / name))
    src = {n: nib.load(str(ds / "sub-01" / n)) for n in ("T1.nii.gz", "FLAIR.nii.gz", "seg.nii.gz", "big.nii")}
    want = {n: (np.asanyarray(i.dataobj).astype(np.float64), i.affine, i.header.get_qform(coded=True)[1],
                bytes(i.header["descrip"])) for n, i in src.items()}
    monkeypatch.setattr(rawpack, "pool_free_gb", lambda: 1e6)
    rawpack.pack(ds, apply=True, workers=2)
    with zipfile.ZipFile(ds / "toy.raw.zip") as zf:
        names = set(zf.namelist())
    assert {"sub-01/T1.nii.gz.jls", "sub-01/FLAIR.nii.gz.jls", "sub-01/seg.nii.gz.jls", "sub-01/big.nii"} <= names
    rawpack.stage(ds, tmp_path / "stage")
    for n, (data, affine, qcode, descrip) in want.items():
        img = nib.load(str(tmp_path / "stage" / "toy" / "sub-01" / n))
        got = np.asanyarray(img.dataobj).astype(np.float64)
        tol = 0.6 * (data.max() - data.min()) / 65535 if n == "FLAIR.nii.gz" else 0.0
        assert np.abs(got - data).max() <= tol, n
        assert np.allclose(img.affine, affine, atol=1e-5) and img.header.get_qform(coded=True)[1] == qcode, n
        assert bytes(img.header["descrip"]) == descrip, n
    assert src["T1.nii.gz"].dataobj.slope == nib.load(str(tmp_path / "stage" / "toy" / "sub-01" / "T1.nii.gz")).dataobj.slope
