"""Format-7 image encoding (int16 + scale) in brain.prep.store, format-6 read support, and brain.prep.reencode."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from brain.prep import reencode, store


def _ct(rng, shape=(16, 24, 20)):
    v = rng.normal(30, 400, shape).astype(np.float32)
    v[0] = -1024
    v[1, :4] = rng.uniform(-1000, -500.01, (4,) + shape[2:])       # sub-air-threshold values to clamp
    v[2, :2] = rng.uniform(2048, 3071, (2,) + shape[2:])            # float16 rounds these to even HU
    v[3, 0, :3] = [-500.0, -499.6, -500.4]
    return np.clip(v, -1024, 3071).astype(np.float16)


def _mri(rng, shape=(16, 24, 20)):
    z = rng.normal(0, 2, shape).astype(np.float32)
    z = np.clip(z, -5, 20)
    z[0] = -6.0
    z[1, :2] = 19.99
    z[2, 0, :4] = [1 / 16, 3 / 16, -1 / 16, 0.0]                   # z * 1000 ties (round half to even)
    return z.astype(np.float16)


def _old_zip(path: Path, vol16: np.ndarray, kind: str, with_brain: bool = True) -> Path:
    """A format-6 image zip as the old writer made it (float16 image, format_version 6 on every array)."""
    arrays = {"image": (vol16[:, None], {"format_version": 6, "kind": kind, "modality": kind[:3], "raw_mad": 1.5,
                                         "grid_size": [20, 24, 16], "brain_ok": True})}
    if with_brain:
        brain = np.zeros(vol16.shape, np.uint8)
        brain[4:12, 6:18, 5:15] = 1
        arrays["brain"] = (brain, {"format_version": 6, "brain_method": "synthstrip", "brain_frac": 0.1})
    store.write_zip(path, arrays)
    return path


def test_encode_decode_ct_clamps_air_and_rounds_to_hu():
    v = _ct(np.random.default_rng(0)).astype(np.float32)
    q, attrs = store.encode_image(v, "ct_hu")
    assert q.dtype == np.int16 and attrs == {"dtype_rev": 7, "scale": 1.0, "air_below": -500.0}
    d = store.decode(q, {"kind": "ct_hu", **attrs})
    air = v < -500
    assert np.all(d[air] == -1024)
    assert np.abs(d[~air] - v[~air]).max() <= 0.5
    assert np.all(d[v >= 2048] == v[v >= 2048])                     # float16 HU above 2048 are even integers: exact
    assert d[3, 0, 0] == -500 and d[3, 0, 2] == -1024                # -500 kept, -500.4 is air


def test_encode_decode_mri_within_half_step_and_floor_exact():
    z = _mri(np.random.default_rng(1)).astype(np.float32)
    q, attrs = store.encode_image(z, "mri_z")
    assert attrs["scale"] == 0.001
    d = store.decode(q, {"kind": "mri_z", **attrs})
    assert d.dtype == np.float32
    assert np.abs(d.astype(np.float64) - z).max() <= 0.0005 + 1e-6
    assert np.all(d[0] == -6.0)                                     # background sentinel decodes exactly
    assert store.decode(q, {"kind": "mri_z", **attrs}, np.float16).dtype == np.float16


def test_encode_rejects_out_of_range_and_nan():
    with pytest.raises(ValueError):
        store.encode_image(np.array([40.0]), "mri_z")               # 40 z * 1000 overflows int16
    with pytest.raises(ValueError):
        store.encode_image(np.array([np.nan]), "ct_hu")


def test_masks_and_brain_are_not_decoded():
    m = np.arange(6, dtype=np.uint8)
    assert store.decode(m, {"labels": [0, 1]}) is m
    assert store.decode(m, {"brain_method": "synthstrip"}) is m


def test_format6_zip_reads_as_float_through_every_reader(tmp_path):
    v = _mri(np.random.default_rng(2))
    p = _old_zip(tmp_path / "old.zip", v, "mri_z")
    img = store.read_image(p)
    assert img.dtype == np.float32 and np.array_equal(img, v.astype(np.float32))
    got = store.read_many(p, ("image", "brain"), dtype=np.float16)
    assert got["image"].dtype == np.float16 and np.array_equal(store.squeeze_channel(got["image"]), v)
    assert got["brain"].dtype == np.uint8
    assert store.read(p, "image").shape == (16, 1, 24, 20)
    raw, attrs = store.read_all(p)
    assert raw["image"].dtype == np.float16 and attrs["image"]["format_version"] == 6


@pytest.mark.parametrize("kind", ["ct_hu", "mri_z"])
def test_reencode_zip_in_place(tmp_path, kind):
    rng = np.random.default_rng(3)
    v = _ct(rng) if kind == "ct_hu" else _mri(rng)
    p = _old_zip(tmp_path / "case.zip", v, kind)
    os.chmod(p, 0o600)
    with zipfile.ZipFile(p) as z:
        before = {n: z.read(n) for n in z.namelist() if not n.startswith("image/")}
    old_size = p.stat().st_size
    rec = reencode.reencode_zip(p, reencode.SpaceGuard(0, 0, quota_file=tmp_path / "none"))
    assert rec["status"] == "ok" and rec["old_bytes"] == old_size and rec["new_bytes"] == p.stat().st_size
    assert not reencode.tmp_for(p).exists()
    assert oct(p.stat().st_mode & 0o777) == "0o600"
    with zipfile.ZipFile(p) as z:
        after = {n: z.read(n) for n in z.namelist() if not n.startswith("image/")}
    assert json.loads(before.pop("zarr.json")) == json.loads(after.pop("zarr.json"))
    assert before == after                                           # brain array bytes and attrs untouched
    raw, attrs = store.read_all(p)
    assert raw["image"].dtype == np.int16 and attrs["image"]["format_version"] == 7
    assert attrs["image"]["dtype_rev"] == 7 and attrs["image"]["raw_mad"] == 1.5
    assert attrs["brain"]["format_version"] == 6                     # copied as is
    new = store.read_image(p)
    ref = v.astype(np.float32)
    if kind == "ct_hu":
        air = ref < -500
        assert np.all(new[air] == -1024) and np.abs(new[~air] - ref[~air]).max() <= 0.5
    else:
        assert np.abs(new.astype(np.float64) - ref).max() <= 0.0005 + 1e-6
    assert reencode.reencode_zip(p)["status"] == "skip"
    assert not reencode.needs_reencode(p)


def test_reencode_failure_leaves_original(tmp_path, monkeypatch):
    p = _old_zip(tmp_path / "case.zip", _ct(np.random.default_rng(4)), "ct_hu")
    data = p.read_bytes()

    def bad(*a, **k):
        raise AssertionError("forced")
    monkeypatch.setattr(reencode, "verify", bad)
    with pytest.raises(AssertionError):
        reencode.reencode_zip(p)
    assert p.read_bytes() == data and not reencode.tmp_for(p).exists()


def test_space_guard_stops_before_writing(tmp_path):
    p = _old_zip(tmp_path / "case.zip", _ct(np.random.default_rng(5)), "ct_hu")
    data = p.read_bytes()
    with pytest.raises(reencode.Stop):
        reencode.reencode_zip(p, reencode.SpaceGuard(min_free_gb=1e12, quota_margin_gb=0, quota_file=tmp_path / "x"))
    q = tmp_path / ".quota"
    q.write_text(" SCRATCH |      1020.0 |     1024.0 |  99.6 | 422.0K |  1.0M |  42.20\n")
    with pytest.raises(reencode.Stop):
        reencode.reencode_zip(p, reencode.SpaceGuard(min_free_gb=0, quota_margin_gb=10, quota_file=q))
    assert p.read_bytes() == data and not reencode.tmp_for(p).exists()
    assert reencode.quota_gb(q) == (1020.0, 1024.0)


def test_cli_shards_resume_and_dry_run(tmp_path):
    root = tmp_path / "store"
    rng = np.random.default_rng(6)
    rels = []
    for i in range(4):
        kind = "ct_hu" if i % 2 else "mri_z"
        rel = f"ds{i % 2}/c{i}.zip"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        _old_zip(root / rel, _ct(rng) if i % 2 else _mri(rng), kind, with_brain=i != 3)
        rels.append(rel)
    lst = tmp_path / "list.txt"
    lst.write_text("\n".join(rels + ["ds0/missing.zip"]) + "\n")
    out = tmp_path / "out"
    common = ["--root", str(root), "--paths", str(lst), "--out", str(out), "--min-free-gb", "0"]
    snap = {r: (root / r).read_bytes() for r in rels}
    assert reencode.main(common + ["--dry-run", "--out", str(tmp_path / "dry")]) == 0
    assert all((root / r).read_bytes() == snap[r] for r in rels)     # dry run keeps the originals
    assert reencode.main(common + ["--shard", "0/2"]) == 0
    assert reencode.main(common + ["--shard", "1/2"]) == 0
    recs = [json.loads(ln) for f in sorted(out.glob("manifest_*.jsonl")) for ln in open(f)]
    assert sorted(r["status"] for r in recs) == ["missing", "ok", "ok", "ok", "ok"]
    assert all(not reencode.needs_reencode(root / r) for r in rels)
    n_lines = len(open(out / "manifest_0of2.jsonl").readlines())
    assert reencode.main(common + ["--shard", "0/2"]) == 0          # resume: the manifest says all done
    assert len(open(out / "manifest_0of2.jsonl").readlines()) == n_lines


def test_write_case_writes_format7_and_resume_check(tmp_path):
    from brain.prep import pipeline
    from brain.prep.sources import Case

    vol = _mri(np.random.default_rng(7))
    res = pipeline.Result(vol, {}, None, {"qc_flag": False})
    store.write_case(tmp_path, Case("ixi_t1", "c1", "mri", None, ()), res)
    p = store.path_for(tmp_path, "ixi_t1", "c1")
    shape = (16, 1, 24, 20)
    assert store.is_current(p, "image", shape, np.int16)
    assert not store.is_current(p, "image", shape, np.float16)
    old = _old_zip(tmp_path / "old.zip", vol, "mri_z")
    assert not store.is_current(old, "image", shape, np.int16)       # a format-6 zip is not current for prep
    assert np.abs(store.read_image(p).astype(np.float64) - vol.astype(np.float32)).max() <= 0.0005 + 1e-6


def test_ct_without_air_clamp_keeps_air_range():
    v = _ct(np.random.default_rng(8)).astype(np.float32)
    q, attrs = store.encode_image(v, "ct_hu", air_below=None)
    assert "air_below" not in attrs
    assert np.abs(store.decode(q, {"kind": "ct_hu", **attrs}) - v).max() <= 0.5


@pytest.mark.parametrize("kind", ["ct_hu", "mri_z"])
def test_float16_decode_matches_float32_then_cast(kind):
    rng = np.random.default_rng(9)
    v = (_ct(rng) if kind == "ct_hu" else _mri(rng)).astype(np.float32)
    q, attrs = store.encode_image(v, kind)
    a = {"kind": kind, **attrs}
    f16 = store.decode(q, a, np.float16)
    assert f16.dtype == np.float16 and np.array_equal(f16, store.decode(q, a).astype(np.float16))


def test_reencode_without_air_clamp(tmp_path):
    v = _ct(np.random.default_rng(10))
    p = _old_zip(tmp_path / "case.zip", v, "ct_hu")
    assert reencode.reencode_zip(p, air_below=None)["n_air_clamped"] == 0
    assert np.abs(store.read_image(p) - v.astype(np.float32)).max() <= 0.5
