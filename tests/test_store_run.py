"""Format-6 store layout and the GPU-batched prep runner (with a stand-in SynthStrip session)."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from brain.common.grid import DEPTH
from brain.prep import pipeline, run, store
from brain.prep.sources import Case


def _res(with_brain=True):
    vol = np.zeros((8, 16, 16), np.float16)
    brain = np.zeros((8, 16, 16), np.uint8)
    brain[2:6, 4:12, 4:12] = 1
    return pipeline.Result(vol, {"seg": brain.copy()}, brain if with_brain else None,
                           {"brain_method": "synthstrip", "brain_frac": 0.1, "brain_ok": True, "qc_flag": False})


def test_case_zip_holds_image_and_brain_and_labels_stay_separate(tmp_path):
    case = Case("cq500", "c1", "ct", None, (("seg", Path("unused.nii.gz")),))
    legacy = store.brain_path(tmp_path, case)
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"old")                       # a format-5 brain zip is cleaned up
    store.write_case(tmp_path, case, _res())
    image = store.path_for(tmp_path, "cq500", "c1")
    assert sorted(p.name for p in image.parent.iterdir()) == ["c1.seg.zip", "c1.zip"]
    got = store.read_many(image, ("image", "brain", "nope"))
    assert set(got) == {"image", "brain"} and got["image"].shape == (8, 1, 16, 16)
    assert int(got["brain"].sum()) == 4 * 8 * 8
    assert store.read_attrs(image, "brain")["brain_method"] == "synthstrip"
    assert store.read_attrs(image, "image")["format_version"] == store.FORMAT_VERSION
    arrays, attrs = store.read_all(image)
    assert set(arrays) == {"image", "brain"} and attrs["image"]["kind"] == "ct_hu"


def test_scratch_quota_parse(tmp_path):
    q = tmp_path / ".quota"
    q.write_text(" Space | Usage (GB) | Limit (GB) | % Used\n HOME | 51.0 | 200.0 | 25.51 | 1K | 1M | 1\n"
                 " SCRATCH |      845.0 |     1024.0 |  82.52 | 422.0K |  1.0M |  42.20\n")
    assert run.scratch_used_pct(q) == 82.52
    assert run.scratch_used_pct(tmp_path / "missing") is None


class _FakeSession:
    """Stands in for the GPU SynthStrip session: a soft-tissue threshold, and it records batch sizes."""

    def __init__(self):
        self.batches = []

    def run_batch(self, volumes, modalities):
        self.batches.append(len(volumes))
        return [(v > -5) & (v < 100) for v in volumes]


def test_gpu_mask_runner_batches_synthstrip_and_writes_one_zip_per_case(tmp_path, monkeypatch):
    from tests.test_prep import ct_phantom
    from brain.prep import synthstrip_runtime

    src = tmp_path / "src"
    src.mkdir()
    cases = []
    for i in range(3):
        p = src / f"ct{i}.nii.gz"
        sitk.WriteImage(ct_phantom((10.0 * i, 0.0), spacing=(1.0, 1.0, 4.0), n=(220, 220, 40), artifacts=False), str(p))
        cases.append(Case("cq500", f"ct{i}", "ct", p))
    fake = _FakeSession()
    monkeypatch.setattr(synthstrip_runtime, "session", lambda gpu=True: fake)
    root = tmp_path / "store"
    opts = pipeline.Options(brain_mask="auto")
    with mp.get_context("spawn").Pool(2) as pool, open(tmp_path / "m.jsonl", "w") as out:
        complete = run.run_gpu(pool, cases, root, "auto", opts, out, batch=2, guard=lambda: False)
    rows = [json.loads(line) for line in open(tmp_path / "m.jsonl")]
    assert complete and len(rows) == 3 and {r["status"] for r in rows} == {"ok"}, rows
    assert {r["brain_method"] for r in rows} == {"synthstrip"} and sum(fake.batches) == 3
    got = store.read_many(store.path_for(root, "cq500", "ct0"), ("image", "brain"))
    assert got["image"].shape == (DEPTH, 1, 256, 256) and got["brain"].shape == (DEPTH, 256, 256)
    assert got["brain"].any()


def test_store_cleanup_keeps_current_format_and_index_outputs(tmp_path):
    import zarr
    from zarr.storage import ZipStore
    from brain.prep import store_cleanup

    case = Case("cq500", "c1", "ct", None, (("seg", Path("unused.nii.gz")),))
    store.write_case(tmp_path, case, _res())                       # current: image zip + seg zip
    old = tmp_path / "cq500" / "c0.zip"                           # a format-5 leftover
    zs = ZipStore(old, mode="w")
    zarr.open_group(zs, mode="w").create_array("image", data=np.zeros((2, 1, 2, 2), np.float16),
                                               attributes={"format_version": 5})
    zs.close()
    (tmp_path / "cq500" / "c0.brain.zip").write_bytes(b"x")
    (tmp_path / "cq500" / "c2.zip.partial.7").write_bytes(b"x")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "masks.csv").write_text("a\n")
    (tmp_path / "sinoct").mkdir()
    for name in ("manifest_v5_0of2.jsonl", "manifest_v6_A_0of8.jsonl", "remask_0of4.jsonl", "volumes.csv"):
        (tmp_path / name).write_text("{}\n")
    doomed = {p.relative_to(tmp_path).as_posix() for p in store_cleanup.plan(tmp_path)}
    assert doomed == {"cq500/c0.zip", "cq500/c0.brain.zip", "cq500/c2.zip.partial.7", "sinoct",
                      "manifest_v5_0of2.jsonl", "remask_0of4.jsonl"}
