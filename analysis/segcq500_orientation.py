"""Seg-CQ500 left-right / placement check (2026-11-12).

Finding: the stored Seg-CQ500 copies of the 32 gantry-tilt studies (sheared sform, no qform; ITK rejects them and
``geometry.read`` takes its nibabel fallback) were prepped on 2026-10-18 (manifest_v6_B), a week before the fallback
was fixed (commit 5e53b1a, 2026-10-25: it used to leave ITK's identity direction under RAS-ordered data, rotating the
image 180 degrees about the superior axis). They were never re-prepped. As stored, their hyperdense blood overlaps the
DICOM CQ500 copy of the same study 0.02 (0.05 mirrored) vs 0.89 for the untilted studies; re-prepared with the
current code, 0.99 (5 tilted studies). Fix: re-prep seg_cq500 (jobs/prep_seg_cq500.sbatch).

``compare``: per Seg-CQ500 study on the store grid (or ``--root`` for a candidate store), the overlap of hyperdense
blood (55-90 HU in the eroded brain) with the DICOM copy, as stored and mirrored, by tilt. ``reprep``: prepares
tilted studies with the current ``pipeline.prepare`` (read -> register -> resample) and reports the same overlap.
Aggregates only, no case ids.

    python analysis/segcq500_orientation.py compare [--root CANDIDATE_STORE]
    python analysis/segcq500_orientation.py reprep --n 5
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.common.paths import DATASETS  # noqa: E402
from brain.downstream.tasks import DEFAULT_VOLUMES  # noqa: E402
from brain.prep import store  # noqa: E402

ROOT = DEFAULT_VOLUMES.parent
PACK = DATASETS / "seg_cq500" / "seg_cq500.raw.zip"


def tilted_cases() -> dict[str, str]:
    """case -> the CT member's path in the raw pack, for studies whose sform is sheared."""
    import nibabel as nib

    from brain.prep.rawpack import META

    def sheared(a):
        a = np.asarray(a, float)[:3, :3]
        u = a / np.linalg.norm(a, axis=0)
        return bool(np.abs(u.T @ u - np.eye(3)).max() > 1e-3)

    out = {}
    with zipfile.ZipFile(PACK) as z:
        for n in z.namelist():
            if n.endswith(META) and "/CT.nii" in n:
                h = nib.Nifti1Header(base64.b64decode(json.loads(z.read(n))["nifti_header"]))
                if sheared(h.get_sform()):
                    out[n.split("/")[-2]] = n[: -len(META)]
    return out


def blood(hu: np.ndarray, brain: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    return binary_erosion(brain, iterations=4) & (hu >= 55) & (hu <= 90)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.ndimage import binary_dilation

    return float((a & binary_dilation(b, iterations=2)).sum() / max(a.sum(), 1))


def stored_blood(path: str, root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    a = store.read_many((root or ROOT) / path, ("image", "brain"))
    brain = store.squeeze_channel(a["brain"]) > 0
    return blood(store.squeeze_channel(a["image"]).astype(np.float32), brain), brain


def compare(root: Path | None = None) -> None:
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
    tilt = tilted_cases()
    seg, dic = v[v["dataset"] == "seg_cq500"], v[v["dataset"] == "cq500"]
    dic = dic.set_index(dic["case_id"].astype(str))
    rows = []
    for r in seg.itertuples():
        cid = str(r.case_id)
        if cid not in dic.index:
            continue
        a, _ = stored_blood(r.image_path, root)
        b, _ = stored_blood(dic.loc[cid, "image_path"])
        if a.sum() < 300 or b.sum() < 300:
            continue
        rows.append({"tilted": cid in tilt, "asis": overlap(a, b), "mirror": overlap(a[:, :, ::-1], b)})
    f = pd.DataFrame(rows)
    print(f.groupby("tilted")[["asis", "mirror"]].median().round(2).to_string())


def reprep(n: int) -> None:
    from brain.prep import pipeline
    from brain.prep.rawpack import JLS, META, restore_nifti
    from brain.prep.sources import Case

    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
    dic = v[v["dataset"] == "cq500"]
    dic = dic.set_index(dic["case_id"].astype(str))
    tilt = {c: m for c, m in tilted_cases().items() if c in dic.index}
    import os

    from brain.prep import registration

    opts = pipeline.Options("affine", os.environ["BRAIN_TEMPLATE"], os.environ["BRAIN_TEMPLATE_BRAIN"], "none", "sitk")
    registration.load_template(opts.template, opts.template_brain)
    rows = []
    with tempfile.TemporaryDirectory(dir=Path.home() / "mask_labels_work") as tmp, zipfile.ZipFile(PACK) as z:
        for cid, member in tilt.items():
            b, brain_b = stored_blood(dic.loc[cid, "image_path"])
            if b.sum() < 300:
                continue
            dest = Path(tmp) / cid / "CT.nii.gz"
            dest.parent.mkdir(parents=True)
            restore_nifti(z.read(member + JLS), json.loads(z.read(member + META)), dest)
            p = pipeline.prepare(Case("seg_cq500", cid, "ct", dest), opts=opts)
            a = blood(p.raw.astype(np.float32), brain_b)
            res = {"current": overlap(a, b), "current_mirror": overlap(a[:, :, ::-1], b)}
            rows.append(res)
            print(f"[reprep] case {len(rows)}: " + " ".join(f"{k} {x:.2f}" for k, x in res.items()), flush=True)
            if len(rows) >= n:
                break
    print(pd.DataFrame(rows).median().round(2).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare", "reprep"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--root", type=Path, default=None, help="compare: store holding the seg_cq500 copies")
    a = ap.parse_args()
    compare(a.root) if a.cmd == "compare" else reprep(a.n)


if __name__ == "__main__":
    main()
