"""RSNA: new direct-DICOM path vs the original NIfTI round trip (orientation, z alignment, HU).

    python -m tests.rsna_regression 12
The old path centers on the volume; the new one on the head, so the in-plane translation between them is
estimated by phase correlation and only the correlation after alignment is judged. Flips must not help.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

from data import preprocess as old
from data.prep import pipeline, sources

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
WINDOW = lambda hu: (np.clip(hu, -200, 200) + 200) / 400  # the original CT window


def best_match(ref: np.ndarray, new: np.ndarray, flip: tuple[int, ...]):
    """Best (corr, dz, dy, dx) over z offsets -3..3 with in-plane translation from phase correlation."""
    mov = np.flip(new, flip) if flip else new
    best = (-1.0, 0, 0, 0)
    for dz in range(-3, 4):
        a, b = ref[64], mov[64 + dz]
        (dy, dx), *_ = phase_cross_correlation(a, b, upsample_factor=2)
        b2 = ndi.shift(b, (dy, dx), order=1, mode="nearest")
        c = float(np.corrcoef(a.ravel(), b2.ravel())[0, 1])
        if c > best[0]:
            best = (c, dz, float(dy), float(dx))
    return best


def main() -> None:
    cases = sources.rsna()
    cases = cases[:: max(1, len(cases) // N)][:N]
    print(f"{'case':28s} {'best flip':>10s} {'corr':>6s} {'dz':>3s} {'dy':>6s} {'dx':>6s} {'corr no-flip':>12s} {'mae(brain)':>10s} "
          f"{'tilt':>5s} {'gapdev':>6s}")
    bad = 0
    for case in cases:
        with tempfile.TemporaryDirectory() as d:
            nii = old.materialize_rsna_study(Path(case.zip_path), case.case_id, [m[0] for m in case.members], Path(d))
            ref = old.volume_to_shared_space(nii, "ct")[:, 0]
        res = pipeline.process(case)
        new = WINDOW(res.vol.astype(np.float32))
        scores = {f: best_match(ref, new, f) for f in ((), (0,), (1,), (2,))}
        flip, (c, dz, dy, dx) = max(scores.items(), key=lambda kv: kv[1][0])
        # Judged on untilted studies: the old path ignored gantry tilt, so tilted ones differ by design.
        # |dz| <= 3: the old z extent counted voxel edges (n*dz), the new one slice centres ((n-1)*dz).
        ok = res.info["tilt_deg"] > 1 or (flip == () and c > 0.95 and abs(dz) <= 3)
        bad += not ok
        brain = (ref[64] > 0.5) & (ref[64] < 0.7)
        shifted = ndi.shift(new[64 + dz], (dy, dx), order=1, mode="nearest")
        mae = float(np.abs(ref[64][brain] - shifted[brain]).mean() * 400)  # in HU
        print(f"{case.case_id[:28]:28s} {str(flip):>10s} {c:6.3f} {dz:3d} {dy:6.1f} {dx:6.1f} {scores[()][0]:12.3f} {mae:10.1f} "
              f"{res.info['tilt_deg']:5.1f} {res.info.get('slice_gap_dev_mm', float('nan')):6.2f}"
              f"{'' if ok else '   <-- CHECK'}")
    print("FAILED" if bad else "all studies: identity orientation, z aligned, high correlation", f"({bad} to check)")


if __name__ == "__main__":
    main()
