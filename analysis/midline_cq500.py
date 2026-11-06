"""Validate brain.multitask.measures midline shift on CQ500 (held out; evaluation only): SynthSeg each volume, measure
every variant (brain-pole line; skull-plane septum all rows / frontal horns / frontal max / third ventricle / max),
score against the 3-reader majority MidlineShift / MassEffect labels. SynthSeg labels are saved (uint8 npz) so later
variants can be re-scored without re-segmenting: ``--from-saved``."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/roku/brain-model")
from sklearn.metrics import roc_auc_score  # noqa: E402

from brain.downstream.tasks import CQ500_LABELS, DEFAULT_VOLUMES, load_task  # noqa: E402
from brain.multitask.measures import head_shell, midline_shift, midline_shift_plane  # noqa: E402
from brain.prep import store  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--from-saved", action="store_true")
a = ap.parse_args()
SEG = Path("/home/roku/orcd/scratch/brain-model/runs/diagnostics/midline_cq500_seg")
SEG.mkdir(parents=True, exist_ok=True)
v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
b = load_task("cq500", v)
seg = None
if not a.from_saved:
    from brain.prep.synthseg import Segmenter

    seg = Segmenter("cuda")
COLS = ["shift_mm", "frontal_mm", "frontal_max_mm", "v3_mm", "max_mm"]
rows = []
for i, vid in enumerate(b.volume_ids):
    img = store.read_image(DEFAULT_VOLUMES.parent / b.image_paths[vid]).astype(np.float32)
    f = SEG / (vid.replace(":", "_") + ".npz")
    if a.from_saved and f.exists():
        lab = np.load(f)["lab"]
    else:
        (lab, _), = seg.segment_store([img], ["ct"])
        np.savez_compressed(f, lab=lab.astype(np.uint8 if lab.max() < 256 else np.uint16))
    old = midline_shift(lab)
    new = midline_shift_plane(lab, head_shell(img, "ct"))
    rows.append({"volume_id": vid, "pole_mm": old["shift_mm"], **{k: abs(new[k]) for k in COLS},
                 "n_frontal": new["n_frontal"], **{k: int(b.y[i, j]) for j, k in enumerate(CQ500_LABELS)}})
    if (i + 1) % 50 == 0:
        print(f"[cq500] {i + 1}/{len(b.volume_ids)}", flush=True)
t = pd.DataFrame(rows)
t.to_csv("/home/roku/brain-model/runs/diagnostics/midline_cq500.csv", index=False)
for col in ["pole_mm"] + COLS:
    s = t[t[col].notna()]
    line = f"{col:15s} measured {len(s):3d}/{len(t)}"
    for lab in ("MidlineShift", "MassEffect", "ICH"):
        line += f" | {lab} AUROC {roc_auc_score(s[lab], s[col]):.3f}"
    pos, neg = s[s.MidlineShift == 1][col], s[s.MidlineShift == 0][col]
    line += f" | median pos {pos.median():.1f} neg {neg.median():.1f} mm"
    for thr in (3, 5):
        line += f" | @{thr}mm sens {(pos > thr).mean():.2f} spec {(neg <= thr).mean():.2f}"
    print(line, flush=True)
