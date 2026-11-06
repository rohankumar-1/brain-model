"""CQ500 orientation check without masks: for studies whose bleed the radiologists read as one-sided (>= 2 of 3, not
both sides), the image side is where the blood-density voxels (55-90 HU, inside the brain mask, away from the midline)
outnumber the other side by 1.5x; agreement with the reads, for the store's current CQ500 images and the seg_cq500 copies."""
import sys
import os
import zipfile

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from brain.prep import store

S = "/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr"
v = pd.read_csv(f"{S}/volumes.csv", low_memory=False)
with zipfile.ZipFile("/home/roku/orcd/pool/datasets/cq500_meta/cq500_meta.raw.zip") as z:
    reads = pd.read_csv(z.open("reads.csv")).set_index("name")
nl = reads[[f"R{i}:BleedLocation-Left" for i in (1, 2, 3)]].sum(axis=1)
nr = reads[[f"R{i}:BleedLocation-Right" for i in (1, 2, 3)]].sum(axis=1)
side = pd.Series(np.where((nl >= 2) & (nr < 2), "L", np.where((nr >= 2) & (nl < 2), "R", "")), index=reads.index)
side = side[side != ""]
for ds in sys.argv[1:] or ["cq500", "seg_cq500"]:
    rows = []
    for r in v[(v.dataset == ds) & v.case_id.isin(side.index)].itertuples():
        if not os.path.exists(f"{S}/{r.image_path}"):        # flagged by prep (not stored)
            continue
        a = store.read_many(f"{S}/{r.image_path}", ("image", "brain"))
        img = store.squeeze_channel(a["image"]).astype(np.float32)
        brain = ndi.binary_erosion(store.squeeze_channel(a["brain"]) > 0, iterations=4)
        blood = ndi.binary_opening(brain & (img > 55) & (img < 90), iterations=1)
        x = img.shape[2]
        right, left = blood[:, :, x // 2 + 4:].sum(), blood[:, :, : x // 2 - 4].sum()   # +X = patient right
        if max(left, right) < 200 or max(left, right) < 1.5 * min(left, right):
            continue
        rows.append({"case": r.case_id, "image": "R" if right > left else "L", "read": side[r.case_id]})
    d = pd.DataFrame(rows)
    if len(d):
        print(f"{ds}: {len(d)} lateralised studies; image side == read side in {(d.image == d.read).mean():.0%}", flush=True)
