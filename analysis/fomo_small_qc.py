"""Visual QC of the small-FOMO prep: per family, mid-axial + mid-sagittal slices of normal (affine) and fallback
(rigid / translation) registrations, with the mean brain outline of 20 affine-registered IXI scans (same grid)."""
import glob
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "/home/roku/brain-model")
from brain.downstream.tasks import DEFAULT_VOLUMES  # noqa: E402
from brain.prep import store  # noqa: E402
from brain.prep.index import folder_of  # noqa: E402

ROOT = DEFAULT_VOLUMES.parent
v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
ref = v[(v.dataset_family == "ixi") & (v.reg_method == "affine")].sample(20, random_state=0)
ref_mask = np.mean([np.asarray(store.read(ROOT / p, "brain")) > 0 for p in ref.image_path], axis=0)

rows = [json.loads(line) for f in sorted(glob.glob(str(ROOT / "manifest_v6_fomo_small_*of4.jsonl"))) for line in open(f)]
m = pd.DataFrame(rows)
m = m[m.status == "ok"]
m["fam"] = m.dataset.str.rsplit("_", n=1).str[0]
pick = []
for fam, g in m.groupby("fam"):
    good = g[g.reg_method == "affine"].sample(min(2, (g.reg_method == "affine").sum()), random_state=0)
    bad = g[g.reg_method != "affine"].sample(min(2, (g.reg_method != "affine").sum()), random_state=0)
    pick.append(pd.concat([good, bad]))
pick = pd.concat(pick)
fams = sorted(pick.fam.unique())
fig, axes = plt.subplots(len(fams), 8, figsize=(16, 2.1 * len(fams)), facecolor="black")
for i, fam in enumerate(fams):
    g = pick[pick.fam == fam].reset_index(drop=True)
    for j in range(4):
        a_ax, s_ax = axes[i, 2 * j], axes[i, 2 * j + 1]
        a_ax.axis("off"); s_ax.axis("off")
        if j >= len(g):
            continue
        r = g.iloc[j]
        p = store.path_for(ROOT, folder_of(r.dataset), r.case_id)
        img = np.asarray(store.read_image(p), np.float32)
        lo, hi = np.percentile(img[img > -5.9], [1, 99.5]) if (img > -5.9).any() else (0, 1)
        z, x = img.shape[0] // 2, img.shape[2] // 2
        a_ax.imshow(img[z], cmap="gray", vmin=lo, vmax=hi, origin="lower")
        a_ax.contour(ref_mask[z], [0.5], colors="#00e5a0", linewidths=0.7)
        s_ax.imshow(img[:, :, x], cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect="auto")
        s_ax.contour(ref_mask[:, :, x], [0.5], colors="#00e5a0", linewidths=0.7)
        col = "white" if r.reg_method == "affine" else "#ff9f1c"
        a_ax.set_title(f"{fam[5:]} {r.reg_method} s={r.reg_scale:.2f}", color=col, fontsize=7)
fig.suptitle("Small-FOMO prep QC: green = mean IXI brain outline on the same grid; orange titles = fallback registration",
             color="white")
fig.tight_layout()
fig.savefig("/home/roku/brain-model/runs/diagnostics/fomo_small_qc.png", dpi=80, facecolor="black")
print("wrote fomo_small_qc.png")
