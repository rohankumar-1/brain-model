"""Midline shift: (1) the healthy range -- every variant on brains with no reason to be shifted (RSNA CT without any
hemorrhage, healthy community MRI from CoRR / NKI, CQ500 scans with no finding), with each variant's healthy 95th
percentile scored as a CQ500 threshold; (2) a figure of CQ500 scans with the skull plane, septum rows (frontal horns
highlighted), the third-ventricle centre and the measured values. Needs midline_cq500.py's saved segmentations."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "/home/roku/brain-model")
from brain.downstream.tasks import DEFAULT_VOLUMES, build_rsna_volume_labels, load_task  # noqa: E402
from brain.multitask.labels import synthseg_volumes  # noqa: E402
from brain.multitask.measures import L_LV, R_LV, V3, _mask, head_shell, midline_shift_plane  # noqa: E402
from brain.prep import store  # noqa: E402
from brain.prep.synthseg import LABELS  # noqa: E402

D = Path("/home/roku/brain-model/runs/diagnostics")
SEG = Path("/home/roku/orcd/scratch/brain-model/runs/diagnostics/midline_cq500_seg")
ROOT = DEFAULT_VOLUMES.parent
COLS = ["shift_mm", "frontal_mm", "frontal_max_mm", "v3_mm", "max_mm"]


def measure(args):
    vid, image_path, seg_path, modality = args
    try:
        img = store.read_image(ROOT / image_path).astype(np.float32)
        seg = store.read_image(ROOT / seg_path, "mask")
        out = midline_shift_plane(seg, head_shell(img, modality), LABELS)
        return {"volume_id": vid, "modality": modality, **{k: abs(out[k]) for k in COLS}}
    except Exception as e:  # noqa: BLE001
        return {"volume_id": vid, "modality": modality, "error": str(e)[:100]}


# ---------------------------------------------------------------- (1) healthy range
v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
sel = pd.read_csv(ROOT / "labels" / "synthseg" / "select.csv")
qc = synthseg_volumes()
ok = set(qc.index[qc["qc_ok"]])
sel = sel[sel.volume_id.isin(ok)]
rsna = build_rsna_volume_labels()
neg_ct = sel[sel.volume_id.isin(set(rsna.loc[rsna["any"] == 0, "volume_id"]))].sample(300, random_state=0)
vi = v.set_index("volume_id")
mri = sel[sel.dataset_family.isin(["fomo_corr", "fomo_nki"]) & sel.volume_id.map(vi["contrast"]).eq("t1w")].sample(
    300, random_state=0)
jobs = [(r.volume_id, r.image_path, store.path_for("", r.folder, str(r.case_id), "synthseg"), m)
        for df, m in ((neg_ct, "ct"), (mri, "mri")) for r in df.itertuples()]
with ProcessPoolExecutor(16) as ex:
    healthy = pd.DataFrame(list(ex.map(measure, jobs, chunksize=4)))
cq = pd.read_csv(D / "midline_cq500.csv")
clean = cq[(cq[["ICH", "MassEffect", "MidlineShift", "CalvarialFracture"]].sum(axis=1) == 0)].assign(modality="cq500_clean")
healthy = pd.concat([healthy, clean[["volume_id", "modality"] + COLS]], ignore_index=True)
healthy.to_csv(D / "midline_healthy.csv", index=False)
print("[healthy] errors:", int(healthy.get("error", pd.Series(dtype=str)).notna().sum()))
print("[healthy] |shift| in mm by cohort: median / 95th percentile / max (n measured)")
for mod, g in healthy.groupby("modality"):
    print(f"  {mod:12s} " + " | ".join(f"{c} {g[c].median():.2f}/{g[c].quantile(.95):.2f}/{g[c].max():.1f} (n={g[c].notna().sum()})"
                                        for c in COLS), flush=True)
print("[threshold] CQ500 sensitivity / specificity at the healthy-CT 95th percentile of each variant:")
hct = healthy[healthy.modality == "ct"]
for c in COLS:
    thr = hct[c].quantile(.95)
    s = cq[cq[c].notna()]
    pos, neg = s[s.MidlineShift == 1][c], s[s.MidlineShift == 0][c]
    print(f"  {c:15s} thr {thr:.2f} mm: sens {(pos > thr).mean():.2f} spec {(neg <= thr).mean():.2f}", flush=True)

# ---------------------------------------------------------------- (2) figure
b = load_task("cq500", v)
best = "max_mm"
cq["err"] = cq[best]
pos = cq[cq.MidlineShift == 1].dropna(subset=[best]).sort_values(best)
neg = cq[cq.MidlineShift == 0].dropna(subset=[best]).sort_values(best)
pick = pd.concat([pos.iloc[np.linspace(len(pos) // 3, len(pos) - 1, 4).round().astype(int)].assign(why="radiologist shift"),
                  pos.head(2).assign(why="missed shift (lowest measured)"),
                  neg.sample(2, random_state=0).assign(why="no shift (typical)"),
                  neg.tail(2).assign(why="false alarm (highest measured)")])
fig, axes = plt.subplots(2, 5, figsize=(25, 11), facecolor="black")
for ax, r in zip(axes.ravel(), pick.itertuples()):
    img = store.read_image(ROOT / b.image_paths[r.volume_id]).astype(np.float32)
    lab = np.load(SEG / (r.volume_id.replace(":", "_") + ".npz"))["lab"]
    out = midline_shift_plane(lab, head_shell(img, "ct"))
    a_, b_, c_ = out["plane"]
    lv_l, lv_r, v3 = _mask(lab, L_LV, None), _mask(lab, R_LV, None), _mask(lab, V3, None)
    v3_front = np.nonzero(v3.any(axis=(0, 2)))[0].max()
    zs = [z for z in np.nonzero(v3.any(axis=(1, 2)))[0] if lv_l[z].any() and lv_r[z].any()]
    z = int(np.median(zs)) if zs else img.shape[0] // 2
    ax.imshow(np.clip(img[z] / 80, 0, 1), cmap="gray", origin="lower")
    ax.contour(lv_l[z] | lv_r[z], [0.5], colors="#4da6ff", linewidths=0.8)
    ax.contour(v3[z], [0.5], colors="#c77dff", linewidths=0.8)
    ys = np.arange(img.shape[1])
    ax.plot(a_ + b_ * ys + c_ * z, ys, color="#00e5a0", lw=1.3, label="skull symmetry plane")
    sy = np.nonzero(lv_l[z].any(axis=1) & lv_r[z].any(axis=1))[0]
    pts = [(0.5 * (np.nonzero(lv_l[z, y])[0].max() + np.nonzero(lv_r[z, y])[0].min()), y) for y in sy
           if np.nonzero(lv_l[z, y])[0].max() < np.nonzero(lv_r[z, y])[0].min()]
    if pts:
        p = np.array(pts)
        fr = p[:, 1] > v3_front
        ax.scatter(p[~fr, 0], p[~fr, 1], s=4, color="#888888", label="septum (posterior rows)")
        ax.scatter(p[fr, 0], p[fr, 1], s=6, color="#ff3b3b", label="septum (frontal horns)")
    if v3[z].any():
        vy, vx = np.nonzero(v3[z])
        ax.scatter([vx.mean()], [vy.mean()], marker="x", s=60, color="#c77dff", label="3rd ventricle centre")
    ax.set_xlim(30, 226); ax.set_ylim(20, 236); ax.axis("off")
    ax.set_title(f"{r.why}\n{r.volume_id.split(':')[-1]}: frontal max {r.frontal_max_mm:.1f} | V3 {r.v3_mm:.1f} | "
                 f"all-rows {r.shift_mm:.1f} mm", color="white", fontsize=10)
axes[0, 0].legend(loc="lower left", fontsize=7, facecolor="black", labelcolor="white")
fig.suptitle("Midline shift on CQ500 (axial at the foramen of Monro; anterior up, patient right on the right)",
             color="white", fontsize=14)
fig.tight_layout()
fig.savefig(D / "midline_viz2.png", dpi=80, facecolor="black")
print("wrote", D / "midline_viz2.png")
