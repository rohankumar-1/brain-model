"""Slide (16:9 PNG) of every task the full-suite run trains: one row per task group, one chip per task, coloured by
output type. Task list from the registry and configs/multitask/full_iso_bank.yaml.
    python analysis/presentation/tasks_slide.py OUT.png"""
import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brain.multitask.registry import load_registry  # noqa: E402

TYPE = {"binary": ("#2a78d6", "Binary"), "regression": ("#eb6834", "Regression"),
        "multiclass": ("#6250d6", "Multiclass"), "segmentation": ("#1baf7a", "Segmentation"),
        "dense": ("#8a8983", "Dense / self-supervised")}
GROUP = [("demographics", "Demographics"), ("diagnosis", "Diagnosis"), ("clinical_scores", "Clinical scores"),
         ("hemorrhage", "Hemorrhage"), ("lesion", "Lesion masks"), ("tumor", "Tumour"),
         ("radiology", "Radiology findings"), ("anatomy_seg", "Anatomy masks"), ("anatomy_volume", "Anatomy volumes"),
         ("derived", "Brain & asymmetry"), ("ct_measures", "CT measures"), ("synthesis", "Sequence synthesis"),
         ("ssl_recon", "Self-supervised")]
NAME = {
    "age": "age", "sex": "sex", "neuro_diagnosis": "neuro diagnosis", "infant_abnormal": "infant abnormal",
    "tumor_type": "tumour type", "mmse_grade": "MMSE grade", "autism_grade": "autism grade",
    "hemorrhage_any": "any bleed", "hemorrhage_epidural": "epidural", "hemorrhage_intraparenchymal": "intraparenchymal",
    "hemorrhage_intraventricular": "intraventricular", "hemorrhage_subarachnoid": "subarachnoid",
    "hemorrhage_subdural": "subdural", "hemorrhage_segmentation": "bleed mask",
    "whole_tumor": "whole tumour", "tumor_core": "tumour core", "enhancing_tumor": "enhancing tumour",
    "resection_cavity": "resection cavity", "tumor_volume": "tumour volume", "glioma_survival_12m": "survival 12 mo",
    "glioma_mgmt": "MGMT", "mri_normal_for_age": "normal for age", "mri_small_vessel_disease": "small vessel disease",
    "mri_wm_lesion": "WM lesion", "mri_postop": "post-op", "mri_mass": "mass", "mri_edema": "edema",
    "mri_enlarged_ventricles": "enlarged ventricles", "mri_infarct": "infarct", "mri_artifact": "artifact",
    "evans_index": "Evans index", "vol_wmh": "WMH volume", "ventriculomegaly": "ventriculomegaly",
    "vol_deep_gm": "deep GM", "vol_parenchymal_fraction": "parenchymal fraction",
    "vol_log_ventricle_brain_ratio": "ventricle / brain", "vol_gm_wm_ratio": "GM / WM",
    "vol_hippocampus_asym": "hippocampus asym.", "vol_ventricle_asym": "ventricle asym.",
    "vol_cortex_asym": "cortex asym.", "brain_volume": "brain volume", "brain_mask": "brain mask",
    "brain_asymmetry": "brain asymmetry", "intensity_asymmetry": "intensity asymmetry",
    "ct_csf_fraction": "CSF fraction", "ct_brain_density": "brain density", "ct_calcification": "calcification",
    "ct_skull_density": "skull density", "seg_ct_skull": "skull mask", "seg_ct_csf": "CSF mask",
    "synth_t1": "→ T1", "synth_t1c": "→ T1c", "synth_t2": "→ T2", "synth_flair": "→ FLAIR", "synth_ct": "→ CT",
    "masked_recon": "masked reconstruction"}

cfg = yaml.safe_load(open(ROOT / "configs/multitask/full_iso_bank.yaml"))
reg = load_registry(cfg["data"].get("registry"))
ids = set(reg.train_ids(cfg["tasks"]))
by = {}
for t in reg.tasks:
    if t.id in ids:
        by.setdefault(t.group, []).append((NAME.get(t.id, t.id.removeprefix("seg_").replace("_", " ")), t.type))
if t_missing := [t for t in ids if t not in NAME and not t.startswith("seg_")]:
    print("unnamed tasks:", t_missing)

fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
W, H = 16.0, 9.0
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
LABEL_W, X0, ROW, CH, FS = 2.75, 3.0, 0.5, 0.36, 9.0
renderer = fig.canvas.get_renderer()
inv = ax.transData.inverted()


def text_w(s):
    t = ax.text(0, 0, s, fontsize=FS)
    bb = t.get_window_extent(renderer=renderer)
    t.remove()
    return inv.transform((bb.width, 0))[0] - inv.transform((0, 0))[0]


y = H - 0.45
for key, label in GROUP:
    tasks = by.get(key, [])
    if not tasks:
        continue
    ax.text(0.35, y, label, ha="left", va="center", fontsize=10.5, fontweight="bold", color="#1d1d1b")
    ax.text(LABEL_W, y, str(len(tasks)), ha="right", va="center", fontsize=10, color="#55544f")
    x = X0
    for name, typ in tasks:
        w = text_w(name) + 0.32
        if x + w > W - 0.3:                          # wrap a long group onto a second line
            x, y = X0, y - ROW * 0.82
        ax.add_patch(FancyBboxPatch((x, y - CH / 2), w, CH, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=TYPE[typ][0], ec="none"))
        ax.text(x + w / 2, y, name, ha="center", va="center", fontsize=FS, color="white", fontweight="bold")
        x += w + 0.12
    y -= ROW

ax.text(0.35, 0.45, f"{len(ids)} tasks in 13 groups", ha="left", va="center", fontsize=11, fontweight="bold",
        color="#1d1d1b")
lx = 4.2
for typ, (c, name) in TYPE.items():
    n = sum(1 for g in by.values() for _, t in g if t == typ)
    ax.add_patch(FancyBboxPatch((lx, 0.3), 0.3, 0.3, boxstyle="round,pad=0,rounding_size=0.06", fc=c, ec="none"))
    s = f"{name} ({n})"
    ax.text(lx + 0.42, 0.45, s, ha="left", va="center", fontsize=10, color="#1d1d1b")
    lx += 0.42 + text_w(s) * 1.1 + 0.45
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, facecolor="white")
print(out)
