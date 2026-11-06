"""Labels v12 sanity check (2026-11-06): row counts of the changed tasks, and real samples through MultitaskDataset:
CT skull / CSF Hounsfield masks (sensible fractions), lesion-ignore on BraTS anatomy targets, ET / TC on T1c only."""
import sys
import numpy as np
import pandas as pd
import yaml

from brain.multitask.data import MultitaskDataset, drop_datasets, drop_volumes, load_labels
from brain.multitask.registry import load_registry

L = "/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr/labels/multitask_v12"
STORE = "/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr"
cfg = yaml.safe_load(open("configs/joint/r15.yaml"))
reg = load_registry(None)
lab, stats = load_labels(L)
vol = pd.read_csv(f"{STORE}/volumes.csv", low_memory=False)
lab = drop_volumes(drop_datasets(lab, vol, cfg["data"]["exclude_datasets"]), cfg["data"]["exclude_volumes"])
cs = vol.drop_duplicates("volume_id").set_index("volume_id")["contrast_std"]
for t in ["tumor_core", "enhancing_tumor", "whole_tumor", "seg_ct_skull", "seg_ct_csf", "ct_skull_density"]:
    g = lab[(lab.task == t) & (lab.use == "train")]
    print(f"{t:18s} rows {len(g):6d} by split {g.split.value_counts().to_dict()} empty {int(g.ref.eq('@empty').sum())} "
          f"contrasts {cs.reindex(g.volume_id).value_counts().head(4).to_dict()}")
bad = lab[lab.task.isin(["tumor_core", "enhancing_tumor"]) & ~lab.ref.eq("@empty")]
assert cs.reindex(bad.volume_id).eq("t1c").all(), "ET / TC rows on a non-T1c scan"

ids = [reg.index[t] for t in ["seg_ct_skull", "seg_ct_csf", "seg_white_matter", "seg_cortex", "whole_tumor"]]
ds = MultitaskDataset(STORE, lab, vol, reg, stats, split="train", train=False, tasks=[reg.tasks[i].id for i in ids],
                      k_max=8, max_seg=8, lesion_ignore=True)
rng = np.random.default_rng(0)
seen = {"ct": 0, "lesion": 0}
for i in rng.permutation(len(ds))[:400]:
    s = ds[int(i)]
    names = [reg.tasks[int(t)].id for t in s["task"]]
    if "seg_ct_skull" in names and seen["ct"] < 3:
        for n in ("seg_ct_skull", "seg_ct_csf"):
            if n in names:
                m = s["seg"][int(s["seg_slot"][names.index(n)])].numpy()
                print(f"[ct] {n}: {100 * (m == 1).mean():.2f}% of voxels")
        seen["ct"] += 1
    if "whole_tumor" in names and any(n.startswith("seg_") and n != "seg_ct_skull" for n in names) and seen["lesion"] < 3:
        for n in names:
            if n in ("seg_white_matter", "seg_cortex"):
                m = s["seg"][int(s["seg_slot"][names.index(n)])].numpy()
                print(f"[lesion] {n}: target {100 * (m == 1).mean():.2f}%, ignored {100 * (m == 2).mean():.2f}%")
        seen["lesion"] += 1
    if seen["ct"] >= 3 and seen["lesion"] >= 3:
        break
print("seen", seen)
sys.exit(0 if seen["ct"] and seen["lesion"] else 1)
