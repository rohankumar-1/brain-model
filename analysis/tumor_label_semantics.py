"""Do the tumour label numbers mean the same thing in every cohort? Reads each label's appearance off the images
instead of trusting the documentation: per label, the median T1c and FLAIR intensity inside it minus the median of
the label-free brain (store units, z-scored MRI), over 25 random cases. Enhancing tumour is bright on T1c, oedema
bright on FLAIR with ~normal T1c, necrosis / non-enhancing core dark on T1c, a resection cavity CSF-like (dark on both).

    python runs/diagnostics/tumor_label_semantics.py"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/roku/brain-model")
from brain.downstream.tasks import DEFAULT_VOLUMES  # noqa: E402
from brain.prep import store  # noqa: E402
from brain.prep.index import folder_of  # noqa: E402

ROOT = DEFAULT_VOLUMES.parent
N = 25


def cohorts():
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False, usecols=["dataset", "case_id", "image_path", "mask_seg"])
    path = dict(zip(v.case_id.astype(str), v.image_path))
    for pre in ("brats2023_gli", "brats2023_men", "brats2023_ped", "brats2024_gli", "ucsf_pdgm"):
        t1 = v[(v.dataset == f"{pre}_t1n") & v.mask_seg.notna()].sample(N, random_state=0)
        sib = (lambda c, s: c[:-3] + s) if pre != "ucsf_pdgm" else (lambda c, s: c.rsplit("_", 1)[0] + "_" + s)
        yield pre, [(ROOT / r.mask_seg, *(ROOT / path[sib(str(r.case_id), s)] for s in ("t1c", "t2f")))
                    for r in t1.itertuples() if all(sib(str(r.case_id), s) in path for s in ("t1c", "t2f"))]
    m = pd.DataFrame([json.loads(line) for line in open(ROOT / "manifest_v6_upenn_gbm_0of1.jsonl")])
    f = folder_of("upenn_gbm_t1n")
    ids = [c for c in m[m.dataset == "upenn_gbm_t1n"].case_id if store.path_for(ROOT, f, c, "seg").exists()]
    pick = np.random.default_rng(0).choice(ids, N, replace=False)
    yield "upenn_gbm", [(store.path_for(ROOT, f, c, "seg"),
                         *(store.path_for(ROOT, folder_of(f"upenn_gbm_{s}"), c[:-3] + s) for s in ("t1c", "t2f")))
                        for c in pick]


rows = []
for name, cases in cohorts():
    for mp, p1, p2 in cases:
        lab = np.asarray(store.read_image(mp, "mask"))
        t1c, flair = (np.asarray(store.read_image(p), np.float32) for p in (p1, p2))
        normal = (lab == 0) & (t1c > -5.9) & (flair > -5.9)
        for k in np.unique(lab[lab > 0]):
            sel = lab == k
            if sel.sum() < 50:
                continue
            rows.append({"cohort": name, "label": int(k), "t1c": np.median(t1c[sel]) - np.median(t1c[normal]),
                         "flair": np.median(flair[sel]) - np.median(flair[normal]), "ml": sel.sum() / 1000})
d = pd.DataFrame(rows)
out = d.groupby(["cohort", "label"]).agg(cases=("ml", "size"), ml=("ml", "median"), t1c=("t1c", "median"),
                                         flair=("flair", "median")).round(2)
print(out.to_string())
