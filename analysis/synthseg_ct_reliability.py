"""How noisy are the SynthSeg targets on CT? Test-retest agreement of the multitask SynthSeg measures (QC-passed scans,
brain.multitask.labels.synthseg_volumes) on subjects scanned twice: RSNA patients with two CT studies vs MRI repeats
(OASIS-1 reliability rescans, days apart; OASIS-2 visits, 1-2 years apart, a conservative bar since brains change).
ICC(1,1) on one random pair per subject; a target with low ICC is mostly noise to the model.
Writes runs/diagnostics/synthseg_ct_reliability.csv."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/roku/brain-model")
from brain.downstream.tasks import DEFAULT_VOLUMES  # noqa: E402
from brain.multitask.labels import synthseg_volumes  # noqa: E402

MEASURES = ["log_ventricles", "log_csf", "hippocampus", "amygdala", "thalamus", "caudate", "putamen", "cortex",
            "white_matter", "cerebellum", "brainstem", "deep_gm", "parenchymal_fraction", "gm_wm_ratio",
            "log_ventricle_brain_ratio"]

# measures that read the field of view (posterior fossa, whole-brain totals): only on pairs with the whole brain in view
FOV_MEASURES = {"cerebellum", "brainstem", "log_csf", "parenchymal_fraction", "log_ventricle_brain_ratio"}


def icc1(a: np.ndarray, b: np.ndarray) -> float:
    """One-way random ICC(1,1) of paired measurements."""
    x = np.stack([a, b], 1)
    ok = np.isfinite(x).all(1)
    x = x[ok]
    n = len(x)
    if n < 10:
        return np.nan
    gm, sm = x.mean(), x.mean(1)
    msb = 2 * ((sm - gm) ** 2).sum() / (n - 1)
    msw = ((x - sm[:, None]) ** 2).sum() / n
    return float((msb - msw) / (msb + msw))


def pairs(m: pd.DataFrame, subject: pd.Series, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = subject[subject.index.isin(m.index)].groupby(subject).groups
    two = [rng.choice(list(ids), 2, replace=False) for ids in g.values() if len(ids) >= 2]
    return m.loc[[p[0] for p in two]].reset_index(drop=True), m.loc[[p[1] for p in two]].reset_index(drop=True)


v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False, usecols=["volume_id", "dataset_family", "dedup_key", "case_id"])
v = v.drop_duplicates("volume_id").set_index("volume_id")
m = synthseg_volumes()
print(f"SynthSeg: {len(m)} volumes, QC pass {m.qc_ok.mean():.3f}")
m = m[m.qc_ok]
rng = np.random.default_rng(0)
cohorts = {
    "RSNA CT (same patient, 2 studies)": v[v.dataset_family == "rsna_ich"].dedup_key,
    "OASIS-1 MRI (rescan, days)": v[v.dataset_family == "oasis1"].case_id.str.extract(r"(OAS1_\d+)")[0]
        .set_axis(v[v.dataset_family == "oasis1"].index),
    "OASIS-2 MRI (visits, 1-2 y)": v[v.dataset_family == "oasis2"].dedup_key,
}
rows = []
for name, subj in cohorts.items():
    a, b = pairs(m, subj, rng)
    fov = (a["fov_ok"] & b["fov_ok"]).to_numpy()           # both scans show the whole brain
    for meas in MEASURES:
        use = fov if meas in FOV_MEASURES else np.ones(len(a), bool)
        rows.append({"cohort": name, "pairs": int(use.sum()), "measure": meas,
                     "icc": icc1(a[meas].to_numpy()[use], b[meas].to_numpy()[use])})
out = pd.DataFrame(rows)
out.to_csv("/home/roku/brain-model/runs/diagnostics/synthseg_ct_reliability.csv", index=False)
print(out.pivot(index="measure", columns="cohort", values="icc").round(2).to_string())
print(out.groupby("cohort").pairs.first().to_string())
