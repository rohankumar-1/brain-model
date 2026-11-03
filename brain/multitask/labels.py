"""Label table for BrainQuery's tasks: one row per (volume, task).

    python -m brain.multitask.labels                      # build labels/multitask_v2/{labels.parquet, stats.json}
    python -m brain.multitask.labels --live-sample 1500   # sample size for the live targets' normalisation

Columns: ``volume_id, task, value, ref, split``. ``value`` is the target (0/1, class index, raw regression value;
NaN for segmentation and live targets). ``ref`` says where a dense or live target comes from: a mask zip relative to
the store with its label ids (``path|1,2,3``), ``@brain`` (the brain mask in the image zip) or ``@live:<name>`` (a
per-scan target computed by the loader from the image and its brain mask, see ``live_targets``).

``use: train`` sources read only pretraining-pool volumes (never a held-out family); which rows are trained on is the
loader's choice (split == train of trained tasks). Regression statistics (mean / std, for standardised targets) come
from the train rows of each trained task; live targets' from a random sample of train volumes.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.windowing import asinh_window, canonical_c_w
from brain.prep.fomo import PT_SHORT
from brain.downstream.tasks import (
    CQ500_LABELS,
    DEFAULT_VOLUMES,
    build_rsna_volume_labels,
    label_file,
    load_task,
)
from brain.multitask.registry import Registry, Task, load_registry
from brain.prep import store

STORE = DEFAULT_VOLUMES.parent
OUT_DIR = STORE / "labels" / "multitask_v2"     # v1 (labels/multitask) stays as built for checkpoints trained on it
GRID_ML = 256 * 256 * 160 / 1000.0          # store grid volume in ml (1 mm voxels): brain_frac -> template-space ml
# volumes.csv family -> FOMO participants.tsv dataset
FOMO_PT = {"fomo_" + short: pt for pt, short in PT_SHORT.items()}
LIVE = ("mask_asym", "intensity_asym", "ct_csf_frac", "ct_brain_hu", "ct_calcification", "ct_skull_hu")
COLS = ["volume_id", "task", "value", "ref", "split"]


# ------------------------------------------------------------------------------------------------ live targets
def _pool2(a: np.ndarray) -> np.ndarray:
    z, y, x = (s - s % 2 for s in a.shape)
    return a[:z, :y, :x].reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean(axis=(1, 3, 5))


def pooled_brain(brain: np.ndarray) -> np.ndarray:
    """Brain mask on the 2x block-averaged grid (a voxel is brain when at least half its block is)."""
    return _pool2(np.asarray(brain, dtype=np.float32)) >= 0.5


def lr_mismatch(b: np.ndarray) -> float:
    """Share of a mask that does not match its mirror across the X midline (0 = left-right symmetric)."""
    bf = b[..., ::-1]
    return float((b ^ bf).sum() / max((b | bf).sum(), 1))


def live_targets(image: np.ndarray, brain: np.ndarray, modality: str) -> dict[str, float]:
    """Per-scan targets from a (Z, Y, X) image and its brain mask, on a 2x block-averaged grid (block averaging, unlike
    strided subsampling, commutes with the mirror). Left-right measures mirror the X axis about the grid centre, as the
    loader's flip does, so they are invariant to that augmentation."""
    img = _pool2(np.asarray(image, dtype=np.float32))
    b = pooled_brain(brain)
    out = {k: float("nan") for k in LIVE}
    if b.sum() < 1000:
        return out
    out["mask_asym"] = lr_mismatch(b)
    both = b & b[..., ::-1]
    x = asinh_window(img, *canonical_c_w(modality))
    d = np.abs(x - x[..., ::-1])[both]
    out["intensity_asym"] = float(d.mean() / (x[both].std() + 1e-6))
    if str(modality).lower().startswith("ct"):
        hu = img[b]
        out["ct_csf_frac"] = float(((hu >= 0) & (hu <= 15)).mean())
        tissue = hu[(hu > 0) & (hu < 100)]
        out["ct_brain_hu"] = float(np.median(tissue)) if tissue.size else float("nan")
        # full-resolution (calcifications are a few voxels): intracranial calcification share, skull bone density
        from brain.multitask.measures import ct_measures

        full = ct_measures(np.asarray(image, dtype=np.float32), np.asarray(brain), None)
        out["ct_calcification"], out["ct_skull_hu"] = full["ct_calcification"], full["ct_skull_hu"]
    return out


def _live_one(args) -> dict[str, float]:
    path, modality = args
    try:
        a = store.read_many(STORE / path, ("image", "brain"))
    except Exception:   # noqa: BLE001  a sample for statistics: skip unreadable volumes
        return {k: float("nan") for k in LIVE}
    return live_targets(store.squeeze_channel(a["image"]), a["brain"], modality)


# ------------------------------------------------------------------------------------------------ sources
def _rows(frame: pd.DataFrame, value, ref="") -> pd.DataFrame:
    out = pd.DataFrame({"volume_id": frame["volume_id"].astype(str).to_numpy()})
    out["value"] = np.asarray(value, dtype=np.float64) if not np.isscalar(value) else float(value)
    out["ref"] = ref if isinstance(ref, str) else np.asarray(ref, dtype=object)
    return out


def _mask_ok(v: pd.DataFrame) -> pd.Series:
    """A SynthStrip brain mask that passed QC (classical / failed masks would teach a wrong outline)."""
    ok = v["brain_ok"].astype(str).str.lower().isin(["true", "1", "1.0"])
    return ok & v["brain_method"].astype(str).eq("synthstrip")


def _participants() -> pd.DataFrame:
    """FOMO participants.tsv, with ages it lacks filled from the original releases (``participants_extra.tsv``,
    ``brain.datasets.fomo_ages``) where available."""
    p = pd.read_csv(label_file("fomo260k", "participants.tsv"), sep="\t", low_memory=False)
    p = p.astype({"dataset": str, "participant_id": str, "session_id": str})
    extra_path = label_file("fomo260k", "participants.tsv").with_name("participants_extra.tsv")
    if extra_path.exists():
        e = pd.read_csv(extra_path, sep="\t").astype({"dataset": str, "participant_id": str, "session_id": str})
        key = ["dataset", "participant_id", "session_id"]
        fill = p[key].merge(e[key + ["age"]], on=key, how="left")["age"].to_numpy()
        missing = pd.to_numeric(p["age"], errors="coerce").isna().to_numpy()
        p["age"] = p["age"].astype(object)                   # read as text ("n/a" rows); values parse downstream
        p.loc[missing, "age"] = fill[missing]
    return p


def src_fomo_participants(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    fams = list(s["families"])
    rows = v[v["dataset_family"].isin(fams)].copy()
    parts = rows["case_id"].astype(str).str.split("_")
    rows["pt"] = rows["dataset_family"].map(FOMO_PT)
    rows["sub"], rows["ses"] = parts.str[0], parts.str[1]
    p = _participants()
    p = p[p["dataset"].isin(rows["pt"].unique())]
    field = s["field"]
    p = p.dropna(subset=[field])
    by_ses = p.drop_duplicates(["dataset", "participant_id", "session_id"]).set_index(
        ["dataset", "participant_id", "session_id"])[field]
    # a scan whose session is not listed takes the subject's value only when every listed session agrees: before
    # 2026-10-25 it took the first row's, which for age in longitudinal / infant cohorts is another session's age
    g = p.groupby(["dataset", "participant_id"])[field]
    by_sub = g.first()[g.nunique() == 1]
    val = [by_ses.get((a, b, c), by_sub.get((a, b))) for a, b, c in zip(rows["pt"], rows["sub"], rows["ses"])]
    n_ses = sum(1 for a, b, c in zip(rows["pt"], rows["sub"], rows["ses"]) if (a, b, c) in by_ses.index)
    print(f"[labels] {task.id} ({field}): {n_ses} scans matched by session, {sum(x is not None for x in val) - n_ses} "
          f"by a subject-constant value, {sum(x is None for x in val)} unmatched", flush=True)
    rows["raw"] = val
    rows = rows.dropna(subset=["raw"])
    # free-text groups: ``drop`` exact values; ``map`` exact values, else ``prefix`` (first matching prefix), else
    # ``default`` (unmatched rows are dropped without one)
    rows = rows[~rows["raw"].astype(str).isin([str(d) for d in s.get("drop", [])])]
    if "map" in s or "prefix" in s:
        m = {str(k): v_ for k, v_ in (s.get("map") or {}).items()}
        pre = [(str(k), v_) for k, v_ in (s.get("prefix") or {}).items()]

        def lookup(x: str):
            if x in m:
                return m[x]
            for k, v_ in pre:
                if x.startswith(k):
                    return v_
            return s.get("default")

        rows["raw"] = rows["raw"].astype(str).map(lookup)
        rows = rows.dropna(subset=["raw"])
        if task.type == "multiclass":
            rows["raw"] = rows["raw"].map({c: i for i, c in enumerate(task.classes)})
    else:
        rows["raw"] = pd.to_numeric(rows["raw"], errors="coerce")
        rows = rows.dropna(subset=["raw"])
    return _rows(rows, rows["raw"].astype(float))


def src_rsna(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    lab = build_rsna_volume_labels()
    rows = v[v["dataset_family"] == "rsna_ich"].merge(lab[["volume_id", s["subtype"]]], on="volume_id")
    return _rows(rows, rows[s["subtype"]])


def src_rsna_ica(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    """RSNA-ICA series labels: ``finding`` is ``Aneurysm Present`` or a side-merged location (tasks.RSNA_ICA_LOCATIONS)."""
    from brain.downstream.tasks import rsna_ica_labels

    rows = rsna_ica_labels(v)
    return _rows(rows, rows[s["finding"]].astype(float))


def src_fastmri_findings(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    """fastMRI+ scan-level findings (labels/fastmri_findings.csv, brain.prep.fastmri_labels): 1 = the radiologist
    labelled the finding, 0 = the file was reviewed without it."""
    f = STORE / "labels" / "fastmri_findings.csv"
    if not f.exists():
        return _rows(v.iloc[:0], [])
    lab = pd.read_csv(f)[["case_id", s["finding"]]]
    rows = v[v["dataset"].astype(str).str.startswith("fastmri")].astype({"case_id": str}).merge(lab, on="case_id")
    return _rows(rows, rows[s["finding"]].astype(float))


def src_family_class(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    cls = {c: i for i, c in enumerate(task.classes)}
    rows = v[v["dataset_family"].isin(list(s["map"]))]
    return _rows(rows, rows["dataset_family"].map(lambda f: cls[s["map"][f]]))


def src_brain_volume(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    rows = v[_mask_ok(v) & v["brain_frac"].notna()]
    return _rows(rows, rows["brain_frac"].astype(float) * GRID_ML)


def src_brain_mask(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    return _rows(v[_mask_ok(v)], np.nan, "@brain")


def src_live(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    if s["name"] not in LIVE:
        raise ValueError(f"unknown live target {s['name']}")
    rows = v[_mask_ok(v)]
    if s.get("modality"):
        rows = rows[rows["modality"].astype(str).str.lower().str.startswith(s["modality"])]
    return _rows(rows, np.nan, f"@live:{s['name']}")


def _ids(s: dict) -> str:
    return ",".join(str(int(i)) for i in s["ids"])


def _brats_frame(v: pd.DataFrame, s: dict) -> pd.DataFrame:
    """Every sequence of a BraTS subject with a segmentation (one mask per subject, on the shared grid)."""
    rows = v[v["dataset_family"].isin(list(s["families"]))].copy()
    if s.get("datasets"):
        rows = rows[rows["dataset"].astype(str).str.startswith(tuple(s["datasets"]))]
    rows["mask"] = [f"{f}/{c.rsplit('-', 1)[0]}.seg.zip" for f, c in zip(rows["folder"], rows["case_id"].astype(str))]
    exists = {m: (STORE / m).exists() for m in rows["mask"].unique()}
    return rows[rows["mask"].map(exists)]


def src_brats_seg(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    rows = _brats_frame(v, s)
    return _rows(rows, np.nan, (rows["mask"] + "|" + _ids(s)).to_numpy())


def _mask_ml(args) -> float:
    path, ids = args
    m = np.asarray(store.read(STORE / path, "mask"))
    return float(np.isin(m, ids).sum()) / 1000.0


def src_brats_seg_volume(v: pd.DataFrame, task: Task, s: dict, workers: int = 8) -> pd.DataFrame:
    rows = _brats_frame(v, s)
    masks = sorted(rows["mask"].unique())
    with ProcessPoolExecutor(workers) as ex:
        ml = dict(zip(masks, ex.map(_mask_ml, [(m, list(s["ids"])) for m in masks], chunksize=16)))
    return _rows(rows, np.log1p(rows["mask"].map(ml).astype(float)))


def src_mask_column(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    rows = v[v["dataset_family"].isin(list(s["families"])) & v[s["column"]].notna()]
    return _rows(rows, np.nan, (rows[s["column"]].astype(str) + "|" + _ids(s)).to_numpy())


def src_store_mask(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    """A label mask written next to a stored volume (``<case>.<mask>.zip``, e.g. ``brain.prep.rsna_ica_vessels``):
    the volumes of ``families`` that have one, ``ids`` the voxel values that count as the target."""
    rows = v[v["dataset_family"].isin(list(s["families"]))]
    rel = [store.path_for(Path(""), f, c, s["mask"]) for f, c in zip(rows["folder"], rows["case_id"].astype(str))]
    have = np.asarray([(STORE / p).exists() for p in rel], dtype=bool)
    refs = np.asarray([f"{p}|{_ids(s)}" for p, h in zip(rel, have) if h], dtype=object)
    return _rows(rows[have], np.nan, refs)


def src_cq500(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    b = load_task("cq500", v)
    j = CQ500_LABELS.index(s["finding"])
    out = _rows(pd.DataFrame({"volume_id": b.volume_ids}), b.y[:, j])
    out["split"] = [b.splits[i] for i in b.volume_ids]
    return out


def src_downstream(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    """A frozen-suite task's labels and splits (``tasks.load_task``), for evaluation cohorts."""
    b = load_task(s["task"], v)
    frame = pd.DataFrame({"volume_id": b.volume_ids})
    if b.task_type == "segmentation":
        ids = list(s.get("ids") or sorted({i for r in b.regions.values() for i in r}))
        ign = "|" + ",".join(str(int(x)) for x in s["ignore"]) if s.get("ignore") else ""   # out of the loss
        ref = [f"{b.mask_paths[i]}|{','.join(str(int(x)) for x in ids)}{ign}" for i in b.volume_ids]
        out = _rows(frame, np.nan, np.asarray(ref, dtype=object))
    else:
        y = np.asarray(b.y, dtype=np.float64)
        if y.ndim > 1:
            raise ValueError(f"{s['task']} has {y.shape[1]} labels; pick one in the source")
        out = _rows(frame, y)
    out["split"] = [b.splits[i] for i in b.volume_ids]
    return out


# ------------------------------------------------------------------------------------------------ SynthSeg
SYNTHSEG = STORE / "labels" / "synthseg"
# FreeSurfer label groups; left / right merged (the loader's flip moves masks with the image, so a sided target would
# swap anatomy) and asymmetry kept as flip-invariant |L - R| / (L + R)
SS_GROUPS = {
    "ventricles": [4, 5, 14, 15, 43, 44], "lateral_ventricles_L": [4, 5], "lateral_ventricles_R": [43, 44],
    "hippocampus": [17, 53], "hippocampus_L": [17], "hippocampus_R": [53], "amygdala": [18, 54],
    "thalamus": [10, 49], "caudate": [11, 50], "putamen": [12, 51], "pallidum": [13, 52],
    "cortex": [3, 42], "cortex_L": [3], "cortex_R": [42], "white_matter": [2, 41],
    "cerebellum": [7, 8, 46, 47], "brainstem": [16], "csf": [24],
    "deep_gm": [10, 11, 12, 13, 26, 28, 49, 50, 51, 52, 58, 60],
}
SS_FLUID = {4, 5, 14, 15, 24, 43, 44}


def synthseg_volumes() -> pd.DataFrame:
    """Per-volume structure volumes (template-space ml: scans are affine-registered, so already head-size
    normalised) for every SynthSeg-labelled volume, with group sums, derived measures and a QC flag."""
    import glob

    from brain.prep.synthseg import LABELS, NAMES

    files = sorted(glob.glob(str(SYNTHSEG / "volumes_*.csv")))
    if not files:
        return pd.DataFrame(columns=["volume_id"])
    d = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("volume_id", keep="last").set_index("volume_id")
    by_id = {int(l): NAMES[int(l)] for l in LABELS[1:]}
    g = pd.DataFrame(index=d.index)
    for name, ids in SS_GROUPS.items():
        g[name] = d[[by_id[i] for i in ids]].sum(axis=1)
    brain = d[[by_id[i] for i in by_id if i not in SS_FLUID]].sum(axis=1)
    fluid = d[[by_id[i] for i in SS_FLUID]].sum(axis=1)
    asym = lambda a, b: (g[a] - g[b]).abs() / (g[a] + g[b]).clip(lower=1e-3)  # noqa: E731
    m = pd.DataFrame(index=d.index)
    for name in ("ventricles", "csf"):                         # heavy-tailed: log
        m[f"log_{name}"] = np.log(g[name].clip(lower=0.1))
    for name in ("hippocampus", "amygdala", "thalamus", "caudate", "putamen", "cortex", "white_matter",
                 "cerebellum", "brainstem", "deep_gm"):
        m[name] = g[name]
    m["parenchymal_fraction"] = brain / (brain + fluid)
    m["log_ventricle_brain_ratio"] = np.log(g["ventricles"].clip(lower=0.1) / brain)
    m["gm_wm_ratio"] = (g["cortex"] + g["deep_gm"]) / g["white_matter"].clip(lower=1.0)
    m["hippocampus_asym"] = asym("hippocampus_L", "hippocampus_R")
    m["ventricle_asym"] = asym("lateral_ventricles_L", "lateral_ventricles_R")
    m["cortex_asym"] = asym("cortex_L", "cortex_R")
    # whole-brain field of view (posterior fossa included): truncated scans (M4Raw, RSNA without its posterior-fossa
    # series, some Nigerian / HBN) still give valid masks of what is visible, but their volumes and ratios would
    # measure the field of view
    m["fov_ok"] = (g["cerebellum"] >= 110.0) & (g["brainstem"] >= 18.0)
    # QC: failed segmentations (misregistered / tilted inputs) mislabel whole hemispheres (cortex asymmetry 0.25+) and
    # lose the thalamus. Healthy MRI stays below 0.08 (99.9th percentile); tumours and haemorrhage with mass effect
    # reach 0.08-0.15, so the cutoff sits above them (0.08 removed them: a bias against pathology). The hippocampus
    # check applies only with the whole brain in view (CT coverage often clips the temporal lobes).
    m["qc_ok"] = ((m["cortex_asym"] <= 0.15) & (g["thalamus"] >= 8.0) & ((g["hippocampus"] >= 4.0) | ~m["fov_ok"])
                  & brain.between(700.0, 2600.0))
    return m


QC_MIN = 0.65            # SynthSeg 2.0 QC: a group score below this predicts a failed segmentation (Billot et al. 2023)
# derived measures that combine structures: every QC group must pass
# combined measures -> the QC groups they read (a failing cerebellum says nothing about a grey/white ratio)
_GW, _CSF = ("general_white_matter", "general_grey_matter"), ("general_csf",)
_COMBINED_QC = {"parenchymal_fraction": _GW + _CSF, "log_ventricle_brain_ratio": _GW + _CSF, "gm_wm_ratio": _GW,
                "ct_gray_white": _GW, "evans_index": _GW + _CSF, "cortex_asym": ("general_grey_matter",),
                "ventricle_asym": _CSF, "hippocampus_asym": ("hippocampus_amygdala",)}
_MEASURE_GROUP = {"log_ventricles": "ventricles", "log_csf": "csf"}


def synthseg_qc() -> pd.DataFrame:
    """Per volume, SynthSeg 2.0 QC scores per structure group (``brain.prep.synthseg_qc``, labels/synthseg/qc_*.csv);
    empty when not computed."""
    import glob

    files = sorted(glob.glob(str(SYNTHSEG / "qc_*.csv")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("volume_id", keep="last").set_index("volume_id")


def _qc_columns(structure: str | None) -> list[str]:
    """The QC-score columns a target depends on: the groups of its FreeSurfer labels, those a combined measure reads
    (``_COMBINED_QC``), or all for anything else."""
    from brain.prep.synthseg import LABELS
    from brain.prep.synthseg_qc import qc_groups

    groups, names = qc_groups()
    cols = [n.replace(" ", "_").replace("+", "_") for n in names]
    if structure in _COMBINED_QC:
        return list(_COMBINED_QC[structure])
    if structure is None or structure not in SS_GROUPS:
        return cols[1:]
    lut = {int(lab): int(g) for lab, g in zip(LABELS, groups)}
    got = sorted({cols[lut[i]] for i in SS_GROUPS[structure] if lut.get(i, 0) > 0})
    return got or ["general_csf"]          # extra-ventricular CSF (label 24) is background to the QC network


def _synthseg_rows(v: pd.DataFrame, need_fov: bool = False,
                   structure: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """QC-passed SynthSeg volumes for a target on ``structure`` (an ``SS_GROUPS`` name or a measure). With QC scores
    (``synthseg_qc``), a volume passes when every QC group of the target scores >= ``QC_MIN``; volumes without scores
    fall back to the hand-set ``qc_ok``. ``need_fov``: the whole brain in view as well (volume targets)."""
    m = synthseg_volumes()
    if "qc_ok" in m:
        ok = m["qc_ok"].copy()
        q = synthseg_qc()
        if len(q):
            cols = _qc_columns(_MEASURE_GROUP.get(structure, structure))
            scored = m.index.isin(q.index)
            ok[scored] = (q.reindex(m.index[scored])[cols] >= QC_MIN).all(axis=1).to_numpy()
        m = m[ok & (m["fov_ok"] if need_fov else True)]
    rows = v[v["volume_id"].isin(m.index)]
    return rows, m.loc[rows["volume_id"]]


def src_synthseg_seg(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    """Binary mask of a SynthSeg structure group (QC-passed volumes); the zip holds indices into ``LABELS``."""
    from brain.prep.synthseg import LABELS

    rows, _ = _synthseg_rows(v, structure=s["group"])
    idx = ",".join(str(int(i)) for i in np.searchsorted(LABELS, SS_GROUPS[s["group"]]))
    refs = [f"{store.path_for(Path(''), f, c, 'synthseg')}|{idx}" for f, c in zip(rows["folder"], rows["case_id"].astype(str))]
    return _rows(rows, np.nan, np.asarray(refs, dtype=object))


def src_synthseg_scan_measure(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    """A per-scan clinical measure from ``labels/synthseg/measures.csv`` (``brain.multitask.measures``) of QC-passed
    SynthSeg scans; ``modality`` restricts it (e.g. CT-only grey-white ratio)."""
    path = SYNTHSEG / "measures.csv"
    if not path.exists():
        return _rows(v.head(0), np.nan)
    meas = pd.read_csv(path).set_index("volume_id")[s["measure"]].dropna()
    rows, _ = _synthseg_rows(v, structure=s["measure"])
    rows = rows[rows["volume_id"].isin(meas.index)]
    if s.get("modality"):
        rows = rows[rows["modality"].astype(str).str.lower().str.startswith(s["modality"])]
    return _rows(rows, meas.loc[rows["volume_id"]].to_numpy())


def src_synthseg_measure(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    """A volume / ratio / asymmetry of QC-passed scans with the whole brain in the field of view."""
    rows, m = _synthseg_rows(v, need_fov=True, structure=s["measure"])
    return _rows(rows, m[s["measure"]].to_numpy())


def src_wmh(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    """log(1 + WMH ml) (``raw``: ml) from ``labels/wmh`` (``brain.prep.wmh_run``; FLAIR / T2 outside tumour cohorts).
    ``contrasts`` keeps those input contrasts; ``min_age`` keeps scans of subjects with a known age (FOMO
    participants) at least that. Checked 2026-11-01 (runs/diagnostics/wmh_validate.py, tool_label_quality): against
    expert masks WMH-SynthSeg ranks burden (Spearman 0.84) but over-calls it 1.8x, site-dependently; in children it
    calls developing white matter WMH (HBN: 80% > 5 ml, falling with age) and on T2 it does not track age."""
    import glob
    from types import SimpleNamespace

    files = sorted(glob.glob(str(STORE / "labels" / "wmh" / "volumes_*.csv")))
    if not files:
        return _rows(v.head(0), np.nan)
    w = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("volume_id", keep="last").set_index("volume_id")
    if s.get("contrasts"):
        w = w[w["contrast"].astype(str).str.lower().isin([str(c).lower() for c in s["contrasts"]])]
    rows = v[v["volume_id"].isin(w.index)]
    if s.get("min_age") is not None:
        fams = [f for f in rows["dataset_family"].unique() if f in FOMO_PT]
        ages = src_fomo_participants(v, SimpleNamespace(id=f"{task.id} ages"), {"families": fams, "field": "age"})
        age = pd.to_numeric(ages.set_index("volume_id")["value"], errors="coerce")
        rows = rows[rows["volume_id"].map(age) >= float(s["min_age"])]
    ml = w.loc[rows["volume_id"], s.get("measure", "wmh_ml")].to_numpy()
    return _rows(rows, ml if s.get("raw") else np.log1p(ml))


# ------------------------------------------------------------------------------------------------ self-supervised
def src_ssl_recon(v: pd.DataFrame, task: Task, s: dict) -> pd.DataFrame:
    """Masked reconstruction: every label-pool volume (the loader hides tokens and keeps the image as the target)."""
    return _rows(v, np.nan, "@recon")


def _affine_key(path: str) -> str:
    try:
        a = store.read_attrs(STORE / path, "image").get("reg_affine")
    except Exception:   # noqa: BLE001  unreadable attrs: no pair
        return ""
    return "" if a is None else ",".join(f"{x:.4f}" for x in np.asarray(a, dtype=float).ravel())


def coregistered_pairs(v: pd.DataFrame, workers: int = 8, path: Path | None = None) -> pd.DataFrame:
    """Volumes on one identical stored grid: same subject (``dedup_key``) and the same registration affine to 4
    decimals (a sequence prepped through its reference's registration, ``Case.ref``: BraTS, UPENN-GBM, SynthRAD, whose
    MR / CT / CBCT sit in separate store folders). Returns ``grid`` per volume (``volume_id, grid``; one id per group);
    the affine read is cached (``pairs_affine.parquet``)."""
    path = path or OUT_DIR / "pairs_affine.parquet"
    cand = v[v.groupby("dedup_key")["volume_id"].transform("size") > 1]
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["volume_id", "affine"])
    todo = cand[~cand["volume_id"].isin(old["volume_id"])]
    if len(todo):
        with ProcessPoolExecutor(workers) as ex:
            keys = list(ex.map(_affine_key, todo["image_path"].astype(str), chunksize=64))
        old = pd.concat([old, pd.DataFrame({"volume_id": todo["volume_id"].to_numpy(), "affine": keys})])
        path.parent.mkdir(parents=True, exist_ok=True)
        old.to_parquet(path, index=False)
    c = cand.merge(old, on="volume_id")
    c = c[c["affine"].astype(str) != ""]
    c["grid"] = c["dedup_key"].astype(str) + "|" + c["affine"].astype(str)
    c = c[c.groupby("grid")["volume_id"].transform("size") > 1]
    return c[["volume_id", "grid"]]


_PAIRS: dict[int, pd.DataFrame] = {}


def src_synth(v: pd.DataFrame, task: Task, s: dict, workers: int = 8, cache_dir: Path | None = None) -> pd.DataFrame:
    """Contrast synthesis: each volume with another contrast ``s.contrast`` (``contrast_std``) on its identical grid
    asks for that contrast; ``ref`` is ``@synth:<partner image path>|<partner modality>``. A cone-beam CT (SynthRAD
    task 2) is never a target but asks for its CT (CBCT -> CT), although both count as contrast ``ct``."""
    key = id(v)
    if key not in _PAIRS:                         # one affine scan for all synth_* tasks of a build
        _PAIRS[key] = coregistered_pairs(v, workers, cache_dir / "pairs_affine.parquet" if cache_dir else None)
    g = _PAIRS[key].merge(v[["volume_id", "image_path", "modality", "contrast_std", "dataset"]], on="volume_id")
    cbct = g["dataset"].astype(str).str.contains("cbct")
    same = g["contrast_std"].astype(str) == str(s["contrast"])
    target = g[same & ~cbct].drop_duplicates("grid")
    pairs = g.merge(target[["grid", "volume_id", "image_path", "modality"]], on="grid", suffixes=("", "_tgt"))
    other = pairs["contrast_std"].astype(str) != str(s["contrast"])
    pairs = pairs[(pairs["volume_id"] != pairs["volume_id_tgt"])
                  & (other | pairs["dataset"].astype(str).str.contains("cbct"))]
    ref = "@synth:" + pairs["image_path_tgt"].astype(str) + "|" + pairs["modality_tgt"].astype(str).str.lower()
    return _rows(pairs, np.nan, ref.to_numpy())


SOURCES = {
    "ssl_recon": src_ssl_recon, "synth": src_synth,
    "fomo_participants": src_fomo_participants, "rsna": src_rsna, "rsna_ica": src_rsna_ica, "fastmri_findings": src_fastmri_findings,
    "family_class": src_family_class,
    "brain_volume": src_brain_volume, "brain_mask": src_brain_mask, "live": src_live,
    "brats_seg": src_brats_seg, "brats_seg_volume": src_brats_seg_volume, "mask_column": src_mask_column, "store_mask": src_store_mask,
    "cq500": src_cq500, "downstream": src_downstream,
    "synthseg_seg": src_synthseg_seg, "synthseg_measure": src_synthseg_measure,
    "synthseg_scan_measure": src_synthseg_scan_measure, "wmh": src_wmh,
}


# ------------------------------------------------------------------------------------------------ registration QC
REG_ASYM_MAX = 0.09     # brain-mask left-right mismatch about the grid midline (tilted / shifted heads); tumour
                        # cohorts stay below ~0.10, misregistered OASIS-1 / RSNA / HBN scans reach 0.10-0.27
REG_DICE_MIN = 0.85     # brain mask vs registered template brain (OASIS-1-style misregistration; children ~0.85+)
# Targets that are only right when the registration is: template-space volumes (registration scale) and left-right
# asymmetries (grid midline). Registration QC drops a failing scan's rows for these tasks only; its study-level labels
# (hemorrhage, age, diagnosis, ...) and image-space masks (brain, tumour, SynthSeg structures) still train.
REG_SENSITIVE = ("brain_volume", "brain_asymmetry", "intensity_asymmetry", "tumor_volume")
REG_SENSITIVE_PREFIXES = ("vol_",)


def registration_sensitive(task_id: str) -> bool:
    return task_id in REG_SENSITIVE or task_id.startswith(REG_SENSITIVE_PREFIXES)


HARD_DROP_REASONS = ("duplicate", "mirrored_reader")      # pool_drop reasons that also keep a volume out of labels


def label_pool(volumes: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """The volumes training labels come from, and the subset kept out of SSL only (``ssl_only_drop``).

    Pretraining-pool volumes, plus BHSD (its RSNA studies were kept out of SSL only), plus the volumes SSL excluded for
    image-quality reasons that do not touch study-level labels: DICOM-assembly QC failures (``ssl_qc_drop``: RSNA
    studies with stripes / missing posterior fossa) and the per-volume prep-QC drops (partial slabs, unscaled
    fallback registrations). Before 2026-10-25 those were silently out of every task too (the hemorrhage tasks lost
    2,200 RSNA studies). ``index.MULTITASK_FAMILIES`` (held out of SSL, multitask training cohorts) are in. Other
    held-out families, dataset-level drops and cross-dataset duplicates stay out of everything.
    ``ssl_only_drop`` volumes lose their registration-sensitive and live (image-measured) rows in ``build``."""
    from brain.prep.index import MULTITASK_FAMILIES, POOL_DROP_DATASETS

    drop_dir = STORE / "labels" / "pool_drop"
    dups: set[str] = set()
    qc_drops: set[str] = set()
    for f in sorted(drop_dir.glob("*.csv")) if drop_dir.exists() else []:
        d = pd.read_csv(f)
        if d.empty:
            continue
        reasons = d["reason"].astype(str) if "reason" in d else pd.Series("duplicate", index=d.index)
        dup = reasons.isin(HARD_DROP_REASONS)
        dups |= set(d.loc[dup, "volume_id"].astype(str))
        qc_drops |= set(d.loc[~dup, "volume_id"].astype(str))
    vid = volumes["volume_id"].astype(str)
    pretrain = volumes["pretrain"].astype(bool)
    bhsd = volumes["bhsd"].astype(bool) if "bhsd" in volumes else False
    ssl_qc = volumes["ssl_qc_drop"].astype(bool) if "ssl_qc_drop" in volumes else False
    ssl_only = (ssl_qc | vid.isin(qc_drops)) & ~pretrain
    mt = volumes["dataset_family"].isin(MULTITASK_FAMILIES) if "dataset_family" in volumes else False
    twin = volumes["twin_drop"].astype(bool) if "twin_drop" in volumes else False    # BraTS copies of UCSF / UPENN
    keep = (pretrain | bhsd | ssl_only | mt) & (~volumes["held_out"].astype(bool) | mt) \
        & ~volumes["dataset"].isin(POOL_DROP_DATASETS) & ~vid.isin(dups) & ~twin
    return volumes[keep], set(vid[keep & ssl_only])


def _mask_asym(path: str) -> float:
    try:
        b = np.asarray(store.read(STORE / path, "brain")) > 0
    except Exception:   # noqa: BLE001  no brain mask: unknown, decided by the other checks
        return float("nan")
    return lr_mismatch(pooled_brain(b))


def registration_qc(volumes: pd.DataFrame, workers: int = 8, path: Path | None = None) -> pd.DataFrame:
    """Per-volume registration check for label-pool volumes, cached (``regqc.parquet``; only new volumes are
    read). ``ok`` is False when the brain mask is left-right asymmetric about the grid midline (> REG_ASYM_MAX),
    overlaps the template brain poorly (< REG_DICE_MIN), a SynthSeg segmentation of the scan failed its QC, or the
    registration fell back to rigid / translation (not scaled to the template; counted only from 2026-10-25).
    Volumes failing it lose their registration-sensitive training rows (``registration_sensitive``) only."""
    path = path or OUT_DIR / "regqc.parquet"
    pool, _ = label_pool(volumes)
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["volume_id", "mask_asym"])
    todo = pool[~pool["volume_id"].isin(old["volume_id"])]
    if len(todo):
        with ProcessPoolExecutor(workers) as ex:
            asym = list(ex.map(_mask_asym, todo["image_path"].astype(str), chunksize=64))
        old = pd.concat([old[["volume_id", "mask_asym"]],
                         pd.DataFrame({"volume_id": todo["volume_id"].to_numpy(), "mask_asym": asym})])
        path.parent.mkdir(parents=True, exist_ok=True)
        old.to_parquet(path, index=False)
    cols = ["volume_id", "dataset_family", "brain_dice_tpl"] + (["reg_method"] if "reg_method" in pool else [])
    q = pool[cols].merge(old, on="volume_id", how="left")
    ss = synthseg_volumes()
    q["synthseg_ok"] = q["volume_id"].map(ss["qc_ok"]) if "qc_ok" in ss else np.nan
    # a missing value (no brain mask / not segmented) never fails a check on its own: plain booleans, no NA
    asym_bad = q["mask_asym"].gt(REG_ASYM_MAX).to_numpy(bool)
    dice_bad = q["brain_dice_tpl"].lt(REG_DICE_MIN).to_numpy(bool)
    seg_bad = q["synthseg_ok"].map(lambda x: x is False or x is np.False_).to_numpy(bool)
    # a rigid or translation-only fallback registration is unscaled: template-space volumes are native ml there
    fit_bad = (q["reg_method"].notna() & q["reg_method"].astype(str).ne("affine")).to_numpy(bool) \
        if "reg_method" in q else np.zeros(len(q), bool)
    q["ok"] = ~(asym_bad | dice_bad | seg_bad | fit_bad)
    return q


# ------------------------------------------------------------------------------------------------ build
def refine(f: pd.DataFrame, s: dict, volumes: pd.DataFrame) -> pd.DataFrame:
    """Source options applied to any source's rows: ``modality`` keeps that modality's volumes (e.g. a SynthSeg
    target that is noise on CT); ``ct_max_thick_mm`` drops CT volumes whose slices are thicker (unknown thickness
    too); ``exclude_datasets`` (fnmatch patterns) drops those datasets' volumes; ``bins``
    (ascending edges) turns a scalar into ordered classes 0..len(bins) (value < bins[0] is class 0); ``threshold``
    turns it into a binary label (value >= threshold)."""
    if s.get("modality") and "modality" not in f:
        mod = dict(zip(volumes["volume_id"].astype(str), volumes["modality"].astype(str).str.lower()))
        f = f[f["volume_id"].astype(str).map(mod).fillna("").str.startswith(str(s["modality"]))]
    elif s.get("modality"):
        f = f[f["modality"].astype(str).str.lower().str.startswith(str(s["modality"]))]
    if s.get("ct_max_thick_mm") is not None:      # thick-slice CT out of the small-structure SynthSeg targets
        u = volumes.drop_duplicates("volume_id")
        vid = f["volume_id"].astype(str)
        ct = vid.map(dict(zip(u["volume_id"].astype(str), u["modality"].astype(str).str.lower()))).fillna("").str.startswith("ct")
        mm = pd.to_numeric(vid.map(dict(zip(u["volume_id"].astype(str), u["thick_mm"]))), errors="coerce")
        f = f[~ct | (mm <= float(s["ct_max_thick_mm"]))]
    if s.get("exclude_datasets"):                 # e.g. contrast CTA out of the non-contrast CT measures
        from fnmatch import fnmatch

        ds = dict(zip(volumes["volume_id"].astype(str), volumes["dataset"].astype(str)))
        pats = list(s["exclude_datasets"])
        f = f[~f["volume_id"].astype(str).map(lambda v: any(fnmatch(ds.get(v, ""), p) for p in pats))]
    if "bins" in s or "threshold" in s:
        x = pd.to_numeric(f["value"], errors="coerce")
        f = f[x.notna()].copy()
        x = x[x.notna()].to_numpy(dtype=float)
        f["value"] = (np.digitize(x, np.asarray(s["bins"], dtype=float)) if "bins" in s
                      else (x >= float(s["threshold"]))).astype(float)
    return f


# labels an external tool computed from the scan itself (SynthSeg, the WMH segmenter, SynthStrip's brain mask and the
# measures taken inside it): wrong whenever the scan or its processing is broken. Human and study-level labels
# (radiologist reads, expert masks, demographics) stay on such scans.
TOOL_SOURCES = {"synthseg_seg", "synthseg_measure", "synthseg_scan_measure", "wmh", "live", "brain_volume", "brain_mask"}


def _mirror_synthrad_ct(f: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """SynthRAD task-1 CT copies of the paired T1 MR's rows (same case id, one stored grid: the CT was resampled
    through the MR's registration), so an MR-derived target reaches CT without trusting CT SynthSeg."""
    v = volumes.astype({"case_id": str})
    mr = v[v["dataset"] == "synthrad_t1w"].set_index("volume_id")["case_id"]
    ct = v[v["dataset"] == "synthrad_ct"].drop_duplicates("case_id").set_index("case_id")["volume_id"]
    m = f[f["volume_id"].isin(mr.index)].copy()
    m["volume_id"] = m["volume_id"].map(mr).map(ct)
    m = m.dropna(subset=["volume_id"])
    return pd.concat([f, m.drop(columns=["split"], errors="ignore")], ignore_index=True)


def build(reg: Registry, volumes: pd.DataFrame, workers: int = 8, qc: pd.DataFrame | None = None,
          cache_dir: Path | None = None, qc_drop: str = "tool") -> pd.DataFrame:
    """``qc_drop``: training rows on scans failing the index's brain-mask or registration check (``brain_ok`` /
    ``reg_ok``) that are dropped: ``tool`` (labels from ``TOOL_SOURCES``), ``all``, or ``none``."""
    # training sources read the label pool (``label_pool``): the pretraining pool, BHSD, and the volumes kept out of
    # SSL only (split == train rows, like every other source)
    pool, ssl_only = label_pool(volumes)
    _PAIRS.clear()
    split = dict(zip(volumes["volume_id"].astype(str), volumes["split"].astype(str)))
    frames = []
    for t in reg.tasks:
        for s in t.sources:
            use = s.get("use", "train" if t.trained else "eval")
            fn = SOURCES[s["source"]]
            src = pool if use == "train" else volumes
            kw = {"workers": workers} if fn is src_brats_seg_volume else \
                {"workers": workers, "cache_dir": cache_dir} if fn is src_synth else {}
            f = refine(fn(src, t, s, **kw), s, src)
            if s.get("mirror_synthrad_ct"):       # after refine, so a ``modality: mri`` filter does not drop the copies
                f = _mirror_synthrad_ct(f, src)
            f["tool"] = s["source"] in TOOL_SOURCES
            if "split" not in f:
                f["split"] = f["volume_id"].map(split)
            f["split"] = f["split"].fillna(f["volume_id"].astype(str).map(split))    # mirrored rows
            f["task"] = t.id
            f["use"] = use
            frames.append(f)
            print(f"[labels] {t.id:28s} {s['source']:18s} {use:5s} rows={len(f):6d} "
                  f"splits={f['split'].value_counts().to_dict()}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["volume_id", "task"], keep="first")
    if qc_drop != "none":                  # index QC (brain mask / registration) failures
        ok = lambda c: volumes[c].astype(str).str.lower().isin(["true", "1", "1.0"]) if c in volumes \
            else pd.Series(True, index=volumes.index)  # noqa: E731
        failing = set(volumes.loc[~(ok("brain_ok") & ok("reg_ok")), "volume_id"].astype(str))
        drop = (out["use"] == "train") & out["volume_id"].astype(str).isin(failing) & \
            (out["tool"].astype(bool) if qc_drop == "tool" else True)
        print(f"[labels] index QC: {len(failing)} volumes fail brain_ok / reg_ok; {int(drop.sum())} training rows "
              f"dropped ({qc_drop} labels)", flush=True)
        out = out[~drop]
    if qc is not None:                     # registration QC: failing volumes lose their alignment-dependent targets
        bad = set(qc.loc[~qc["ok"], "volume_id"])
        sensitive = out["task"].map(registration_sensitive).astype(bool)
        drop = (out["use"] == "train") & out["volume_id"].isin(bad) & sensitive
        print(f"[labels] registration QC: {len(bad)} volumes fail; {int(drop.sum())} training rows of "
              f"registration-sensitive tasks dropped (their other labels kept)", flush=True)
        out = out[~drop]
    skull = {t.id for t in reg.tasks if t.needs_skull}
    if skull and "stripped_input" in volumes:   # a scan distributed skull-stripped cannot answer a skull task
        stripped = set(volumes.loc[volumes["stripped_input"].astype(str).str.lower().isin(["true", "1", "1.0"]),
                                   "volume_id"].astype(str))
        drop = out["task"].isin(skull) & out["volume_id"].astype(str).isin(stripped)
        print(f"[labels] {int(drop.sum())} rows of skull tasks {sorted(skull)} dropped on skull-stripped scans",
              flush=True)
        out = out[~drop]
    if ssl_only:                           # SSL-only drops: study-level labels and masks stay, image measures go
        live = {t.id for t in reg.tasks if any(s.get("source") == "live" for s in t.sources)}
        unsafe = out["task"].map(lambda t: registration_sensitive(t) or t in live).astype(bool)
        drop = (out["use"] == "train") & out["volume_id"].isin(ssl_only) & unsafe
        print(f"[labels] {len(ssl_only)} SSL-only-dropped volumes back in the label pool; {int(drop.sum())} of their "
              f"registration-sensitive / live rows dropped", flush=True)
        out = out[~drop]
    return out[COLS + ["use"]].reset_index(drop=True)


def regression_stats(reg: Registry, labels: pd.DataFrame, volumes: pd.DataFrame, live_sample: int,
                     workers: int, seed: int = 0) -> dict[str, dict[str, float]]:
    stats = {}
    for t in reg.trained:
        if t.type != "regression":
            continue
        rows = labels[(labels["task"] == t.id) & (labels["use"] == "train") & (labels["split"] == "train")]
        live = rows["ref"].astype(str).str.startswith("@live:")
        if not live.any():
            y = rows["value"].astype(float)
            stats[t.id] = {"mean": float(y.mean()), "std": float(y.std() + 1e-6), "n": int(len(y))}
    live_tasks = {t.id: str(r) for t in reg.trained
                  for r in labels.loc[labels["task"] == t.id, "ref"].head(1) if str(r).startswith("@live:")}
    if live_tasks:
        cand = labels[labels["task"].isin(list(live_tasks)) & (labels["split"] == "train")]["volume_id"].unique()
        mod = volumes.set_index("volume_id").loc[cand, "modality"].astype(str).str.lower()
        ct, mr = mod.index[mod.str.startswith("ct")].to_numpy(), mod.index[~mod.str.startswith("ct")].to_numpy()
        rng = np.random.default_rng(seed)
        # a third of the sample is CT, so the CT-only targets get enough values
        n_ct = min(len(ct), live_sample // 3)
        pick = np.concatenate([rng.choice(ct, size=n_ct, replace=False),
                               rng.choice(mr, size=min(len(mr), live_sample - n_ct), replace=False)])
        v = volumes.set_index("volume_id").loc[pick]
        args = list(zip(v["image_path"].astype(str), v["modality"].astype(str)))
        with ProcessPoolExecutor(workers) as ex:
            vals = pd.DataFrame(list(ex.map(_live_one, args, chunksize=8)), index=pick)
        for tid, ref in live_tasks.items():
            name = ref.split(":", 1)[1]
            ok = labels[(labels["task"] == tid)]["volume_id"]
            y = vals.loc[vals.index.isin(set(ok)), name].dropna()
            stats[tid] = {"mean": float(y.mean()), "std": float(y.std() + 1e-6), "n": int(len(y)), "live": name}
    return stats


def summary(reg: Registry, labels: pd.DataFrame) -> str:
    lines = []
    for t in reg.tasks:
        r = labels[labels["task"] == t.id]
        tr = r[(r["use"] == "train") & (r["split"] == "train")]
        extra = ""
        if t.type in ("binary",) and len(tr):
            extra = f" pos={tr['value'].mean():.3f}"
        if t.type == "multiclass" and len(tr):
            extra = " classes=" + str(tr["value"].value_counts().sort_index().astype(int).to_dict())
        lines.append(f"{t.id:28s} {t.type:12s} {t.role:9s} train={len(tr):6d} "
                     f"all={len(r):6d}{extra}")
    tr = labels[(labels["use"] == "train") & (labels["split"] == "train")]
    trained = {t.id for t in reg.trained}
    per_vol = tr[tr["task"].isin(trained)].groupby("volume_id").size()
    lines.append(f"train volumes with labels: {len(per_vol)}; tasks per volume: mean {per_vol.mean():.2f}, "
                 f"quantiles {per_vol.quantile([0.1, 0.5, 0.9]).to_dict()}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--live-sample", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--qc-drop", choices=["tool", "all", "none"], default="tool",
                    help="labels dropped on scans failing brain_ok / reg_ok (build's qc_drop)")
    a = ap.parse_args()
    reg = load_registry(a.registry)
    volumes = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
    # each label version computes its own registration-QC cache: re-registered volumes (infants) change their masks
    qc = registration_qc(volumes, a.workers, path=a.out / "regqc.parquet")
    print("[labels] registration QC fail share per family:\n"
          + (1 - qc.groupby("dataset_family")["ok"].mean()).round(3).to_string(), flush=True)
    labels = build(reg, volumes, a.workers, qc, cache_dir=a.out, qc_drop=a.qc_drop)
    stats = regression_stats(reg, labels, volumes, a.live_sample, a.workers)
    a.out.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(a.out / "labels.parquet", index=False)
    (a.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    qc.to_csv(a.out / "regqc_summary.csv", index=False)
    text = summary(reg, labels)
    (a.out / "summary.txt").write_text(text + "\n")
    print(text)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
