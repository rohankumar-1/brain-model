"""Dataset map for the 2026-11-09 meeting: per dataset, the scans the full-suite run trains on (left of centre) and the
scans it is tested on (right): a training cohort's held-out test split, or an external cohort's benchmark-scored scans.
Rows grouped and coloured by modality + target. Counts are scans (each MRI sequence counts once), from the store index,
labels v12 and configs/multitask/r16.yaml's exclusions; external sizes are the most scans any of that cohort's frozen
benchmark tasks scored (runs/bench/page/scores.json), or for segmentation-transfer sets the cohort's cases.
    python analysis/presentation/datasets_figure.py OUT.html"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from brain.common.paths import resolve

def dataset_data() -> dict:
    """The figure's rows and totals (also embedded by build_presentation.py)."""
    ROOT = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(open(ROOT / "configs/multitask/r16.yaml"))
    d = cfg["data"]
    store = resolve(d["store_root"])
    v = pd.read_csv(store / "volumes.csv", usecols=["volume_id", "dataset_family", "pretrain", "split"],
                    low_memory=False).drop_duplicates("volume_id")
    labels = pd.read_parquet(resolve(d["labels_dir"]) / "labels.parquet", columns=["volume_id", "split", "use"])
    excl = set(d.get("exclude_datasets") or [])
    xv = set(pd.read_csv(resolve(d["exclude_volumes"]))["volume_id"])
    fam = v.set_index("volume_id")["dataset_family"]
    ok = lambda s: ~s.map(fam).isin(excl) & ~s.isin(xv)  # noqa: E731
    lt = labels[labels.use == "train"]
    lt = lt[ok(lt.volume_id)]
    train_ids = set(lt[lt.split.isin(["train", "val"])].volume_id) | set(
        v[(v.pretrain == True) & ok(v.volume_id) & (v.split != "test")].volume_id)  # noqa: E712
    test_ids = set(lt[lt.split == "test"].volume_id)
    v["tr"], v["te"] = v.volume_id.isin(train_ids), v.volume_id.isin(test_ids)
    by = v.groupby("dataset_family")[["tr", "te"]].sum().astype(int)
    scores = json.load(open(ROOT / "runs/bench/page/scores.json"))
    bench_n = {t: max((m.get(k) or {}).get("n", 0) for k in m if isinstance(m.get(k), dict))
               for t, m in scores.items() if ":" not in t}


    def ext(*tasks):
        return max(bench_n.get(t, 0) for t in tasks)


    FOMO_HEALTHY = ["fomo_corr", "fomo_slim", "fomo_wand", "fomo_m4raw", "fomo_brainlat", "fomo_wayne", "fomo_nigerian",
                    "fomo_yale_highres", "fomo_chbmp", "fomo_beijing", "fomo_abd", "fomo_hbnssi", "fomo_acpi",
                    "fomo_ageility", "fomo_ahead", "fomo_cunmet", "fomo_mica", "fomo_taowu", "fomo_cleveland"]


    def own(f):
        fs = f if isinstance(f, list) else [f]
        return int(by.loc[fs, "tr"].sum()), int(by.loc[fs, "te"].sum())


    # group label, rows: (name, family or families | None, external scans or None, note)
    GROUPS = [
        ("CT · intracranial hemorrhage", [
            ("RSNA ICH", "rsna_ich", None, "study labels for 5 bleed types"),
            ("HemSeg-200", "hemseg", None, "bleed masks on RSNA studies"),
            ("BHSD", "bhsd", None, "bleed masks on RSNA studies"),
            ("CQ500", None, ext("cq500"), "3-reader findings, external site"),
            ("INSTANCE 2022", None, 99, "bleed masks, segmentation transfer"),
        ]),
        ("CT · stroke and vessels", [
            ("RSNA Aneurysm (CTA + MRA)", None, ext("rsna_ica_cta_aneurysm") + ext("rsna_ica_mr_aneurysm"),
             "removed from training for the benchmark"),
            ("ISLES 2024 (NCCT + CTA)", None, ext("isles24_lvo_side", "isles24_mrs"), "LVO side, 90-day outcome"),
            ("AISD", None, ext("aisd_remote_infarct"), "acute infarct CT"),
        ]),
        ("CT · pediatric and synthesis", [
            ("SynthRAD 2023 (CT + MR)", "synthrad", None, "paired CT / MR"),
            ("Pediatric Cranial CT", "pedi_cranial_ct", None, "age, sex, bleed negatives"),
        ]),
        ("MRI · tumour", [
            ("Yale brain metastases", "fomo_yale_mets", None, "4 sequences"),
            ("BraTS glioma", "brats_gli", None, "masks, 4 sequences"),
            ("BraTS meningioma", "brats_men", None, "masks, 4 sequences"),
            ("UPenn-GBM", "upenn_gbm", None, "survival, MGMT"),
            ("BraTS-GoAT", "brats_goat", None, "masks, 4 sequences"),
            ("BraTS pediatric", "brats_ped", None, "masks, 4 sequences"),
            ("UCSF-PDGM", None, ext("ucsf_idh", "ucsf_lgg"), "IDH, grade, MGMT, survival"),
            ("MSD brain tumour", None, ext("msd"), "tumour burden"),
        ]),
        ("MRI · neurodegeneration", [
            ("ADNI", None, ext("adni_dx3", "adni_mmse"), "CN / MCI / AD, MMSE"),
            ("OASIS-2", None, ext("oasis2_cdr"), "CDR, MMSE"),
            ("PPMI", None, ext("ppmi_pd"), "Parkinson's vs control"),
            ("OASIS-1", None, ext("oasis1_cdr"), "CDR, MMSE"),
        ]),
        ("MRI · development and psychiatry", [
            ("FOMO infant", "fomo_infant", None, "0-2 years"),
            ("ID1000", "id1000", None, "adults, demographics"),
            ("PIOP2", "piop2", None, "young adults"),
            ("PIOP1", "piop1", None, "young adults"),
            ("Pixar", None, ext("pixar_age"), "children and adults, age"),
            ("ABIDE I", None, ext("abide_asd"), "autism vs control"),
        ]),
        ("MRI · lesions and small vessels", [
            ("ISLES 2022", None, 249, "infarct masks, segmentation transfer"),
            ("ATLAS", None, ext("atlas_lesion_volume"), "stroke lesion volume, time since stroke"),
            ("WMH challenge", None, 170, "white-matter hyperintensity masks"),
            ("MS3SEG", None, ext("ms3seg_age", "ms3seg_lesion_load"), "MS lesion load, age"),
            ("VALDO", None, ext("valdo_epvs"), "perivascular spaces"),
        ]),
        ("MRI · healthy and clinical", [
            ("FOMO healthy cohorts (19)", FOMO_HEALTHY, None, "pooled: CoRR, SLIM, WAND, M4Raw ..."),
            ("fastMRI+ clinical", "fastmri", None, "radiology findings"),
            ("IXI", None, ext("ixi_age"), "healthy adults, age"),
        ]),
    ]

    rows = []
    for gi, (gname, items) in enumerate(GROUPS):
        for name, fam_, external, note in items:
            tr, te = own(fam_) if fam_ is not None else (0, 0)
            rows.append({"g": gi, "name": name, "train": tr, "test": te, "ext": int(external or 0), "note": note})
    groups = [g for g, _ in GROUPS]
    tot = {"train": sum(r["train"] for r in rows), "test": sum(r["test"] for r in rows),
           "ext": sum(r["ext"] for r in rows), "n_train_sets": sum(r["train"] > 0 for r in rows),
           "n_ext_sets": sum(r["ext"] > 0 for r in rows)}

    return {"groups": groups, "rows": rows, "tot": tot}


ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    data = json.dumps(dataset_data())
    page = (Path(__file__).with_name("datasets_figure.html.tpl").read_text()
            .replace("/*DATA*/null", data))
    Path(sys.argv[1]).write_text(page)
    print(data[-200:])
