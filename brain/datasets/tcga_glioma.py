"""TCGA-GBM / TCGA-LGG patient clinical data (cBioPortal PanCancer Atlas studies, public) for the BraTS 2023 GLI cases
that came from TCGA (``BraTS2023_2017_GLI_Mapping.xlsx``: BraTS id -> TCGA patient id).

Writes ``labels/sources/tcga_glioma/clinical.tsv``: one row per mapped BraTS case with the TCGA patient id, cancer type
(GBM / LGG), overall survival (months, deceased flag), IDH / 1p19q subtype and age.

    python -m brain.datasets.tcga_glioma
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile

import pandas as pd

from brain.common.paths import DATASETS
from brain.downstream.tasks import LABEL_SOURCES

STUDIES = ("gbm_tcga_pan_can_atlas_2018", "lgg_tcga_pan_can_atlas_2018")
API = "https://www.cbioportal.org/api/studies/{}/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY&pageSize=100000"
KEEP = ["CANCER_TYPE_ACRONYM", "OS_MONTHS", "OS_STATUS", "SUBTYPE", "AGE", "SEX"]


def clinical() -> pd.DataFrame:
    parts = []
    for s in STUDIES:
        with urllib.request.urlopen(API.format(s), timeout=120) as r:
            d = pd.DataFrame(json.load(r))
        parts.append(d.pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first"))
    c = pd.concat(parts)[KEEP].reset_index().rename(columns={"patientId": "tcga_id"})
    c["os_months"] = pd.to_numeric(c.pop("OS_MONTHS"), errors="coerce")
    c["deceased"] = c.pop("OS_STATUS").astype(str).str.startswith("1").astype(int)
    sub = c.pop("SUBTYPE").astype(str)
    c["idh_mutant"] = sub.str.contains("IDHmut").astype(float).where(sub.str.contains("IDH"))
    c["codeleted"] = sub.str.contains("codel") & ~sub.str.contains("non-codel")
    c["codeleted"] = c["codeleted"].astype(float).where(sub.str.contains("IDH"))
    return c.rename(columns={"CANCER_TYPE_ACRONYM": "cancer_type", "AGE": "age", "SEX": "sex"})


def mapping() -> pd.DataFrame:
    z = zipfile.ZipFile(DATASETS / "brats" / "brats.raw.zip")
    m = pd.read_excel(io.BytesIO(z.read("2023_GLI/BraTS2023_2017_GLI_Mapping.xlsx")))
    m = m.rename(columns={"BraTS2023": "brats_id", "Local ID ": "tcga_id"})
    return m[m["tcga_id"].astype(str).str.startswith("TCGA")][["brats_id", "tcga_id"]]


def main() -> None:
    out = mapping().merge(clinical(), on="tcga_id", how="inner")
    path = LABEL_SOURCES / "tcga_glioma" / "clinical.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep="\t", index=False)
    print(f"[tcga_glioma] {len(out)} BraTS cases -> {path}")
    print(out[["cancer_type", "deceased", "idh_mutant", "codeleted"]].describe(include="all").loc[["count", "mean"]])


if __name__ == "__main__":
    main()
