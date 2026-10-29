"""Leakage and subject-key tests for v4 index splits. Run: python -m tests.test_index"""

from __future__ import annotations

from pathlib import Path

from brain.prep.index import (
    HELD_OUT,
    SPLIT_RANK,
    all_held_out,
    bhsd_case_ids,
    brats_patient,
    dataset_family,
    dedup_key,
    folder_of,
    ixi_subject,
    merge_split,
    oasis_subject,
    volume_id,
)


def test_brats_timepoints_are_one_person():
    a = "BraTS-GLI-00005-000-t1n"
    b = "BraTS-GLI-00005-100-t1c"
    assert brats_patient(a) == brats_patient(b) == "BraTS-GLI-00005"
    assert dedup_key("brats2023_gli_t1n", a) == dedup_key("brats2024_gli_t1c", b)
    assert brats_patient("BraTS-MEN-RT-0420-1-t1n") == "BraTS-MEN-RT-0420"
    assert dataset_family("brats_goat_t1n") == "brats_goat"
    assert dataset_family("brats2024_men_rt_t1c") == "brats_men_rt"
    assert dedup_key("brats_goat_t1n", "BraTS-GoAT-00739-t1n") == "brats:goat:BraTS-GoAT-00739"


def test_oasis2_sessions_are_one_person():
    assert oasis_subject("OAS2_0001_MR1") == oasis_subject("OAS2_0001_MR2") == "OAS2_0001"
    assert dedup_key("oasis2", "OAS2_0001_MR1") == dedup_key("oasis2", "OAS2_0001_MR2")
    assert dedup_key("oasis1", "OAS1_0001_MR1") != dedup_key("oasis2", "OAS2_0001_MR1")


def test_ixi_t1_t2_are_one_person():
    assert ixi_subject("IXI016-Guys-0697-T1") == "IXI016"
    assert dedup_key("ixi_t1", "IXI016-Guys-0697-T1") == dedup_key("ixi_t2", "IXI016-Guys-0697-T2")


def test_isles_and_wmh_sequences_share_a_subject():
    assert dedup_key("isles22_dwi", "sub-strokecase0002_dwi") == dedup_key("isles22_flair", "sub-strokecase0002_flair")
    assert dedup_key("wmh_flair", "test_Utrecht_14") == dedup_key("wmh_t1", "test_Utrecht_14")
    assert dedup_key("isles24_ncct", "sub-stroke0001") == dedup_key("isles24_cta", "sub-stroke0001")


def test_piop1_and_piop2_do_not_collide():
    assert dedup_key("piop1", "sub-0001") != dedup_key("piop2", "sub-0001")
    assert volume_id("piop1", "sub-0001", "mri") != volume_id("piop2", "sub-0001", "mri")


def test_folder_of_matches_the_writer():
    from brain.prep.sources import Case
    for ds in ("brats2023_gli_t1n", "isles22_flair", "isles22_dwi", "wmh_flair", "wmh_t1", "brainmetshare_flair",
               "brainmetshare_t1_gd", "fomo_hbn_pd", "brats_goat_t2f", "aisd", "rsna_ich"):
        assert folder_of(ds) == Case(ds, "x", "mri", Path("x")).folder, ds
    assert folder_of("brats2023_gli_t1n") == "brats2023_gli"
    assert folder_of("isles22_flair") == "isles22"
    assert folder_of("isles22_dwi") == "isles22_dwi"
    assert folder_of("wmh_flair") == "wmh"
    assert folder_of("wmh_t1") == "wmh_t1"
    assert folder_of("aisd") == "aisd"
    assert folder_of("fomo_hbn_t1w") == "fomo_hbn"
    assert folder_of("fomo_hbn_flair") == "fomo_hbn"
    assert folder_of("fomo_yale_mets_t1c") == "fomo_yale_mets"


def test_held_out_families():
    assert dataset_family("cq500") == dataset_family("seg_cq500") == "cq500"
    assert dataset_family("isles22_dwi") == "isles22"
    assert dataset_family("wmh_flair") == "wmh"
    assert dataset_family("isles24_cta") == "isles24"
    assert dataset_family("instance") == "instance"
    assert dedup_key("instance", "001") == "instance:001"
    assert "instance" in HELD_OUT and "aisd" in HELD_OUT and "brainmetshare" in HELD_OUT
    assert dataset_family("aisd") == "aisd"
    assert dataset_family("brainmetshare_bravo") == dataset_family("brainmetshare_flair") == "brainmetshare"
    assert dedup_key("brainmetshare_bravo", "Mets_001") == dedup_key("brainmetshare_flair", "Mets_001")
    assert dataset_family("brats2023_gli_t1n") == dataset_family("brats2024_gli_t2f") == "brats_gli"
    assert dataset_family("fomo_hbn_t1w") == dataset_family("fomo_hbn_flair") == "fomo_hbn"
    assert dedup_key("fomo_hbn_t1w", "sub-01_ses-HBNsiteRU_t1w") == \
        dedup_key("fomo_hbn_flair", "sub-01_ses-HBNsiteRU_flair") == "fomo_hbn:sub-01"
    assert dedup_key("fomo_yale_mets_t1c", "sub-a_ses-02_t1c") == \
        dedup_key("fomo_yale_mets_t1w", "sub-a_ses-01_t1w") == "fomo_yale_mets:sub-a"
    assert dataset_family("fomo_openneuro_ds000228_t1w") == "fomo_openneuro_ds000228"
    assert "cq500" in all_held_out()
    assert "msd_braintumor" in HELD_OUT
    assert dataset_family("msd_braintumor") == "msd_braintumor"
    assert volume_id("msd_braintumor", "BRATS_001", "mri") == "msd_braintumor:BRATS_001"
    assert dedup_key("msd_braintumor", "BRATS_001") == "msd_braintumor:BRATS_001"


def test_inherit_current_splits_keeps_existing_and_assigns_new():
    import pandas as pd
    from brain.prep.index import inherit_current_splits

    old = pd.DataFrame({"dedup_key": ["ixi:a", "abide1:b"], "split": ["train", "val"]})
    vols = pd.DataFrame({
        "dedup_key": ["ixi:a", "abide1:b", "instance:001", "instance:002", "instance:003"],
        "modality": ["mri", "mri", "ct", "ct", "ct"],
        "dataset_family": ["ixi", "abide1", "instance", "instance", "instance"],
        "case_id": ["a", "b", "001", "002", "003"],
    })
    out = inherit_current_splits(vols, old)
    assert out.loc[out.dedup_key == "ixi:a", "split"].iloc[0] == "train"
    assert out.loc[out.dedup_key == "abide1:b", "split"].iloc[0] == "val"
    assert set(out.loc[out.dataset_family == "instance", "split"]) <= {"train", "val", "test"}


def test_aisd_official_test_ids_are_pinned():
    import pandas as pd
    from brain.datasets.download_aisd import splits_for
    from brain.prep.index import pin_official_splits
    from brain.prep.sources import AISD_TEST_IDS, aisd_id

    ids = [aisd_id(x) for x in AISD_TEST_IDS[:3]] + ["0019983", "0021023"]
    vols = pd.DataFrame({
        "dataset_family": ["aisd"] * len(ids),
        "case_id": ids,
        "dedup_key": [f"aisd:{i}" for i in ids],
        "split": ["train"] * len(ids),
    })
    out = pin_official_splits(vols)
    pinned = splits_for(ids)
    assert list(out["split"]) == [pinned[i] for i in ids]
    assert (out.loc[out.case_id.isin([aisd_id(x) for x in AISD_TEST_IDS[:3]]), "split"] == "test").all()


def test_merge_split_prefers_test():
    assert merge_split({"train", "test"}) == "test"
    assert merge_split({"train", "val"}) == "val"
    assert SPLIT_RANK["test"] < SPLIT_RANK["val"] < SPLIT_RANK["train"]


def test_bhsd_ids_are_rsna_case_ids():
    ids = bhsd_case_ids()
    if not ids:
        return
    assert len(ids) == 192
    assert all(i.startswith("RSNA-ID_") and len(i) == 18 for i in ids)


def test_rsna_volume_id_matches_v3_label_table():
    assert volume_id("rsna_ich", "RSNA-ID_000421322f", "ct") == "ct:RSNA-ID_000421322f"
    assert volume_id("cq500", "CQ500-CT-0", "ct") == "ct:CQ500-CT-0"
    assert volume_id("oasis1", "OAS1_0001_MR1", "mri") == "mri:OAS1_0001_MR1"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
