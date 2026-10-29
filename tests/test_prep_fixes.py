"""2026-10-25 preprocessing fixes: N4 rules and the store-level pass, infant templates, prep-time dedup, RSNA
re-keying. python tests/run.py tests/test_prep_fixes.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import SimpleITK as sitk

from brain.prep import registration
from brain.prep.intensity import MRI_FLOOR
from brain.prep.n4_store import correct
from brain.prep.pipeline import needs_n4


def test_n4_rule_covers_structural_mri_but_not_diffusion_or_ct():
    for ds in ("brats2023_gli_t1n", "brats_goat_t2f", "msd_braintumor", "valdo_t2s", "ms3seg_flair", "fomo_hbn_t1w",
               "brainmetshare_bravo", "ixi_t1", "rsna_ica_mra", "rsna_ica_t1c"):
        assert needs_n4(ds), ds
    for ds in ("isles22_dwi", "isles22_adc", "oasis1", "rsna_ich", "cq500", "rsna_ica_cta"):
        assert not needs_n4(ds), ds


def test_store_n4_removes_a_multiplicative_field_and_restandardises():
    rng = np.random.default_rng(0)
    zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, s) for s in (80, 96, 96)], indexing="ij")   # N4 needs room after 4x shrink
    head = zz ** 2 + yy ** 2 + xx ** 2 < 0.8
    brain = zz ** 2 + yy ** 2 + xx ** 2 < 0.5
    v = np.where(head, 300.0, 0.0) * np.exp(0.3 * xx) + rng.normal(0, 3, head.shape)   # raw with a smooth gain
    med, mad = float(np.median(v[brain])), float(1.4826 * np.median(np.abs(v[brain] - np.median(v[brain]))))
    z = np.where(head, (v - med) / mad, MRI_FLOOR).astype(np.float16)
    out, st = correct(z, brain, med, mad)
    x = out[brain].astype(np.float32)
    assert abs(np.median(x)) < 1e-2 and abs(1.4826 * np.median(np.abs(x - np.median(x))) - 1) < 1e-2
    left, right = out[brain & (xx < -0.3)].astype(np.float32), out[brain & (xx > 0.3)].astype(np.float32)
    zl, zr = z[brain & (xx < -0.3)].astype(np.float32), z[brain & (xx > 0.3)].astype(np.float32)
    assert abs(np.median(right) - np.median(left)) < 0.3 * abs(np.median(zr) - np.median(zl))   # the gain is gone
    assert (out[~head] == MRI_FLOOR).all() and st["n4"] == "store"


def test_nihpd_bins_and_affine_composition():
    assert registration.nihpd_bin(0.5) == "00-02" and registration.nihpd_bin(12.0) == "11-14"
    assert registration.nihpd_bin(35.0) == "33-44" and registration.nihpd_bin(80.0) == "44-60"
    a, b = sitk.AffineTransform(3), sitk.AffineTransform(3)
    a.SetMatrix((1.2, 0, 0, 0, 0.9, 0.1, 0, 0, 1.0))
    a.SetTranslation((3, -2, 1))
    b.SetMatrix((0.7, 0, 0, 0, 0.7, 0, 0, 0, 0.7))
    b.SetCenter((10, 5, -4))
    b.SetTranslation((1, 1, 1))
    c = registration.compose(a, b)
    for p in ((0, 0, 0), (12.5, -7, 30)):
        assert np.allclose(c.TransformPoint(p), a.TransformPoint(b.TransformPoint(p)), atol=1e-6)


def test_dedup_keeps_the_original_labelled_training_copy():
    from brain.prep.dedup import keep_rank

    gli = {"volume_id": "mri:BraTS-GLI-1-t1n", "dataset_family": "brats_gli", "has_mask": True, "held_out": False}
    goat = {"volume_id": "mri:BraTS-GoAT-9-t1n", "dataset_family": "brats_goat", "has_mask": False, "held_out": False}
    msd = {"volume_id": "msd_braintumor:BRATS_1", "dataset_family": "msd_braintumor", "has_mask": True, "held_out": True}
    fomo = {"volume_id": "fomo_x:sub-1", "dataset_family": "fomo_x", "has_mask": False, "held_out": False}
    ixi = {"volume_id": "ixi:sub-1", "dataset_family": "ixi", "has_mask": False, "held_out": False}
    assert keep_rank(gli) > keep_rank(goat) and keep_rank(gli) > keep_rank(msd) and keep_rank(ixi) > keep_rank(fomo)


def test_rekeyed_subjects_keep_their_majority_split():
    from brain.prep.index import inherit_current_splits

    old = pd.DataFrame({"volume_id": ["a", "b", "c", "d"], "dedup_key": ["s:a", "s:b", "s:c", "s:d"],
                        "split": ["train", "test", "train", "test"]})
    new = pd.DataFrame({"volume_id": ["a", "b", "c", "d"], "dedup_key": ["p:1", "p:1", "p:1", "s:d"],
                        "modality": "ct", "dataset_family": "rsna_ich"})
    out = inherit_current_splits(new, old).set_index("volume_id")["split"]
    assert (out[["a", "b", "c"]] == "train").all() and out["d"] == "test"     # one patient, one split


def test_prep_skips_and_removes_listed_duplicates(tmp_path):
    from brain.prep import store
    from brain.prep.run import skip_hard_drops
    from brain.prep.sources import Case

    (tmp_path / "labels" / "pool_drop").mkdir(parents=True)
    pd.DataFrame({"volume_id": ["mri:BraTS-GoAT-00001-t1n"], "reason": ["duplicate"]}).to_csv(
        tmp_path / "labels" / "pool_drop" / "duplicates.csv", index=False)
    dup, keep = Case("brats_goat_t1n", "BraTS-GoAT-00001-t1n", "mri"), Case("brats_goat_t1n", "BraTS-GoAT-00002-t1n", "mri")
    img = store.paths(tmp_path, dup)[0]
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"x")
    todo, dropped = skip_hard_drops(tmp_path, [dup, keep])
    assert todo == [keep] and dropped == [dup] and not img.exists()


def test_co_registered_sequences_reuse_their_references_fit(tmp_path):
    """2026-10-29: a sequence with a stored reference takes that reference's transform (one fit per subject), and the
    runner writes references first."""
    from pathlib import Path

    from brain.prep import native, store
    from brain.prep.pipeline import shared_registration
    from brain.prep.run import phases, with_ref_images
    from brain.prep.sources import Case

    tf = sitk.AffineTransform(3)
    tf.SetMatrix((0.9, 0.1, 0.0, -0.1, 0.92, 0.0, 0.0, 0.0, 0.95))
    tf.SetTranslation((3.0, -2.0, 7.0))
    t1 = Case("ucsf_pdgm_t1n", "S-1_t1n", "mri", Path("/d/S-1_T1.nii.gz"))
    t1c = Case("ucsf_pdgm_t1c", "S-1_t1c", "mri", Path("/d/S-1_T1c.nii.gz"), ref=t1.path)
    lone = Case("ixi_t1", "IXI1", "mri", Path("/d/IXI1.nii.gz"))
    todo = with_ref_images([t1c, lone], [t1, t1c, lone], tmp_path)    # the reference need not be in this run
    assert todo[0].ref_image == store.paths(tmp_path, t1)[0] and todo[1].ref_image is None
    assert [[c.case_id for c in p] for p in phases(todo)] == [["IXI1"], ["S-1_t1c"]]

    assert shared_registration(todo[0]) is None                      # reference not written yet: fit as before
    attrs = {"reg_affine": native._affine4(tf), "reg_method": "affine", "reg_scale": 0.92, "reg_mi": -0.6}
    store.write_arrays(todo[0].ref_image, {"image": (np.zeros((4, 1, 4, 4), np.float16), attrs)})
    got, info = shared_registration(todo[0])
    assert np.allclose(native._affine4(got), attrs["reg_affine"])
    assert info == {"reg_method": "affine", "reg_scale": 0.92, "reg_mi": -0.6, "reg_shared": t1.path.name}
    p = (12.0, -5.0, 30.0)
    assert np.allclose(got.TransformPoint(p), tf.TransformPoint(p))


def test_ct_with_stored_hu_and_a_stale_rescale_header_is_undone():
    from brain.prep.dicom import stored_is_hu

    rng = np.random.default_rng(0)
    hu = np.concatenate([np.full(5000, -1000.0), rng.normal(35, 10, 2000), np.full(1000, 700.0)])  # air, brain, bone
    assert not stored_is_hu(hu, 1.0, -1024.0)                    # a correct GE header: tissue is already in range
    assert stored_is_hu(hu - 1024, 1.0, -1024.0)                 # pixels already HU, header still shifts by -1024
    assert stored_is_hu(hu * 10 - 10240, 10.0, -10240.0)         # the Siemens slope 10 / intercept -10240 case
    assert not stored_is_hu(hu - 1024, 1.0, 0.0)                 # identity tags: nothing to undo


def test_old_brats_numbering_is_stored_as_brats_2023():
    from brain.prep.pipeline import remap_seg

    m = np.array([0, 1, 2, 4], np.uint8)
    assert remap_seg("ucsf_pdgm_t1n", "seg", m).tolist() == [0, 1, 2, 3]
    assert remap_seg("upenn_gbm_t1n", "seg", m).tolist() == [0, 1, 2, 3]
    assert remap_seg("brats2024_gli_t1n", "seg", m).tolist() == [0, 1, 2, 4]   # 4 is the resection cavity there
    assert remap_seg("ucsf_pdgm_t1n", "brain", m).tolist() == [0, 1, 2, 4]
