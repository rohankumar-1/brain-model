"""Phantom tests for v4 preprocessing. Run: python -m tests.test_prep (pytest also works)."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from brain.prep import centering, geometry, intensity, pipeline
from brain.prep.sources import Case

RNG = np.random.default_rng(0)


def ct_phantom(offset_mm=(40.0, -30.0), spacing=(0.45, 0.45, 5.0), n=(512, 512, 40), shift=0.0, artifacts=True, hz=0.0):
    """Ellipsoid head (scalp, skull, brain) in air, with table, wires, FOV ring and padding."""
    sx, sy, sz = spacing
    x = (np.arange(n[0]) - (n[0] - 1) / 2) * sx
    y = (np.arange(n[1]) - (n[1] - 1) / 2) * sy
    z = (np.arange(n[2]) - (n[2] - 1) / 2) * sz
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    hx, hy = offset_mm
    r = np.sqrt(((X - hx) / 75) ** 2 + ((Y - hy) / 95) ** 2 + ((Z - hz) / 80) ** 2)
    vol = np.full(Z.shape, -1000.0, np.float32)
    vol[r < 1.0] = 40.0                           # scalp
    vol[r < 0.92] = 1000.0                        # skull
    vol[r < 0.86] = 35.0                          # brain
    if artifacts:
        vol[(Y > hy + 95) & (Y < hy + 115) & (np.abs(X - hx) < 150)] = 120.0          # table touching the head
        vol[(np.abs(X - hx - 0.6 * (Y - hy)) < 4) & (Y > hy + 90)] = 60.0               # wire to the FOV edge
        rad = np.hypot(X, Y)
        vol[(rad > 200) & (rad < 204)] = 300.0                                          # ring artifact
        vol[rad > 225] = -3024.0                                                        # FOV padding
    vol += RNG.normal(0, 15, vol.shape).astype(np.float32) + shift
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing(spacing)
    img.SetOrigin((-(n[0] - 1) / 2 * sx, -(n[1] - 1) / 2 * sy, -(n[2] - 1) / 2 * sz))
    return img


def estimated_xy(img, modality="ct"):
    small = geometry.coarse(img)
    (cx, cy, cz), _, info = centering.estimate(sitk.GetArrayFromImage(small), modality, geometry.COARSE_MM)
    return np.array(small.TransformContinuousIndexToPhysicalPoint((cx, cy, cz))[:2]), info


def test_ct_center_with_artifacts():
    for off in [(40.0, -30.0), (-55.0, 25.0), (0.0, 0.0)]:
        img = intensity.clamp_ct(ct_phantom(off))
        xy, info = estimated_xy(img)
        err = np.hypot(*(xy - np.array(off)))
        assert err < 8.0, (off, xy, info, err)
        assert info["center_method"] == "brain", info


def test_ct_head_fallback_when_brain_missing():
    img = intensity.clamp_ct(ct_phantom((30.0, 10.0)))
    a = sitk.GetArrayFromImage(img)
    a[(a > -10) & (a < 150)] = 400.0            # no voxels in the brain window
    img2 = sitk.GetImageFromArray(a)
    img2.CopyInformation(img)
    xy, info = estimated_xy(img2)
    assert info["center_method"] in ("head", "head_disagree"), info
    assert np.hypot(*(xy - np.array([30.0, 10.0]))) < 12.0, (xy, info)


def test_target_grid_is_centered_and_contains_head():
    off = (40.0, -30.0)
    img = intensity.clamp_ct(ct_phantom(off))
    xy, _ = estimated_xy(img)
    g = geometry.target_grid(img, xy)
    raw = geometry.resample(img, g, -1024.0)
    assert raw.shape == (128, 256, 256)
    ys, xs = np.nonzero(raw[64] > 800)              # the skull shell (table, wire and ring are below 800 HU)
    assert abs((xs.min() + xs.max()) / 2 - 127.5) < 8 and abs((ys.min() + ys.max()) / 2 - 127.5) < 8
    assert raw.min() >= -1024.0 - 1e-3          # padding never leaks below the floor


def test_ct_offset_repair_and_clamp():
    shifted = ct_phantom(artifacts=False, shift=1024.0)
    fixed, repaired = intensity.fix_ct_offset(shifted)
    assert repaired and abs(float(np.median(sitk.GetArrayFromImage(fixed)[20, 256 - 60:256 + 60, 256 - 40:256 + 40])) - 35) < 25
    normal, repaired = intensity.fix_ct_offset(ct_phantom())
    assert not repaired
    clamped = sitk.GetArrayFromImage(intensity.clamp_ct(ct_phantom()))
    assert clamped.min() >= -1024.0 and clamped.max() <= 3071.0


def mri_phantom(scale=1.0, offset=0.0, skull=True, noise=True):
    n = 96
    a = np.arange(n) - (n - 1) / 2
    Z, Y, X = np.meshgrid(a, a, a, indexing="ij")
    r = np.sqrt((X / 32) ** 2 + (Y / 40) ** 2 + (Z / 34) ** 2)
    v = np.zeros((n, n, n), np.float32)
    if skull:
        v[r < 1.25] = 60.0 * 1.0                   # scalp / fat
        v[r < 1.05] = 10.0                         # skull (dark on T1)
    v[r < 0.95] = 120.0                            # brain
    v[r < 0.35] = 40.0                             # ventricles (dark): a hole to fill
    if noise:
        v += np.abs(RNG.normal(0, 4, v.shape)).astype(np.float32)
    return v * scale + offset


def test_mri_normalization_is_scale_and_offset_invariant():
    base, _ = intensity.mri_native(mri_phantom(noise=False))
    for scale, off in [(3.0, 0.0), (0.2, 50.0)]:
        other, _ = intensity.mri_native(mri_phantom(scale, off, noise=False))
        fg = base.astype(np.float32) > -5.5
        assert np.abs(base.astype(np.float32)[fg] - other.astype(np.float32)[fg]).max() < 0.05
    assert base.astype(np.float32).min() == intensity.MRI_FLOOR


def test_mri_foreground_fills_ventricles_and_ignores_background_noise():
    v = mri_phantom()
    m = intensity.mri_mask(v)
    assert m[48, 48, 48] and not m[0, 0, 0] and 0.05 < m.mean() < 0.6


def test_mri_center_skull_on_and_off():
    for skull in (True, False):
        v = mri_phantom(skull=skull)
        (cx, cy, cz), _, info = centering.estimate(v, "mri", 4.0)
        assert np.hypot(cx - 47.5, cy - 47.5) * 4.0 < 10.0, (skull, cx, cy, info)


def test_pipeline_end_to_end_nifti(tmp_dir=None):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "CT.nii.gz"
        sitk.WriteImage(ct_phantom((40.0, -30.0), spacing=(0.5, 0.5, 2.0), n=(400, 400, 100)), str(path))
        res = pipeline.process(Case("cq500", "phantom", "ct", path, (("brain", path),)))
    assert res.vol.shape == (128, 256, 256) and res.vol.dtype == np.float16
    assert res.masks["brain"].shape == (128, 256, 256)
    assert res.info["center_method"] == "brain" and res.info["shift_mm"] > 30


def test_z_anchor_covers_brain_when_head_is_off_center_in_z():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "CT.nii.gz"
        sitk.WriteImage(ct_phantom((0.0, 0.0), spacing=(0.6, 0.6, 3.0), n=(400, 400, 100), hz=60.0), str(path))
        case = Case("cq500", "phantom_z", "ct", path)
        vol_res = pipeline.process(case, z_anchor="volume")
        brain_res = pipeline.process(case, z_anchor="brain")
    assert vol_res.info["brain_cover"] < 0.8, vol_res.info
    assert brain_res.info["brain_cover"] > 0.88, brain_res.info
    assert brain_res.info["fg_frac"] > vol_res.info["fg_frac"] * 0.9


def test_coregistered_sequences_share_one_grid():
    """A sequence with signal dropout would center differently alone; with the reference both use one grid."""
    import tempfile
    from pathlib import Path

    def write(v, path):
        img = sitk.GetImageFromArray(v)
        img.SetSpacing((1.0, 1.0, 1.0))
        img.SetOrigin((-30.0, -50.0, -40.0))
        sitk.WriteImage(img, str(path))

    a = mri_phantom(skull=False)
    b = a.copy()
    b[:, :, 60:] = 0.0                                     # one side lost: the foreground (and its center) moves
    with tempfile.TemporaryDirectory() as d:
        pa, pb = Path(d) / "t1n.nii.gz", Path(d) / "t2f.nii.gz"
        write(a, pa), write(b, pb)
        ref = pipeline.process(Case("brats_t1n", "s-t1n", "mri", pa), keep_debug=True)
        alone = pipeline.process(Case("brats_t2f", "s-t2f", "mri", pb), keep_debug=True)
        shared = pipeline.process(Case("brats_t2f", "s-t2f", "mri", pb, ref=pa), keep_debug=True)
    assert np.abs(np.array(alone.debug["grid"].origin) - ref.debug["grid"].origin).max() > 1.0
    assert np.allclose(shared.debug["grid"].origin, ref.debug["grid"].origin)


def test_instance_pairs_data_and_label_folders():
    import tempfile
    from pathlib import Path
    from brain.prep.sources import instance

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "data").mkdir()
        (root / "label").mkdir()
        (root / "data" / "001.nii.gz").write_bytes(b"x")
        (root / "label" / "001.nii.gz").write_bytes(b"y")
        (root / "data" / "002.nii.gz").write_bytes(b"x")
        (root / "label" / "002_label.nii.gz").write_bytes(b"y")
        cases = instance(root)
    assert [c.case_id for c in cases] == ["001", "002"]
    assert all(c.dataset == "instance" and c.modality == "ct" and c.masks[0][0] == "ICH" for c in cases)


def test_msd_source_lists_labeled_train_cases():
    import tempfile
    from pathlib import Path
    from brain.prep.sources import msd

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "imagesTr").mkdir()
        (root / "labelsTr").mkdir()
        (root / "imagesTr" / "BRATS_001.nii.gz").write_bytes(b"x")
        (root / "labelsTr" / "BRATS_001.nii.gz").write_bytes(b"y")
        (root / "imagesTr" / "._BRATS_002.nii.gz").write_bytes(b"skip")
        cases = msd(root)
    assert len(cases) == 1
    assert cases[0].dataset == "msd_braintumor" and cases[0].case_id == "BRATS_001"
    assert cases[0].modality == "mri" and cases[0].masks[0][0] == "seg"


def test_sinoct_source_groups_reconstructed_series(tmp_path=None):
    import tempfile
    from pathlib import Path
    from brain.prep.sources import sinoct

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        keep = root / "batch_1" / "series_9" / "reconstructed_image"
        skip = root / "batch_1" / "series_10" / "reconstructed_image"
        keep.mkdir(parents=True)
        skip.mkdir(parents=True)
        for i in range(12):
            (keep / f"image_{i}.dcm").write_bytes(b"dicom")
        for i in range(3):
            (skip / f"image_{i}.dcm").write_bytes(b"dicom")
        cases = sinoct(root)
        assert len(cases) == 1
        assert cases[0].dataset == "sinoct" and cases[0].modality == "ct"
        assert cases[0].case_id == "batch_1_series_9"
        assert cases[0].path == keep


def test_sinoct_pixel_fingerprint_ignores_fov_pad_and_matches():
    import pandas as pd
    from brain.datasets.sinoct_overlap import canon_plane, fingerprints_from_hu, score_probe

    brain = np.full((512, 512), -1024, np.int16)
    brain[80:432, 80:432] = np.arange(352 * 352, dtype=np.int16).reshape(352, 352) % 80 + 20
    padded = brain.copy()
    padded[:12] = -3024
    padded[-12:] = -3024
    a = fingerprints_from_hu(brain)
    b = fingerprints_from_hu(padded)
    assert a["exact"] == b["exact"] and a["coarse"] == b["coarse"] and a["ahash"] == b["ahash"]
    other_img = np.full((512, 512), -1024, np.int16)
    other_img[80:432, 80:432] = 200
    other = fingerprints_from_hu(other_img)
    assert other["exact"] != a["exact"]
    assert canon_plane(brain).shape == (384, 384)

    fp = pd.DataFrame([{"study_id": "ID_rsna", "member": "a.dcm", **a}])
    hit = pd.DataFrame([{"series": "series_1", "batch": "batch_1", **b}])
    miss = pd.DataFrame([{"series": "series_2", "batch": "batch_1", **other}])
    scores = score_probe(pd.concat([hit, miss], ignore_index=True), fp, hamming=8)
    by = scores.set_index("series")
    assert bool(by.loc["series_1", "drop"]) and by.loc["series_1", "reason"] == "exact"
    assert not bool(by.loc["series_2", "drop"])


def test_sinoct_pull_skips_dropped_series():
    import pandas as pd
    from brain.datasets.sinoct_overlap import select_pull_files

    recon = pd.DataFrame({
        "series": ["series_1", "series_1", "series_2", "series_3"],
        "rel": ["a.dcm", "b.dcm", "c.dcm", "d.dcm"],
        "size": [1, 1, 1, 1],
    })
    got = select_pull_files(recon, keep={"series_1", "series_2", "series_3"}, drop={"series_2"})
    assert set(got["series"]) == {"series_1", "series_3"}
    assert "c.dcm" not in set(got["rel"])


def test_sinoct_drops_sinograms_and_rsna_ids():
    import pandas as pd
    from brain.datasets.download_sinoct import classify, is_sinogram, normalize_id

    assert is_sinogram("Data/sinograms/0073410.npy")
    assert not is_sinogram("Data/0073410/ct.dcm")
    block = {normalize_id("ID_1f6d1e8aeb"), normalize_id("ID_000000e27")}
    files = pd.DataFrame({
        "file_id": ["a", "b", "c", "d"],
        "file_name": [
            "Data/ID_1f6d1e8aeb/slice.dcm",
            "Data/keepme/slice.dcm",
            "Data/sinograms/keepme.npy",
            "Data/other/ID_000000e27.dcm",
        ],
        "size": [1, 1, 1, 1],
    })
    labels = pd.DataFrame({"patient_id": ["ID_1f6d1e8aeb", "keepme", "other"], "label": ["abnormal", "normal", "normal"]})
    out = classify(files, labels, block)
    keep = set(out.loc[out.keep, "file_name"])
    assert keep == {"Data/keepme/slice.dcm"}
    assert set(out.loc[out.drop_reason == "rsna", "patient_id"]) >= {"ID_1f6d1e8aeb", "other"}
    assert (out.kind == "sinogram").sum() == 1


def test_brainmetshare_pairs_sequences_and_seg_on_bravo():
    import tempfile
    from pathlib import Path
    from brain.prep.sources import brainmetshare

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train = root / "train" / "Mets_001"
        test = root / "test" / "Mets_002"
        train.mkdir(parents=True)
        test.mkdir(parents=True)
        for name in ("bravo", "t1_pre", "t1_gd", "flair"):
            (train / f"{name}.nii.gz").write_bytes(b"x")
            (test / f"{name}.nii.gz").write_bytes(b"x")
        (train / "seg.nii.gz").write_bytes(b"y")
        cases = brainmetshare(root)
    tags = {(c.dataset, c.case_id, bool(c.masks), c.ref is not None) for c in cases}
    assert ("brainmetshare_bravo", "Mets_001", True, False) in tags
    assert ("brainmetshare_flair", "Mets_001", False, True) in tags
    assert ("brainmetshare_bravo", "Mets_002", False, False) in tags
    assert all(c.modality == "mri" for c in cases)
    assert len(cases) == 8


def test_aisd_id_pads_numeric_manifest_stems():
    from brain.prep.sources import aisd_id
    assert aisd_id(72762) == "0072762" and aisd_id("73410") == "0073410"


def test_aisd_pairs_image_and_mask_and_pads_ids():
    import tempfile
    from pathlib import Path
    from brain.prep.sources import aisd

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "image").mkdir()
        (root / "mask").mkdir()
        (root / "image" / "73410.nii.gz").write_bytes(b"x")
        (root / "mask" / "0073410.nii.gz").write_bytes(b"y")
        cases = aisd(root)
    assert len(cases) == 1 and cases[0].case_id == "0073410" and cases[0].masks[0][0] == "lesion"


def test_aisd_png_to_hu_keeps_ventricles_and_air():
    from brain.datasets.download_aisd import png_to_hu

    vol = np.zeros((5, 32, 32), np.uint8)
    vol[:, 8:24, 8:24] = 80
    vol[:, 12:20, 12:20] = 0
    hu = png_to_hu(vol)
    assert hu[2, 0, 0] == -1024.0
    assert abs(float(hu[2, 16, 16])) < 1e-3
    assert 20 < float(hu[2, 9, 9]) < 30


def test_aisd_png_stack_writes_nifti_and_pairs():
    import tempfile
    from pathlib import Path
    from brain.datasets.download_aisd import PNG_SPACING_XYZ, pngs_to_nifti
    from brain.prep.sources import aisd

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for kind in ("image", "mask"):
            d = root / kind / "73410"
            d.mkdir(parents=True)
            for z in range(4):
                a = np.zeros((16, 16), np.uint8)
                a[4:12, 4:12] = 80 if kind == "image" else 3
                a[6:10, 6:10] = 0
                sitk.WriteImage(sitk.GetImageFromArray(a), str(d / f"{z:03d}.png"))
        pngs_to_nifti(root, workers=1)
        cases = aisd(root)
        assert len(cases) == 1 and cases[0].case_id == "0073410" and cases[0].masks[0][0] == "lesion"
        img = sitk.ReadImage(str(cases[0].path))
        assert img.GetSize() == (16, 16, 4)
        assert img.GetSpacing() == PNG_SPACING_XYZ
        arr = sitk.GetArrayFromImage(img)
        assert arr[1, 0, 0] == -1024.0
        assert arr[1, 5, 5] > 20


def test_fomo_ingest_zip_keeps_structural_and_deletes_archive():
    import zipfile
    import tempfile
    from pathlib import Path
    from brain.datasets.download_fomo import already_extracted, ingest_zip

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        zp = root / "PT018_HBN" / "sub-01" / "ses-01.zip"
        zp.parent.mkdir(parents=True)
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("anat/sub-01_ses-01_T1w.nii.gz", b"t1")
            zf.writestr("dwi/sub-01_ses-01_dwi.nii.gz", b"dwi")
        n_ok, n_skip = ingest_zip(zp, root, delete=True)
        out = root / "extracted" / "PT018_HBN" / "sub-01" / "ses-01"
        assert n_ok == 1 and n_skip == 0 and not zp.exists()
        assert (out / "sub-01_ses-01_T1w.nii.gz").read_bytes() == b"t1"
        assert not (out / "sub-01_ses-01_dwi.nii.gz").exists()
        assert already_extracted(root, "PT018_HBN/sub-01/ses-01.zip")


def test_fomo_mapping_holds_out_entire_openneuro_ds():
    import pandas as pd
    from brain.prep.fomo import holdout_from_mapping, is_structural, match_hold_family

    assert is_structural("sub-01_ses-01_T1w.nii.gz") and is_structural("t2f.nii.gz")
    assert not is_structural("sub-01_dwi.nii.gz") and not is_structural("adc_trace.nii.gz")
    assert match_hold_family("Yale_Brain_Mets_Longitudinal/sub-01/t1w.nii.gz") is None
    assert match_hold_family("stanford/brainmetshare/Mets_001/bravo.nii.gz") == "brainmetshare"
    df = pd.DataFrame({
        "old_path": [
            "ds004351/ISLES-2022/sub-strokecase0001/ses-0001/dwi/foo.nii.gz",
            "ds000228/sub-01/anat/sub-01_T1w.nii.gz",
            "HBN/sub-01/anat/sub-01_T1w.nii.gz",
        ],
        "old_filename": ["foo.nii.gz", "sub-01_T1w.nii.gz", "sub-01_T1w.nii.gz"],
        "new_path": [
            "PT030_OpenNeuro/sub-1/ses-01/dwi.nii.gz",
            "PT030_OpenNeuro/sub-2/ses-01/t1w.nii.gz",
            "PT018_HBN/sub-01/ses-01/t1w.nii.gz",
        ],
        "new_filename": ["dwi.nii.gz", "t1w.nii.gz", "t1w.nii.gz"],
        "mapping_pt": ["PT030_OpenNeuro", "PT030_OpenNeuro", "PT018_HBN"],
    })
    hits = holdout_from_mapping(df)
    assert set(hits["fomo_family"]) == {"fomo_openneuro_ds004351"}
    assert set(hits["held_out"]) == {"isles22"}


def test_fomo_brats_restores_ids_and_joins_local_seg():
    import tempfile
    from pathlib import Path
    from brain.prep.sources import brats_fomo

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fomo = tmp / "fomo"
        local = tmp / "brats" / "extracted"
        dest = fomo / "extracted" / "PT009_BraTS-GEN" / "sub-0001" / "ses-01"
        dest.mkdir(parents=True)
        (dest / "T1w.nii.gz").write_bytes(b"x")
        (dest / "T1c.nii.gz").write_bytes(b"x")
        gli = local / "2023_GLI" / "BraTS-GLI-00005-000"
        gli.mkdir(parents=True)
        (gli / "BraTS-GLI-00005-000-seg.nii.gz").write_bytes(b"y")
        (gli / "BraTS-GLI-00005-000-t1n.nii.gz").write_bytes(b"z")
        mapping = fomo / "PT009_BraTS-GEN" / "mapping.tsv"
        mapping.parent.mkdir(parents=True)
        mapping.write_text(
            "old_path\tnew_path\told_filename\tnew_filename\tparticipant_id\tsession_id\tmodality\n"
            "BraTS2023/BraTS-GLI-00005-000/BraTS-GLI-00005-000-t1n.nii.gz\t"
            "sub-0001/ses-01/anat/sub-0001_ses-01_T1w.nii.gz\t"
            "BraTS-GLI-00005-000-t1n.nii.gz\tsub-0001_ses-01_T1w.nii.gz\tsub-0001\tses-01\tanat\n"
            "BraTS2023/BraTS-GLI-00005-000/BraTS-GLI-00005-000-t1c.nii.gz\t"
            "sub-0001/ses-01/anat/sub-0001_ses-01_T1c.nii.gz\t"
            "BraTS-GLI-00005-000-t1c.nii.gz\tsub-0001_ses-01_T1c.nii.gz\tsub-0001\tses-01\tanat\n"
        )
        cases = brats_fomo(fomo, local)
    by = {c.dataset: c for c in cases}
    assert "brats2023_gli_t1n" in by and "brats2023_gli_t1c" in by
    assert by["brats2023_gli_t1n"].case_id == "BraTS-GLI-00005-000-t1n"
    assert by["brats2023_gli_t1n"].masks and by["brats2023_gli_t1n"].masks[0][0] == "seg"
    assert by["brats2023_gli_t1c"].ref == by["brats2023_gli_t1n"].path
    assert not by["brats2023_gli_t1c"].masks


def test_fomo_case_ids_keep_contrasts_in_one_folder():
    from brain.prep.pipeline import needs_n4
    from brain.prep.sources import Case

    t1 = Case("fomo_hbn_t1w", "sub-01_ses-01_t1w", "mri")
    fl = Case("fomo_hbn_flair", "sub-01_ses-01_flair", "mri")
    assert t1.folder == fl.folder == "fomo_hbn"
    assert t1.case_id != fl.case_id
    assert needs_n4("fomo_hbn_t1w") and needs_n4("fomo_nki_t2w")
    assert not needs_n4("brats2023_gli_t1n") and not needs_n4("oasis1")


def test_fomo_jpegls_roundtrip_uint16():
    from brain.prep.archive import decode_volume, encode_volume, as_uint16_volume, archive_for_pt, pt_from_family
    from pathlib import Path

    vol = RNG.integers(0, 4000, size=(32, 32, 8), dtype=np.uint16)
    blob = encode_volume(vol)
    got = decode_volume(blob)
    assert got.shape == vol.shape and got.dtype == vol.dtype
    assert np.array_equal(got, vol)
    signed = RNG.integers(-200, 200, size=(16, 16, 4), dtype=np.int16)
    u, meta = as_uint16_volume(signed)
    assert meta["bias"] == 32768 and np.array_equal(u.astype(np.int32) - 32768, signed)
    q, qmeta = as_uint16_volume(np.linspace(1.1, 150.0, 32, dtype=np.float32).reshape(4, 4, 2))
    assert qmeta["kind"] == "uint16_quant" and q.dtype == np.uint16
    assert pt_from_family("fomo_hbn") == "PT018_HBN"
    assert archive_for_pt("PT002_Nigerian_Clinical", Path("/tmp")).name == "fomo_group_small.jls.zip"
    assert archive_for_pt("PT018_HBN", Path("/tmp")).name == "PT018_HBN.jls.zip"


def test_qc_flags():
    blank = np.full((128, 256, 256), 0.0, np.float32)            # blank converted volume: 0 HU everywhere
    _, stats = intensity.ct_native(blank)
    assert stats["fg_odd"]
    assert intensity.ct_native(np.full((128, 256, 256), -1000.0, np.float32))[1]["fg_odd"]  # all air: no anatomy


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
