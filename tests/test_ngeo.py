"""Native-geometry CT copy (brain.prep.ngeo): slice plan, dedupe, de-shear, store-frame recovery, affine round trips."""

import numpy as np
import pytest
import SimpleITK as sitk

from brain.prep import ngeo


def _tilted_stack(K=20, tilt_deg=20.0, dz=5.0, H=64, W=64, ps=0.5, seed=0, sheared=True):
    """A gantry-tilted axial series: planes rotated about x by ``tilt_deg``, table steps along z (sheared) or along
    the plane normal (``sheared=False``). IPPs / IOP as a scanner writes them (LPS)."""
    t = np.radians(tilt_deg)
    rowc = np.array([1.0, 0.0, 0.0])                    # along columns j
    colc = np.array([0.0, np.cos(t), -np.sin(t)])       # along rows i
    n = np.cross(rowc, colc)
    step = np.array([0.0, 0.0, dz]) if sheared else n * dz
    p = np.array([-16.0, -16.0, -50.0]) + np.arange(K)[:, None] * step[None]
    rng = np.random.default_rng(seed)
    pix = rng.normal(0, 1, (K, H, W)).astype(np.float32)
    return ngeo.SliceStack(pix, p, colc * ps, rowc * ps)


def _attrs(R=None):
    """1 mm grid attrs (RAS-index grid like prep's) and a registration matrix."""
    R = np.eye(4) if R is None else R
    return {"grid_origin": [127.5, 127.5, -80.0], "grid_spacing": [1.0, 1.0, 1.0],
            "grid_direction": [-1, 0, 0, 0, -1, 0, 0, 0, 1], "grid_size": [256, 256, 160],
            "reg_affine": R.ravel().tolist()}


# --------------------------------------------------------------------------------------------------------- plan

def test_plan_thick_native_slices_copied():
    t = np.arange(30) * 5.0 + 0.3
    pl = ngeo.plan(t)
    assert pl.D == pytest.approx(5.0) and not pl.thin and pl.n_gaps == 0 and pl.n_synth == 0
    assert all(len(w) == 1 and w[0][1] == 1.0 for w in pl.weights)
    np.testing.assert_allclose(pl.T, t)


def test_plan_thin_box_average_into_5mm_slabs():
    d = 0.625
    t = np.arange(160) * d
    pl = ngeo.plan(t)
    assert pl.thin and pl.D == pytest.approx(5.0) and pl.thick_mm == pytest.approx(5.0)
    assert len(pl.T) == 20
    for w in pl.weights:
        assert len(w) == 8
        np.testing.assert_allclose([x for _, x in w], 1 / 8)
    # slab centres: the mean of the 8 slices they average
    for Tj, w in zip(pl.T, pl.weights):
        assert Tj == pytest.approx(np.mean([t[k] for k, _ in w]))


def test_plan_non_divisible_thin_spacing():
    pl = ngeo.plan(np.arange(100) * 0.8)            # 6 slices -> 4.8 mm slabs
    assert pl.D == pytest.approx(4.8) and all(len(w) == 6 for w in pl.weights)


def test_plan_mixed_rsna_protocol():
    """2.5 mm posterior fossa then 5 mm above (RSNA): thick grid, thin part box-averaged onto it, thick part copied."""
    thin = np.arange(15) * 2.5
    thick = thin[-1] + 2.85 + np.arange(20) * 5.0
    t = np.r_[thin, thick]
    pl = ngeo.plan(t)
    assert not pl.thin and pl.D == pytest.approx(5.0) and pl.uneven
    n_thick = sum(1 for w in pl.weights if len(w) == 1 and w[0][0] >= 15)
    assert n_thick == 20
    assert pl.n_synth >= 6
    for w in pl.weights:
        assert sum(x for _, x in w) == pytest.approx(1.0)


def test_plan_gap_is_interpolated_and_counted():
    t = np.delete(np.arange(30) * 5.0, 12)
    pl = ngeo.plan(t)
    assert pl.n_gaps == 1 and len(pl.T) == 30
    w = pl.weights[12]
    assert [k for k, _ in w] == [11, 12] and [x for _, x in w] == pytest.approx([0.5, 0.5])


def test_plan_range_clip():
    pl = ngeo.plan(np.arange(40) * 5.0, lo=50.0, hi=120.0)
    assert pl.T.min() >= 50.0 and pl.T.max() <= 120.0 and len(pl.T) == 15


# -------------------------------------------------------------------------------------------------------- dedupe

def test_dedupe_removes_repeats_and_sorts():
    st = _tilted_stack(K=10)
    rev = st.take(np.r_[np.arange(10)[::-1], [3, 3, 7]])           # reversed + 3 repeats
    out, n_dup = ngeo.sort_dedupe(rev)
    assert n_dup == 3 and out.K == 10
    t = out.p @ out.normal
    assert np.all(np.diff(t) > 0)
    np.testing.assert_allclose(np.sort(t), np.sort(st.p @ st.normal))


def test_tilt_angle_measured_from_ipp_steps():
    assert ngeo.stack_tilt_deg(_tilted_stack(tilt_deg=17.0)) == pytest.approx(17.0, abs=1e-6)
    assert ngeo.stack_tilt_deg(_tilted_stack(tilt_deg=17.0, sheared=False)) == pytest.approx(0.0, abs=1e-6)


# -------------------------------------------------------------------------------------------- in-plane de-shear

def test_deshear_column_lands_on_one_pixel():
    """A structure fixed in space along the plane normal (a rod) shows at the same output pixel on every slice of a
    sheared series."""
    st = _tilted_stack(K=12, H=96, W=96, sheared=True)
    st.pix[:] = 0
    n = st.normal
    rod0 = st.p[0] + 40 * st.a + 50 * st.b                          # a point on slice 0
    L = np.linalg.pinv(np.stack([st.a, st.b], axis=1))
    for k in range(st.K):
        x = rod0 + ((st.p[k] - rod0) @ n) * n                       # the rod's point in plane k
        i, j = L @ (x - st.p[k])
        st.pix[k, int(round(i)) - 1:int(round(i)) + 2, int(round(j)) - 1:int(round(j)) + 2] = 1000
    pl = ngeo.plan(st.p @ n)
    g = ngeo.make_grid(st, pl, rod0)
    img = ngeo.resample_image(st, pl, g, cval=0)
    peaks = [np.unravel_index(np.argmax(img[s]), img[s].shape) for s in range(g.S)]
    assert max(abs(pr - peaks[0][0]) + abs(pc - peaks[0][1]) for pr, pc in peaks) <= 1
    # centred on the requested point
    assert abs(peaks[0][0] - 255.5) <= 2 and abs(peaks[0][1] - 255.5) <= 2


def test_grid_slices_lie_on_source_planes():
    st = _tilted_stack(K=15, sheared=True)
    pl = ngeo.plan(st.p @ st.normal)
    g = ngeo.make_grid(st, pl, st.p[7])
    A = g.affine()
    for s, w in enumerate(pl.weights):
        k = w[0][0]
        P = A @ np.array([s, 100.0, 300.0, 1.0])
        assert (P[:3] - st.p[k]) @ st.normal == pytest.approx(0.0, abs=1e-6)


def test_inplane_axes_radiological():
    st = _tilted_stack(tilt_deg=0.0)
    u_r, u_c = ngeo.inplane_axes(st.a, st.b, st.normal)
    np.testing.assert_allclose(u_c, [1, 0, 0], atol=1e-9)           # columns toward patient left
    np.testing.assert_allclose(u_r, [0, 1, 0], atol=1e-9)           # rows toward posterior
    np.testing.assert_allclose(st.normal, [0, 0, 1], atol=1e-9)     # slices inferior -> superior


# ------------------------------------------------------------------------------------------- store frame (sitk)

def _sitk_native_attrs(st_files: ngeo.SliceStack):
    """What prep stored for a SimpleITK series read of these files: ITK origin = first sorted IPP, direction
    [row, col, row x col], z spacing |IPP_last - IPP_first| / (N - 1); then DICOMOrient to RAS."""
    rowc, colc = ngeo.unit(st_files.b), ngeo.unit(st_files.a)
    n0 = np.cross(rowc, colc)
    order = np.argsort(st_files.p @ n0)
    p = st_files.p[order]
    dz = np.linalg.norm(p[-1] - p[0]) / (len(p) - 1)
    H, W = st_files.pix.shape[1:]
    img = sitk.Image([W, H, len(p)], sitk.sitkUInt8)
    img.SetOrigin(tuple(p[0]))
    img.SetSpacing((float(np.linalg.norm(st_files.b)), float(np.linalg.norm(st_files.a)), float(dz)))
    img.SetDirection(tuple(np.stack([rowc, colc, n0], axis=1).ravel()))
    o = sitk.DICOMOrient(img, "RAS")
    attrs = {"native_origin": list(o.GetOrigin()), "native_spacing": list(o.GetSpacing()),
             "native_direction": list(o.GetDirection()), "native_size": list(o.GetSize())}
    expected = np.array([img.TransformContinuousIndexToPhysicalPoint((0.0, 0.0, float(np.flatnonzero(order == k)[0])))
                         for k in range(len(p))])
    return attrs, expected


@pytest.mark.parametrize("sheared", [True, False])
def test_sitk_store_origins_recovered(sheared):
    st = _tilted_stack(K=14, sheared=sheared)
    shuffled = st.take(np.random.default_rng(1).permutation(st.K))
    attrs, expected = _sitk_native_attrs(shuffled)
    q, info = ngeo.sitk_store_origins(shuffled.p, shuffled.a, shuffled.b, attrs, np.arange(shuffled.K))
    assert q is not None and info["store_frame_check_mm"] < 1e-6
    np.testing.assert_allclose(q, expected, atol=1e-6)
    if not sheared:
        np.testing.assert_allclose(q, shuffled.p, atol=1e-6)         # untilted, even: both frames agree


# ---------------------------------------------------------------------------------------------- affine round trip

def _random_affine(seed=3):
    rng = np.random.default_rng(seed)
    R = np.eye(4)
    R[:3, :3] = np.eye(3) * 0.9 + rng.normal(0, 0.05, (3, 3))
    R[:3, 3] = rng.normal(0, 10, 3)
    return R


@pytest.mark.parametrize("sheared", [True, False])
def test_affine_template_round_trip_and_matches_per_slice_chain(sheared):
    """template point -> ngeo voxel -> template point; and affine_template agrees with the explicit chain
    ngeo voxel -> store frame (per slice) -> inv(reg_affine)."""
    st = _tilted_stack(K=20, H=32, W=32, sheared=sheared)
    attrs_nat, _ = _sitk_native_attrs(st)
    q, _ = ngeo.sitk_store_origins(st.p, st.a, st.b, attrs_nat)
    st.q = q
    R = _random_affine()
    attrs = _attrs(R) | attrs_nat
    res = ngeo.build(st, attrs, None)
    A = np.asarray(res.attrs["affine_template"]).reshape(4, 4)
    assert res.attrs["tpl_resid_mm"] < 1e-6
    rng = np.random.default_rng(0)
    for _ in range(20):
        vox = np.r_[rng.integers(0, res.grid.S), rng.uniform(0, 511, 2), 1.0]
        tp = A @ vox                                                # ngeo voxel -> template point
        np.testing.assert_allclose(np.linalg.solve(A, tp), vox, atol=1e-9)   # template point -> the same voxel
        s, r, c = vox[:3]
        store_pt = res.Q[int(s)] + r * res.Ur + c * res.Uc
        chain = np.linalg.inv(R) @ np.r_[store_pt, 1.0]
        np.testing.assert_allclose(chain, tp, atol=1e-6)
    # per-slice template origins agree with the affine
    so = np.asarray(res.attrs["tpl_slice_origin"])
    np.testing.assert_allclose(so, (A @ np.c_[np.arange(len(so)), np.zeros((len(so), 2)), np.ones(len(so))].T)[:3].T,
                               atol=2e-3)


def test_uneven_store_frame_reports_residual():
    """SimpleITK's even-spacing assumption on a mixed 2.5 / 5 mm series is not affine: the fit says so, the per-slice
    map stays exact."""
    t = np.r_[np.arange(10) * 2.5, 22.5 + 2.5 + np.arange(15) * 5.0]
    K = len(t)
    st = ngeo.SliceStack(np.zeros((K, 16, 16), np.float32), np.c_[np.zeros(K), np.zeros(K), t],
                         np.array([0.0, 0.5, 0.0]), np.array([0.5, 0.0, 0.0]))
    attrs_nat, expected = _sitk_native_attrs(st)
    q, _ = ngeo.sitk_store_origins(st.p, st.a, st.b, attrs_nat)
    np.testing.assert_allclose(q, expected, atol=1e-6)
    st.q = q
    res = ngeo.build(st, _attrs() | attrs_nat, None)
    assert res.attrs["tpl_resid_mm"] > 3.0
    assert res.attrs["assembly"]["uneven"]


def test_template_mask_maps_back_exactly():
    """A label painted on the 1 mm grid at the template position of an ngeo voxel comes back at that voxel."""
    st = _tilted_stack(K=20, H=48, W=48, sheared=True)
    attrs_nat, _ = _sitk_native_attrs(st)
    st.q, _ = ngeo.sitk_store_origins(st.p, st.a, st.b, attrs_nat)
    R = _random_affine(5)
    attrs = _attrs(R) | attrs_nat
    res = ngeo.build(st, attrs, None)
    s, r, c = 6, 260, 250
    store_pt = res.Q[s] + r * res.Ur + c * res.Uc
    tp = np.linalg.inv(R) @ np.r_[store_pt, 1.0]
    xyz = np.linalg.inv(ngeo.grid_affine(attrs)) @ tp
    lab = np.zeros((160, 256, 256), np.uint8)
    x, y, z = (int(round(v)) for v in xyz[:3])
    lab[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] = 7
    m = ngeo.map_template_array(lab, attrs, res.Q, res.Ur, res.Uc, res.grid)
    assert m[s, r, c] == 7


def test_label_vote_in_thin_slabs():
    K = 16
    st = ngeo.SliceStack(np.zeros((K, 32, 32), np.float32), np.c_[np.zeros(K), np.zeros(K), np.arange(K) * 0.625],
                         np.array([0.0, 0.5, 0.0]), np.array([0.5, 0.0, 0.0]))
    lab = np.zeros((K, 32, 32), np.uint8)
    lab[:5] = 2                                                    # 5 of the first slab's 8 slices
    lab[8:11] = 3                                                  # 3 of the second slab's 8
    ls = ngeo.SliceStack(lab, st.p, st.a, st.b)
    pl = ngeo.plan(st.p @ st.normal)
    g = ngeo.make_grid(st, pl, st.p[0] + 8 * st.a + 8 * st.b)
    out = ngeo.resample_labels(ls, pl, g)
    assert out.shape[0] == 2
    assert (out[0] == 2).sum() > 0 and (out[1] == 3).sum() == 0


def test_hu_int16_air_clamp():
    v = np.array([-3000.0, -600.0, -499.6, 40.4, 5000.0])
    np.testing.assert_array_equal(ngeo.to_hu_int16(v), np.array([-1024, -1024, -500, 40, 3071], np.int16))


def test_sitk_store_origins_instance_order_fallback():
    """With repeated positions GDCM orders files by instance number: the order whose first IPP and spacing reproduce
    the stored grid wins."""
    st = _tilted_stack(K=12, tilt_deg=0.0)
    files = st.take(np.r_[np.arange(12), [4, 7]])                   # two repeats, instance = member order
    rowc, colc = ngeo.unit(files.b), ngeo.unit(files.a)
    n0 = np.cross(rowc, colc)
    p = files.p                                                     # ITK in member (instance) order
    dz = np.linalg.norm(p[-1] - p[0]) / (len(p) - 1)
    img = sitk.Image([64, 64, len(p)], sitk.sitkUInt8)
    img.SetOrigin(tuple(p[0]))
    img.SetSpacing((0.5, 0.5, float(dz)))
    img.SetDirection(tuple(np.stack([rowc, colc, n0], axis=1).ravel()))
    o = sitk.DICOMOrient(img, "RAS")
    attrs = {"native_origin": list(o.GetOrigin()), "native_spacing": list(o.GetSpacing()),
             "native_direction": list(o.GetDirection()), "native_size": list(o.GetSize())}
    q, info = ngeo.sitk_store_origins(files.p, files.a, files.b, attrs, np.arange(len(p)))
    assert q is not None and info["store_frame_order"].startswith(("instance", "member"))
    exp = np.array([img.TransformContinuousIndexToPhysicalPoint((0.0, 0.0, float(k))) for k in range(len(p))])
    np.testing.assert_allclose(q, exp, atol=1e-6)


def test_plan_prefers_coarse_class_in_mixed_series():
    """Mostly 2.5 mm slices by count but a long 5 mm part: 5 mm grid, 5 mm slices copied, no interpolated holes."""
    t = np.r_[np.arange(20) * 2.5, 47.5 + 2.5 + np.arange(19) * 5.0]
    pl = ngeo.plan(t)
    assert pl.D == pytest.approx(5.0) and pl.n_gaps == 0


def test_nifti_store_frame_flip_recovered():
    """A 1 mm store written in a flipped frame (old nibabel fallback): the flip is found by correlation and the
    template mapping lands."""
    rng = np.random.default_rng(0)
    vol = np.full((20, 64, 64), -1024.0, np.float32)
    zz, yy, xx = np.mgrid[:20, :64, :64]
    vol[((yy - 30) ** 2 + (xx - 26) ** 2 < 400)] = 40.0
    vol[((yy - 20) ** 2 + (xx - 40) ** 2 < 30)] = 80.0
    vol += rng.normal(0, 3, vol.shape).astype(np.float32)
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((1.0, 1.0, 5.0))
    img.SetOrigin((-32.0, -32.0, -50.0))
    st = ngeo.stack_from_sitk(img)
    # the store: same voxels, x and y index flipped, identity frame elsewhere (what the old fallback did)
    stored = sitk.GetImageFromArray(vol[:, ::-1, ::-1].copy())
    stored.SetSpacing((1.0, 1.0, 5.0))
    stored.SetOrigin((-20.0, -12.0, -50.0))                         # the old reader's (wrong) origin
    attrs = _attrs() | {"native_origin": list(stored.GetOrigin()), "native_spacing": list(stored.GetSpacing()),
                        "native_direction": list(stored.GetDirection()), "native_size": list(stored.GetSize()),
                        "grid_origin": [-80.0, -80.0, -80.0], "grid_direction": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
    # the 1 mm image: the stored volume resampled onto the grid (identity registration)
    ref = sitk.Image([256, 256, 160], sitk.sitkFloat32)
    ref.SetOrigin((-80.0, -80.0, -80.0))
    one = sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(stored, sitk.sitkFloat32), ref, sitk.Transform(),
                                               sitk.sitkLinear, -1024.0))
    info = ngeo.nifti_store_frame(st, img, attrs, one)
    assert info["store_frame"] == "native_flip_xy" and info["store_frame_corr"] > 0.9
    res = ngeo.build(st, attrs, None, image_1mm_zyx=one)
    assert res.attrs["qc_corr_1mm"] > 0.9
