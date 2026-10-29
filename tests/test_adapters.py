"""Geometry of the published-model adapters: the stored grid in world space, resampling, and features back onto our
token grid (a feature map that holds its own world coordinates must come back as our token centres)."""

import numpy as np
import torch

from brain.common.grid import GRID_XYZ
from brain.downstream.models import ExternalEncoder

# the store's geometry (ITK, LPS): x index -> right, y index -> anterior, z index -> superior
ATTRS = {"grid_origin": [126.94, 149.03, -69.67], "grid_spacing": [1.0, 1.0, 1.0],
         "grid_direction": [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0], "grid_size": list(GRID_XYZ)}


class _Enc(ExternalEncoder):
    def __init__(self):
        super().__init__({}, torch.device("cpu"))


def test_grid_affine_is_ras():
    a = ExternalEncoder.grid_affine(ATTRS)
    assert np.allclose(a[:3, :3], np.eye(3))                   # RAS-aligned 1 mm grid
    assert np.allclose(a[:3, 3], [-126.94, -149.03, -69.67])


def test_resample_onto_itself_is_identity():
    e = _Enc()
    small = dict(ATTRS, grid_size=[12, 10, 8])
    vol = np.random.default_rng(0).normal(size=(12, 10, 8)).astype(np.float32)
    out = e.resample(vol, small, e.grid_affine(small), (12, 10, 8))
    assert torch.allclose(out, torch.from_numpy(vol), atol=1e-4)


def test_resample_flip_and_fill():
    """An LPS-ordered 1 mm target over the same box: x and y reversed; points outside get ``fill``."""
    e = _Enc()
    small = dict(ATTRS, grid_size=[6, 5, 4])
    vol = np.arange(6 * 5 * 4, dtype=np.float32).reshape(6, 5, 4)
    a = e.grid_affine(small) @ np.array([[-1, 0, 0, 5], [0, -1, 0, 4], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    assert torch.allclose(e.resample(vol, small, a, (6, 5, 4)), torch.from_numpy(vol[::-1, ::-1].copy()))
    shifted = e.grid_affine(small) @ np.array([[1, 0, 0, 100], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    assert torch.all(e.resample(vol, small, shifted, (6, 5, 4), fill=-7.0) == -7.0)


def test_to_token_grid_returns_token_centres():
    """A coarse feature map whose channels are its voxels' RAS coordinates, read at our tokens, gives each token's
    centre in RAS: the token order, axis order and offsets agree with our encoder's."""
    e = _Enc()
    aff = np.diag([4.0, 4.0, 4.0, 1.0])
    aff[:3, 3] = [-140.0, -160.0, -80.0]
    shape = (72, 68, 44)                                       # covers the stored box
    idx = torch.stack(torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape], indexing="ij"), 0)
    fmap = torch.einsum("ij,jabc->iabc", torch.as_tensor(aff[:3, :3], dtype=torch.float32), idx) + \
        torch.as_tensor(aff[:3, 3], dtype=torch.float32)[:, None, None, None]
    vol = np.full(GRID_XYZ, -6.0, np.float32)                 # the stored MRI background
    vol[64:192, 64:192, 40:120] = 1.0                          # MRI foreground in the middle
    out = e.to_token_grid(fmap, aff, vol, "mri", ATTRS)
    assert out["tokens"].shape == (5120, 3) and out["xyz"].shape == (5120, 3)
    g = ExternalEncoder.grid_affine(ATTRS)
    # token t = (z, y, x) flattened, X fastest: its centre voxel index (16 x + 7.5, 16 y + 7.5, 8 z + 3.5)
    z, y, x = np.unravel_index(np.arange(5120), (20, 16, 16))
    centre = np.stack([16 * x + 7.5, 16 * y + 7.5, 8 * z + 3.5], 1) @ g[:3, :3].T + g[:3, 3]
    assert np.allclose(out["tokens"].numpy(), centre, atol=1e-3)
    assert np.allclose(out["xyz"][:, 0].numpy(), x / 15) and np.allclose(out["xyz"][:, 2].numpy(), z / 19)
    keep = out["keep"].numpy().reshape(20, 16, 16)
    assert keep[10, 8, 8] and not keep[0, 0, 0]


def test_ct_fm_input_orientation():
    """CT-FM's input axes run superior, posterior, left (SPL) at 3 x 1 x 1 mm, centred on the stored box."""
    from brain.downstream.adapters.published import CTFM, _box_centre

    e = CTFM.__new__(CTFM)
    e.spacing, e.shape = (3.0, 1.0, 1.0), (64, 256, 256)
    a = e._affine(ATTRS)
    assert np.allclose(a[:3, 0], [0, 0, 3]) and np.allclose(a[:3, 1], [0, -1, 0]) and np.allclose(a[:3, 2], [-1, 0, 0])
    assert np.allclose(a @ np.r_[31.5, 127.5, 127.5, 1.0], np.r_[_box_centre(ATTRS), 1.0])


def test_ct_fm_triage_input_stacks_windows_along_s():
    """CT-FM's head-CT pipeline: each HU window scaled to [0, 1], the copies stacked along S."""
    from brain.downstream.adapters.published import CTFM

    e = CTFM.__new__(CTFM)
    e.windows = [(0.0, 80.0), (-125.0, 175.0), (28.0, 36.0), (-900.0, 2100.0)]
    hu = torch.tensor([-1024.0, 0.0, 32.0, 40.0, 1000.0]).view(5, 1, 1).expand(5, 2, 2)
    x = e._input(hu)
    assert x.shape == (20, 2, 2) and float(x.min()) >= 0 and float(x.max()) <= 1
    assert torch.allclose(x[3:5, 0, 0], torch.tensor([0.5, 1.0]))                    # blood window: 40 HU -> 0.5
    assert torch.allclose(x[12, 0, 0], torch.tensor(0.5))                             # stroke window: 32 HU -> 0.5
    e.windows = []
    assert torch.allclose(e._input(hu)[:, 0, 0], ((hu[:, 0, 0] + 1024) / 3072).clamp(0, 1))
