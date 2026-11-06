"""Synthetic tissue contrast from a SynthSeg label map (brain.data.synth_contrast)."""
import numpy as np

from brain.data.synth_contrast import LR_PAIRS, synth_image
from brain.prep.synthseg import LABELS


def _head(shape=(24, 32, 32)):
    lab = np.zeros(shape, dtype=np.uint8)
    pos = {int(f): i for i, f in enumerate(LABELS)}
    lab[4:20, 6:26, 4:16] = pos[2]                       # left white matter
    lab[4:20, 6:26, 16:28] = pos[41]                     # right white matter
    lab[8:16, 12:20, 12:20] = pos[4]                     # a ventricle
    lab[4:8, 6:26, 4:28] = pos[3]                        # cortex
    img = np.full(shape, -6.0, dtype=np.float32)
    img[2:22, 4:28, 2:30] = 1.5                          # skull and scalp
    img[lab > 0] = 0.0
    return img, lab, pos


def test_brain_rerendered_head_kept():
    img, lab, _ = _head()
    out = synth_image(img, lab, np.random.default_rng(0), blur=(0.0, 0.0))
    brain = lab > 0
    assert out.dtype == img.dtype and out.shape == img.shape
    assert np.array_equal(out[~brain], img[~brain])          # real skull, scalp and background
    assert abs(np.median(out[brain])) < 0.2 and 0.8 < out[brain].std() < 1.2   # the store's tissue scale


def test_homologous_labels_share_intensity_and_draws_differ():
    img, lab, pos = _head()
    a = synth_image(img, lab, np.random.default_rng(1), std=(0.0, 0.0), blur=(0.0, 0.0))
    b = synth_image(img, lab, np.random.default_rng(2), std=(0.0, 0.0), blur=(0.0, 0.0))
    left, right = a[lab == pos[2]], a[lab == pos[41]]
    assert np.allclose(left.mean(), right.mean(), atol=1e-4)
    assert not np.allclose(a[lab == pos[4]].mean(), b[lab == pos[4]].mean())  # each draw a new contrast
    assert all(l_ in pos and r_ in pos for l_, r_ in LR_PAIRS)
