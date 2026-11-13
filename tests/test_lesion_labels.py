"""Mask-derived lesion labels (brain.multitask.lesion_measures) and the left-right flip of sided targets."""
import math

import numpy as np
import pytest
import torch

from brain.multitask import lesion_measures as LM
from brain.multitask.data import lr_swap_table, swap_flipped
from brain.multitask.registry import Registry, Task, load_registry

SHAPE = (40, 60, 256)          # (Z, Y, X); x increasing = patient's right


def _priors() -> dict[str, np.ndarray]:
    z = np.zeros(SHAPE, bool)
    p = {k: z.copy() for k in ("ventricles", "lateral_ventricles", "posterior_fossa", "deep_grey", "intracranial")}
    p["intracranial"][5:35, 5:55, 60:196] = True
    p["ventricles"][15:25, 25:45, 110:146] = True
    p["lateral_ventricles"][15:25, 25:45, 110:146] = True
    p["posterior_fossa"][5:12, 10:30, 90:166] = True
    p["deep_grey"][15:25, 15:25, 100:156] = True
    return p


def test_side_class_thresholds():
    assert LM.side_class(9.0, 1.0) == 0          # 90% left
    assert LM.side_class(1.0, 9.0) == 1
    assert LM.side_class(6.0, 4.0) == 2          # 60 / 40: bilateral
    assert LM.side_class(8.5, 1.5) == 2          # 85% one side: still bilateral at the 0.9 rule
    assert LM.side_class(0.0, 0.0) == -1
    assert LM.SIDE_CLASSES[:2] == ("left", "right")


def test_lesion_stats_side_compartments_components():
    pri = _priors()
    brain = pri["intracranial"].copy()
    m = np.zeros(SHAPE, bool)
    m[20:24, 30:34, 70:80] = True                # left hemisphere (low x), outside the other priors
    m[20:24, 30:34, 85:95] = True                # a second, separate focus
    st = LM.lesion_stats(m, brain, pri)
    assert st["components"] == 2
    assert st["left_ml"] > 0 and st["right_ml"] == 0
    assert LM.side_class(st["left_ml"], st["right_ml"]) == 0
    assert LM.LOCATION_CLASSES[LM.location_class(st)] == "hemispheric"
    m2 = np.zeros(SHAPE, bool)
    m2[16:20, 30:40, 120:140] = True             # inside the ventricle prior (which overlaps deep grey: ventricle wins)
    assert LM.LOCATION_CLASSES[LM.location_class(LM.lesion_stats(m2, brain, pri))] == "intraventricular"
    m3 = np.zeros(SHAPE, bool)
    m3[20:24, 2:4, 150:160] = True               # outside the intracranial prior
    assert LM.LOCATION_CLASSES[LM.location_class(LM.lesion_stats(m3, brain, pri))] == "extra-axial"


def _ventricles(septum_x: float, gap: int = 3) -> np.ndarray:
    v = np.zeros(SHAPE, bool)
    left_end = int(round(septum_x - gap / 2))
    v[15:25, 25:45, left_end - 12:left_end] = True
    v[15:25, 25:45, left_end + gap:left_end + gap + 12] = True
    return v


@pytest.mark.parametrize("shift", [0, 4, -7])
def test_midline_shift_recovers_septum_offset(shift):
    pri = _priors()
    out = LM.midline_shift(_ventricles(LM.GRID_MID_X + shift), pri)
    assert out["mls_rows"] >= LM.MLS_MIN_ROWS
    assert abs(out["mls_signed"] - shift) <= 1.0
    assert out["mls_mm"] == pytest.approx(abs(out["mls_signed"]))


def test_midline_shift_plane_reference_and_missing_ventricles():
    pri = _priors()
    v = _ventricles(LM.GRID_MID_X + 6)
    tilted = LM.midline_shift(v, pri, (LM.GRID_MID_X + 6, 0.0, 0.0))        # plane through the septum: no shift
    assert abs(tilted["mls_signed"]) <= 1.0
    none = LM.midline_shift(np.zeros(SHAPE, bool), pri)
    assert math.isnan(none["mls_mm"]) and none["mls_rows"] == 0


def _reg_with_side() -> Registry:
    return Registry([
        Task("age", "regression", ["age"], []),
        Task("lesion_side", "multiclass", ["side"], [], classes=["left", "right", "bilateral"], lr_swap=[[0, 1]]),
        Task("tumor_type", "multiclass", ["type"], [], classes=["a", "b", "c", "d"]),
    ])


def test_flip_swaps_only_sided_targets_of_flipped_rows():
    reg = _reg_with_side()
    tab = lr_swap_table(reg)
    assert tab.shape == (3, 4) and tab[1, :3].tolist() == [1, 0, 2] and tab[2].tolist() == [0, 1, 2, 3]
    task = torch.tensor([[1, 2, 0], [1, 2, -1], [1, 0, -1]])
    y = torch.tensor([[0.0, 0.0, 40.0], [2.0, 1.0, float("nan")], [1.0, 70.0, float("nan")]])
    flip = torch.tensor([True, True, False])
    out = swap_flipped(y, task, flip, tab)
    assert out[0].tolist() == [1.0, 0.0, 40.0]           # side left -> right; tumour class and age untouched
    assert out[1, 0].item() == 2.0 and out[1, 1].item() == 1.0   # bilateral stays bilateral
    assert out[2].tolist()[:2] == [1.0, 70.0]            # row not flipped
    assert torch.isnan(out[1, 2])
    assert lr_swap_table(Registry([Task("age", "regression", ["age"], [])])) is None


def test_registry_rejects_lr_swap_on_non_multiclass(tmp_path):
    p = tmp_path / "reg.yaml"
    p.write_text("tasks:\n  - id: x\n    type: binary\n    descriptions: [x]\n    lr_swap: [[0, 1]]\n")
    with pytest.raises(ValueError):
        load_registry(p)


def test_default_registry_sided_tasks_are_multiclass_with_left_right_pair():
    reg = load_registry()
    for t in reg.tasks:
        if t.lr_swap:
            assert t.type == "multiclass" and t.classes[0] == "left" and t.classes[1] == "right"
