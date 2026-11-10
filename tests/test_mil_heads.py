"""Multiple-instance binary readout (registry ``pool: mil``, ``model.mil``) and thin-CT routing of ngeo samples."""

from __future__ import annotations

import math

import torch

from brain.multitask.data import needs_thin_ct
from brain.multitask.model import BrainQuery
from brain.multitask.registry import load_registry
from brain.train.defaults import default_config, smoke_config
from brain.train.multitask import build_encoder

CPU = torch.device("cpu")
REG = load_registry()
MIL = ["hemorrhage_any", "hemorrhage_epidural", "hemorrhage_intraparenchymal", "hemorrhage_intraventricular",
       "hemorrhage_subarachnoid", "hemorrhage_subdural", "mri_mass", "mri_edema", "mri_infarct", "mri_wm_lesion",
       "mri_postop", "mri_small_vessel_disease"]


def _model(mil: bool):
    torch.manual_seed(0)
    cfg = smoke_config(default_config("multitask"))
    cfg["model"]["task_tokens"] = {"enabled": True}
    cfg["model"]["token_map"] = {"enabled": True, "dim": 16}
    cfg["model"]["mil"] = {"enabled": mil}
    model = BrainQuery(build_encoder(cfg, CPU, synthetic=True), REG, cfg["model"]).eval()
    model.set_token_tasks([REG.index[t] for t in REG.train_ids(cfg["tasks"])])
    return model


def test_registry_marks_the_finding_tasks_and_thin_ct_limits():
    assert sorted(t.id for t in REG.tasks if t.pool == "mil") == sorted(MIL)
    assert all(REG[t].type == "binary" for t in MIL)
    assert REG["seg_thalamus"].ct_max_thick_mm == 2.0 and REG["seg_ventricles"].ct_max_thick_mm is None


def test_mil_off_keeps_older_models_unchanged():
    m = _model(mil=False)
    assert m.mil_logr is None and not bool(m.mil.any())
    assert "mil_logr" not in m.state_dict()


@torch.no_grad()
def test_mil_logits_pool_only_mil_slots_over_kept_tokens():
    m = _model(mil=True)
    b, n, d = 2, 12, m.backbone.embed_dim
    enc = {"tokens": torch.randn(b, n, d)}
    z = torch.randn(b, 2, m.out_norm.normalized_shape[0])
    task = torch.tensor([[REG.index["hemorrhage_any"], REG.index["age"]],
                         [REG.index["sex"], -1]])
    keep = torch.ones(b, n, dtype=torch.bool)
    keep[0, 6:] = False
    add = m.mil_logits(enc, z, task, keep)
    assert add.shape == (b, 2)
    assert add[0, 1] == 0 and (add[1] == 0).all()                       # not MIL / padded slots untouched
    lg = m.token_logits(enc, z, torch.tensor([[0, 0]]))[0, :6] - m.tok_bias   # a soft max over kept tokens, no prior
    r = math.exp(float(m.mil_logr[REG.index["hemorrhage_any"]]))
    ref = (torch.logsumexp(r * lg, 0) - math.log(6)) / r
    assert torch.allclose(add[0, 0], ref, atol=1e-5)
    assert lg.mean() - 1e-5 <= add[0, 0] <= lg.max() + 1e-5
    assert torch.isfinite(m.mil_logits(enc, z, task, torch.zeros(b, n, dtype=torch.bool))).all()   # no kept token


def test_mil_adds_to_the_binary_logit_and_gets_gradients():
    m = _model(mil=True).train()
    b, n, d = 2, 12, m.backbone.embed_dim
    enc = {"tokens": torch.randn(b, n, d)}
    z = torch.randn(b, 1, m.out_norm.normalized_shape[0], requires_grad=True)
    task = torch.tensor([[REG.index["hemorrhage_subdural"]], [REG.index["mri_mass"]]])
    add = m.mil_logits(enc, z, task, torch.ones(b, n, dtype=torch.bool))
    add.sum().backward()
    assert z.grad.abs().sum() > 0 and m.mil_logr.grad.abs().sum() > 0 and m.mil_bias.grad.abs().sum() > 0
    assert m.tok_bias.grad is None or float(m.tok_bias.grad.abs()) < 1e-6        # the pooled logit leaves the prior alone


def test_thin_ct_tasks_are_read_at_1mm_from_a_resliced_copy():
    thin = {REG.index[t]: REG[t].ct_max_thick_mm for t in ("seg_thalamus", "seg_hippocampus")}
    drawn = [(REG.index["seg_thalamus"], 0.0, "x.synthseg.zip|7,25"), (REG.index["age"], 1.0, "")]
    assert needs_thin_ct(drawn, thin, 5.0)                    # SynthRAD's 1 mm CT re-sliced to 5 mm
    assert not needs_thin_ct(drawn, thin, 1.0)                # a thin copy stays native
    assert not needs_thin_ct([(REG.index["seg_ventricles"], 0.0, "")], thin, 5.0)
