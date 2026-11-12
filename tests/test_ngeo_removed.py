"""Native-geometry CT input was removed (2026-11-11): checkpoints of the runs that had it (R17, full_ngeo_bank) still
load for evaluation and resume without their in-plane stem / decoder weights, and a new run asking for it stops."""

from __future__ import annotations

import pytest
import torch

from brain.train import engine

CPU = torch.device("cpu")
BASE = ["multitask", "--synthetic", "--log-every", "1", "--set", "train.eval_every=100", "--set", "train.ckpt_every=2",
        "--set", "train.ema_decay=0.9"]
# a few keys of each legacy prefix, shaped as in R17 (values are never read)
LEGACY_ENC = {"ngeo_embed.proj.weight": torch.zeros(64, 64, 1, 8, 8), "ngeo_embed.stem.0.0.weight": torch.zeros(32, 1, 1, 3, 3)}
LEGACY_BQ = {"backbone." + k: v for k, v in LEGACY_ENC.items()} | {"seg_ngeo.head.weight": torch.zeros(1, 16, 1, 1, 1),
                                                                     "seg_ngeo.stem.0.weight": torch.zeros(128, 192, 1, 1, 1)}


def _legacy_checkpoint(tmp_path, monkeypatch):
    """A phantom run's last.pt dressed as a native-geometry run's: legacy weights in the encoder, BrainQuery and
    weight-average states, an optimizer with parameters the model no longer has, ngeo enabled in its config."""
    monkeypatch.setenv("BRAIN_PERSIST_SYNTHETIC", "1")
    src = tmp_path / "src"
    engine.run(BASE + ["--max-steps", "2", "--out-dir", str(src)])
    ck = torch.load(src / "last.pt", map_location="cpu", weights_only=False)
    clean = {k: v.clone() for k, v in ck["multitask"].items()}
    ck["model"] = {**ck["model"], **LEGACY_ENC}
    ck["multitask"] = {**ck["multitask"], **LEGACY_BQ}
    ck["ema"]["avg"] = {**ck["ema"]["avg"], **{k: v.float() for k, v in LEGACY_BQ.items()}}
    groups = ck["optimizer"]["param_groups"]
    n = sum(len(g["params"]) for g in groups)
    groups[0]["params"] = list(groups[0]["params"]) + list(range(n, n + len(LEGACY_BQ)))
    ck["config"]["model"]["ngeo"] = {"enabled": True, "stem": [8, 16], "patch_xyz": [32, 32, 1], "seg_stride": [1, 4, 4]}
    ck["config"]["data"]["ngeo"] = {"enabled": True, "mixed": True, "index_name": "volumes_ngeo.csv"}
    path = tmp_path / "legacy.pt"
    torch.save(ck, path)
    return path, clean


def test_legacy_checkpoint_evaluates_on_1mm_input(tmp_path, monkeypatch, capsys):
    from brain.downstream.embed3d import load_brain3d
    from brain.multitask.data import task_loaders
    from brain.multitask.evaluate import run_eval
    from brain.multitask.fewshot import load_model

    path, clean = _legacy_checkpoint(tmp_path, monkeypatch)
    capsys.readouterr()
    model, reg, cfg = load_model(path, CPU)                 # the saved config still enables ngeo: ignored
    out = capsys.readouterr().out
    assert f"dropped {len(LEGACY_BQ)} legacy native-geometry weights" in out
    assert not any("ngeo" in k for k in model.state_dict())
    assert all(torch.equal(model.state_dict()[k], v) for k, v in clean.items())
    enc = load_brain3d(torch.load(path, map_location="cpu", weights_only=False), CPU)
    assert f"encoder: dropped {len(LEGACY_ENC)} legacy" in capsys.readouterr().out
    assert not hasattr(enc.model, "ngeo_embed")
    ids = [t.id for t in reg.trained]
    _, val, _, _ = task_loaders(cfg, reg, ids, batch=2, workers=0, seed=0, main=True, distributed=False,
                                synthetic=True)
    res = run_eval(model, val, cfg, reg, {}, CPU, tuple(model.backbone.patch_xyz),
                   dict(device_type="cpu", dtype=torch.bfloat16, enabled=False))
    assert res


@pytest.mark.parametrize("biases", [False, True])
def test_legacy_checkpoint_resumes(tmp_path, monkeypatch, capsys, biases):
    """A branched resume from a native-geometry checkpoint drops its legacy weights; with ``biases`` it also adds
    zero-init depthwise convolutions and mirror mixing (R18's start). The rest of the state stays strict, the
    optimizer is reset (the saved one has parameters the model lost) and the weight average keeps no legacy keys."""
    path, _ = _legacy_checkpoint(tmp_path, monkeypatch)
    dst = tmp_path / "dst"
    capsys.readouterr()
    extra = ["--set", "model.dw_conv_blocks=2", "--set", "model.mirror_blocks=[1]"] if biases else []
    engine.run(BASE + ["--max-steps", "4", "--out-dir", str(dst), "--reset-schedule",
                       "--set", f"output.resume_from={path}", *extra])
    out = capsys.readouterr().out
    assert f"[resume] dropped {len(LEGACY_BQ)} legacy native-geometry weights" in out
    assert "optimizer state reset" in out
    ck = torch.load(dst / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 4
    for state in (ck["multitask"], ck["ema"]["avg"]):
        assert not any("ngeo" in k for k in state)
        assert any(".dw." in k for k in state) == biases and any(k.startswith("backbone.mirror.") for k in state) == biases


def test_legacy_checkpoint_with_other_unexpected_keys_stays_strict(tmp_path, monkeypatch):
    path, _ = _legacy_checkpoint(tmp_path, monkeypatch)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["multitask"]["not_a_module.weight"] = torch.zeros(1)
    torch.save(ck, path)
    with pytest.raises(SystemExit, match="state mismatch"):
        engine.run(BASE + ["--max-steps", "4", "--out-dir", str(tmp_path / "dst"), "--reset-schedule",
                           "--set", f"output.resume_from={path}"])


@pytest.mark.parametrize("key", ["model.ngeo", "data.ngeo"])
def test_new_run_with_native_geometry_is_refused(tmp_path, key):
    with pytest.raises(SystemExit, match="native-geometry input was removed 2026-11-11; all CT is read at 1 mm"):
        engine.run(BASE + ["--max-steps", "1", "--out-dir", str(tmp_path), "--set", f"{key}={{enabled: true}}"])


def test_legacy_keys_off_are_accepted():
    from brain.train.defaults import default_config, refuse_ngeo

    cfg = default_config("multitask")
    refuse_ngeo(cfg)                                                             # None
    cfg["data"]["ngeo"] = {"enabled": False, "mixed": True}
    cfg["model"]["ngeo"] = {"enabled": False}
    refuse_ngeo(cfg)
