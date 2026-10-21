"""Experiment records -> SQLite: what a run writes, how metrics flatten, and how the report picks heads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from brain.downstream import db, records
from brain.downstream.results import finalize_run, new_run_dir


def _ckpt(tmp: Path, step: int = 7) -> Path:
    p = tmp / "runs" / "brainmodel_tiny" / "last.pt"
    p.parent.mkdir(parents=True)
    torch.save({"model": {"w": torch.zeros(3)}, "step": step, "config": {"model": {"size": "tiny"}},
                "provenance": {"method": "lejepa3d", "store_format": 6}}, p)
    return p


def _run(tmp: Path, ckpt: Path, heads: dict, task: str = "cq500", task_type: str = "multilabel") -> list[Path]:
    run_dir = new_run_dir(tmp / "out", task, ckpt, "r1")
    cfg = {"task": task, "task_type": task_type, "ckpt": str(ckpt), "seed": 42, "slices_root": str(tmp),
           "n_train": 10, "n_val": 3, "n_test": 5, "pretrain_overlap": 0.0, "started_utc": records.now_utc()}
    finalize_run(run_dir, cfg, heads)
    return sorted(records.record_dir().glob("*/*.json"))


def test_finalize_run_writes_one_complete_record_per_head(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "RESULTS_ROOT", tmp_path / "results")
    ck = _ckpt(tmp_path)
    m = {"auc_ovr_macro": 0.8, "auc_ovr_macro_ci": [0.7, 0.9], "auc_per_label": {"ICH": 0.85, "SDH": 0.75},
         "by_cohort": {"cq500": {"auc_ovr_macro": 0.81, "n": 4}}, "val": {"auc_ovr_macro": 0.77},
         "history": [{"step": 1}], "view": "raw", "n_test": 5}
    paths = _run(tmp_path, ck, {"logreg_pool": m, "ridge": {**m, "val": {"auc_ovr_macro": 0.9}}})
    assert len(paths) == 2
    rec = json.loads(paths[0].read_text())
    assert rec["model"]["step"] == 7 and rec["model"]["provenance"]["method"] == "lejepa3d"
    assert rec["model"]["model_id"].startswith("brainmodel_tiny/last@")
    assert rec["data"]["store_format"] == 6 and rec["data"]["grid_xyz"] == [256, 256, 160]
    assert rec["protocol"]["seed"] == 42 and "history" not in rec["metrics"]["test"]
    assert rec["metrics"]["val"]["auc_ovr_macro"] in (0.77, 0.9) and rec["env"]["git"]["commit"]
    assert Path(rec["artifacts"]["run_dir"]).is_dir()


def test_flatten_keeps_ci_labels_and_cohorts():
    rows = db.flatten({"auc": 0.8, "auc_ci": [0.7, 0.9], "auc_per_label": {"ICH": 0.85},
                       "by_cohort": {"siteA": {"auc": 0.6}}, "dice_per_class": {"ET": 0.5},
                       "head": "x", "chosen_param": 1.0, "bad": float("nan")}, "test")
    got = {(n, g): (v, lo, hi) for _, n, g, v, lo, hi in rows}
    assert got[("auc", "")] == (0.8, 0.7, 0.9)
    assert got[("auc_per_label", "label:ICH")][0] == 0.85
    assert got[("auc", "cohort:siteA")][0] == 0.6 and got[("dice_per_class", "class:ET")][0] == 0.5
    assert ("bad", "") not in got and ("head", "") not in got


def test_ingest_is_idempotent_and_the_report_picks_heads_on_val(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "RESULTS_ROOT", tmp_path / "results")
    ck = _ckpt(tmp_path)
    _run(tmp_path, ck, {"a": {"auc_ovr_macro": 0.9, "val": {"auc_ovr_macro": 0.6}},     # best test, worse val
                        "b": {"auc_ovr_macro": 0.8, "val": {"auc_ovr_macro": 0.7}}})
    con = db.connect(tmp_path / "x.sqlite")
    assert db.ingest(con) == 2 and db.ingest(con) == 0
    r = db.report(con)
    assert len(r) == 1 and r.iloc[0]["head"] == "b" and r.iloc[0]["picked_on"] == "val"
    assert abs(r.iloc[0]["test"] - 0.8) < 1e-9
    full = db.report(con, all_heads=True)
    assert set(full["head"]) == {"a", "b"} and np.isclose(full.set_index("head").loc["a", "val"], 0.6)


def test_probe_records_land_in_the_probe_report(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from brain.downstream import probe3d

    monkeypatch.setattr(records, "RESULTS_ROOT", tmp_path / "results")
    ck = _ckpt(tmp_path)
    prov = SimpleNamespace(name="brainmodel_tiny/last", ckpt=ck, variant=None)
    ctrl = SimpleNamespace(name="control/pixel", ckpt=None, variant=None)
    one = {"ap": 0.2, "auroc": 0.8, "prevalence": 0.05, "lift": 4.0}
    les = {"atlas": {"linear": one, "mlp": one, "n_train": 3, "n_test": 2},
           **{f"mean_{k}_{m}": one[m] for k in ("linear", "mlp") for m in ("ap", "lift", "auroc")}}
    res = {"lesion": les, "fingerprint": {"dataset_bal_acc": 0.5, "modality_bal_acc": 1.0},
           "location": {"r2_mean": 0.9, "token_participation_ratio": 3.0},
           "ci": {"lift_lin": {"lo": 3.0, "hi": 5.0}, "loc_r2": {"lo": 0.8, "hi": 0.95}}}
    args = SimpleNamespace(root=tmp_path, out=tmp_path / "p.json", seed=0, reps=10)
    n = probe3d.record_probes([prov, ctrl], {prov.name: res, ctrl.name: res}, args, records.now_utc(), {})
    assert n == 2 * (2 + 2 + 1 + 1)
    con = db.connect(tmp_path / "x.sqlite")
    db.ingest(con)
    r = db.report(con, probes=True)
    lift = r[(r.task == "probe:lesion:mean") & (r["head"] == "linear")]
    assert len(lift) == 2 and np.allclose(lift.test, 4.0) and set(lift.ci_lo.dropna()) == {3.0}
    assert set(r.model_id) == {"control/pixel", next(m for m in r.model_id if m.startswith("brainmodel_tiny"))}
