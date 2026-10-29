"""Config loading and the shipped YAMLs. No torch needed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from brain.common.config import deep_update, load_config, parse_overrides, save_config

ROOT = Path(__file__).resolve().parents[1]


def _defaults():
    # defaults.py is plain Python; load it by path so this test does not import the torch-heavy train package
    spec = importlib.util.spec_from_file_location("brain_defaults", ROOT / "brain" / "train" / "defaults.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_deep_update_merges_nested_and_replaces_scalars():
    out = deep_update({"a": {"b": 1, "c": 2}, "d": 3}, {"a": {"b": 9}, "d": [1]})
    assert out == {"a": {"b": 9, "c": 2}, "d": [1]}


def test_parse_overrides_builds_nested_dict_with_yaml_values():
    assert parse_overrides(["optim.lr=1e-3", "train.max_steps=10", "views.global.size_xyz=[64,64,32]"]) == {
        "optim": {"lr": 0.001}, "train": {"max_steps": 10}, "views": {"global": {"size_xyz": [64, 64, 32]}}}


def test_load_config_precedence_defaults_file_overrides(tmp_path):
    td = tmp_path
    path = Path(td) / "c.yaml"
    path.write_text("optim: {lr: 0.5}\nextra: 1\n")
    cfg = load_config(path, defaults={"optim": {"lr": 1.0, "wd": 0.1}}, overrides={"optim": {"wd": 0.2}})
    assert cfg == {"optim": {"lr": 0.5, "wd": 0.2}, "extra": 1}
    save_config(cfg, Path(td) / "out" / "saved.yaml")
    assert yaml.safe_load((Path(td) / "out" / "saved.yaml").read_text()) == cfg


def test_every_shipped_config_only_uses_known_keys():
    d = _defaults()
    bad = []

    def walk(base, user, path):
        for k, v in user.items():
            if k not in base:
                bad.append(".".join(path + [k]))
            elif isinstance(v, dict) and isinstance(base[k], dict):
                walk(base[k], v, path + [k])

    files = sorted(f for f in (ROOT / "configs").glob("*/*.yaml")                    # not the task registry or the
                   if f.name != "registry.yaml" and f.parent.name != "bench")        # benchmark spec / model registry
    assert files
    for f in files:
        walk(d.default_config(f.parent.name), yaml.safe_load(f.read_text()), [f.name])
    assert not bad, bad


def test_new_augmentation_behaviour_is_the_default_for_3d():
    aug = _defaults().default_config("lejepa3d")["aug"]
    for key in ("curve", "bias", "through_plane", "skull"):
        assert float(aug[key]["p"]) > 0, key
    assert aug["curve"]["flip"] > 0


def test_smoke_config_shrinks_the_3d_model():
    d = _defaults()
    cfg = d.smoke_config(d.default_config("lejepa3d"))
    assert cfg["data"]["volume_size_xyz"] == [32, 32, 16] and cfg["model"]["size"] == "test"
