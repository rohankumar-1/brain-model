"""One YAML/JSON config loader for every method: defaults, file, then dotted CLI overrides."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _parse_value(raw: str) -> Any:
    """int, then float (YAML 1.1 reads ``1e-3`` as a string), then YAML for lists, bools and null."""
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return yaml.safe_load(raw)


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """``["optim.lr=1e-3", "train.max_steps=10"]`` -> nested dict. Values are parsed as YAML scalars/lists."""
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override must look like a.b=value, got {item!r}")
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_value(raw)
    return out


def load_config(
    path: Path | str | None = None,
    defaults: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge ``defaults`` <- file <- ``overrides``. Unknown keys are kept, so configs can extend defaults."""
    cfg = deepcopy(defaults or {})
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text()
        if path.suffix.lower() == ".json":
            user = json.loads(text)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            user = yaml.safe_load(text) or {}
        else:
            raise ValueError(f"unsupported config format {path.suffix!r} (use .yaml/.yml/.json)")
        if not isinstance(user, dict):
            raise ValueError(f"config root must be a mapping, got {type(user).__name__}")
        cfg = deep_update(cfg, user)
    return deep_update(cfg, overrides) if overrides else cfg


def save_config(cfg: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    else:
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
