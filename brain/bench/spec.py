"""The benchmark spec (``configs/bench/benchmark.yaml``) and what it means for one model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from brain.common.paths import REPO_ROOT

SPEC = REPO_ROOT / "configs" / "bench" / "benchmark.yaml"


@dataclass
class Benchmark:
    version: int
    index_sha1: str | None
    reference: str
    groups: dict[str, list[str]]           # transfer group -> tasks
    seg_groups: dict[str, list[str]]
    modality: dict[str, str]               # task -> ct | mri
    seen: dict[str, list[str]]             # benchmark task -> registry tasks whose training makes it in-domain
    raw: dict

    @property
    def tasks(self) -> list[str]:
        return [t for ts in self.groups.values() for t in ts]

    @property
    def seg_tasks(self) -> list[str]:
        return [t for ts in self.seg_groups.values() for t in ts]

    def group_of(self, task: str) -> str:
        return next(g for g, ts in {**self.groups, **self.seg_groups}.items() if task in ts)

    def tasks_for(self, model, with_seg: bool = False) -> list[str]:
        """The benchmark tasks ``model`` (a ``ModelRef``) is scored on: its modalities, segmentation for our 3D
        encoders only (``with_seg``), its seen tasks left out."""
        mods = {str(m).lower()[:2] for m in (model.spec.get("modalities") or ("ct", "mri"))}
        tasks = self.tasks + (self.seg_tasks if with_seg and model.is_checkpoint else [])
        return [t for t in tasks if self.modality.get(t, "")[:2] in mods and t not in self.seen_by(model)]

    def seen_by(self, model) -> set[str]:
        """Benchmark tasks whose labels ``model`` trained on: a registry model's ``seen_tasks``; for ours, the tasks
        ``seen`` maps to a registry task it trained."""
        if not model.is_checkpoint:
            return set(model.seen_tasks)
        trained = set(model.seen_tasks)
        return {t for t, reg in self.seen.items() if set(reg) & trained}


def load(path: Path = SPEC) -> Benchmark:
    raw = yaml.safe_load(Path(path).read_text())
    modality = {t: m for m, ts in (raw.get("modality") or {}).items() for t in ts}
    b = Benchmark(version=int(raw.get("version", 1)), index_sha1=raw.get("index_sha1"),
                  reference=str(raw["reference"]), groups=dict(raw["transfer"]),
                  seg_groups=dict(raw.get("transfer_seg") or {}), modality=modality,
                  seen={k: list(v) for k, v in (raw.get("seen") or {}).items()}, raw=raw)
    missing = [t for t in b.tasks + b.seg_tasks if t not in modality]
    if missing:
        raise SystemExit(f"{path}: tasks without a modality: {missing}")
    return b
