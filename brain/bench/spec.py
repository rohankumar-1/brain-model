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
    labels_dir: str | None                 # in-domain / few-shot label table
    readouts_file: Path | None             # the pinned readout per task (``python -m brain.bench pin``)
    raw: dict
    # later indexes that only add rows (``python -m brain.bench index-check``): results on them count as current
    index_compatible: tuple[str, ...] = ()

    @property
    def index_shas(self) -> tuple[str, ...]:
        return tuple(x for x in (self.index_sha1, *self.index_compatible) if x)

    def readouts(self) -> dict[str, str] | None:
        """task -> pinned readout head, or None before ``pin`` (then the leaderboard's validation choice)."""
        if self.readouts_file is None or not self.readouts_file.exists():
            return None
        return dict((yaml.safe_load(self.readouts_file.read_text()) or {}).get("readouts") or {})

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
                  seen={k: list(v) for k, v in (raw.get("seen") or {}).items()}, labels_dir=raw.get("labels_dir"),
                  readouts_file=REPO_ROOT / raw["readouts"] if raw.get("readouts") else None, raw=raw,
                  index_compatible=tuple(raw.get("index_compatible") or ()))
    missing = [t for t in b.tasks + b.seg_tasks if t not in modality]
    if missing:
        raise SystemExit(f"{path}: tasks without a modality: {missing}")
    return b


def index_check(old: Path, new: Path) -> bool:
    """Is ``new`` the ``old`` index plus rows? Every old volume must still be there with the same value in every old
    column (split, dedup_key, pool flags, ...); only then may ``new``'s sha1 join ``index_compatible``. ``vol_idx`` is
    the row number (it shifts when rows are inserted) and only the index's own slice table reads it: not compared."""
    import hashlib

    import pandas as pd

    key = ["volume_id", "dataset"]                          # volume_id alone repeats (cq500 / seg_cq500)
    a = pd.read_csv(old, low_memory=False).set_index(key)
    b = pd.read_csv(new, low_memory=False).set_index(key)
    gone, lost_cols = a.index.difference(b.index), [c for c in a.columns if c not in b.columns]
    common, cols = a.index.intersection(b.index), [c for c in a.columns if c in b.columns and c != "vol_idx"]
    x, y = (d.loc[common, cols].astype(str).fillna("<NA>") for d in (a, b))   # str keeps NaN, and NaN != NaN
    changed = {c: int((x[c] != y[c]).sum()) for c in cols if (x[c] != y[c]).any()}
    ok = not len(gone) and not lost_cols and not changed
    print(f"[index-check] {len(a):,} -> {len(b):,} volumes (+{len(b.index.difference(a.index)):,}); "
          f"missing {len(gone):,}; columns lost {lost_cols}; changed values {changed or 'none'}")
    print(f"[index-check] {'append-only' if ok else 'NOT append-only'}; new sha1 "
          f"{hashlib.sha1(Path(new).read_bytes()).hexdigest()}")
    return ok
