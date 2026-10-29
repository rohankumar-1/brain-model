"""Task registry: what can be asked of the model (``configs/multitask/registry.yaml``).

A task is a semantic request (a description, an output type and, for multiclass, class names) with label sources.
Its index in the registry is its identity everywhere (label table, batches, the task-ID ablation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

TYPES = ("binary", "regression", "multiclass", "segmentation", "survival")
ROLES = ("train", "held_out")
DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "multitask" / "registry.yaml"


# Task groups for balancing (sampling and loss are balanced over groups first, then over tasks inside a group, so
# the 30 SynthSeg tasks count as two groups, not as 30 tasks). A task's ``group`` in the YAML overrides these.
GROUP_RULES = (
    ("seg_", "anatomy_seg"), ("vol_", "anatomy_volume"), ("hemorrhage_", "hemorrhage"),
)
GROUP_OF = {
    "age": "demographics", "sex": "demographics",
    "neuro_diagnosis": "diagnosis", "tumor_type": "diagnosis",
    "whole_tumor": "tumor", "tumor_core": "tumor", "enhancing_tumor": "tumor", "resection_cavity": "tumor",
    "tumor_volume": "tumor",
    "brain_volume": "derived", "brain_mask": "derived", "brain_asymmetry": "derived",
    "intensity_asymmetry": "derived", "ct_csf_fraction": "derived", "ct_brain_density": "derived",
    "ct_calcification": "derived", "ct_skull_density": "derived",
}


def default_group(task_id: str, role: str) -> str:
    if role != "train":
        return "held_out"
    for prefix, group in GROUP_RULES:
        if task_id.startswith(prefix):
            return group
    return GROUP_OF.get(task_id, task_id)


@dataclass
class Task:
    id: str
    type: str
    descriptions: list[str]
    sources: list[dict]
    classes: list[str] = field(default_factory=list)
    role: str = "train"
    analog: str | None = None
    survival_bins: list[float] = field(default_factory=list)
    group: str = ""
    needs_skull: bool = False        # the target is about the skull / head outline: never asked of a skull-stripped scan

    @property
    def trained(self) -> bool:
        return self.role == "train"


TRAINABLE_TYPES = ("binary", "regression", "multiclass", "segmentation")     # types slot_losses has a loss for


@dataclass
class Registry:
    tasks: list[Task]

    def __post_init__(self):
        self.index = {t.id: i for i, t in enumerate(self.tasks)}
        if len(self.index) != len(self.tasks):
            raise ValueError("duplicate task ids in the registry")

    def __getitem__(self, key: str | int) -> Task:
        return self.tasks[self.index[key] if isinstance(key, str) else key]

    def __len__(self) -> int:
        return len(self.tasks)

    @property
    def groups(self) -> list[str]:
        return list(dict.fromkeys(t.group for t in self.tasks))

    @property
    def trained(self) -> list[Task]:
        return [t for t in self.tasks if t.trained]

    def train_ids(self, tk: dict | None) -> list[str]:
        """The trained tasks a run's ``tasks`` config section selects: ``groups`` / ``ids`` (None = all), minus
        ``exclude``."""
        tk = tk or {}
        only, ids, drop = set(tk.get("groups") or []), set(tk.get("ids") or []), set(tk.get("exclude") or [])
        out = [t.id for t in self.trained if (not only or t.group in only) and (not ids or t.id in ids)
               and t.id not in drop]
        lossless = [i for i in out if self[i].type not in TRAINABLE_TYPES]
        if lossless:                          # slot_losses has no branch for them: they would train silently on nothing
            raise SystemExit(f"tasks {lossless} have a type with no training loss (trainable: {TRAINABLE_TYPES})")
        return out

    def subset(self, ids: list[str]) -> "Registry":
        """The registry of a checkpoint trained on ``ids`` (its saved task order), from this one's definitions."""
        missing = [i for i in ids if i not in self.index]
        if missing:
            raise SystemExit(f"tasks {missing[:5]} of the checkpoint are no longer in the registry")
        return Registry([self[i] for i in ids])

    def max_classes(self) -> int:
        return max([len(t.classes) for t in self.tasks] + [1])


def load_registry(path: str | Path | None = None) -> Registry:
    raw = yaml.safe_load(Path(path or DEFAULT_REGISTRY).read_text())
    tasks = []
    for t in raw["tasks"]:
        desc = t["descriptions"] if isinstance(t["descriptions"], list) else [t["descriptions"]]
        task = Task(id=str(t["id"]), type=str(t["type"]), descriptions=[str(d) for d in desc],
                    sources=list(t.get("sources") or []), classes=[str(c) for c in t.get("classes") or []],
                    role=str(t.get("role", "train")), analog=t.get("analog"),
                    survival_bins=[float(b) for b in t.get("survival_bins") or []],
                    needs_skull=bool(t.get("needs_skull", False)))
        task.group = str(t.get("group") or default_group(task.id, task.role))
        if task.type not in TYPES:
            raise ValueError(f"task {task.id}: unknown type {task.type}")
        if task.role not in ROLES:
            raise ValueError(f"task {task.id}: unknown role {task.role}")
        if task.type == "multiclass" and len(task.classes) < 2:
            raise ValueError(f"task {task.id}: multiclass needs at least two classes")
        tasks.append(task)
    return Registry(tasks)
