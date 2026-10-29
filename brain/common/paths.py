"""Repository root and path resolution shared by trainers, prep and downstream code."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Cluster locations; override with BRAIN_SCRATCH / BRAIN_DATASETS on any other machine.
SCRATCH = Path(os.environ.get("BRAIN_SCRATCH", "/home/roku/orcd/scratch/brain-model"))
DATASETS = Path(os.environ.get("BRAIN_DATASETS", "/home/roku/orcd/pool/datasets"))
PROCESSED_ROOT = SCRATCH / "data" / "processed" / "slices_zarr"


def resolve(path: str | Path, root: Path = REPO_ROOT) -> Path:
    """Absolute paths pass through; relative ones are taken from the repository root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else (root / path).resolve()
