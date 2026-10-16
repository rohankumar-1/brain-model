"""Persist downstream suite results in a comparable layout."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from downstream.metrics import HEADLINE, summarize_for_table


def new_run_dir(root: Path, task: str, vae_ckpt: Path, run_name: str | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name or stamp
    vae_tag = f"{vae_ckpt.parent.name}_{vae_ckpt.stem}"
    out = root / task / vae_tag / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "heads").mkdir(exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    return out


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not serializable: {type(o)}")


def save_head_result(
    run_dir: Path,
    head_name: str,
    metrics: dict,
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    volume_ids: list[str],
) -> None:
    write_json(run_dir / "heads" / f"{head_name}.json", metrics)
    np.savez_compressed(
        run_dir / "predictions" / f"{head_name}.npz",
        y_true=y_true,
        y_score=y_score,
        volume_ids=np.asarray(volume_ids),
    )


def eval_banner(title: str, *detail: str, width: int = 72) -> None:
    """Large boxed header so a long embed log still has an obvious eval name."""
    title = title.replace("_", " ")
    width = max(width, len(title) + 6, *(len(d) + 4 for d in detail if d))
    bar = "=" * width
    print(f"\n{bar}\n  {title}", flush=True)
    for d in detail:
        if d:
            print(f"  {d}", flush=True)
    print(bar, flush=True)


def format_rows(headers: list[str], rows: list[list[str]], *, left: int = 2) -> str:
    """Spaced columns; the first ``left`` are left-aligned, the rest right-aligned."""
    widths = [max(len(h), 6, *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        parts = [f"{c:<{w}}" if i < left else f"{c:>{w}}" for i, (c, w) in enumerate(zip(cells, widths))]
        return "   ".join(parts)

    return "\n".join([line(headers), *(line(r) for r in rows)])


def print_head_table(task_name: str, task_type: str, rows: list[tuple[str, str, dict]]) -> None:
    """One row per head after every model for this task has been scored."""
    key = HEADLINE[task_type]
    finite = [m[key] for _, _, m in rows if isinstance(m.get(key), (int, float)) and np.isfinite(m[key])]
    best = max(finite) if finite else float("nan")
    table = []
    for name, view, m in rows:
        val = float(m[key]) if key in m else float("nan")
        mark = " *" if np.isfinite(val) and abs(val - best) < 1e-9 else "  "
        lo, hi = m.get(f"{key}_ci", (float("nan"), float("nan")))
        param = f"{m['chosen_param']:g}" if "chosen_param" in m else ""
        table.append([name, view, f"{val:.4f}{mark}", f"[{lo:.4f}, {hi:.4f}]", param])
    eval_banner(task_name, "summary")
    print(format_rows(["head", "view", key, "95% CI", "param"], table), flush=True)
    print(flush=True)


def finalize_run(run_dir: Path, config: dict, all_metrics: dict[str, dict]) -> Path:
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "metrics.json", all_metrics)
    rows = [summarize_for_table(name, m) for name, m in all_metrics.items()]
    df = pd.DataFrame(rows)
    csv_path = run_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    # also append to a rolling leaderboard under task/
    board = run_dir.parent.parent / "leaderboard.csv"
    df2 = df.copy()
    df2.insert(0, "run_dir", str(run_dir))
    df2.insert(1, "vae", config.get("vae_ckpt"))
    df2.insert(2, "time_utc", config.get("time_utc"))
    if board.exists():
        prev = pd.read_csv(board)
        df2 = pd.concat([prev, df2], ignore_index=True)
    df2.to_csv(board, index=False)
    return csv_path
