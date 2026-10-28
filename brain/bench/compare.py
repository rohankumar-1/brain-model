"""Every model x every benchmark task, in the task's own metric (python -m brain.bench compare).

The leaderboard (``board``) reports skill paired with the reference; this table is the plain read-out: per task, each
model's test score (AUROC or R2) on the pinned readout with its 95% subject-bootstrap interval, scored as the model
cards score it (``card.transfer_rows``: saved per-scan predictions, every CV repeat). All models share each task's
splits, so a row compares the same test subjects. A task whose labels a model trained on is shown as ``seen``; one of
a modality the model does not take is blank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from brain.bench.board import OUT

METRIC = {"auc": "AUROC", "auc_ovr_macro": "AUROC (macro)", "r2": "R²"}


def compare(models: list[str] | None = None, out: Path = OUT) -> pd.DataFrame:
    from brain.bench.card import headline, transfer_rows
    from brain.bench.spec import load
    from brain.downstream.models import registry, resolve_model
    from brain.eval.report import suite_heads

    b = load()
    t = suite_heads()
    t = t[t.task.isin(b.tasks)]
    if models:
        refs = [resolve_model(x) for x in models]
    else:                              # the registered models (configs/bench/models.yaml) that have results
        have, refs = set(t.model_id), []
        for name in registry():
            try:
                m = resolve_model(name)
            except (SystemExit, FileNotFoundError, KeyError) as e:   # e.g. a planned run not trained yet
                print(f"[compare] skipping {name}: {e}")
                continue
            if m.model_id in have:
                refs.append(m)
    rows, heads = [], {}
    for m in refs:
        mine = b.tasks_for(m)
        seen = b.seen_by(m)
        df = transfer_rows(m, b, mine)
        for mod in sorted({b.modality[x][:2] for x in mine}):
            heads[(m.name, mod)] = headline(df[df.modality.str[:2] == mod])
        for task in b.tasks:
            if task in seen:
                rows.append({"model": m.name, "task": task, "cell": "seen"})
        for r in df.itertuples():
            if pd.notna(getattr(r, "test", np.nan)):
                rows.append({"model": m.name, "task": r.task, "test": r.test, "lo": r.lo, "hi": r.hi,
                             "metric": r.metric, "head": r.head, "n_test": r.n_test, "skill": r.skill})
    long = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"compare_v{b.version}"
    long.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(render(long, heads, b))
    print(f"[compare] {len(refs)} models x {len(b.tasks)} tasks -> {stem}.md / .csv")
    return long


def render(long: pd.DataFrame, heads: dict, b) -> str:
    lines = [f"# Benchmark v{b.version}: every model on every task", "",
             "Test score on the pinned readout with its 95% subject-bootstrap interval (CV tasks: mean over repeats). "
             "All models share each task's splits. Bold: best on the task. `seen`: the model trained on these labels "
             "(not transfer). Skill row: 0 = chance, 1 = perfect, mean over groups of the mean over tasks, over the "
             "tasks the model has (the leaderboard's Δ pairs each model with the reference instead).", ""]
    scored = long.dropna(subset=["test"]) if "test" in long else long.iloc[0:0]
    for mod in ("ct", "mri"):
        tasks = [x for x in b.tasks if b.modality[x][:2] == mod]
        names = [n for (n, md), _ in sorted(heads.items(), key=lambda kv: -np.nan_to_num(kv[1], nan=-9))
                 if md == mod]
        if not names:
            continue
        lines += [f"## {mod.upper()}", "", "| Group | Task | Metric | n test | " + " | ".join(names) + " |",
                  "|---|---|---|---:|" + "---:|" * len(names)]
        for task in tasks:
            s = scored[scored.task == task]
            if s.empty:
                continue
            best = s[s.model.isin(names)].test.max()
            cells = []
            for n in names:
                hit = long[(long.model == n) & (long.task == task)]
                if hit.empty:
                    cells.append("")
                elif "cell" in hit and hit.cell.notna().any():
                    cells.append("seen")
                else:
                    r = hit.iloc[0]
                    v = f"{r.test:.3f} [{r.lo:.2f}, {r.hi:.2f}]"
                    cells.append(f"**{v}**" if np.isclose(r.test, best) else v)
            metric = METRIC.get(str(s.metric.iloc[0]), str(s.metric.iloc[0]))
            lines.append(f"| {b.group_of(task)} | {task} | {metric} | {int(s.n_test.max())} | "
                         + " | ".join(cells) + " |")
        lines += ["| | **skill** | | | " + " | ".join(f"**{heads[(n, mod)]:.3f}**" for n in names) + " |", ""]
    return "\n".join(lines)
