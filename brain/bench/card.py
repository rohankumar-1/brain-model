"""A model's card: every number the evaluation suite has recorded for it, in one place.

    python -m brain.bench card --model BQ-A1          # -> runs/bench/cards/<name>.md

Sections (each only when its results exist; ``docs/benchmark.md`` explains every column):

1. Transfer (frozen encoder): per benchmark task the readout used (the validation-fixed head of the leaderboard), the
   task's metric on test with its 95% interval, on validation, its skill, the test size, the pretraining overlap; then
   the skill per group and per modality (mean over groups of the mean over tasks).
2. Segmentation transfer (BrainQuery checkpoints): Dice per task on test with its interval.
3. Token probes (``probe3d``, BrainQuery checkpoints): lesion detection from frozen tokens, scanner / dataset
   fingerprint, token location.
4. In-domain (BrainQuery models with task heads): their own heads on the trained tasks' test splits, with the paired
   difference to the reference.
5. Adaptation: few-shot on the held-out tasks (task embedding fitted on k scans) next to a linear probe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import REPO_ROOT

CARDS = REPO_ROOT / "runs" / "bench" / "cards"


def _fmt(v, d: int = 3) -> str:
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def transfer_rows(model, b, tasks: list[str]) -> pd.DataFrame:
    """One row per benchmark task: the leaderboard's readout for this model on the pinned index, scored like the
    leaderboard scores it (its saved per-scan predictions, every CV repeat saved, subject bootstrap)."""
    from brain.bench.board import _pick, current_results
    from brain.eval.report import _suite_preds, fixed_heads, score_ci, suite_heads
    from brain.eval.skill import skill

    t = suite_heads()
    t = t[t.task.isin(tasks)]
    t = t[current_results(t, b)]
    heads = b.readouts() or fixed_heads(t)
    rows = []
    for task in tasks:
        r = _pick(t, model.model_id, task, heads.get(task))
        row = {"group": b.group_of(task), "task": task, "modality": b.modality[task]}
        if r is not None:
            p = _suite_preds(r)
            test, lo, hi = score_ci(str(r.metric), p) if p else (r.test, r.ci_lo, r.ci_hi)
            row.update(metric=r.metric, head=r["head"], test=test, lo=lo, hi=hi, val=r.val,
                       n_test=len(p[0][0]) if p else r.n_test, overlap=r.pretrain_overlap, cv=bool(r.cv),
                       repeats=len(p) if p else 0,
                       skill=float(skill(str(r.metric), test)) if pd.notna(test) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def headline(df: pd.DataFrame) -> float:
    d = df.dropna(subset=["skill"]) if "skill" in df else df.iloc[0:0]
    return float(d.groupby("group").skill.mean().mean()) if len(d) else np.nan


def probe_rows(model) -> pd.DataFrame:
    from brain.downstream import db

    t = db.table(db.connect())
    t = t[(t.model_id == model.model_id) & (t.kind == "probe")]
    t = t.sort_values("finished_utc").drop_duplicates(["task", "head"], keep="last")
    return t[["task", "head", "metric", "test", "ci_lo", "ci_hi", "n_test"]].sort_values(["task", "head"])


def card(target: str, out: Path = CARDS) -> str:
    from brain.bench.run import HEADS_DIR, has_heads
    from brain.bench.spec import load
    from brain.downstream.models import resolve_model

    b = load()
    m = resolve_model(target)
    ref = resolve_model(b.reference)
    seen = sorted(b.seen_by(m))
    lines = [f"# {m.name}", "",
             f"`{m.model_id}` · class **{m.training_class}** · modalities "
             f"{', '.join(m.spec.get('modalities') or ['ct', 'mri'])}" + (f" · {m.spec['note']}" if m.spec.get("note")
                                                                          else ""),
             f"Benchmark v{b.version} (index `{(b.index_sha1 or '-')[:10]}`); reference {ref.name}."
             + (f" Trained on {len(m.seen_tasks)} registry tasks; benchmark tasks with seen labels: "
                f"{', '.join(seen) or 'none'}." if m.is_checkpoint and m.seen_tasks else ""), ""]

    tasks = [t for t in b.tasks if b.modality[t][:2] in {x[:2] for x in (m.spec.get("modalities") or ["ct", "mri"])}]
    tr = transfer_rows(m, b, tasks)
    lines += ["## 1. Transfer (frozen encoder)", "",
              "| Group | Task | Mod | Metric | Readout | Test [95% CI] | Val | Skill | n test | Overlap |",
              "|---|---|---|---|---|---|---:|---:|---:|---:|"]
    for r in tr.itertuples():
        if not hasattr(r, "test") or pd.isna(getattr(r, "test", np.nan)):
            lines.append(f"| {r.group} | {r.task} | {r.modality} | | | not run | | | | |")
            continue
        flag = " (seen)" if r.task in seen else ""
        cv = (f" (CV, {r.repeats} rep.)" if r.repeats > 1 else " (CV, 1st rep.)") if r.cv else ""
        lines.append(f"| {r.group} | {r.task}{flag} | {r.modality} | {r.metric} | {r.head}{cv} "
                     f"| {_fmt(r.test)} [{_fmt(r.lo)}, {_fmt(r.hi)}] | {_fmt(r.val)} | {_fmt(r.skill)} | "
                     f"{int(r.n_test) if pd.notna(r.n_test) else '-'} | {_fmt(r.overlap, 2)} |")
    if "cv" in tr and tr.cv.fillna(False).any():
        lines += ["", "Test is scored from the saved per-scan predictions, as on the leaderboard. CV tasks: the mean over "
                  "the saved repeats (`1st rep.`: a result from before 2026-10-27, which kept only the first of 3)."]
    scored = tr[~tr.task.isin(seen)] if "skill" in tr else tr
    if "skill" in scored and scored.skill.notna().any():
        g = scored.dropna(subset=["skill"]).groupby("group").skill.agg(["mean", "count"])
        lines += ["", "Skill by group: " + ", ".join(f"{k} {v['mean']:.3f} ({int(v['count'])})" for k, v in g.iterrows()),
                  "Headline (mean over groups): " + ", ".join(
                      f"{mod.upper()} {headline(scored[scored.modality == mod]):.3f}" for mod in sorted(scored.modality.unique())
                      if scored[scored.modality == mod].skill.notna().any())]
    lines.append("")

    if m.is_checkpoint:
        seg = transfer_rows(m, b, b.seg_tasks)
        if "test" in seg and seg.test.notna().any():
            lines += ["## 2. Segmentation transfer", "", "| Task | Readout | Dice [95% CI] | n test |", "|---|---|---|---:|"]
            for r in seg.dropna(subset=["test"]).itertuples():
                lines.append(f"| {r.task} | {r.head} | {_fmt(r.test)} [{_fmt(r.lo)}, {_fmt(r.hi)}] | {int(r.n_test)} |")
            lines.append("")
        pr = probe_rows(m)
        if len(pr):
            lines += ["## 3. Token probes (probe3d)", "", "| Probe | Head | Metric | Test [95% CI] | n test |",
                      "|---|---|---|---|---:|"]
            for r in pr.itertuples():
                lines.append(f"| {r.task.removeprefix('probe:')} | {r.head} | {r.metric} | {_fmt(r.test)} "
                             f"[{_fmt(r.ci_lo)}, {_fmt(r.ci_hi)}] | {int(r.n_test) if pd.notna(r.n_test) else '-'} |")
            lines.append("")

    if has_heads(m):
        from brain.eval.report import default_out

        f = HEADS_DIR / m.cache_key / "test_eval_heads.csv"
        if not f.exists():                                     # the run report's own-head evaluation
            f = default_out(m.ckpt) / "test_eval_heads.csv"
        if f.exists():
            h = pd.read_csv(f)
            mine = h[h.model == f"{m.ckpt.parent.name}/{m.ckpt.stem}"]
            # tasks it trained (an untrained head's output means nothing) with enough test scans
            # (brain.eval.report.MIN_IN_DOMAIN_SCANS)
            mine = mine[mine.task.isin(m.seen_tasks) & (mine.n >= 20)]
            lines += ["## 4. In-domain (own heads, test splits)", "",
                      f"The tasks it trained, with at least 20 test scans; Δ is the paired difference to {ref.name} on "
                      "the same scans.",
                      "", "| Group | Task | Metric | n | Score [95% CI] | Δ vs reference [95% CI] |",
                      "|---|---|---|---:|---|---|"]
            for r in mine.sort_values(["group", "task"]).itertuples():
                d = f"{r.delta_vs_ref:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]" if "delta_vs_ref" in h and \
                    pd.notna(getattr(r, "delta_vs_ref", np.nan)) else "-"
                lines.append(f"| {r.group} | {r.task} | {r.metric} | {r.n} | {_fmt(r.score)} [{_fmt(r.ci_lo)}, "
                             f"{_fmt(r.ci_hi)}] | {d} |")
            lines.append("")
        fs = m.ckpt.parent / f"fewshot_{m.ckpt.stem}.json"
        if fs.exists():
            d = pd.DataFrame(json.loads(fs.read_text()).get("fewshot", []))
            if len(d):
                g = d.groupby(["task", "k", "metric"]).agg(value=("value", "mean"), sd=("value", "std"),
                                                          probe=("linear_probe", "mean"), seeds=("seed", "nunique"),
                                                          n_test=("n_test", "first")).reset_index()
                lines += ["## 5. Adaptation (few-shot)", "",
                          "Held-out tasks: the task embedding fitted on k labelled scans (mean ± sd over seeds), next "
                          "to a linear probe on the frozen pooled embedding with the same scans.", "",
                          "| Task | k | Metric | Few-shot | Linear probe | Seeds | n test |", "|---|---:|---|---|---:|---:|---:|"]
                for r in g.itertuples():
                    lines.append(f"| {r.task} | {r.k} | {r.metric} | {_fmt(r.value)} ± {_fmt(r.sd)} | {_fmt(r.probe)} | "
                                 f"{r.seeds} | {r.n_test} |")
                lines.append("")
    text = "\n".join(lines)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{m.name.replace(' ', '_').replace('/', '_')}.md").write_text(text + "\n")
    return text
