"""The leaderboard: every model paired with the spec's reference on the benchmark transfer tasks, per modality.

For each model and modality: the tasks both it and the reference can be scored on (their modalities; neither trained
on the task's labels), one readout per task fixed on validation across all models (``brain.eval.report.fixed_heads``),
per-scan predictions paired on the scans both answered, and the paired subject bootstrap of ``brain.eval.report``
(skill: 0 = chance, 1 = perfect; a headline is the mean over groups of the mean over tasks). Results computed on a
store index other than the spec's ``index_sha1`` are left out and counted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import REPO_ROOT

CLASS_ORDER = {"specialist": 0, "supervised": 1, "ssl": 2, "baseline": 3}
OUT = REPO_ROOT / "runs" / "bench"


def current_results(t: pd.DataFrame, b) -> pd.Series:
    """Rows computed under this benchmark version: on its store index, with the suite's default eval window (multi)
    and plain encoding (no refinement / fine grid / taps variant) and, for cross-validated tasks, with cross-fitted
    validation (v2; v1 CV results are left out and counted as stale)."""
    ok = t.index_sha1.isin(b.index_shas) if b.index_sha1 else pd.Series(True, index=t.index)
    ok &= t.variant == "multi"        # the suite's default window (CT: brain, subdural, bone), plain encoding
    return ok & (t.cv != "cv") if b.version >= 2 else ok


def discover(t: pd.DataFrame, tasks: list[str], min_tasks: int = 5) -> list[str]:
    """Models with results on at least ``min_tasks`` benchmark tasks: their checkpoint path, or registry name."""
    from brain.downstream import db

    have = t[t.task.isin(tasks)].groupby("model_id").task.nunique()
    ids = set(have[have >= min_tasks].index)
    m = pd.read_sql("select model_id, ckpt, run_name from models", db.connect())
    out = []
    for r in m[m.model_id.isin(ids)].itertuples():
        out.append(r.ckpt if isinstance(r.ckpt, str) and r.ckpt.endswith(".pt") and Path(r.ckpt).exists()
                   else r.run_name)
    return sorted(set(out))


def _pick(t: pd.DataFrame, model_id: str, task: str, head: str | None):
    m = t[(t.model_id == model_id) & (t.task == task) & t.encoder]
    row = m[m["head"] == head]
    if row.empty and len(m) and m.val.notna().any():             # this model lacks the fixed head: its best on val
        row = m.loc[[m.val.idxmax()]]
    return row.iloc[-1] if len(row) else None


def pairs(model, ref, b, t: pd.DataFrame, heads: dict, tasks: list[str]) -> tuple[list[dict], list[dict]]:
    """Paired rows for ``model`` vs ``ref`` over ``tasks``, and notes on the tasks left out."""
    from brain.eval.report import _suite_preds, pair_rows

    seen = b.seen_by(model) | b.seen_by(ref)
    mods = [{str(m).lower()[:2] for m in (x.spec.get("modalities") or ("ct", "mri"))} for x in (model, ref)]
    out, notes = [], []
    for task in tasks:
        mod = b.modality[task][:2]
        if not all(mod in m for m in mods):
            continue
        if task in seen:
            notes.append({"task": task, "why": "labels seen in training"})
            continue
        a, r = _pick(t, model.model_id, task, heads.get(task)), _pick(t, ref.model_id, task, heads.get(task))
        if a is None or r is None:
            notes.append({"task": task, "why": "no result for " + ("model" if a is None else "reference")})
            continue
        if a.cv != r.cv:
            notes.append({"task": task, "why": f"protocols differ ({a.cv or 'fixed'} vs {r.cv or 'fixed'})"})
            continue
        pa, pr = _suite_preds(a), _suite_preds(r)
        if pa is None or pr is None:
            notes.append({"task": task, "why": "prediction file missing"})
            continue
        row = pair_rows("transfer", b.group_of(task), task, a.metric, pa, pr)
        if row:
            row["head"] = a["head"] if a["head"] == r["head"] else f"{a['head']} / {r['head']}"
            row["modality"] = b.modality[task]
            row["overlap"] = a.pretrain_overlap
            out.append(row)
    return out, notes


def board(models: list[str] | None = None, ref: str | None = None, with_seg: bool = False, reps: int = 1000,
          out: Path = OUT) -> pd.DataFrame:
    from brain.bench.spec import load
    from brain.downstream.models import resolve_model
    from brain.eval.report import fixed_heads, suite_heads, summarise

    b = load()
    tasks = b.tasks + (b.seg_tasks if with_seg else [])
    t_all = suite_heads()
    t_all = t_all[t_all.task.isin(tasks)]
    current = current_results(t_all, b)
    t = t_all[current]
    heads = b.readouts() or fixed_heads(t)
    refm = resolve_model(ref or b.reference)
    targets = models if models else discover(t, tasks)
    resolved = [refm] + [m for m in (resolve_model(x) for x in targets) if m.model_id != refm.model_id]
    rows, per_task, notes = [], [], []
    for m in resolved:
        stale = int((t_all[~current].model_id == m.model_id).sum())
        prs, nts = pairs(m, refm, b, t, heads, tasks)
        notes += [{"model": m.name, **n} for n in nts]
        for mod in sorted({r["modality"] for r in prs}):
            sub = [r for r in prs if r["modality"] == mod]
            per, h = summarise(sub, reps=reps)
            ov = [r["overlap"] for r in sub if r["overlap"] is not None and np.isfinite(r["overlap"])]
            rows.append({"modality": mod, "model": m.name, "model_id": m.model_id, "training_class": m.training_class,
                         "tasks": h["n_tasks"], "skill": h["run"], "ref_skill": h["ref"],
                         "delta": h["delta"] if m is not refm else 0.0, "lo": h["lo"] if m is not refm else 0.0,
                         "hi": h["hi"] if m is not refm else 0.0, "overlap": float(np.mean(ov)) if ov else np.nan,
                         "stale_results": stale, "left_out": sum(1 for n in nts)})
            if len(per):
                per_task.append(per.assign(model=m.name, model_class=m.training_class, modality=mod))
    tab = pd.DataFrame(rows)
    if tab.empty:
        raise SystemExit("no benchmark results to compare (run python -m brain.bench run --model ...)")
    tab = tab.sort_values(["modality", "training_class", "skill"],
                          key=lambda s: s.map(CLASS_ORDER) if s.name == "training_class"
                          else (-s if s.name == "skill" else s)).reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"board_v{b.version}"
    tab.to_csv(stem.with_suffix(".csv"), index=False)
    if per_task:
        per = pd.concat(per_task, ignore_index=True)
        per.to_csv(stem.with_name(stem.name + "_tasks.csv"), index=False)
        wide, md = render_tasks(per, tab, b)
        wide.to_csv(stem.with_name(stem.name + "_all_tasks.csv"))
        stem.with_name(stem.name + "_all_tasks.md").write_text(md)
    if notes:
        pd.DataFrame(notes).to_csv(stem.with_name(stem.name + "_left_out.csv"), index=False)
    stem.with_suffix(".md").write_text(render(tab, refm, b))
    print(render(tab, refm, b))
    print(f"[board] -> {stem}.md / .csv / _tasks.csv / _all_tasks.md" + (" / _left_out.csv" if notes else ""))
    return tab


def render_tasks(per: pd.DataFrame, tab: pd.DataFrame, b) -> tuple[pd.DataFrame, str]:
    """Every task x every model: the skill each model gets on it (paired scans with the reference, its readout), per
    modality in the leaderboard's model order, the best per task in bold. Returns the wide table and its markdown."""
    order = {(m, mod): i for i, (m, mod) in enumerate(zip(tab.model, tab.modality))}
    rank = {t: i for i, t in enumerate(b.tasks)}
    lines = [f"# Benchmark v{b.version}: every task", "",
             "Skill per task (0 = chance, 1 = perfect) on the scans each model shares with the reference; the "
             "headline row is the leaderboard's (mean over groups of the mean over tasks). Readout: the pinned head "
             "(a model without it uses its own best on validation). Bold: the best on the task. -: not scored "
             "(another modality, labels seen in training, or no result yet).", ""]
    wides = []
    for mod, g in per.groupby("modality", sort=True):
        models = sorted(g.model.unique(), key=lambda m: order.get((m, mod), 1e9))
        w = g.pivot_table(index=["group", "task"], columns="model", values="run", aggfunc="last")[models]
        w = w.reindex(sorted(w.index, key=lambda gt: rank.get(gt[1], 1e9)))
        info = g.groupby("task").agg(metric=("metric", "first"), n=("n", "max"),
                                     head=("head", lambda h: h.mode().iat[0]))
        lines += [f"## {mod.upper()}", "", "| Group | Task | Metric | Readout | n | " + " | ".join(models) + " |",
                  "|---|---|---|---|---:|" + "---:|" * len(models)]
        for (grp, task), row in w.iterrows():
            best = row.max()
            cells = ["-" if not np.isfinite(v) else (f"**{v:.3f}**" if v == best else f"{v:.3f}") for v in row]
            i = info.loc[task]
            lines.append(f"| {grp} | {task} | {i.metric} | {i['head']} | {int(i.n)} | " + " | ".join(cells) + " |")
        head = tab[tab.modality == mod].set_index("model").skill
        lines += ["| | **headline** | | | | " + " | ".join(f"{head.get(m, np.nan):.3f}" for m in models) + " |", ""]
        wides.append(w.assign(modality=mod).set_index("modality", append=True))
    return pd.concat(wides), "\n".join(lines)


def render(tab: pd.DataFrame, ref, b) -> str:
    lines = [f"# Benchmark v{b.version}: transfer, frozen encoders", "",
             f"Reference: `{ref.name}`. Skill: 0 = chance, 1 = perfect; mean over task groups of the mean over tasks, "
             "on the tasks a model shares with the reference. Δ: model − reference with its paired 95% interval "
             "(subject bootstrap). Overlap: mean fraction of a task's scans the model may have seen in pretraining.", ""]
    for mod, g in tab.groupby("modality", sort=True):
        lines += [f"## {mod.upper()} tasks", "",
                  "| Model | Class | Tasks | Skill | Δ vs reference [95% CI] | Overlap | Left out | Stale |",
                  "|---|---|---:|---:|---|---:|---:|---:|"]
        full = g.loc[g.model_id == ref.model_id, "tasks"].max() if (g.model_id == ref.model_id).any() else g.tasks.max()
        for r in g.itertuples():
            d = "reference" if r.model_id == ref.model_id else f"{r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" + \
                (" *" if (r.lo > 0 or r.hi < 0) else "")
            if r.tasks + r.left_out < full:                     # still being evaluated: not a comparable headline
                d += " (partial)"
            ov = "-" if not np.isfinite(r.overlap) else f"{r.overlap:.2f}"
            lines.append(f"| {r.model} | {r.training_class} | {r.tasks} | {r.skill:.3f} | {d} | {ov} | {r.left_out} | "
                         f"{r.stale_results} |")
        lines.append("")
    lines += ["`*`: the interval excludes 0. Left out: tasks dropped for that model (seen labels, missing results, "
              "protocol mismatch; see `_left_out.csv`). Stale: results on another store index, or cross-validated "
              "without cross-fitted validation (before v2), not used. "
              "(partial): results missing on some of its tasks, so its numbers are not yet comparable.", ""]
    return "\n".join(lines)


def pin(models: list[str] | None = None) -> dict[str, str]:
    """Choose each benchmark task's readout once: the head with the best mean validation score over ``models`` (default:
    every model with results on all benchmark tasks of its modalities, on the pinned index), and write it to the spec's
    ``readouts`` file. Pinning makes a model's score independent of which other models have been evaluated."""
    import datetime

    import yaml

    from brain.bench.spec import load
    from brain.downstream.models import resolve_model
    from brain.eval.report import fixed_heads, suite_heads

    b = load()
    if b.readouts_file is None:
        raise SystemExit("the spec has no readouts file")
    t = suite_heads()
    t = t[t.task.isin(b.tasks + b.seg_tasks)]
    t = t[current_results(t, b)]
    refs = [resolve_model(x) for x in (models or discover(t, b.tasks))]
    complete = []
    for m in refs:
        have = set(t[t.model_id == m.model_id].task)
        if set(b.tasks_for(m)) <= have:
            complete.append(m)
    if not complete:
        raise SystemExit("no model has results on all of its benchmark tasks yet")
    heads = fixed_heads(t[t.model_id.isin({m.model_id for m in complete})])
    doc = {"version": b.version, "pinned_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "from_models": [m.name for m in complete], "index_sha1": b.index_sha1,
           "readouts": {k: heads[k] for k in b.tasks + b.seg_tasks if k in heads}}
    b.readouts_file.write_text("# The pinned readout per benchmark task (python -m brain.bench pin): the head with the best "
                               "mean validation\n# score over the models listed, never chosen on test. Re-pin only with a "
                               "new benchmark version.\n" + yaml.safe_dump(doc, sort_keys=False))
    print(f"[pin] {len(doc['readouts'])} readouts from {', '.join(doc['from_models'])} -> {b.readouts_file}")
    return doc["readouts"]
