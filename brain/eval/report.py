"""One report per run: three headline numbers against a reference run, then group scores; per-task detail to CSV.

    python -m brain.eval.report runs/joint/x/best.pt --ref runs/multitask/v7nosmall_s42/best.pt
    jobs/submit_report.sh RUN_CKPT REF_CKPT          # runs the evaluations below, then this report

* **In-domain**: the model's own heads on the trained tasks' test splits (``brain.multitask.test_eval``, paired with
  the reference on the same scans), grouped by registry task group, on the tasks both models trained.
* **Transfer**: the frozen encoder on the core downstream tasks (``configs/eval/core.yaml``; results from the records
  DB, ``brain.downstream.suite``).
* **Adaptation**: few-shot on the held-out tasks at one k (``brain.multitask.fewshot``, mean over seeds).

Every metric is on the skill scale (``brain.eval.skill``: 0 = chance, 1 = perfect). A headline is the mean over groups
of the mean over tasks, on the tasks both models have; its interval combines the per-task intervals assuming tasks are
independent (paired for in-domain, unpaired for transfer, over seeds for adaptation), so treat it as approximate.
Writes ``<out>/report.md`` and ``<out>/report_tasks.csv`` (default ``runs/eval/<run>``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from brain.common.paths import REPO_ROOT, resolve
from brain.eval.skill import scale, skill

CORE = REPO_ROOT / "configs" / "eval" / "core.yaml"
Z = 1.96


def model_id(ckpt: Path) -> str:
    """The checkpoint's records-DB id (``<run>/<file>@<fingerprint>``): suite results are matched on it exactly."""
    from brain.downstream.records import model_info

    return model_info(ckpt)["model_id"]


def label(ckpt: Path) -> str:
    """The checkpoint's display name (``run @17k`` / ``run/best``)."""
    from brain.downstream.results_view import model_label

    return model_label(model_id(ckpt))


def trained(ckpt: Path) -> set[str]:
    """The tasks a multitask / joint checkpoint was trained on (its registry minus its ``tasks`` selection); none for
    an SSL encoder."""
    import torch

    from brain.multitask.registry import load_registry

    ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    if "tasks" not in ck:                                       # an SSL encoder: no heads, transfer only
        return set()
    cfg = ck["config"]
    return set(load_registry(cfg["data"].get("registry")).subset(list(ck["tasks"])).train_ids(cfg.get("tasks")))


def default_out(ckpt: Path) -> Path:
    return REPO_ROOT / "runs" / "eval" / label(ckpt).replace("/", "_").replace(" @", "_")


def core_tasks(core: Path = CORE, with_seg: bool = False) -> list[str]:
    """The transfer tasks of the core set, in order (``jobs/submit_report.sh`` runs these through the suite)."""
    c = yaml.safe_load(open(core))
    groups = dict(c["transfer"]) | (dict(c["transfer_seg"]) if with_seg else {})
    return [t for ts in groups.values() for t in ts]


def heads_rows(csv: Path, run: str, ref: str, tasks: set[str]) -> pd.DataFrame:
    """In-domain rows on ``tasks``: per task, run and reference skill, paired difference and its standard error."""
    from brain.multitask.registry import load_registry

    if not csv.exists():
        return pd.DataFrame()
    d = pd.read_csv(csv)
    d = d[(d.kind == "heads") & d.task.isin(tasks)]
    reg = load_registry()
    ncls = {t.id: max(len(t.classes), 2) for t in reg.tasks}
    a, b = d[d.model == run].set_index("task"), d[d.model == ref].set_index("task")
    out = []
    for t in a.index.intersection(b.index):
        r, f = a.loc[t], b.loc[t]
        n = ncls.get(t, 2)
        p = r if pd.notna(r.get("delta_lo")) else f            # the paired interval sits on the non-reference row
        se = (p.delta_hi - p.delta_lo) / (2 * Z) * scale(r.metric, n) if pd.notna(p.get("delta_lo")) else np.nan
        out.append({"part": "in_domain", "group": r.group, "task": t, "metric": r.metric,
                    "run": skill(r.metric, r.score, n), "ref": skill(f.metric, f.score, n), "se": se})
    return pd.DataFrame(out)


def transfer_rows(groups: dict[str, list[str]], run: str, ref: str, rep: pd.DataFrame | None = None) -> pd.DataFrame:
    """Transfer rows for model ids ``run`` and ``ref``: the suite's chosen head per task, unpaired difference. Records
    match on the weights' fingerprint, so a copy of a checkpoint scored under another run name counts."""
    from brain.downstream.results_view import load

    def fp(mid):
        return mid.split("+")[0].rpartition("@")[2]

    rep = load(ingest=True) if rep is None else rep
    rep = rep[~rep.model_id.str.contains("+", regex=False)].copy()          # variants (other readouts) are other models
    rep["model_id"] = rep.model_id.map(lambda m: {fp(run): run, fp(ref): ref}.get(fp(m), m))
    rep = rep[rep.model_id.isin([run, ref])].drop_duplicates(["model_id", "task"])
    out = []
    for g, tasks in groups.items():
        for t in tasks:
            a, b = rep[(rep.model_id == run) & (rep.task == t)], rep[(rep.model_id == ref) & (rep.task == t)]
            if a.empty or b.empty:
                out.append({"part": "transfer", "group": g, "task": t, "metric": "", "run": np.nan, "ref": np.nan,
                            "se": np.nan, "missing": "run" if a.empty else "ref"})
                continue
            a, b = a.iloc[0], b.iloc[0]
            k = scale(a.metric)
            se = np.hypot((a.ci_hi - a.ci_lo) / (2 * Z), (b.ci_hi - b.ci_lo) / (2 * Z)) * k
            out.append({"part": "transfer", "group": g, "task": t, "metric": a.metric,
                        "run": skill(a.metric, a.test), "ref": skill(b.metric, b.test), "se": se})
    return pd.DataFrame(out)


def adaptation_rows(run_ckpt: Path, ref_ckpt: Path, k: int) -> pd.DataFrame:
    def load(ck: Path) -> pd.DataFrame:
        f = ck.parent / f"fewshot_{ck.stem}.json"
        if not f.exists():
            return pd.DataFrame()
        d = pd.DataFrame(json.load(open(f)).get("fewshot", []))
        d = d[d.k == k].copy()
        d["skill"] = [skill(m, v) for m, v in zip(d.metric, d.value)]
        return d.groupby("task").agg(metric=("metric", "first"), mean=("skill", "mean"), var=("skill", "var"),
                                     n=("skill", "size"))

    a, b = load(run_ckpt), load(ref_ckpt)
    if a.empty or b.empty:
        return pd.DataFrame()
    out = []
    for t in a.index.intersection(b.index):
        r, f = a.loc[t], b.loc[t]
        group = "ct" if t.startswith("cq500") else "lesion_seg" if r.metric == "dice" else "mri_dx"
        out.append({"part": "adaptation", "group": group, "task": t, "metric": r.metric, "run": r["mean"],
                    "ref": f["mean"], "se": float(np.sqrt(r["var"] / r.n + f["var"] / f.n))})
    return pd.DataFrame(out)


def headline(rows: pd.DataFrame) -> dict:
    """Mean over groups of task means for run and reference, the difference and its approximate 95% interval."""
    d = rows.dropna(subset=["run", "ref"])
    if d.empty:
        return {"run": np.nan, "ref": np.nan, "delta": np.nan, "lo": np.nan, "hi": np.nan, "n_tasks": 0}
    gs = d.groupby("group")
    w = d.group.map(1.0 / (gs.size() * gs.ngroups))                  # each task's weight in the mean of group means
    delta = float((w * (d.run - d.ref)).sum())
    se = float(np.sqrt((w ** 2 * d.se.fillna(0) ** 2).sum()))
    return {"run": float((w * d.run).sum()), "ref": float((w * d.ref).sum()), "delta": delta,
            "lo": delta - Z * se, "hi": delta + Z * se, "n_tasks": len(d)}


def compare(models: dict[str, Path], ref: str, heads_csv: Path, core: Path = CORE, with_seg: bool = False):
    """Every model in ``models`` (display name -> checkpoint) against ``models[ref]``: one row per (model, part, task)
    with run / ref skill and standard error, plus ``model`` / ``kind``. The notebook view of many reports at once;
    a model without heads (an SSL encoder) or without few-shot results just has no rows for that part."""
    from brain.downstream.results_view import load

    c = yaml.safe_load(open(core))
    groups = dict(c["transfer"]) | (dict(c["transfer_seg"]) if with_seg else {})
    rep = load(ingest=True)
    ref_ck = resolve(models[ref])
    ref_id, ref_heads, ref_tr = model_id(ref_ck), f"{ref_ck.parent.name}/{ref_ck.stem}", trained(ref_ck)
    parts = []
    for name, ck in models.items():
        ck = resolve(ck)
        mine = trained(ck)
        rows = [heads_rows(heads_csv, f"{ck.parent.name}/{ck.stem}", ref_heads, mine & ref_tr) if mine else None,
                transfer_rows(groups, model_id(ck), ref_id, rep),
                adaptation_rows(ck, ref_ck, int(c["adaptation"]["k"]))]
        for r in rows:
            if r is not None and len(r):
                parts.append(r.assign(model=name))
    return pd.concat(parts, ignore_index=True)


def headlines(rows: pd.DataFrame) -> pd.DataFrame:
    """``compare`` rows -> one row per (model, part): run and ref score, Δ, its 95% interval, task count."""
    out = [{"model": m, "part": p, **headline(g)} for (m, p), g in rows.groupby(["model", "part"], sort=False)]
    return pd.DataFrame(out)


def val_history(run_dir: Path) -> pd.DataFrame:
    """A run's validation history on the skill scale, recomputed from ``metrics.json`` (runs from before ``val_score``
    was logged included): one row per (step, group) plus group ``score`` (mean of group means)."""
    import re

    from brain.eval.skill import summarize
    from brain.multitask.registry import load_registry

    reg = load_registry()
    info = {t.id: (t.group, max(len(t.classes), 2)) for t in reg.tasks}
    pat = re.compile(r"^val_(.+)_(auc|r2|bacc|dice)$")
    out = []
    for h in json.load(open(resolve(run_dir) / "metrics.json")).get("history", []):
        rows = []
        for k, v in h.items():
            m = pat.match(k)
            if m and m.group(1) in info and v is not None:
                g, n = info[m.group(1)]
                rows.append((g, skill(m.group(2), v, n)))
        out += [{"step": int(h["step"]), "group": k.removeprefix("group_"), "skill": v}
                for k, v in summarize(rows).items()]
    return pd.DataFrame(out)


def render(run: str, ref: str, parts: dict[str, pd.DataFrame]) -> str:
    names = {"in_domain": "In-domain (heads)", "transfer": "Transfer (frozen)", "adaptation": "Adaptation (few-shot)"}
    lines = [f"run: {run}   vs   ref: {ref}", "skill: 0 = chance, 1 = perfect; mean over groups of task means", "",
             f"{'':24}{'run':>7}{'ref':>7}{'Δ':>8}   95% CI          tasks"]
    for p, rows in parts.items():
        h = headline(rows) if len(rows) else headline(pd.DataFrame(columns=["run", "ref", "group", "se"]))
        ci = f"[{h['lo']:+.3f}, {h['hi']:+.3f}]" if np.isfinite(h["delta"]) else ""
        lines.append(f"{names[p]:24}{h['run']:7.3f}{h['ref']:7.3f}{h['delta']:+8.3f}   {ci:16}{h['n_tasks']:>5}")
    for p, rows in parts.items():
        d = rows.dropna(subset=["run", "ref"]) if len(rows) else rows
        if not len(d):
            lines += ["", f"{names[p]}: no results for both models"]
            continue
        g = d.groupby("group").agg(run=("run", "mean"), ref=("ref", "mean"), n=("task", "size"))
        g["delta"] = g.run - g.ref
        lines += ["", f"{names[p]} by group", f"  {'group':18}{'run':>7}{'ref':>7}{'Δ':>8}{'tasks':>7}"]
        lines += [f"  {i:18}{r.run:7.3f}{r.ref:7.3f}{r.delta:+8.3f}{int(r.n):>7}" for i, r in g.iterrows()]
        miss = rows[rows.run.isna() | rows.ref.isna()] if len(rows) else rows
        if len(miss):
            lines.append(f"  missing: {', '.join(miss.task)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--heads", type=Path, default=None, help="test_eval heads CSV (default <out>/test_eval_heads.csv)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--core", type=Path, default=CORE)
    ap.add_argument("--with-seg", action="store_true", help="add the segmentation transfer group")
    a = ap.parse_args()
    ckpt, ref_ckpt = resolve(a.ckpt), resolve(a.ref)
    run, ref = label(ckpt), label(ref_ckpt)
    out = a.out or default_out(ckpt)
    out.mkdir(parents=True, exist_ok=True)
    core = yaml.safe_load(open(a.core))
    groups = dict(core["transfer"]) | (dict(core["transfer_seg"]) if a.with_seg else {})
    heads_csv = a.heads or out / "test_eval_heads.csv"
    run_heads = f"{ckpt.parent.name}/{ckpt.stem}"                     # test_eval's model names
    ref_heads = f"{ref_ckpt.parent.name}/{ref_ckpt.stem}"
    both = trained(ckpt) & trained(ref_ckpt)
    parts = {"in_domain": heads_rows(heads_csv, run_heads, ref_heads, both),
             "transfer": transfer_rows(groups, model_id(ckpt), model_id(ref_ckpt)),
             "adaptation": adaptation_rows(ckpt, ref_ckpt, int(core["adaptation"]["k"]))}
    text = render(run, ref, parts)
    print(text, end="")
    (out / "report.md").write_text("```\n" + text + "```\n")
    pd.concat([p for p in parts.values() if len(p)], ignore_index=True).to_csv(out / "report_tasks.csv", index=False)
    print(f"[report] -> {out}/report.md, report_tasks.csv")


if __name__ == "__main__":
    main()
