"""One report per run: three headline numbers against a reference run, then group scores; per-task detail to CSV.

    python -m brain.eval.report runs/joint/x/best.pt --ref runs/multitask/v7nosmall_s42/best.pt
    jobs/submit_report.sh RUN_CKPT REF_CKPT          # runs the evaluations below, then this report

* **In-domain**: the model's own heads on the trained tasks' test splits (``brain.multitask.test_eval`` per-scan
  predictions), grouped by registry task group, on the tasks both models trained.
* **Transfer**: the frozen encoder on the core downstream tasks (``configs/eval/core.yaml``; per-scan predictions of
  the frozen suite, ``brain.downstream.suite``). One head per task for every model, fixed on validation (the encoder
  readout with the best mean validation score across all evaluated models); the multitask models' own outputs and
  slots (``taskout`` / ``slots`` / ``general`` readouts) are not transfer and never count. A task whose labels a model
  trained on (``core.yaml: seen``) is left out of that model's comparisons.
* **Adaptation**: few-shot on the held-out tasks at one k (``brain.multitask.fewshot``), paired seed by seed.

Every metric is on the skill scale (``brain.eval.skill``: 0 = chance, 1 = perfect). A headline is the mean over groups
of the mean over tasks, on the tasks both models have, scored on the scans both models answered. Intervals come from
one paired bootstrap per headline: each subject (``dedup_key``) gets a Poisson(1) weight shared by every task and
model it appears in, so scans of one person, and tasks sharing scans, move together; the headline is recomputed in
each replicate and its 2.5 / 97.5 percentiles are the interval (per-task intervals come from the same replicates).
Writes ``<out>/report.md`` and ``<out>/report_tasks.csv`` (default ``runs/eval/<run>``).
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from brain.common.paths import REPO_ROOT, resolve
from brain.eval.skill import skill

CORE = REPO_ROOT / "configs" / "eval" / "core.yaml"
HEADS_PREDS = REPO_ROOT / "runs" / "eval" / "heads_all" / "test_eval_preds.parquet"
SUPERVISED_READOUTS = ("slots", "taskout", "general")       # the multitask model's own outputs, not its encoder
PARTS = {"in_domain": "In-domain (heads)", "transfer": "Transfer (frozen)", "adaptation": "Adaptation (few-shot)"}


# ------------------------------------------------------------------------------------------------ models
def model_id(ckpt: Path) -> str:
    """The checkpoint's records-DB id (``<run>/<file>@<fingerprint>``)."""
    from brain.downstream.records import model_info

    return model_info(ckpt)["model_id"]


def fingerprint(mid: str) -> str:
    return mid.split("+")[0].rpartition("@")[2]


def label(ckpt: Path) -> str:
    """The checkpoint's display name (``run @17k`` / ``run/best``)."""
    from brain.downstream.results_view import model_label

    return model_label(model_id(ckpt))


def heads_name(ckpt: Path) -> str:
    """The model name ``test_eval`` writes (``<run dir>/<file stem>``)."""
    return f"{ckpt.parent.name}/{ckpt.stem}"


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


@lru_cache(maxsize=1)
def subject_of() -> dict[str, str]:
    """volume_id -> subject key (``dedup_key``, what the splits group by)."""
    from brain.downstream.tasks import DEFAULT_VOLUMES

    v = pd.read_csv(DEFAULT_VOLUMES, usecols=["volume_id", "dedup_key"], low_memory=False)
    return dict(zip(v.volume_id.astype(str), v.dedup_key.astype(str)))


@lru_cache(maxsize=1)
def n_classes() -> dict[str, int]:
    from brain.multitask.registry import load_registry

    return {t.id: max(len(t.classes), 2) for t in load_registry().tasks}


# ------------------------------------------------------------------------------------------------ weighted metrics
def _auc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted ROC AUC for every row of weights ``w`` (R, n) at once (ties count one half)."""
    y = y.astype(bool)
    if y.all() or not y.any():
        return np.full(w.shape[0], np.nan)
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[:, order]
    _, start = np.unique(s, return_index=True)                  # tie groups of equal scores
    grp = np.searchsorted(start, np.arange(len(s)), side="right") - 1
    wn = np.where(~y, w, 0.0)
    wp = np.where(y, w, 0.0)
    g_neg = np.zeros((w.shape[0], len(start)))
    np.add.at(g_neg.T, grp, wn.T)
    below = np.cumsum(g_neg, axis=1) - g_neg                     # negative weight strictly below each tie group
    pairs = (wp * (below[:, grp] + 0.5 * g_neg[:, grp])).sum(1)
    tot = wp.sum(1) * wn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, pairs / tot, np.nan)


def _r2(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = w.sum(1, keepdims=True)
    mean = (w * y).sum(1, keepdims=True) / np.maximum(sw, 1e-12)
    ss_res = (w * (y - p) ** 2).sum(1)
    ss_tot = (w * (y - mean) ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)


def _bacc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> np.ndarray:
    recalls = []
    for c in np.unique(y):
        m = y == c
        den = w[:, m].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            recalls.append(np.where(den > 0, w[:, m & (p == c)].sum(1) / den, np.nan))
    return np.nanmean(np.stack(recalls), axis=0)


def metric_reps(metric: str, y: np.ndarray, p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The task metric under each row of bootstrap weights ``w`` (R, n). Multilabel targets (n, L): macro AUC over the
    labels that have both classes."""
    fam = metric.split("_")[0]
    if fam == "auc":
        if y.ndim == 1 and p.ndim == 2 and p.shape[1] > 2:          # multiclass, one-vs-rest macro
            y = (y.astype(int)[:, None] == np.arange(p.shape[1])[None]).astype(float)
        if y.ndim == 2 and y.shape[1] > 1:
            if p.ndim == 1:
                return np.full(w.shape[0], np.nan)
            cols = [_auc(y[:, j], p[:, j], w) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
            return np.nanmean(np.stack(cols), axis=0) if cols else np.full(w.shape[0], np.nan)
        y = y.reshape(-1)
        p = p[:, -1] if p.ndim == 2 else p
        return _auc(y, p, w)
    if fam == "r2":
        return _r2(y.reshape(-1).astype(float), (p[:, 0] if p.ndim == 2 else p).astype(float), w)
    if fam == "bacc":
        p = p.argmax(1) if p.ndim == 2 else p
        return _bacc(y.reshape(-1).astype(int), p.astype(int), w)
    if fam == "dice":                                           # per-scan Dice rows: the weighted mean
        return (w * p).sum(1) / np.maximum(w.sum(1), 1e-12)
    raise ValueError(f"no metric {metric}")


# ------------------------------------------------------------------------------------------------ inputs
def pair_rows(part: str, group: str, task: str, metric: str, runs: list, refs: list, ncls: int = 2) -> dict | None:
    """One paired task: ``runs`` / ``refs`` are lists (one per seed; heads and transfer have one) of
    (volume_ids, y, p). Restricted to the scans present in every list."""
    common = set(runs[0][0])
    for v, _, _ in runs + refs:
        common &= set(v)
    if len(common) < 3:
        return None
    vids = sorted(common)

    def align(item):
        idx = {v: i for i, v in enumerate(item[0])}
        sel = [idx[v] for v in vids]
        return np.asarray(item[1])[sel], np.asarray(item[2])[sel]

    return {"part": part, "group": group, "task": task, "metric": metric, "ncls": ncls, "vids": vids,
            "run": [align(r) for r in runs], "ref": [align(r) for r in refs]}


def heads_pairs(preds: pd.DataFrame, run: str, ref: str, tasks: set[str]) -> list[dict]:
    from brain.multitask.registry import load_registry

    group = {t.id: t.group for t in load_registry().tasks}
    metric = {"binary": "auc", "regression": "r2", "multiclass": "bacc", "segmentation": "dice"}
    out = []
    d = preds[preds.task.isin(tasks) & preds.model.isin([run, ref])]
    for t, g in d.groupby("task"):
        a, b = g[g.model == run], g[g.model == ref]
        if a.empty or b.empty:
            continue
        kind = a.kind.iloc[0]
        r = pair_rows("in_domain", group.get(t, "?"), t, metric[kind],
                      [(a.volume_id.tolist(), a.y.to_numpy(), a.p.to_numpy())],
                      [(b.volume_id.tolist(), b.y.to_numpy(), b.p.to_numpy())], n_classes().get(t, 2))
        if r:
            out.append(r)
    return out


@lru_cache(maxsize=1)
def suite_heads() -> pd.DataFrame:
    """Every (model, task, head) of the frozen suite with its val / test score, protocol and prediction file."""
    from brain.downstream import db

    con = db.connect()
    db.ingest(con)
    t = db.table(con)
    t = t[t.kind == "task"].sort_values("finished_utc").drop_duplicates(["model_id", "task", "head", "view", "seed"],
                                                                          keep="last")
    t = t[~t.model_id.str.contains("+", regex=False)].copy()
    t["fp"] = t.model_id.map(fingerprint)
    t["pred"] = [json.loads(a).get(h) if isinstance(a, str) and a else None for a, h in zip(t.artifacts, t["head"])]
    t["cv"] = [bool((json.loads(p) if isinstance(p, str) and p else {}).get("cv")) for p in t.protocol]
    t["encoder"] = ~t["head"].astype(str).str.startswith(SUPERVISED_READOUTS)
    return t


def fixed_heads(t: pd.DataFrame) -> dict[str, str]:
    """Per task, the encoder readout with the best mean validation score over every evaluated (non-random) model."""
    e = t[t.encoder & t.val.notna() & ~t.model_id.str.startswith("random")]
    means = e.groupby(["task", "head"]).val.mean()
    return {task: g.idxmax()[1] for task, g in means.groupby(level=0)}


def _suite_preds(row) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    if not row.pred or not Path(row.pred).exists():
        return None
    z = np.load(row.pred, allow_pickle=False)
    return [str(v) for v in z["volume_ids"]], np.asarray(z["y_true"]), np.asarray(z["y_score"])


def transfer_pairs(run_ck: Path, ref_ck: Path, groups: dict[str, list[str]], seen: dict[str, list[str]],
                   run_tr: set[str], ref_tr: set[str]) -> tuple[list[dict], list[dict]]:
    """Paired transfer rows, and notes on tasks left out (missing results, protocol mismatch, seen labels)."""
    t = suite_heads()
    heads = fixed_heads(t)
    fp_run, fp_ref = fingerprint(model_id(run_ck)), fingerprint(model_id(ref_ck))
    out, notes = [], []
    for g, tasks in groups.items():
        for task in tasks:
            if set(seen.get(task, [])) & (run_tr | ref_tr):
                notes.append({"task": task, "why": "labels seen in training"})
                continue
            picked = []
            for fp in (fp_run, fp_ref):
                m = t[(t.fp == fp) & (t.task == task) & t.encoder]
                row = m[m["head"] == heads.get(task)]
                if row.empty and len(m) and m.val.notna().any():      # this model lacks the fixed head
                    row = m.loc[[m.val.idxmax()]]
                picked.append(row.iloc[-1] if len(row) else None)
            if any(p is None for p in picked):
                notes.append({"task": task, "why": "no result for " + ("run" if picked[0] is None else "ref")})
                continue
            if picked[0].cv != picked[1].cv:
                notes.append({"task": task, "why": "protocols differ (CV vs fixed split)"})
                continue
            a, b = _suite_preds(picked[0]), _suite_preds(picked[1])
            if a is None or b is None:
                notes.append({"task": task, "why": "prediction file missing"})
                continue
            r = pair_rows("transfer", g, task, picked[0].metric, [a], [b])
            if r:
                r["head"] = picked[0]["head"] if picked[0]["head"] == picked[1]["head"] else \
                    f"{picked[0]['head']} / {picked[1]['head']}"
                out.append(r)
    return out, notes


def adaptation_pairs(run_ck: Path, ref_ck: Path, k: int) -> list[dict]:
    def load(ck: Path) -> dict:
        f = ck.parent / f"fewshot_{ck.stem}.json"
        if not f.exists():
            return {}
        rows = json.load(open(f)).get("preds", [])
        return {(r["task"], int(r["seed"])): r for r in rows if int(r["k"]) == k}

    a, b = load(run_ck), load(ref_ck)
    out = []
    for task in sorted({t for t, _ in a} & {t for t, _ in b}):
        seeds = sorted({s for t, s in a if t == task} & {s for t, s in b if t == task})
        if not seeds:
            continue
        m = a[(task, seeds[0])]["metric"]
        group = "ct" if task.startswith("cq500") else "lesion_seg" if m == "dice" else "mri_dx"
        r = pair_rows("adaptation", group, task, m,
                      [(a[(task, s)]["volume_id"], a[(task, s)]["y"], a[(task, s)]["p"]) for s in seeds],
                      [(b[(task, s)]["volume_id"], b[(task, s)]["y"], b[(task, s)]["p"]) for s in seeds])
        if r:
            out.append(r)
    return out


# ------------------------------------------------------------------------------------------------ bootstrap
def summarise(rows: list[dict], reps: int = 1000, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Per-task and headline skill for run and ref, with the paired subject-level Poisson bootstrap."""
    empty = {"run": np.nan, "ref": np.nan, "delta": np.nan, "lo": np.nan, "hi": np.nan, "n_tasks": 0}
    if not rows:
        return pd.DataFrame(), empty
    subj = subject_of()
    keys = sorted({subj.get(v, v) for r in rows for v in r["vids"]})
    pos = {k: i for i, k in enumerate(keys)}
    rng = np.random.default_rng(seed)
    w_subj = np.vstack([np.ones(len(keys)), rng.poisson(1.0, size=(reps, len(keys)))])   # row 0: the point estimate
    per_task, run_s, ref_s = [], [], []
    for r in rows:
        w = w_subj[:, [pos[subj.get(v, v)] for v in r["vids"]]]
        sk = lambda pairs: np.nanmean(np.stack([skill(r["metric"], metric_reps(r["metric"], y, p, w), r["ncls"])  # noqa: E731
                                                for y, p in pairs]), axis=0)
        a, b = sk(r["run"]), sk(r["ref"])
        run_s.append(a)
        ref_s.append(b)
        d = a - b
        per_task.append({"part": r["part"], "group": r["group"], "task": r["task"], "metric": r["metric"],
                         "n": len(r["vids"]), "seeds": len(r["run"]), "head": r.get("head", ""),
                         "run": a[0], "ref": b[0], "delta": d[0],
                         "lo": np.nanpercentile(d[1:], 2.5), "hi": np.nanpercentile(d[1:], 97.5)})
    groups = [r["group"] for r in rows]

    def headline(m: np.ndarray) -> np.ndarray:                  # (tasks, R+1) -> mean of group means per replicate
        g = pd.DataFrame(m.T, columns=pd.MultiIndex.from_arrays([groups, range(len(groups))]))
        return g.T.groupby(level=0).mean().mean(axis=0).to_numpy()

    hr, hf = headline(np.stack(run_s)), headline(np.stack(ref_s))
    hd = hr - hf
    return pd.DataFrame(per_task), {"run": hr[0], "ref": hf[0], "delta": hd[0], "lo": np.nanpercentile(hd[1:], 2.5),
                                    "hi": np.nanpercentile(hd[1:], 97.5), "n_tasks": len(rows)}


def compare_pair(run_ck: Path, ref_ck: Path, heads_preds: Path = HEADS_PREDS, core: Path = CORE,
                 with_seg: bool = False, reps: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(per-task rows, headline rows, notes on left-out tasks) for one run against one reference."""
    c = yaml.safe_load(open(core))
    groups = dict(c["transfer"]) | (dict(c["transfer_seg"]) if with_seg else {})
    run_ck, ref_ck = resolve(run_ck), resolve(ref_ck)
    run_tr, ref_tr = trained(run_ck), trained(ref_ck)
    preds = pd.read_parquet(heads_preds) if Path(heads_preds).exists() else pd.DataFrame(columns=["model", "task"])
    tr_rows, notes = transfer_pairs(run_ck, ref_ck, groups, c.get("seen") or {}, run_tr, ref_tr)
    parts = {"in_domain": heads_pairs(preds, heads_name(run_ck), heads_name(ref_ck), run_tr & ref_tr),
             "transfer": tr_rows,
             "adaptation": adaptation_pairs(run_ck, ref_ck, int(c["adaptation"]["k"]))}
    tasks, heads = [], []
    for p, rows in parts.items():
        per, h = summarise(rows, reps=reps)
        tasks.append(per)
        heads.append({"part": p, **h})
    return pd.concat(tasks, ignore_index=True), pd.DataFrame(heads), pd.DataFrame(notes)


def compare(models: dict[str, Path], ref: str, heads_preds: Path = HEADS_PREDS, core: Path = CORE,
            with_seg: bool = False, reps: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every model in ``models`` (display name -> checkpoint) against ``models[ref]``: per-task rows, headline rows and
    left-out-task notes, each with a ``model`` column. The notebook's view of many reports at once."""
    tasks, heads, notes = [], [], []
    for name, ck in models.items():
        if name == ref:
            continue
        t, h, n = compare_pair(ck, models[ref], heads_preds, core, with_seg, reps)
        tasks.append(t.assign(model=name))
        heads.append(h.assign(model=name))
        notes.append(n.assign(model=name))
    return (pd.concat(tasks, ignore_index=True), pd.concat(heads, ignore_index=True),
            pd.concat(notes, ignore_index=True))


# ------------------------------------------------------------------------------------------------ text
def render(run: str, ref: str, tasks: pd.DataFrame, heads: pd.DataFrame, notes: pd.DataFrame) -> str:
    lines = [f"run: {run}   vs   ref: {ref}",
             "skill: 0 = chance, 1 = perfect; mean over groups of task means; paired subject bootstrap 95% CI", "",
             f"{'':24}{'run':>7}{'ref':>7}{'Δ':>8}   95% CI            tasks"]
    for _, h in heads.iterrows():
        ci = f"[{h.lo:+.3f}, {h.hi:+.3f}]" if np.isfinite(h.delta) else ""
        lines.append(f"{PARTS[h.part]:24}{h.run:7.3f}{h.ref:7.3f}{h.delta:+8.3f}   {ci:18}{int(h.n_tasks):>5}")
    for p in PARTS:
        d = tasks[tasks.part == p] if len(tasks) else tasks
        if not len(d):
            lines += ["", f"{PARTS[p]}: no results for both models"]
            continue
        g = d.groupby("group").agg(run=("run", "mean"), ref=("ref", "mean"), n=("task", "size"))
        g["delta"] = g.run - g.ref
        lines += ["", f"{PARTS[p]} by group", f"  {'group':18}{'run':>7}{'ref':>7}{'Δ':>8}{'tasks':>7}"]
        lines += [f"  {i:18}{r.run:7.3f}{r.ref:7.3f}{r.delta:+8.3f}{int(r.n):>7}" for i, r in g.iterrows()]
    if len(notes):
        lines += ["", "left out of transfer:"] + [f"  {r.task}: {r.why}" for _, r in notes.iterrows()]
    return "\n".join(lines) + "\n"


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--heads", type=Path, default=None, help="test_eval predictions (default <out>/test_eval_preds.parquet)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--core", type=Path, default=CORE)
    ap.add_argument("--with-seg", action="store_true", help="add the segmentation transfer group")
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    ckpt, ref_ckpt = resolve(a.ckpt), resolve(a.ref)
    out = a.out or default_out(ckpt)
    out.mkdir(parents=True, exist_ok=True)
    tasks, heads, notes = compare_pair(ckpt, ref_ckpt, a.heads or out / "test_eval_preds.parquet", a.core,
                                       a.with_seg, a.reps)
    text = render(label(ckpt), label(ref_ckpt), tasks, heads, notes)
    print(text, end="")
    (out / "report.md").write_text("```\n" + text + "```\n")
    tasks.to_csv(out / "report_tasks.csv", index=False)
    print(f"[report] -> {out}/report.md, report_tasks.csv")


if __name__ == "__main__":
    main()
