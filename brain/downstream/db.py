"""SQLite view of the experiment records (``brain.downstream.records``): ingest, report, query.

    python -m brain.downstream.db ingest                       # fold new record files into the database
    python -m brain.downstream.db report                       # model x task headline table (head chosen on val)
    python -m brain.downstream.db report --task cq500 --all-heads
    python -m brain.downstream.db report --probes              # lesion / fingerprint / location probes
    python -m brain.downstream.db query "select task, head, value from metrics join evals using(eval_id) limit 5"

Tables
  models   model_id, ckpt, fingerprint, run_name, step, method, size_name, control, variant, train_config,
           provenance (git / store format / grid / index hash the model was trained on)
  evals    eval_id, model_id, suite, kind (task|probe), task, task_type, head, view, seed, protocol, store_format,
           grid, index_sha1, pretrain_overlap, n_train/n_val/n_test, git_commit, git_dirty, host, slurm_job_id,
           started_utc, finished_utc, artifacts, record_path
  metrics  eval_id, split (test|val), name, grp ('' or 'label:<x>' / 'cohort:<x>' / 'class:<x>'), value, ci_lo, ci_hi

The JSON records are the source of truth: deleting the database and re-ingesting rebuilds it. Run ``ingest`` from
one process at a time (the suite's final job does it; SQLite locking is unreliable on network filesystems).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from brain.downstream.metrics import HEADLINE
from brain.downstream.records import RESULTS_ROOT, record_dir

DB_PATH = RESULTS_ROOT / "experiments.sqlite"
PROBE_HEADLINE = {"probe:lesion": "lift", "probe:fingerprint": "dataset_bal_acc", "probe:location": "r2_mean",
                  "probe:registers": "dataset_bal_acc_mean", "probe:token_norm": "outlier_frac"}

SCHEMA = """
create table if not exists models (
  model_id text primary key, ckpt text, fingerprint text, run_name text, file text, step integer, best_step integer,
  method text, size_name text, control integer, variant text, train_config text, provenance text);
create table if not exists evals (
  eval_id text primary key, model_id text, suite text, kind text, task text, task_type text, head text, view text,
  seed integer, protocol text, store_format integer, grid text, index_sha1 text, pretrain_overlap real,
  n_train integer, n_val integer, n_test integer, git_commit text, git_dirty integer, host text, slurm_job_id text,
  started_utc text, finished_utc text, artifacts text, record_path text);
create table if not exists metrics (
  eval_id text, split text, name text, grp text, value real, ci_lo real, ci_hi real,
  primary key (eval_id, split, name, grp));
create table if not exists ingested (path text primary key);
create index if not exists evals_task on evals(task, model_id);
create index if not exists metrics_name on metrics(name);
"""


def _num(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)) and np.isfinite(v):
        return float(v)
    return None


def flatten(metrics: dict, split: str) -> list[tuple]:
    """(split, name, grp, value, ci_lo, ci_hi) rows. Scalars at the top; ``<name>_ci`` pairs become CI bounds;
    dicts of scalars are groups (per label / class), dicts of dicts are cohorts; lists other than CIs are skipped."""
    rows: dict[tuple, list] = {}

    def put(name: str, grp: str, value):
        v = _num(value)
        if v is not None:
            rows.setdefault((name, grp), [None, None, None])[0] = v

    for k, v in metrics.items():
        if k.endswith("_ci") and isinstance(v, (list, tuple)) and len(v) == 2:
            r = rows.setdefault((k[:-3], ""), [None, None, None])
            r[1], r[2] = _num(v[0]), _num(v[1])
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, dict):                       # by_cohort: {cohort: {metric: value}}
                    for mk, mv in sv.items():
                        put(mk, f"{k.removeprefix('by_')}:{sk}", mv)
                else:                                          # per label / per class: {label: value}
                    put(k, f"{'class' if 'class' in k else 'label'}:{sk}", sv)
        else:
            put(k, "", v)
    return [(split, n, g, *vals) for (n, g), vals in rows.items()]


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=300)          # parallel suite jobs ingest at once
    con.executescript(SCHEMA)
    return con


def ingest(con: sqlite3.Connection, root: Path | None = None) -> int:
    root = root or record_dir()
    done = {r[0] for r in con.execute("select path from ingested")}
    n = 0
    for path in sorted(root.glob("*/*.json")):
        if str(path) in done:
            continue
        rec = json.loads(path.read_text())
        m, d, env, proto = rec["model"], rec.get("data", {}), rec.get("env", {}), rec.get("protocol", {})
        con.execute("insert or replace into models values (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            m["model_id"], m.get("ckpt"), m.get("fingerprint"), m.get("run_name"), m.get("file"), m.get("step"),
            m.get("best_step"), m.get("method"), m.get("size_name"), int(bool(m.get("control"))), m.get("variant"),
            json.dumps(m.get("train_config")), json.dumps(m.get("provenance"))))
        git = env.get("git") or {}
        con.execute("insert or replace into evals values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            rec["eval_id"], m["model_id"], rec.get("suite"), rec.get("kind"), rec.get("task"), rec.get("task_type"),
            rec.get("head"), rec.get("view"), proto.get("seed"), json.dumps(proto), d.get("store_format"),
            json.dumps(d.get("grid_xyz")), d.get("index_sha1"), d.get("pretrain_overlap"), d.get("n_train"),
            d.get("n_val"), d.get("n_test"), git.get("commit"), int(bool(git.get("dirty"))), env.get("host"),
            env.get("slurm_job_id"), env.get("started_utc"), env.get("finished_utc"), json.dumps(rec.get("artifacts")),
            str(path)))
        for split, ms in (rec.get("metrics") or {}).items():
            if isinstance(ms, dict):
                con.executemany("insert or replace into metrics values (?,?,?,?,?,?,?)",
                                [(rec["eval_id"], *row) for row in flatten(ms, split)])
        con.execute("insert into ingested values (?)", (str(path),))
        n += 1
    con.commit()
    return n


def headline_name(task: str, task_type: str) -> str:
    if task.startswith("probe:"):
        return next(v for k, v in PROBE_HEADLINE.items() if task.startswith(k))
    return HEADLINE.get(task_type, "auc")


def table(con: sqlite3.Connection) -> pd.DataFrame:
    """One row per eval with its headline test value, CI and val value."""
    ev = pd.read_sql("select * from evals", con)
    if ev.empty:
        return ev
    ev["metric"] = [headline_name(t, tt or "") for t, tt in zip(ev.task, ev.task_type)]
    mt = pd.read_sql("select eval_id, split, name, value, ci_lo, ci_hi from metrics where grp = ''", con)
    test = mt[mt.split == "test"].rename(columns={"value": "test"}).drop(columns="split")
    val = mt[mt.split == "val"][["eval_id", "name", "value"]].rename(columns={"value": "val"})
    out = ev.merge(test, left_on=["eval_id", "metric"], right_on=["eval_id", "name"], how="left").drop(columns="name")
    return out.merge(val, left_on=["eval_id", "metric"], right_on=["eval_id", "name"], how="left").drop(columns="name")


def report(con: sqlite3.Connection, *, task: str | None = None, all_heads: bool = False, probes: bool = False,
           latest: bool = True) -> pd.DataFrame:
    t = table(con)
    if t.empty:
        return t
    t = t[t.kind == ("probe" if probes else "task")]
    if task:
        t = t[t.task == task]
    if latest:                                   # re-runs of the same (model, task, head, view, protocol seed) -> newest
        t = t.sort_values("finished_utc").drop_duplicates(["model_id", "task", "head", "view", "seed"], keep="last")
    if all_heads or probes:
        cols = ["model_id", "task", "head", "view", "metric", "val", "test", "ci_lo", "ci_hi", "n_test",
                "pretrain_overlap", "finished_utc"]
        return t[cols].sort_values(["task", "model_id", "head"]).reset_index(drop=True)
    # one head per (model, task): best on val among the heads that have a val score. Never on test: before
    # 2026-10-25 a task whose val split had no positives fell back to the best test score ("test!"), an optimistic
    # pick (CQ500 EDH: 2 test positives); such a task now gets picked_on "none" and should be cross-validated.
    rows = []
    for (mid, tk), g in t.groupby(["model_id", "task"]):
        if g.val.notna().any():
            pick, by = g.loc[g.val.idxmax()], "val" if g.val.notna().all() else "val (some heads lack it)"
        else:
            pick, by = g.iloc[0], "none"
        rows.append({"model_id": mid, "task": tk, "metric": pick.metric, "test": pick.test, "ci_lo": pick.ci_lo,
                     "ci_hi": pick.ci_hi, "head": pick["head"], "picked_on": by, "overlap": pick.pretrain_overlap})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["ingest", "report", "query", "export"])
    ap.add_argument("sql", nargs="?", default=None)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--task", default=None)
    ap.add_argument("--all-heads", action="store_true")
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--wide", action="store_true", help="report: pivot to models x tasks")
    ap.add_argument("--out", type=Path, default=None, help="export: CSV path for the flat eval table")
    a = ap.parse_args()
    con = connect(a.db)
    if a.action == "ingest":
        print(f"[db] ingested {ingest(con)} new records -> {a.db}")
        return
    ingest(con)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    if a.action == "query":
        print(pd.read_sql(a.sql, con).to_string(index=False))
    elif a.action == "export":
        out = a.out or a.db.with_suffix(".csv")
        table(con).to_csv(out, index=False)
        print(f"[db] wrote {out}")
    else:
        r = report(con, task=a.task, all_heads=a.all_heads, probes=a.probes)
        if r.empty:
            print("[db] no records yet")
        elif a.wide and not (a.all_heads or a.probes):
            print(r.pivot(index="model_id", columns="task", values="test").round(3).to_string())
        else:
            print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
