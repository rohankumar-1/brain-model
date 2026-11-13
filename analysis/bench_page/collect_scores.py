"""Scores + 95% CI for every model x benchmark task, with per-label rows for the multilabel bleed tasks
(feeds build_page.py, the "Brain Model Task Scores" artifact).   python collect_scores.py scores.json"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/roku/brain-model")
from brain.bench.spec import load
from brain.bench.board import _pick, current_results
from brain.downstream.models import resolve_model
from brain.downstream.tasks import CQ500_LABELS, RSNA_LABELS
from brain.eval.report import _suite_preds, fixed_heads, score_ci, suite_heads

MODELS = ["BQ-random", "ct_fm", "ct_fm_triage", "brainiac", "BQ-v3", "BQ-R7", "BQ-R7-SSL", "BQ-R9", "BQ-R10", "BQ-R10-EMA", "BQ-R12", "BQ-R13", "BQ-R13-EMA", "BQ-R14", "BQ-R14-EMA", "BQ-R14F", "BQ-R16", "BQ-FISO", "BQ-FNGEO", "BQ-R17", "BQ-R18", "brainfm", "hlip"]
LABELS = {"cq500": CQ500_LABELS, "rsna_ich": RSNA_LABELS}
b = load()
t = suite_heads(); t = t[current_results(t, b)]
heads = b.readouts() or fixed_heads(t)
out = {}
for name in MODELS:
    m = resolve_model(name); seen = b.seen_by(m)
    for task in b.tasks:
        r = _pick(t, m.model_id, task, heads.get(task))
        p = _suite_preds(r) if r is not None else None
        if not p:
            continue
        v, lo, hi = score_ci(str(r.metric), p)
        out.setdefault(task, {})[name] = {"v": round(v, 3), "lo": round(lo, 3), "hi": round(hi, 3), "n": len(p[0][0]),
                                          "seen": task in seen, "metric": str(r.metric)}
        if task in LABELS and np.asarray(p[0][1]).ndim == 2:
            for j, lab in enumerate(LABELS[task]):
                pj = [(vids, np.asarray(y)[:, j], np.asarray(s)[:, j]) for vids, y, s in p]
                v, lo, hi = score_ci("auc", pj)
                out.setdefault(f"{task}:{lab}", {})[name] = {
                    "v": round(v, 3), "lo": round(lo, 3), "hi": round(hi, 3), "n": len(p[0][0]),
                    "pos": int(np.asarray(p[0][1])[:, j].sum()), "seen": task in seen, "metric": "auc"}
json.dump(out, open(sys.argv[1], "w"), indent=0)
print(len(out), "rows")
