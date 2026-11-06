"""The "BrainQuery Headline Scores" artifact page (user, 2026-11-03): the task rows of the full scores page, four models.
Left to right: the random-init floor, the best from-scratch single-task model (ViT or ResNet, picked per task family by
its mean test score), the best earlier BrainQuery run (by mean frozen skill) and the newest run (+ its weight average).
    python collect_scores.py scores.json && python build_focus.py scores.json OUT.html"""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_page import COMMON, FP_TASKS, G, HERE, IND, REPO, RSNA_HEAD, S, SKILL, TE, TEFP, cell, marks  # noqa: E402

NEWEST = [("BQ-R14", "R14"), ("BQ-R14-EMA", "R14 avg")]          # the newest run; two columns while it has two
EARLIER = max((m for m in SKILL if m not in dict(NEWEST) and m != "BQ-random"), key=SKILL.get)
# from-scratch single-task runs (configs/single/*_scratch_*), scored by brain.multitask.test_eval on the test split
FAMILIES = {fam: {f"{fam}_scratch_vit": "ViT", f"{fam}_scratch_resnet": "ResNet", f"{fam}_scratch_hybrid": "CS-ViT"}
            for fam in ("hemorrhage", "tumor", "demographics", "anatomy", "ct_measures", "radiology")}
# oldest first, so the newest evaluation of a model / task wins (drop_duplicates keep="last" below): segmentation was
# re-scored 2026-11-03 with Dice on scans with a target only (test_eval_seg_v2, test_eval_hemseg_v2)
_te = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{REPO}/runs/diagnostics/test_eval_*/test_eval_heads.csv"),
                                                    key=lambda f: Path(f).stat().st_mtime)])
_te = _te[(_te.kind == "heads") & (_te.metric != "fp")]
sys.path.insert(0, REPO)
import yaml  # noqa: E402
from brain.multitask.registry import load_registry  # noqa: E402

_reg = load_registry(None)


def trained_tasks(run: str) -> set[str]:
    """The tasks a single-task run trained; its checkpoint carries (untrained) heads for every registry task."""
    cfg = Path(f"{REPO}/runs/single/{run}/config.yaml")
    return set(_reg.train_ids(yaml.safe_load(cfg.read_text())["tasks"])) if cfg.exists() else set()


SPEC, SPEC_NOTE = {}, []
for fam, runs in FAMILIES.items():
    scored = {r: _te[(_te.model == f"{r}/last") & _te.task.isin(trained_tasks(r))].drop_duplicates("task", keep="last")
              for r in runs}
    scored = {r: d for r, d in scored.items() if len(d)}
    if not scored:
        continue
    best = max(scored, key=lambda r: scored[r].score.mean())
    others = [runs[r] for r in scored if r != best]
    SPEC_NOTE.append(f"{fam}, {runs[best]}" + (f" (ahead of the {', '.join(others)})" if others else
                                                " (the other architecture not yet scored)"))
    for r in scored[best].itertuples():
        SPEC[r.task] = {"v": r.score, "lo": r.ci_lo, "hi": r.ci_hi, "n": int(r.n), "own": True, "arch": runs[best]}

COLS = [("BQ-random", "Random init"), ("SPEC", "Single-task, from scratch"),
        (EARLIER, f"Best earlier BQ ({EARLIER.removeprefix('BQ-')})")] + NEWEST
SPLIT = NEWEST[0][0]                        # dotted rule before the newest run


def cells(vals, lower: bool = False):
    # the best number in the row (the lowest for a false-positive row), and those within noise of it
    mk = marks({k: vals.get(k) for k, _ in COLS}, lower)
    out = []
    for k, _ in COLS:
        x = vals.get(k)
        c = cell(x, mk.get(k, ""), " split" if k == SPLIT else "")
        if k == "SPEC" and x is not None:
            c = c.replace("</td>", f'<span class="pub"> {x["arch"]}</span></td>')
        out.append(c)
    return "".join(out)


def table(body):
    head = ('<thead><tr><th>Task</th><th>Metric</th><th class="num">n (pos)</th>' + "".join(
        f'<th class="num{" split" if k == SPLIT else ""}">{n}</th>' for k, n in COLS) + "</tr></thead>")
    return f'<div class="tablebox"><table>{head}<tbody>{"".join(body)}</tbody></table></div>'


def frozen():
    rows = []
    for title, tasks in G:
        rows.append(f'<tr class="group"><td colspan="{3 + len(COLS)}">{title}</td></tr>')
        for key, label in tasks:
            r = dict(S.get(key, {}))
            if key in RSNA_HEAD:
                for m, x in TE.get(RSNA_HEAD[key], {}).items():
                    r.setdefault(m, x)
                if RSNA_HEAD[key] in SPEC:
                    r["SPEC"] = SPEC[RSNA_HEAD[key]]
            anyx = next((x for x in r.values() if not x.get("own")), {})
            metric = "R²" if anyx.get("metric", "auc").startswith("r2") else "AUC"
            n, pos = anyx.get("n", ""), anyx.get("pos")
            ncell = f'{n}<span class="pos"> ({pos})</span>' if pos is not None else f"{n}"
            rows.append(f'<tr><td>{label}</td><td>{metric}</td><td class="num">{ncell}</td>{cells(r)}</tr>')
    top = max(SKILL[k] for k, _ in COLS if k in SKILL)
    rows.append(f'<tr class="sum"><td colspan="3">Mean skill, the {len(COMMON)} tasks every BQ model has</td>' + "".join(
        f'<td class="num{" split" if k == SPLIT else ""}{" best" if SKILL.get(k) == top else ""}">'
        + (f"{SKILL[k]:.3f}" if k in SKILL else "–") + "</td>" for k, _ in COLS) + "</tr>")
    return table(rows)


def trained():
    rows = []
    for title, tasks in IND:
        rows.append(f'<tr class="group"><td colspan="{3 + len(COLS)}">{title}</td></tr>')
        for key, label, metric in tasks:
            r = dict(TE.get(key, {}))
            if key in SPEC:
                r["SPEC"] = SPEC[key]
            n = next((x["n"] for x in r.values()), "")
            rows.append(f'<tr><td>{label}</td><td>{metric}</td><td class="num">{n}</td>{cells(r)}</tr>')
            if key in FP_TASKS and key in TEFP:      # its false positives on lesion-free scans (labels v11)
                f = TEFP[key]
                rows.append(f'<tr><td>&ensp;false positives, lesion-free scans</td><td>rate ↓</td>'
                            f'<td class="num">{next((x["n"] for x in f.values()), "")}</td>{cells(f, lower=True)}</tr>')
    return table(rows)


page = f'''<title>BrainQuery Headline Scores</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
{(HERE / "style.css").read_text()}
</style>

<div class="page">
  <h1>Headline scores: the newest BrainQuery run against its baselines</h1>
  <p class="date">3 November 2026 · updated 5 November</p>
  <p class="lede">Hover a score for its 95% interval. Dark green: the best number in the row; light green: within noise of it (95% intervals overlap).</p>

  <h2>New tasks: frozen encoders</h2>
  <p class="sub">Same readout and test scans for every model. † trained on that task’s labels; ‡ the model’s own trained
    head on its held-out test split (400 scans).</p>
  {frozen()}

  <h2>Trained tasks: own heads on the held-out test split</h2>
  {trained()}

  <ol class="notes">
    <li>Provisional (6 November): R14’s ATLAS, aneurysm, ADNI and PPMI cells (9 tasks) lack the 3D-conv probe until their re-run finishes, so they show R14’s best other readout and R14’s mean skill is understated; R14 avg’s CTA aneurysm cell is being re-run.</li>
    <li>AUC: AUROC, 0.5 = chance. R²: variance explained. Skill rescales each metric so 0 = chance, 1 = perfect.
      n (pos): test scans (positives).</li>
    <li>Random init: the BrainQuery encoder untrained, the floor for frozen transfer.</li>
    <li>Single-task, from scratch: one task family trained from random initialisation, no self-supervised
      pre-training; same data, heads and augmentation as the multitask runs. Per family, the best of a ViT, a 3D ResNet and a ViT with a two-stage convolution stem (CS-ViT), all ~22M
      parameters, by mean test score. Scored so far: {"; ".join(SPEC_NOTE) or "none"}.</li>
    <li>Best earlier BQ: the earlier BrainQuery run with the highest mean frozen skill
      ({EARLIER.removeprefix("BQ-")}, {SKILL[EARLIER]:.3f}).</li>
    <li>R14: R13 (token maps, fastMRI+ findings, 25% token gate) continued to 100k steps with true negatives: empty
      tumour masks on healthy-volunteer MRI, empty bleed masks on bleed-free RSNA CT and 3,006 two-tool RSNA bleed
      pseudo-masks; skull-strip matching, synthetic contrast and the audit fixes. R14 avg: its weight average (decay
      0.999). Lesion Dice is over scans with a lesion; the row below it is the share of lesion-free test scans where
      the head predicted any lesion (lower is better, so green marks the lowest).</li>
    <li>Single training seed per model: differences of ~0.02 in mean skill are within run-to-run noise.</li>
  </ol>
</div>
'''
if __name__ == "__main__":
    Path(sys.argv[2]).write_text(page)
    print(len(page), SPEC_NOTE)
