"""The "BrainQuery Results" artifact for the 2026-11-09 meeting: every number to present on one page, regenerated from
the same sources as the benchmark pages (re-run it when new evaluations land; missing ones show as "pending").

Sections: datasets (the dataset map), how we evaluate, the frozen benchmark (random init, published foundation models,
BrainQuery runs), trained tasks with the model's own heads (random init, single-task from scratch, BrainQuery runs),
CQ500 read zero-shot by the RSNA heads, the literature comparison, caveats.

    python analysis/bench_page/collect_scores.py runs/bench/page/scores.json
    python analysis/presentation/build_presentation.py runs/bench/page/scores.json OUT.html"""
import html
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/bench_page"))
sys.path.insert(0, str(ROOT / "analysis/presentation"))
sys.path.insert(0, str(ROOT))
SCORES, OUT = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], SCORES]                    # build_page reads the scores file from argv[1] on import
import build_page as bp  # noqa: E402
import build_focus as bf  # noqa: E402
import build_literature as bl  # noqa: E402
from datasets_figure import dataset_data  # noqa: E402

DIAG = ROOT / "runs/diagnostics"
NEW, NEW_EMA = "BQ-FISO", "BQ-FISO-EMA"
NEW_NAME = "New run (R14-F + window bank)"

# ---- own-head test results of the new run (paired with R14-F on the same scans when available) ---------------------
TE = {k: dict(v) for k, v in bp.TE.items()}
_new = DIAG / "te_full_iso/test_eval_heads.csv"
NEW_TE = _new.exists()
if NEW_TE:
    d = pd.read_csv(_new)
    pairs = {"full_iso_bank/last": NEW, "full_iso_bank/last_ema": NEW_EMA, "r14f/last": "BQ-R14F"}
    for r in d[(d.kind == "heads") & (d.metric != "fp")].itertuples():
        if r.model in pairs:
            TE.setdefault(r.task, {})[pairs[r.model]] = {"v": r.score, "lo": r.ci_lo, "hi": r.ci_hi, "n": int(r.n),
                                                         "own": True}

# ---- validation history (the newest runs) ----------------------------------------------------------------------
def last_val(run: str, step=None):
    p = ROOT / f"runs/multitask/{run}/metrics.json"
    if not p.exists():
        return None
    h = [r for r in json.load(open(p))["history"] if isinstance(r, dict) and "val_score" in r]
    if step is not None:
        h = [r for r in h if r["step"] <= step]
    return h[-1] if h else None


VAL = {"R16 (stopped before cooldown)": last_val("r16"), NEW_NAME: last_val("full_iso_bank"),
       "Native-CT twin (training)": last_val("full_ngeo_bank")}


# ---- helpers -------------------------------------------------------------------------------------------------------
def tcell(x, mk, extra=""):
    return bp.cell(x, mk, extra)


def table(cols, groups, getter, ncol=True):
    head = "<thead><tr><th>Task</th><th>Metric</th>" + ("<th class=num>n</th>" if ncol else "") + "".join(
        f"<th class=num>{n}</th>" for _, n in cols) + "</tr></thead>"
    body = []
    for title, rows in groups:
        body.append(f'<tr class="group"><td colspan="{2 + ncol + len(cols)}">{title}</td></tr>')
        for key, label, metric in rows:
            vals = getter(key)
            mk = bp.marks({k: vals.get(k) for k, _ in cols})
            n = next((x.get("n") for x in vals.values() if x and x.get("n")), "")
            body.append(f"<tr><td>{label}</td><td>{metric}</td>" + (f"<td class=num>{n}</td>" if ncol else "")
                        + "".join(tcell(vals.get(k), mk.get(k, "")) for k, _ in cols) + "</tr>")
    return f'<div class="tablebox"><table>{head}<tbody>{"".join(body)}</tbody></table></div>'


# ---- frozen benchmark ------------------------------------------------------------------------------------------------
FCOLS = [("BQ-random", "Random init"), ("ct_fm", "CT-FM"), ("brainiac", "BrainIAC"), ("brainfm", "BrainFM"),
         ("hlip", "HLIP"), ("BQ-R14F", "R14-F"), (NEW, "New run")]
have_new_frozen = any(NEW in v for v in bp.S.values() if isinstance(v, dict))
if not have_new_frozen:
    FCOLS = FCOLS[:-1]


def frozen_get(key):
    r = dict(bp.S.get(key, {}))
    if key in bp.RSNA_HEAD:                     # RSNA ICH rows: the BQ runs answer with their own trained heads
        for m, x in TE.get(bp.RSNA_HEAD[key], {}).items():
            r.setdefault(m, x)
    return r


def frozen_groups():
    out = []
    for title, tasks in bp.G:
        rows = []
        for key, label in tasks:
            anyx = next((x for x in bp.S.get(key, {}).values() if isinstance(x, dict)), {})
            metric = "R²" if str(anyx.get("metric", "auc")).startswith("r2") else "AUC"
            rows.append((key, label, metric))
        out.append((title, rows))
    return out


skill_cols = [k for k, _ in FCOLS if k in bp.SKILL]
skill_row = ("<tr class=sum><td colspan=3>Mean skill over the tasks every model answers</td>"
             + "".join(f"<td class=num>{bp.SKILL[k]:.3f}</td>" if k in bp.SKILL else "<td class=num>–</td>"
                       for k, _ in FCOLS) + "</tr>")

# ---- trained tasks, own heads ----------------------------------------------------------------------------------------
OCOLS = [("BQ-random", "Random init"), ("SPEC", "Single-task, scratch"), ("BQ-R14F", "R14-F")]
if NEW_TE:
    OCOLS += [(NEW, "New run"), (NEW_EMA, "New run, EMA")]


def own_get(key):
    r = dict(TE.get(key, {}))
    if key in bf.SPEC:
        r["SPEC"] = bf.SPEC[key]
    return r


# ---- CQ500 zero-shot --------------------------------------------------------------------------------------------------
CHIL = {"ICH": .942, "IPH": .954, "IVH": .931, "SDH": .952, "SAH": .957, "EDH": .973}
FROZEN_KEY = {"ICH": "cq500:ICH", "IPH": "cq500:IPH", "IVH": "cq500:IVH", "SDH": "cq500:SDH", "SAH": "cq500:SAH",
              "EDH": "cq500_edh"}
zs = {}
for m in ("BQ-R14F", "BQ-FISO", "BQ-FNGEO"):
    p = DIAG / f"head_init_cq500/{m}_cv.csv"
    if p.exists():
        c = pd.read_csv(p)
        for r in c[c.arm == "zero_shot"].itertuples():
            zs.setdefault(r.target, {})[m] = {"v": r.auc, "lo": r.lo, "hi": r.hi, "n": int(r.n), "pos": int(r.pos)}
zcols = [m for m in ("BQ-R14F", "BQ-FISO", "BQ-FNGEO") if any(m in v for v in zs.values())]
zname = {"BQ-R14F": "R14-F", "BQ-FISO": "New run", "BQ-FNGEO": "Native-CT twin"}
zrows = []
for t, label in [("ICH", "Any bleed"), ("IPH", "Intraparenchymal"), ("IVH", "Intraventricular"), ("SDH", "Subdural"),
                 ("SAH", "Subarachnoid"), ("EDH", "Epidural")]:
    fr = (bp.S.get(FROZEN_KEY[t], {}) or {}).get("BQ-R14F")
    vals = {m: zs.get(t, {}).get(m) for m in zcols}
    mk = bp.marks(vals)
    pos = next((x["pos"] for x in vals.values() if x), "")
    zrows.append(f"<tr><td>{label}</td><td class=num>{pos}</td>" + tcell(fr, "") +
                 "".join(tcell(vals[m], mk.get(m, "")) for m in zcols) + f"<td class=num>{CHIL[t]:.3f}</td></tr>")
ztable = ('<div class="tablebox"><table class="narrow"><thead><tr><th>Finding</th><th class=num>Positives</th>'
          '<th class=num>R14-F, CQ500-fitted frozen readout</th>' + "".join(
              f"<th class=num>{zname[m]}, RSNA head, no CQ500 labels</th>" for m in zcols) +
          '<th class=num>Supervised, 290k scans [1]</th></tr></thead><tbody>' + "".join(zrows) + "</tbody></table></div>")

# ---- CQ500 best readout (all readouts incl. the attentive probe; the head picked by validation macro AUC) ----------
def best_readout(model_dir: str, task: str = "cq500"):
    runs = sorted((ROOT / f"runs/downstream/{task}/{model_dir}").glob("*/metrics.json"))
    for p in reversed(runs):                              # newest run that has the attentive probe
        m = json.load(open(p))
        if "attn_probe" not in m:
            continue
        val = {h: m[h]["val"]["auc_ovr_macro"] for h in m if isinstance(m[h].get("val"), dict)}
        h = max(val, key=val.get)
        return h, val[h], m[h]
    return None


BR = {name: best_readout(d) for name, d in (("New run", "full_iso_bank_last"), ("R14-F", "r14f_last"))}
BR = {k: v for k, v in BR.items() if v}
LAB = [("ICH", "Any bleed"), ("IPH", "Intraparenchymal"), ("IVH", "Intraventricular"), ("SDH", "Subdural"),
       ("SAH", "Subarachnoid"), ("CalvarialFracture", "Calvarial fracture"), ("MassEffect", "Mass effect"),
       ("MidlineShift", "Midline shift")]
CHIL8 = {"ICH": .942, "IPH": .954, "IVH": .931, "SDH": .952, "SAH": .957, "CalvarialFracture": .962,
         "MassEffect": .922, "MidlineShift": .970}
if BR:
    hdr = "".join(f"<th class=num>{k}: {v[0].replace('attn_probe', 'attention probe').replace('tok_conv', 'token conv')}"
                  f"<br><span class=pos>val {v[1]:.3f}</span></th>" for k, v in BR.items())
    brows = []
    for key, label in LAB:
        old = (bp.S.get(f"cq500:{key}", {}) or {}).get("BQ-R14F")
        cells_ = [f"<td class=num>{v[2]['per_label'][key]['auc']:.3f}</td>" for v in BR.values()]
        brows.append(f"<tr><td>{label}</td>{tcell(old, '')}{''.join(cells_)}<td class=num>{CHIL8[key]:.3f}</td></tr>")
    brows.append("<tr class=sum><td>Macro, 8 findings</td>" + tcell((bp.S.get('cq500', {}) or {}).get('BQ-R14F'), '')
                 + "".join(f"<td class=num>{v[2]['auc_ovr_macro']:.3f}</td>" for v in BR.values())
                 + "<td class=num>–</td></tr>")
    brtable = ('<div class="tablebox"><table class="narrow"><thead><tr><th>Finding</th>'
               '<th class=num>R14-F, pinned readout</th>' + hdr + '<th class=num>Supervised, 290k scans [1]</th>'
               '</tr></thead><tbody>' + "".join(brows) + "</tbody></table></div>")
else:
    brtable = '<p class="pending">CQ500 best-readout runs still in progress.</p>'

# ---- validation summary ----------------------------------------------------------------------------------------------
vgroups = sorted({k for v in VAL.values() if v for k in v if k.startswith("val_group_")})
vhead = "<thead><tr><th>Validation group</th>" + "".join(
    f"<th class=num>{html.escape(n)}{'' if v is None else f'<br><span class=pos>step {v[chr(115)+chr(116)+chr(101)+chr(112)]:,}</span>'}</th>"
    for n, v in VAL.items()) + "</tr></thead>"
vbody = "".join(
    f"<tr><td>{g[10:].replace('_', ' ')}</td>" + "".join(
        f"<td class=num>{v[g]:.3f}</td>" if v and g in v else "<td class=num>–</td>" for v in VAL.values()) + "</tr>"
    for g in vgroups)
vbody += "<tr class=sum><td>Validation score (mean of groups)</td>" + "".join(
    f"<td class=num>{v['val_score']:.3f}</td>" if v else "<td class=num>–</td>" for v in VAL.values()) + "</tr>"
vtable = f'<div class="tablebox"><table class="narrow">{vhead}<tbody>{vbody}</tbody></table></div>'

# ---- datasets chart (the dataset map, embedded) ---------------------------------------------------------------------
DS = dataset_data()
tpl = (ROOT / "analysis/presentation/datasets_figure.html.tpl").read_text()
chart_css = tpl[tpl.index(".stats {"):tpl.index("</style>")]
chart_html = tpl[tpl.index('<div class="stats" id="stats">'):tpl.index('<div class="notes">')]
chart_js = tpl[tpl.index("<script>"):tpl.index("</script>") + 9].replace("/*DATA*/null", json.dumps(DS))
chart_notes = tpl[tpl.index('<div class="notes">'):tpl.index("<details>")]

pending = []
if not NEW_TE:
    pending.append("the new run's own-head test evaluation (job te-full-iso)")
if not have_new_frozen:
    pending.append("the new run's frozen benchmark (job eval-full_iso_bank)")
pend = (f'<p class="pending">Still running, shown as soon as it lands (re-run the generator): {"; ".join(pending)}.</p>'
        if pending else "")

nv = VAL[NEW_NAME]
page = f'''<title>BrainQuery Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=Source+Sans+3:wght@400;600&display=swap">
<style>
{(ROOT / "analysis/bench_page/style.css").read_text()}
:root {{ --ink-2: #55544f; --ink-3: #8a8983; --chip: #f1f0ec; --bg: #ffffff;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --sans: "Source Sans 3", "Segoe UI", system-ui, sans-serif; }}
{chart_css}
nav.toc {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 14px; margin: 6px 0 20px; }}
nav.toc a {{ color: var(--ink); }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 10px 0 6px; }}
.card {{ border: 1px solid var(--grid); border-radius: 8px; padding: 12px 14px; }}
.card b {{ display: block; font-size: 15px; margin-bottom: 4px; }}
.card p {{ margin: 0; font-size: 14px; color: var(--soft); }}
.kpis {{ display: flex; flex-wrap: wrap; gap: 14px 34px; margin: 12px 0 4px; font-family: var(--sans); }}
.kpi b {{ display: block; font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }}
.kpi span {{ color: var(--soft); font-size: 13px; }}
tr.sum td {{ font-weight: 600; border-top: 0.75px solid var(--rule); }}
.pending {{ background: #fff6dd; border-radius: 6px; padding: 6px 10px; font-size: 14px; }}
td.gap.up {{ background: var(--best); }} td.gap.near {{ background: var(--tie); }} td.gap.down {{ background: #f6dcd6; }}
tr.note td {{ font-size: 13px; color: var(--soft); padding-top: 0; }}
td.ours {{ font-weight: 600; }} td.small, .lab {{ font-size: 13px; color: var(--soft); }}
ol.refs {{ font-size: 13px; color: var(--soft); max-width: 90ch; padding-left: 26px; }}
</style>

<div class="page">
  <h1>BrainQuery: results for the 9 November meeting</h1>
  <p class="date">Generated from the benchmark records, the test-split evaluations and the training logs</p>
  <nav class="toc"><a href="#data">Datasets</a><a href="#eval">How we evaluate</a><a href="#val">Training runs</a>
    <a href="#frozen">Frozen benchmark</a><a href="#own">Trained tasks</a><a href="#cq500">CQ500 zero-shot</a>
    <a href="#lit">Literature</a><a href="#notes">Caveats</a></nav>
  <div class="kpis">
    <div class="kpi"><b>{DS["tot"]["train"]:,}</b><span>training scans, {DS["tot"]["n_train_sets"]} cohorts</span></div>
    <div class="kpi"><b>{DS["tot"]["n_ext_sets"]}</b><span>external test cohorts ({DS["tot"]["ext"]:,} scans)</span></div>
    <div class="kpi"><b>{len(bp.COMMON)}</b><span>frozen-benchmark tasks shared by all models</span></div>
    <div class="kpi"><b>{nv["val_score"]:.3f}</b><span>validation score, new run (R16: {VAL["R16 (stopped before cooldown)"]["val_score"]:.3f})</span></div>
  </div>
  {pend}

  <h2 id="data">Datasets</h2>
  {chart_html}
  {chart_notes}
  {chart_js}

  <h2 id="eval">How we evaluate</h2>
  <div class="cards">
    <div class="card"><b>Frozen benchmark (v3)</b><p>The encoder is frozen; a small readout (linear, MLP, spatial or token-conv, pinned per task on validation) is fitted on each task's training split. Small cohorts are cross-validated 5 x 3; scans that duplicate any training scan are removed.</p></div>
    <div class="card"><b>Trained tasks, own heads</b><p>The model's own heads on each training cohort's held-out test split: what the multitask model learned directly.</p></div>
    <div class="card"><b>Zero-shot transfer</b><p>A head trained on one cohort read unchanged on an external one (CQ500 with the RSNA hemorrhage heads): no labels from the external site.</p></div>
    <div class="card"><b>Random init</b><p>The same architecture, untrained, through the same readouts: the floor that shows what training adds.</p></div>
    <div class="card"><b>Single-task, from scratch</b><p>One model per task family (ViT, ResNet or conv-stem ViT of the same size, best kept), trained only on that family: the supervised specialist baseline.</p></div>
    <div class="card"><b>Published models and literature</b><p>CT-FM, BrainIAC, BrainFM and HLIP run through our frozen benchmark; published results from the papers for the same tasks.</p></div>
  </div>

  <h2 id="val">Training runs: validation</h2>
  <p class="sub">Same validation set (labels v12). The new run continues R14-F with R16's recipe plus the learned window bank on the re-encoded store; the native-CT twin reads CT at native geometry and is still training.</p>
  {vtable}

  <h2 id="frozen">Frozen benchmark: new tasks, frozen encoders</h2>
  <p class="sub">Dark green: best in the row; light green: within noise of it (95% intervals overlap); hover for intervals. † the model trained on that task's labels; ‡ its own trained head (RSNA ICH rows).</p>
  {table(FCOLS, frozen_groups(), frozen_get)}
  <div class="tablebox"><table><tbody>{skill_row}</tbody></table></div>

  <h2 id="own">Trained tasks: the model's own heads on held-out test splits</h2>
  <p class="sub">Single-task: the best of three from-scratch architectures per task family. R14-F and the new run are scored on the same scans when the new run's evaluation is in.</p>
  {table(OCOLS, bp.IND, own_get)}

  <h2 id="cq500">CQ500, read by the RSNA heads with no CQ500 labels</h2>
  <p class="sub">All 471 CQ500 scans, the benchmark's own folds. The zero-shot heads beat every readout fitted on CQ500's labels.</p>
  {ztable}
  <h3>CQ500 with the best readout</h3>
  <p class="sub">Every readout fitted on CQ500's folds (linear, MLP, octants, token conv and the attentive probe: learned
    queries attending over the scan's tokens); the column shows the readout with the best <em>validation</em> macro AUC,
    so the pick never sees the test folds.</p>
  {brtable}

  <h2 id="lit">Literature comparison</h2>
  <p class="sub">Ours: R14-F. Comparable: the best published result under a similar protocol (frozen or linear-probe foundation models). Best published: any protocol. Gap = ours minus the reference.</p>
  <div class="tablebox"><table class="lit"><thead><tr><th>Task</th><th>Metric</th><th class="num">Ours</th>
    <th>Best earlier BQ</th><th>Comparable published</th><th class="num">Gap</th><th>Best published (any protocol)</th>
    <th class="num">Gap</th></tr></thead><tbody>{"".join(bl.rows)}</tbody></table></div>
  <ol class="refs">{bl.refs}</ol>

  <h2 id="notes">Caveats</h2>
  <ol class="notes">
    <li>Test sets of 40-130 cases carry ±0.05-0.10 AUC intervals; small differences are noise.</li>
    <li>Published numbers use their own splits and often more labelled data or several sequences.</li>
    <li>The new run changed several things at once (window bank, re-encoded store, batch 12, fresh optimizer); its gains over R16 cannot be attributed to one of them. R16 was stopped before its learning-rate cooldown.</li>
    <li>The native-CT twin starts its CT stem from random weights; its CT groups are still catching up and a stem warm-up is planned before judging native geometry.</li>
    <li>Single training seed per model.</li>
    <li>Brain-volume regression test rows (62 scans) give negative R² for every model, including R14-F in its earlier
      evaluation: that split is too narrow to measure; read volumes from validation. Tiny test sets (e.g. glioma IDH, 9
      cases) are shown but carry no weight.</li>
    <li>Paired on the same test scans, over the 48 tasks both runs trained with at least 50 test scans, the new run
      averages 0.828 against R14-F's 0.814 (17 wins, 6 losses at |Δ| &gt; 0.01). The CT skull / CSF masks were
      redefined in labels v12, so R14-F is not scored on them.</li>
  </ol>
</div>
'''
Path(OUT).write_text(page)
print(len(page), "pending:", pending)
