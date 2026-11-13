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
NEW, NEW_EMA = "BQ-R18", "BQ-R18-EMA"
NEW_NAME = "R18 (ViT-B)"
PREV, PREV_EMA = "BQ-FISO", "BQ-FISO-EMA"          # full_iso_bank: the ViT-S chain R18 is compared against

# ---- own-head test results of the new run (paired with R14-F on the same scans when available) ---------------------
TE = {k: dict(v) for k, v in bp.TE.items()}
# te_full_iso: full_iso_bank (+ EMA) and R14-F; te_r17 / te_r18: R17 cooldown and R18 (+ EMA), each against
# full_iso_bank on the same scans (te_r18's full_iso_bank rows win: R18 and full_iso_bank on identical scans)
TE_FILES = [("te_full_iso", {"full_iso_bank/last": PREV, "full_iso_bank/last_ema": PREV_EMA, "r14f/last": "BQ-R14F"}),
            ("te_r17", {"r17_cool/last": "BQ-R17", "r17_cool/last_ema": "BQ-R17-EMA"}),
            ("te_r18", {"r18/last": NEW, "r18/last_ema": NEW_EMA, "full_iso_bank/last": PREV})]
NEW_TE = (DIAG / "te_r18/test_eval_heads.csv").exists()
for sub, pairs in TE_FILES:
    f = DIAG / sub / "test_eval_heads.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
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
    if not h:
        return None
    r = h[-1]                                     # the weight average's scores where the run kept one (as reported)
    return {**r, **r["ema"], "step": r["step"]} if isinstance(r.get("ema"), dict) and "val_score" in r["ema"] else r


VAL = {"full_iso_bank (ViT-S)": last_val("full_iso_bank"), "R17 cooldown (ViT-B)": last_val("r17_cool"),
       NEW_NAME: last_val("r18")}


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
         ("hlip", "HLIP"), ("BQ-R14F", "R14-F"), (PREV, "full_iso_bank"), ("BQ-R17", "R17"), (NEW, "R18")]
have_new_frozen = any(NEW in v for v in bp.S.values() if isinstance(v, dict))
FCOLS = [c for c in FCOLS if any(c[0] in v for v in bp.S.values() if isinstance(v, dict))]


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
OCOLS = [("BQ-random", "Random init"), ("SPEC", "Single-task, scratch"), ("BQ-R14F", "R14-F"),
         (PREV, "full_iso_bank"), (PREV_EMA, "full_iso_bank, EMA"), ("BQ-R17", "R17"), (NEW, "R18"),
         (NEW_EMA, "R18, EMA")]
OCOLS = [c for c in OCOLS if c[0] == "SPEC" or any(c[0] in v for v in TE.values())]


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
for m in ("BQ-R14F", "BQ-FISO", "BQ-R17", "BQ-R18"):
    p = DIAG / f"head_init_cq500/{m}_cv.csv"
    if p.exists():
        c = pd.read_csv(p)
        for r in c[c.arm == "zero_shot"].itertuples():
            zs.setdefault(r.target, {})[m] = {"v": r.auc, "lo": r.lo, "hi": r.hi, "n": int(r.n), "pos": int(r.pos)}
zcols = [m for m in ("BQ-R14F", "BQ-FISO", "BQ-R17", "BQ-R18") if any(m in v for v in zs.values())]
zname = {"BQ-R14F": "R14-F", "BQ-FISO": "full_iso_bank", "BQ-R17": "R17", "BQ-R18": "R18"}
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


BR = {name: best_readout(d) for name, d in (("R14-F", "r14f_last"), ("full_iso_bank", "full_iso_bank_last"),
                                            ("R17", "r17_cool_last"), ("R18", "r18_last"))}
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

# ---- every task with readout runs that include the attentive probe: the validation-picked readout --------------------
HEADNAME = {"linear": "linear", "mlp": "MLP", "spatial": "octants", "tok_conv": "token conv", "attn_probe": "attention"}


def picked(model_dir: str, task: str):
    runs = sorted((ROOT / f"runs/downstream/{task}/{model_dir}").glob("*/metrics.json"))
    for p in reversed(runs):
        m = json.load(open(p))
        if "attn_probe" not in m:
            continue
        key = next(k for k in ("auc_ovr_macro", "auc", "r2") if k in m["attn_probe"])

        def v(h):
            x = m[h].get("val")
            return x.get(key) if isinstance(x, dict) else x
        h = max((h for h in m if v(h) is not None), key=v)
        return h, v(h), m[h][key], key
    return None


labels_by_key = {k: l for _, ts in bp.G for k, l in ts}
labels_by_key.update({"cq500_chronic_bleed": "CQ500 chronic bleed"})
RDIRS = [("full_iso_bank", "full_iso_bank_last"), ("R17", "r17_cool_last"), ("R18", "r18_last")]
rtasks = sorted({p.parts[-4] for _, d in RDIRS for p in (ROOT / "runs/downstream").glob(f"*/{d}/*/metrics.json")})
rrows = []
for t in rtasks:
    got = [picked(d, t) for _, d in RDIRS]
    if not any(got):
        continue
    metric = "R²" if next(x for x in got if x)[3] == "r2" else "AUC"
    best = max((x[2] for x in got if x), default=None)
    fmt = lambda x: (f"<td class=\"num{' best' if x[2] == best else ''}\">{x[2]:.3f}"  # noqa: E731
                     f"<span class=pos> {HEADNAME.get(x[0], x[0])}</span></td>" if x else "<td class=\"num quiet\">–</td>")
    rrows.append(f"<tr><td>{labels_by_key.get(t, t)}</td><td>{metric}</td>{''.join(fmt(x) for x in got)}</tr>")
rtable = ('<div class="tablebox"><table class="narrow"><thead><tr><th>Task</th><th>Metric</th>' + "".join(
          f"<th class=num>{n}, best readout</th>" for n, _ in RDIRS) + '</tr></thead><tbody>' + "".join(rrows)
          + "</tbody></table></div>") if rrows else '<p class="pending">Readout runs in progress.</p>'

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
    pending.append("R18's own-head test evaluation")
missing18 = sorted(k for k in bp.COMMON if NEW not in (bp.S.get(k, {}) or {}))
if missing18:
    pending.append("R18 frozen benchmark tasks not finished: " + ", ".join(missing18))
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
  <h1>BrainQuery: results through R18 (13 November)</h1>
  <p class="date">Generated from the benchmark records, the test-split evaluations and the training logs</p>
  <nav class="toc"><a href="#data">Datasets</a><a href="#eval">How we evaluate</a><a href="#val">Training runs</a>
    <a href="#frozen">Frozen benchmark</a><a href="#own">Trained tasks</a><a href="#cq500">CQ500 zero-shot</a>
    <a href="#lit">Literature</a><a href="#notes">Caveats</a></nav>
  <div class="kpis">
    <div class="kpi"><b>{DS["tot"]["train"]:,}</b><span>training scans, {DS["tot"]["n_train_sets"]} cohorts</span></div>
    <div class="kpi"><b>{DS["tot"]["n_ext_sets"]}</b><span>external test cohorts ({DS["tot"]["ext"]:,} scans)</span></div>
    <div class="kpi"><b>{len(bp.COMMON)}</b><span>frozen-benchmark tasks shared by all models</span></div>
    <div class="kpi"><b>{nv["val_score"]:.3f}</b><span>validation score, R18 (full_iso_bank: {VAL["full_iso_bank (ViT-S)"]["val_score"]:.3f})</span></div>
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
  <p class="sub">Same validation set (labels v12), averaged-weight (EMA) heads where the run kept them. full_iso_bank: ViT-S, the R10 → R14-F chain (self-supervised start) plus 30k steps. R17: ViT-B from scratch, 76.5k steps + a 10k-step cooldown. R18: R17 at 76.5k + 60k steps with all CT read at 1 mm (native-geometry stem removed), depthwise convolutions, left-right mirror mixing and the hemorrhage any-of loss.</p>
  {vtable}

  <h2 id="frozen">Frozen benchmark: new tasks, frozen encoders</h2>
  <p class="sub">Dark green: best in the row; light green: within noise of it (95% intervals overlap); hover for intervals. † the model trained on that task's labels; ‡ its own trained head (RSNA ICH rows).</p>
  {table(FCOLS, frozen_groups(), frozen_get)}
  <div class="tablebox"><table><tbody>{skill_row}</tbody></table></div>

  <h3>Best readout per task (attention probe included)</h3>
  <p class="sub">Every readout fitted (linear, MLP, octants, token conv, attentive probe); shown is the one with the best
    <em>validation</em> score and its test score, so the choice never sees test data. The attentive probe loses left vs
    right (bleed side, LVO side), where the spatial readouts win. Tasks fill in as the chunked runs finish.</p>
  {rtable}

  <h2 id="own">Trained tasks: the model's own heads on held-out test splits</h2>
  <p class="sub">Single-task: the best of three from-scratch architectures per task family. full_iso_bank, R17 and R18 are scored on the same test scans; R14-F on the scans of its own evaluation.</p>
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
    <li>R14-F and full_iso_bank trained on the eight benchmark cohorts that R17 / R18 exclude (ABIDE, IXI, OASIS-1/2, Pixar, NKI, HBN, RSNA-ICA) for ~115k steps of their chain; only full_iso_bank's last 30k steps excluded them. Their IXI and Pixar age scores are therefore not clean transfers, and their anatomy-volume lead partly reflects that data.</li>
    <li>R17 / R18 read CT through the 1 mm stem on the frozen benchmark; R17 trained CT at native geometry, so its CT rows understate it.</li>
    <li>R18 is not a pure size test: it differs from full_iso_bank in size, initialisation (no self-supervised start) and training data.</li>
    <li>Single training seed per model.</li>
    <li>Brain-volume regression test rows (62 scans) give negative R² for every model, including R14-F in its earlier
      evaluation: that split is too narrow to measure; read volumes from validation. Tiny test sets (e.g. glioma IDH, 9
      cases) are shown but carry no weight.</li>
    <li>The CT skull / CSF masks were redefined in labels v12, so R14-F is not scored on them.</li>
  </ol>
</div>
'''
Path(OUT).write_text(page)
print(len(page), "pending:", pending)
