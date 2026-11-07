"""The "Brain Model Task Scores" artifact page: one combined frozen-transfer table (+ own-head RSNA rows), a held-out
test-split table for trained tasks, the head-transfer table, short notes.
    python collect_scores.py scores.json && python build_page.py scores.json OUT.html"""
import json, sys
from pathlib import Path
import pandas as pd
HERE, REPO = Path(__file__).parent, "/home/roku/brain-model"
S = json.load(open(sys.argv[1]))
# benchmark v3 (2026-11-06): the models re-scored on it (BQ-SSL v3, R6, R7, R7-SSL have v2 results only)
COLS = [("BQ-random", "Random"), ("PUB", "Best published"), ("BQ-R10", "R10"), ("BQ-R10-EMA", "R10 avg"),
        ("BQ-R12", "R12"), ("BQ-R13", "R13"), ("BQ-R13-EMA", "R13 avg"), ("BQ-R14", "R14"), ("BQ-R14-EMA", "R14 avg"),
        ("BQ-R14F", "R14-F")]
SPLITCOL = "BQ-R10"
PUB = {"ct_fm": "CT-FM", "ct_fm_triage": "CT-FM head", "brainiac": "BrainIAC", "brainfm": "BrainFM", "hlip": "HLIP"}
PAPER = {"adni_cn_ad": (0.850, 0.754, 0.947), "adni_cn_mci": (0.701, 0.608, 0.794), "ppmi_pd": (0.610, 0.571, 0.648)}
# own trained heads on the held-out test split (brain.multitask.test_eval), for tasks the benchmark leaves out
te7 = pd.read_csv(f"{REPO}/runs/diagnostics/test_eval_r7/test_eval_heads.csv")
te9 = pd.read_csv(f"{REPO}/runs/diagnostics/test_eval_r9/test_eval_heads.csv")
te10 = pd.read_csv(f"{REPO}/runs/diagnostics/test_eval_r10/test_eval_heads.csv")
# R10 / R12 / R12 avg on labels v9's test split (test_eval_r12); before it exists, R10 from its own (labels v7) run
_te12 = Path(f"{REPO}/runs/diagnostics/test_eval_r12/test_eval_heads.csv")
te12 = pd.read_csv(_te12) if _te12.exists() else te10.iloc[:0]
# R13 (labels v10 = v9 + fastMRI+; same test split for every older task): R10 / R12 / R13 / R13 avg, and R13's own
# run for the tasks only it has (fastMRI+ findings)
_te13, _te13o = (Path(f"{REPO}/runs/diagnostics/{d}/test_eval_heads.csv") for d in ("test_eval_r13", "test_eval_r13_own"))
te13 = pd.read_csv(_te13) if _te13.exists() else te10.iloc[:0]
te13o = pd.read_csv(_te13o) if _te13o.exists() else te10.iloc[:0]
# segmentation re-scored 2026-11-03 (Dice over scans with a target only; empty targets became "fp" rows): overrides
_tes = Path(f"{REPO}/runs/diagnostics/test_eval_seg_v2/test_eval_heads.csv")
tes = pd.read_csv(_tes) if _tes.exists() else te10.iloc[:0]
# R14 against R13 on labels v11 (true negatives: Dice over scans with a lesion, "fp" rows over lesion-free scans)
_te14 = Path(f"{REPO}/runs/diagnostics/test_eval_r14/test_eval_heads.csv")
te14 = pd.read_csv(_te14) if _te14.exists() else te10.iloc[:0]
P14 = {"r13/last": "BQ-R13", "r13/last_ema": "BQ-R13-EMA", "r14/last": "BQ-R14", "r14/last_ema": "BQ-R14-EMA"}
# R14-F (study-level input) on the same split: single scans, and each scan with its co-registered sequences (--study 3)
_te14f, _te14fs = (Path(f"{REPO}/runs/diagnostics/{d}/test_eval_heads.csv") for d in ("test_eval_r14f", "test_eval_r14f_study"))
te14f = pd.read_csv(_te14f) if _te14f.exists() else te10.iloc[:0]
te14fs = pd.read_csv(_te14fs) if _te14fs.exists() else te10.iloc[:0]
P14F, P14FS = {"r14f/last": "BQ-R14F"}, {"r14f/last": "BQ-R14F-seq"}
TE = {}
for df, pairs in ((te7, {"r7_targets/last": "BQ-R7", "r7_ssl/last": "BQ-R7-SSL"}), (te9, {"r9/last": "BQ-R9"}),
                  (te10, {"r10/last": "BQ-R10", "r10/last_ema": "BQ-R10-EMA"}),
                  (te12, {"r10/last": "BQ-R10", "r12/last": "BQ-R12", "r12/last_ema": "BQ-R12-EMA"}),
                  (te13o, {"r13/last": "BQ-R13", "r13/last_ema": "BQ-R13-EMA"}),
                  (te13, {"r10/last": "BQ-R10", "r12/last": "BQ-R12", "r13/last": "BQ-R13", "r13/last_ema": "BQ-R13-EMA"}),
                  (tes, {"r10/last": "BQ-R10", "r12/last": "BQ-R12", "r13/last": "BQ-R13", "r13/last_ema": "BQ-R13-EMA"}),
                  (te14, P14), (te14f, P14F), (te14fs, P14FS)):
    for r in df[df.metric != "fp"].itertuples():           # fp: false-positive rows (labels v11), not scores
        if r.model in pairs:
            TE.setdefault(r.task, {})[pairs[r.model]] = {"v": r.score, "lo": r.ci_lo, "hi": r.ci_hi, "n": int(r.n), "own": True}
# share of lesion-free test scans where a lesion head predicted anything (labels v11 negatives; lower is better)
TEFP = {}
for df, pairs in ((te14, P14), (te14f, P14F), (te14fs, P14FS)):
    for r in df[df.metric == "fp"].itertuples():
        if r.model in pairs:
            TEFP.setdefault(r.task, {})[pairs[r.model]] = {"v": r.score, "lo": r.ci_lo, "hi": r.ci_hi, "n": int(r.n),
                                                           "own": True}
FP_TASKS = ("whole_tumor", "tumor_core", "enhancing_tumor", "resection_cavity", "hemorrhage_segmentation")
RSNA_HEAD = {"rsna_ich:any": "hemorrhage_any", "rsna_ich:intraparenchymal": "hemorrhage_intraparenchymal",
             "rsna_ich:intraventricular": "hemorrhage_intraventricular", "rsna_ich:subdural": "hemorrhage_subdural",
             "rsna_ich:subarachnoid": "hemorrhage_subarachnoid", "rsna_ich:epidural": "hemorrhage_epidural"}
G = [
 ("CT · intracranial haemorrhage, CQ500 (external)", [
   ("cq500:ICH", "Any bleed"), ("cq500:IPH", "Intraparenchymal"), ("cq500:IVH", "Intraventricular"),
   ("cq500:SDH", "Subdural"), ("cq500:SAH", "Subarachnoid"), ("cq500_edh", "Epidural"),
   ("cq500_bleed_side", "Bleed side"), ("cq500", "All 8 findings (macro)")]),
 ("CT · intracranial haemorrhage, RSNA (BQ-R6 onward trained on it)", [
   ("rsna_ich:any", "Any bleed"), ("rsna_ich:intraparenchymal", "Intraparenchymal"),
   ("rsna_ich:intraventricular", "Intraventricular"), ("rsna_ich:subdural", "Subdural"),
   ("rsna_ich:subarachnoid", "Subarachnoid"), ("rsna_ich:epidural", "Epidural")]),
 ("CT · fracture, mass effect, stroke", [
   ("cq500:CalvarialFracture", "CQ500 skull fracture"), ("cq500_other_fracture", "CQ500 other fracture"),
   ("cq500:MassEffect", "CQ500 mass effect"), ("cq500:MidlineShift", "CQ500 midline shift"),
   ("aisd_remote_infarct", "AISD old infarct"), ("isles24_lvo_side", "ISLES24 LVO side"),
   ("isles24_mrs", "ISLES24 mRS, 90 d"), ("isles24_mrs_discharge", "ISLES24 mRS, discharge")]),
 ("MRI · stroke, age, lesions", [
   ("atlas_lesion_volume", "ATLAS lesion volume"), ("atlas_days_post_stroke", "ATLAS time since stroke"),
   ("ixi_age", "IXI age"), ("pixar_age", "Pixar age"), ("ms3seg_age", "MS3SEG age"),
   ("msd", "MSD enhancing tumour"), ("ms3seg_lesion_load", "MS3SEG lesion load"), ("valdo_epvs", "VALDO EPVS")]),
 ("MRI · cognition (R7 onward trained on these labels)", [
   ("oasis1_cdr", "OASIS-1 CDR"), ("oasis2_cdr", "OASIS-2 CDR"), ("oasis1_mmse", "OASIS-1 MMSE"),
   ("oasis2_mmse", "OASIS-2 MMSE"), ("abide_asd", "ABIDE autism")]),
 ("MRI · glioma, UCSF-PDGM (external)", [
   ("ucsf_idh", "IDH mutation"), ("ucsf_lgg", "Lower grade"), ("ucsf_mgmt", "MGMT methylation"),
   ("ucsf_survival_12m", "Survival 12 mo")]),
 ("CT / MRI · aneurysm, RSNA-ICA (R13 not trained on it)", [
   ("rsna_ica_cta_aneurysm", "CTA aneurysm present"), ("rsna_ica_mr_aneurysm", "MR aneurysm present")]),
 ("MRI · neurodegeneration, ADNI / PPMI (external)", [
   ("adni_cn_ad", "ADNI CN vs AD"), ("adni_cn_mci", "ADNI CN vs MCI"), ("adni_dx3", "ADNI CN / MCI / AD"),
   ("adni_mmse", "ADNI MMSE"), ("ppmi_pd", "PPMI Parkinson’s vs control")]),
]
# mean skill over the tasks every BQ model in COLS has (the 16 common transfer tasks); a model missing one gets none
sys.path.insert(0, REPO)
from brain.eval.skill import skill  # noqa: E402
_BQ = ["BQ-R10", "BQ-R10-EMA", "BQ-R12", "BQ-R13", "BQ-R13-EMA", "BQ-R14", "BQ-R14-EMA", "BQ-R14F"]
COMMON = [k for k in S if ":" not in k and all(m in S[k] and not S[k][m]["seen"] for m in _BQ)]
SKILL = {m: sum(skill(S[k][m]["metric"], S[k][m]["v"]) for k in COMMON) / len(COMMON)
         for m in _BQ + ["BQ-random"] if all(m in S[k] for k in COMMON)}


def marks(vals: dict, lower: bool = False) -> dict:
    """Per column: "best" for the top number of the row (lowest when ``lower``), "tie" for any other whose 95% interval
    overlaps the best one's (statistically indistinguishable at this resolution), "" otherwise."""
    have = {k: x for k, x in vals.items() if x}
    if not have:
        return {}
    kb = (min if lower else max)(have, key=lambda k: have[k]["v"])
    b = have[kb]
    out = {}
    for k, x in have.items():
        if x["v"] == b["v"]:
            out[k] = "best"
        elif None not in (x.get("lo"), x.get("hi"), b.get("lo"), b.get("hi")) and (
                x["lo"] <= b["hi"] if lower else x["hi"] >= b["lo"]):
            out[k] = "tie"
        else:
            out[k] = ""
    return out


def cell(x, best, extra=""):
    """``best``: True / "best" (top of the row), "tie" (within noise of it), or False / ""."""
    if x is None:
        return f'<td class="num quiet{extra}">–</td>'
    t = f' title="95% interval {x["lo"]:.3f}–{x["hi"]:.3f}"' if x.get("lo") is not None else ""
    mark = "<sup>‡</sup>" if x.get("own") else "<sup>†</sup>" if x.get("seen") else ""
    cls = " best" if best is True or best == "best" else " tie" if best == "tie" else ""
    return f'<td class="num{cls}{extra}"{t}>{x["v"]:.3f}{mark}</td>'


def table(groups):
    rows = []
    for title, tasks in groups:
        rows.append(f'<tr class="group"><td colspan="{3 + len(COLS)}">{title}</td></tr>')
        for key, label in tasks:
            r = dict(S.get(key, {}))
            if key in RSNA_HEAD:
                for m, x in TE.get(RSNA_HEAD[key], {}).items():
                    r.setdefault(m, x)
            anyx = next((x for x in r.values() if not x.get("own")), {})
            metric = "R²" if anyx.get("metric", "auc").startswith("r2") else "AUC"
            n, pos = anyx.get("n", ""), anyx.get("pos")
            ncell = f'{n}<span class="pos"> ({pos})</span>' if pos is not None else f"{n}"
            pubs = [(r[p], PUB[p]) for p in PUB if p in r]
            if key in PAPER:
                v, lo, hi = PAPER[key]; pubs.append(({"v": v, "lo": lo, "hi": hi}, "BrainDINO paper"))
            pub = max(pubs, key=lambda z: z[0]["v"]) if pubs else None
            vals = {k: (pub[0] if k == "PUB" and pub else r.get(k)) for k, _ in COLS}
            mk = marks(vals)                         # the highest number in the row, and those within noise of it
            cells = []
            for k, _ in COLS:
                x = vals[k]
                c = cell(x, mk.get(k, ""), " split" if k == SPLITCOL else "")
                if k == "PUB" and x is not None:
                    c = c.replace("</td>", f'<span class="pub"> {pub[1]}</span></td>')
                cells.append(c)
            rows.append(f'<tr><td>{label}</td><td>{metric}</td><td class="num">{ncell}</td>{"".join(cells)}</tr>')
    top = max(SKILL.values())
    rows.append(f'<tr class="sum"><td colspan="3">Mean skill, the {len(COMMON)} tasks every BQ model has</td>' + "".join(
        f'<td class="num{" split" if k == SPLITCOL else ""}{" best" if SKILL[k] == top else ""}">{SKILL[k]:.3f}</td>'
        if k in SKILL else '<td class="num">–</td>' for k, _ in COLS) + "</tr>")
    head = ('<thead><tr><th>Task</th><th>Metric</th><th class="num">n (pos)</th>' +
            "".join(f'<th class="num{" split" if k == SPLITCOL else ""}">{n}</th>' for k, n in COLS) + "</tr></thead>")
    return f'<div class="tablebox"><table>{head}<tbody>{"".join(rows)}</tbody></table></div>'


IND = [("Tumour (BraTS, UPENN)", [("whole_tumor", "Whole tumour", "Dice"), ("tumor_core", "Tumour core", "Dice"),
        ("enhancing_tumor", "Enhancing tumour", "Dice"), ("resection_cavity", "Resection cavity", "Dice"),
        ("glioma_survival_12m", "UPENN survival 12 mo", "AUC"), ("glioma_mgmt", "UPENN MGMT", "AUC")]),
       ("Lesion, aneurysm", [("hemorrhage_segmentation", "Bleed segmentation", "Dice"),
        ("aneurysm_present", "Aneurysm present (RSNA-ICA)", "AUC")]),
       ("Anatomy (SynthSeg targets)", [("seg_hippocampus", "Hippocampus", "Dice"), ("seg_thalamus", "Thalamus", "Dice"),
        ("seg_ventricles", "Ventricles", "Dice"), ("vol_parenchymal_fraction", "Parenchymal fraction", "R²")]),
       ("Demographics, diagnosis, CT measures", [("age", "Age", "R²"), ("sex", "Sex", "AUC"),
        ("neuro_diagnosis", "Neurological diagnosis", "bal. acc."), ("tumor_type", "Tumour type", "bal. acc."),
        ("ct_calcification", "CT calcification", "R²"), ("ct_skull_density", "CT skull density", "R²")]),
       ("Radiology findings, fastMRI+ (R13 onward)", [("mri_normal_for_age", "Normal for age", "AUC"),
        ("mri_small_vessel_disease", "Small-vessel disease", "AUC"), ("mri_wm_lesion", "White-matter lesion", "AUC"),
        ("mri_postop", "Post-op / treatment", "AUC"), ("mri_mass", "Mass", "AUC"), ("mri_edema", "Edema", "AUC"),
        ("mri_enlarged_ventricles", "Enlarged ventricles", "AUC"), ("mri_infarct", "Lacune / encephalomalacia", "AUC"),
        ("mri_artifact", "Artifact", "AUC")])]
ICOLS = [("BQ-R7", "R7"), ("BQ-R7-SSL", "R7-SSL"), ("BQ-R10", "R10"), ("BQ-R12", "R12"), ("BQ-R13", "R13"),
         ("BQ-R13-EMA", "R13 avg"), ("BQ-R14", "R14"), ("BQ-R14-EMA", "R14 avg"), ("BQ-R14F", "R14-F"),
         ("BQ-R14F-seq", "R14-F, all sequences")]
d9 = te9[te9.model == "r9/last"].set_index("task")
d10 = te10[te10.model == "r10/last"].set_index("task")
irows = []
for title, tasks in IND:
    irows.append(f'<tr class="group"><td colspan="{3 + len(ICOLS)}">{title}</td></tr>')
    rows_t = []
    for key, label, metric in tasks:
        rows_t.append((TE.get(key, {}), label, metric, False))
        if key in FP_TASKS and key in TEFP:          # its false positives on lesion-free scans, right below
            rows_t.append((TEFP[key], "&ensp;false positives, lesion-free scans", "rate ↓", True))
    for r, label, metric, lower in rows_t:
        n = next((x["n"] for x in r.values()), "")
        mk = marks(r, lower)
        cells = []
        for k, _ in ICOLS:
            x = r.get(k)
            if x is None:
                cells.append('<td class="num quiet">–</td>'); continue
            cls = {"best": " best", "tie": " tie"}.get(mk.get(k, ""), "")
            cells.append(f'<td class="num{cls}" title="95% interval {x["lo"]:.3f}–{x["hi"]:.3f}">'
                         f'{x["v"]:.3f}</td>')
        irows.append(f'<tr><td>{label}</td><td>{metric}</td><td class="num">{n}</td>{"".join(cells)}</tr>')
ind = ('<div class="tablebox"><table class="narrow"><thead><tr><th>Task</th><th>Metric</th><th class="num">n</th>' +
       "".join(f'<th class="num">{n}</th>' for _, n in ICOLS) + f'</tr></thead><tbody>{"".join(irows)}</tbody></table></div>')

TR = [("CQ500 any bleed", "RSNA", 423, 0.904, 0.804, 0.813), ("CQ500 intraparenchymal", "RSNA", 423, 0.89, 0.8, 0.798),
      ("CQ500 intraventricular", "RSNA", 423, 0.93, 0.982, 0.938), ("CQ500 subdural", "RSNA", 423, 0.885, 0.714, 0.76),
      ("CQ500 subarachnoid", "RSNA", 423, 0.858, 0.877, 0.832), ("CQ500 epidural", "RSNA", 423, 0.792, 0.829, 0.749),
      ("IXI age (R²)", "FOMO", 85, 0.864, 0.817, 0.864), ("Pixar age (R²)", "FOMO", 23, 0.864, 0.757, 0.897),
      ("MS3SEG age (R²)", "FOMO", 100, 0.864, -2.355, 0.374)]
tr = []
for t, src, n, own, unch, new in TR:
    top = max(unch, new)
    tr.append(f'<tr><td>{t}</td><td>{src}</td><td class="num">{n}</td><td class="num quiet">{own:.3f}</td>'
              f'<td class="num split{" best" if unch == top else ""}">{unch:.3f}</td>'
              f'<td class="num{" best" if new == top else ""}">{new:.3f}</td></tr>')
trans = ('<div class="tablebox"><table class="narrow"><thead><tr><th>Task</th><th>Head trained on</th><th class="num">n</th>'
         '<th class="num">Own test</th><th class="num split">Unchanged head</th><th class="num">New readout</th></tr></thead>'
         f'<tbody>{"".join(tr)}</tbody></table></div>')

# frozen-encoder segmentation transfer (brain.downstream.seg3d: a 7.5M-parameter 3D U-Net decoder on the frozen token
# grid only, no image skip; small datasets, 2026-11-05)
SEGT = [("wmh", "WMH challenge: white-matter hyperintensities", "MRI"), ("ms3seg", "MS3SEG: MS lesion, WMH, ventricles (mean)", "MRI"),
        ("isles22", "ISLES22: acute infarct", "MRI"), ("instance", "INSTANCE: intracranial bleed", "CT")]
SCOLS = [("random_v3_last", "Random"), ("r10_last", "R10"), ("r13_last", "R13"), ("r14_last", "R14")]


def _seg_score(task, key):
    runs = sorted(Path(f"{REPO}/runs/downstream/{task}/{key}").glob("2026110[5-9]T*/metrics.json"))
    if not runs:
        return None
    m = json.load(open(runs[-1]))["tok_unet3d"]
    lo, hi = m["dice_macro_ci"]
    return {"v": m["dice_macro"], "lo": lo, "hi": hi, "n": m["n_test"], "ntr": m["n_train"]}


srows = []
for task, label, mod in SEGT:
    r = {k: _seg_score(task, k) for k, _ in SCOLS}
    if not any(r.values()):
        continue
    mk = marks(r)
    x0 = next(x for x in r.values() if x)
    srows.append(f'<tr><td>{label}</td><td>{mod}</td><td class="num">{x0["ntr"]} / {x0["n"]}</td>' +
                 "".join(cell(r[k], mk.get(k, "")) for k, _ in SCOLS) + "</tr>")
segt = ('<div class="tablebox"><table class="narrow"><thead><tr><th>Task</th><th>Scans</th><th class="num">n train / test</th>' +
        "".join(f'<th class="num">{n}</th>' for _, n in SCOLS) + f'</tr></thead><tbody>{"".join(srows)}</tbody></table></div>')

# training-set fit (jobs/train_fit.sh, labels v10): each family's best from-scratch specialist and R14 on up to 400 of
# their own training scans and on the test split
FITS = {"anatomy": ("Anatomy", "ResNet"), "tumor": ("Tumour", "ViT"), "hemorrhage": ("Haemorrhage", "CS-ViT"),
        "ct_measures": ("CT measures", "ViT"), "demographics": ("Demographics", "ResNet"), "radiology": ("Radiology findings", "ResNet")}
_fit = []
for fam in FITS:
    for split in ("train", "test"):
        f = Path(f"{REPO}/runs/diagnostics/train_fit/{fam}_{split}")
        for p in f.glob("test_eval_heads*.csv"):
            d = pd.read_csv(p); d = d[d.metric != "fp"]
            d["fam"], d["split"] = fam, split
            d["who"] = ["R14" if "r14" in m else "spec" for m in d.model]
            _fit.append(d)
frows = []
if _fit:
    fd = pd.concat(_fit)
    fd["kind"] = ["Segmentation (Dice)" if m == "dice" else "Other heads (AUC, R², bal. acc.)" for m in fd.metric]
    for fam, (name, arch) in FITS.items():
        for kind, g in fd[fd.fam == fam].groupby("kind", sort=False):
            v = g.groupby(["who", "split"])["score"].mean()
            nt = g.task.nunique()
            if len(v) < 4:
                continue
            gap_s, gap_r = v["spec", "train"] - v["spec", "test"], v["R14", "train"] - v["R14", "test"]
            frows.append(f'<tr><td>{name}: {kind.split(" (")[0].lower()} <span class="pos">({nt})</span></td>'
                         f'<td>{arch}</td>'
                         f'<td class="num">{v["spec", "train"]:.3f}</td><td class="num">{v["spec", "test"]:.3f}</td>'
                         f'<td class="num quiet">{gap_s:+.3f}</td>'
                         f'<td class="num split">{v["R14", "train"]:.3f}</td><td class="num">{v["R14", "test"]:.3f}</td>'
                         f'<td class="num quiet">{gap_r:+.3f}</td></tr>')
fit = ('<div class="tablebox"><table class="narrow"><thead><tr><th>Family: heads (tasks)</th><th>Specialist</th>'
       '<th class="num">Specialist train</th><th class="num">test</th><th class="num">gap</th>'
       '<th class="num split">R14 train</th><th class="num">test</th><th class="num">gap</th></tr></thead>'
       f'<tbody>{"".join(frows)}</tbody></table></div>')

page = f'''<title>Brain Model Task Scores</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
{(HERE / "style.css").read_text()}
</style>

<div class="page">
  <h1>Task scores on held-out test scans: our models against published models</h1>
  <p class="date">28 October 2026 · updated 6 November</p>
  <p class="lede">Hover a score for its 95% interval. Dark green: the best number in the row; light green: within noise of it (95% intervals overlap).</p>

  <h2>New tasks: frozen encoders (benchmark v3)</h2>
  <p class="sub">Same readout and test scans for every model. Benchmark v3: readouts on the patch tokens (linear / MLP
    on their mean, a linear probe on 8 octant means, a 3D-conv probe on the token grid), one per task chosen on
    validation across models; 234 test scans that were copies of training scans removed. † trained on that task’s
    labels; ‡ the model’s own trained head on its held-out test split (400 scans).</p>
  {table(G)}

  <h2>Trained tasks: own heads on the held-out test split</h2>
  {ind}

  <h2>Trained heads applied to another dataset</h2>
  {trans}

  <h2>Segmentation from frozen encoders</h2>
  <p class="sub">Dice on test scans. One 7.5M-parameter 3D U-Net decoder is trained per model on the frozen token grid
    alone (no image input). The untrained encoder ties every trained one: at this decoder size the frozen tokens add no
    segmentation detail beyond the raw patches. AISD (CT infarct) diverged for every encoder and is being re-run.</p>
  {segt}

  <h2>Fit on the training set: specialist vs multitask</h2>
  <p class="sub">Mean score on up to 400 of the model’s own training scans and on the test split (labels v10).
    Segmentation: R14 is 0.01–0.02 below the specialist on the training scans themselves, so the shortfall is fit, not
    overfitting; lesion masks lose 0.12–0.23 from train to test in both models. Other heads: R14 fits training scans far
    better than the specialist (demographics 1.00, tumour 0.99) and is better on test too, but drops 0.26–0.31 on
    demographics, tumour and anatomy measures.</p>
  {fit}

  <ol class="notes">
    <li>AUC: AUROC, 0.5 = chance. R²: variance explained. Skill rescales each metric so 0 = chance, 1 = perfect. n (pos): test scans (positives).</li>
    <li>Best published: CT-FM, CT-FM head (Pai et al., 2025), BrainIAC (Tak et al., 2026), BrainFM (Liu et al., 2025; multitask on real + synthetic scans; its training data include ADNI, OASIS and ATLAS) and HLIP (Zhao et al., 2026; CLIP on clinical brain-MRI reports), run here on the same scans; for ADNI / PPMI also BrainDINO’s published number (different cohort: longitudinal ADNI1).</li>
    <li>BQ-SSL v3: self-supervised encoder only. R7: multitask on the v4 encoder, 12k steps. R7-SSL: + masked reconstruction and contrast synthesis. R10: R7 at 30k steps on 4 GPUs (batch 32), with synthesis, 10% masked reconstruction, task-aware augmentation and aneurysm location + vessel tasks; R10 avg: its weight average (decay 0.999). R12: R10 continued to 60k steps on QC-filtered tool labels (v9) with HL-Gauss regression; R12 avg: its weight average. R13: R12 at 50k continued to 75k with token-map targets (mask occupancy, RSNA slice labels, fastMRI+ boxes), fastMRI+ radiology findings and a 25% token gate, without aneurysm tasks or RSNA-ICA scans; R13 avg: its weight average. Trained-task table: R10, R12 and R13 on the same test split, R7 on its own run's.</li>
    <li>R14: R13 continued to 100k steps with true negatives (empty tumour masks on healthy-volunteer MRI, empty bleed masks on bleed-free RSNA CT, 3,006 two-tool RSNA bleed pseudo-masks), skull-strip matching, synthetic contrast and the audit fixes; R14 avg: its weight average. Lesion Dice is over scans with a lesion; the row below it is the share of lesion-free test scans where the head predicted any lesion (lower is better; R13 and R14 on labels v11).</li>
    <li>R14-F: R14 continued 15k steps with study-level input: a study’s co-registered sequences (BraTS / UPENN mpMRI, SynthRAD MR + CT) pass the first 8 encoder blocks separately and are fused per token position before the last 4; a scan without partners takes the unfused path. Frozen benchmark: single scans. “All sequences”: its heads given each test scan’s co-registered sequences (scans without any stay single).</li>
    <li>Single training seed per model: differences of ~0.02 in mean skill are within run-to-run noise.</li>
  </ol>
</div>
'''
if __name__ == "__main__":                 # build_focus.py imports the tables' data from here
    Path(sys.argv[2]).write_text(page)
    print(len(page))
