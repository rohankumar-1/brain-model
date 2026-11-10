"""Three slide tables (16:9 PNGs): in-training (task trained, held-out test split), external (task trained, unseen
cohort, no labels from it), transfer (task never trained, frozen encoder + readout fitted on the task).
Columns: random init, single-task from scratch, literature, ours (best of the new run and R14-F per row).
    python analysis/presentation/eval_tables.py OUT_DIR"""
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1])
S = json.load(open(ROOT / "runs/bench/page/scores.json"))
sys.argv = [sys.argv[0], str(ROOT / "runs/bench/page/scores.json")]
sys.path.insert(0, str(ROOT / "analysis/bench_page"))
sys.path.insert(0, str(ROOT))
import build_focus as bf  # noqa: E402

te = pd.read_csv(ROOT / "runs/diagnostics/te_full_iso/test_eval_heads.csv")
te = te[(te.kind == "heads") & (te.metric != "fp")]
OWN = {m: dict(zip(te[te.model == m].task, te[te.model == m].score)) for m in ("full_iso_bank/last", "r14f/last")}


def own(task):
    """The new run's own head on the test scans. One model, fixed in advance: picking per row between two models by
    their test score (as the 2026-11-09 tables did) is a selection on test."""
    a = OWN["full_iso_bank/last"].get(task)
    return None if a is None else (a, False)


def rand(key):
    x = (S.get(key, {}) or {}).get("BQ-random")
    return x["v"] if x else None


def spec(task):
    x = bf.SPEC.get(task)
    return x["v"] if x else None


def zero_shot(target):
    """The new run's RSNA head read on CQ500 unchanged (no labels, no choice: zero-shot has no validation to pick a
    model on, so the model is fixed in advance)."""
    p = ROOT / "runs/diagnostics/head_init_cq500/BQ-FISO_cv.csv"
    if not p.exists():
        return None
    c = pd.read_csv(p)
    r = c[(c.arm == "zero_shot") & (c.target == target)]
    return (float(r.auc.iloc[0]), False) if len(r) else None


def best_readout(task, label=None, run="full_iso_bank_last"):
    """(test, val) of ``run``'s readout picked on validation, over every readout of its latest full run (the attentive
    probe included); None without such a run."""
    for p in sorted((ROOT / f"runs/downstream/{task}/{run}").glob("*/metrics.json"), reverse=True):
        m = json.load(open(p))
        if "attn_probe" not in m:
            continue
        key = next(k for k in ("auc_ovr_macro", "auc", "r2") if k in m["attn_probe"])
        v = lambda h: (m[h]["val"].get(key) if isinstance(m[h].get("val"), dict) else None)  # noqa
        h = max((h for h in m if isinstance(m[h], dict) and v(h) is not None), key=v)
        test = m[h]["per_label"][label]["auc"] if label else m[h][key]
        return test, v(h)
    return None


def frozen(key, task=None, label=None):
    """The validation-picked readout of whichever model (new run, R14-F) scores higher on validation: models and
    readouts are both chosen on validation, never on test. (Per-label CQ500 rows use the multilabel task's
    validation.)"""
    cand = {r: best_readout(task, label, r) for r in ("full_iso_bank_last", "r14f_last")} if task else {}
    cand = {r: x for r, x in cand.items() if x is not None and x[1] is not None}
    if not cand:
        return None
    r = max(cand, key=lambda k: cand[k][1])
    return cand[r][0], r == "r14f_last"


T1 = ("In-training evaluation: trained task, held-out test split of a training cohort",
      "own trained heads", [
          ("RSNA ICH · any bleed", "AUC", rand("rsna_ich:any"), spec("hemorrhage_any"), ".967*", own("hemorrhage_any")),
          ("RSNA ICH · subdural", "AUC", rand("rsna_ich:subdural"), spec("hemorrhage_subdural"), ".983*",
           own("hemorrhage_subdural")),
          ("RSNA ICH · epidural", "AUC", rand("rsna_ich:epidural"), spec("hemorrhage_epidural"), ".984*",
           own("hemorrhage_epidural")),
          ("BraTS · whole tumour (one sequence)", "Dice", None, spec("whole_tumor"), ".908*", own("whole_tumor")),
          ("BraTS · enhancing tumour (T1c)", "Dice", None, spec("enhancing_tumor"), ".737*", own("enhancing_tumor")),
          ("UPenn-GBM · 12-month survival", "AUC", None, spec("glioma_survival_12m"), ".72*",
           own("glioma_survival_12m")),
          ("Age (pooled cohorts)", "R²", None, spec("age"), None, own("age")),
          ("Sex", "AUC", None, spec("sex"), None, own("sex"))])
T2 = ("External evaluation: trained task, unseen cohort, no labels from it",
      "CQ500 (471 scans) read by the RSNA heads", [
          ("CQ500 · any bleed", "AUC", .5, None, ".942*", zero_shot("ICH")),
          ("CQ500 · intraparenchymal", "AUC", .5, None, ".954*", zero_shot("IPH")),
          ("CQ500 · intraventricular", "AUC", .5, None, ".931*", zero_shot("IVH")),
          ("CQ500 · subdural", "AUC", .5, None, ".952*", zero_shot("SDH")),
          ("CQ500 · subarachnoid", "AUC", .5, None, ".957*", zero_shot("SAH")),
          ("CQ500 · epidural", "AUC", .5, None, ".973*", zero_shot("EDH"))])
T3 = ("Transfer: task never trained, frozen encoder + small readout fitted on the task",
      "frozen benchmark", [
          ("ADNI · CN vs AD", "AUC", rand("adni_cn_ad"), None, ".922", frozen("adni_cn_ad", "adni_cn_ad")),
          ("ADNI · CN vs MCI", "AUC", rand("adni_cn_mci"), None, ".757", frozen("adni_cn_mci", "adni_cn_mci")),
          ("ADNI · MMSE", "R²", rand("adni_mmse"), None, ".43", frozen("adni_mmse", "adni_mmse")),
          ("PPMI · Parkinson's vs control", "AUC", rand("ppmi_pd"), None, ".610", frozen("ppmi_pd", "ppmi_pd")),
          ("UCSF · IDH mutation", "AUC", rand("ucsf_idh"), None, ".901", frozen("ucsf_idh", "ucsf_idh")),
          ("CQ500 · calvarial fracture", "AUC", rand("cq500:CalvarialFracture"), None, ".962*",
           frozen("cq500:CalvarialFracture", "cq500", "CalvarialFracture")),
          ("CQ500 · midline shift", "AUC", rand("cq500:MidlineShift"), None, ".970*",
           frozen("cq500:MidlineShift", "cq500", "MidlineShift")),
          ("ISLES 2024 · LVO side (CTA)", "AUC", rand("isles24_lvo_side"), None, None,
           frozen("isles24_lvo_side", "isles24_lvo_side")),
          ("MS3SEG · MS lesion load", "R²", rand("ms3seg_lesion_load"), None, None, frozen("ms3seg_lesion_load", "ms3seg_lesion_load"))])
NOTES = {
    "T1": "Single-task: best of three from-scratch architectures per task "
          "family. * supervised or fine-tuned published model on its own split (Pillar-0; RSNA 2019 winner; nnU-Net and "
          "missing-modality models on T1c alone; BrainIAC).",
    "T2": "Single-task specialist not yet scored on CQ500. * Chilamkurthy et al. 2018, "
          "supervised on ~290k head CTs, CQ500 as its external test.",
    "T3": "Single-task models exist only for trained tasks. Literature: the best "
          "published frozen / linear-probe foundation model; * supervised model (no frozen result published). Ours: the "
          "readout picked on validation.",
}
COLS = ["Single-task", "Literature", "Ours"]


def fmt(v):
    if v is None:
        return "–"
    if isinstance(v, str):
        return v
    return f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.2f}"


def draw(tag, spec_):
    title, sub, rows = spec_
    fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.6, 8.35, title, fontsize=17, fontweight="bold", color="#1d1d1b", va="center")
    ax.text(0.6, 7.85, sub, fontsize=11.5, color="#55544f", va="center")
    xs = [9.6, 12.0, 14.6]
    top, rh = 7.15, min(0.62, 5.6 / max(len(rows), 1))
    for x, c in zip(xs, COLS):
        ax.text(x, top, c, ha="center", va="center", fontsize=12.5, fontweight="bold",
                color="#1d1d1b" if c != "Ours" else "#1c5cab")
    ax.text(5.9, top, "Metric", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#1d1d1b")
    ax.plot([0.6, 15.6], [top - 0.32, top - 0.32], color="#1d1d1b", lw=1.2)
    ax.add_patch(plt.Rectangle((13.75, top - 0.32 - rh * len(rows) - 0.05), 1.7, rh * len(rows) + 0.05,
                               color="#e8f0fb", zorder=0))
    any_r14 = False
    for i, (name, metric, r, s, lit, ours) in enumerate(rows):
        y = top - 0.32 - rh * (i + 0.5)
        ax.text(0.6, y, name, va="center", fontsize=12.5, color="#1d1d1b")
        ax.text(5.9, y, metric, ha="center", va="center", fontsize=11.5, color="#55544f")
        ov, from14 = (ours if ours else (None, False))
        any_r14 |= bool(from14)
        for x, v, bold in ((xs[0], s, False), (xs[1], lit, False)):
            ax.text(x, y, fmt(v), ha="center", va="center", fontsize=12.5, color="#1d1d1b" if v else "#a3a29b")
        ax.text(xs[2], y, fmt(ov) + ("ᵃ" if from14 else ""), ha="center", va="center", fontsize=13.5,
                fontweight="bold", color="#1c5cab")
        ax.plot([0.6, 15.6], [y - rh / 2, y - rh / 2], color="#e1e0da", lw=0.8)
    note = NOTES[tag] + (" ᵃ R14-F; otherwise the new run." if any_r14 else " Ours: the new run.")
    ax.text(0.6, 0.55, note, fontsize=9.5, color="#55544f", va="center", wrap=True)
    fig.savefig(OUT / f"eval_{tag.lower()}.png", dpi=200, facecolor="white")
    plt.close(fig)


OUT.mkdir(parents=True, exist_ok=True)
for tag, sp in (("T1", T1), ("T2", T2), ("T3", T3)):
    draw(tag, sp)
    print(tag, [(r[0], fmt(r[2]), fmt(r[3]), fmt(r[4]), fmt(r[5][0]) if r[5] else "–", "R14F" if r[5] and r[5][1] else "")
                for r in sp[2]])


# ---- the same tables as Markdown ---------------------------------------------------------------------------------
md = ["# BrainQuery evaluation, 9 November 2026", "",
      "**Ours**: the new full-suite run's own heads (tables 1, 2); in table 3 the model (new run or R14-F, ᵃ) and its "
      "readout are both picked on validation, never on test. "
      "**Literature**: \\* = supervised or fine-tuned published model, usually on its own split; unstarred = frozen or "
      "linear-probe foundation model. – = no comparable value.", ""]
for i, (tag, (title, sub, rows)) in enumerate((("T1", T1), ("T2", T2), ("T3", T3)), 1):
    md += [f"## {i}. {title}", "", f"{sub[0].upper() + sub[1:]}.", "", "| Task | Metric | Single-task | Literature | Ours |",
           "|---|---|---|---|---|"]
    for name, metric, r, s_, lit, ours in rows:
        ov, f14 = ours if ours else (None, False)
        md.append(f"| {name} | {metric} | {fmt(s_)} | {fmt(lit).replace('*', chr(92) + '*')} | **{fmt(ov)}**"
                  f"{'ᵃ' if f14 else ''} |")
    md += ["", NOTES[tag].replace("*", "\\*"), ""]
(OUT / "eval_tables.md").write_text("\n".join(md))
print("markdown ->", OUT / "eval_tables.md")
