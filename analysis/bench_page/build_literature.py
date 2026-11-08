"""The "Literature Comparison" artifact page: per benchmark task, our number next to the best comparable published result
(frozen-encoder / linear-probe foundation models, same task unit) and the best published result under any protocol.
Literature numbers come from the 2026-11-07 survey (runs/diagnostics/literature_sota/*.md; every value read in the
cited paper's text or tables). Ours: R14-F (newest scored run), frozen benchmark v3 unless marked "own head".
    python build_literature.py OUT.html"""
import sys
from pathlib import Path

HERE = Path(__file__).parent

REFS = {
    1: "Chilamkurthy et al., Lancet 2018 (arXiv 1803.05854): supervised 2D CNNs on ~290k proprietary head CTs; CQ500 as external test, same 3-reader labels.",
    2: "Wang X et al., NeuroImage: Clinical 32:102785, 2021: RSNA 2019 1st-place model, scan-level on CQ500 with the authors' own labels.",
    3: "Zhu et al., FM-HCT, arXiv 2502.02779 (Nat Biomed Eng 2026): SSL on 361k private head CTs, fine-tuned; CQ500 10-task mean 0.863; RSNA 5-class macro 0.923.",
    4: "Agrawal et al., Pillar-0, arXiv 2511.17803, Table 10: fine-tuned, exam-level any-ICH on their RSNA split (Teneggi et al. arXiv 2211.15924: 0.961).",
    5: "Li et al., INSTANCE 2022 challenge, arXiv 2301.03281, Table IV: hidden test, winner 0.693, best Dice 0.721.",
    6: "Song et al., HemSeg-200, arXiv 2405.14559, Table I: nnU-Net, IPH/IVH masks.",
    7: "Wu et al., BHSD, MLMI 2023 (arXiv 2308.11298): binary bleed Dice 0.45-0.50 (supervised / semi-supervised).",
    8: "Liang et al., SEAN, MICCAI 2021 (arXiv 2110.05039), Table 1: AISD official split, 3D supervised.",
    9: "Liu et al., Radiology 2024;313:e240137: 90-day mRS from imaging AUC 0.85 (NCCT days 1-7, 6 cohorts); Hilbert et al. 2019 CTA 0.71. No ISLES'24 outcome results published.",
    10: "RSNA 2025 Intracranial Aneurysm challenge: blended metric 0.5 x AUC(present) + 0.5 x mean(13 location AUCs); 2nd place 0.867 private (arXiv 2606.26706).",
    11: "Ergun et al., arXiv 2606.13315, Tables 2-3: frozen 3D MAE / SL-MAE with an attentive probe; ADNI all timepoints, subject-level 5-fold CV.",
    12: "Wu et al., BrainDINO, arXiv 2604.27277: frozen 2D ViT-B + MLP head; ADNI1 longitudinal, PPMI baseline, UCSF (T1c + FLAIR; pretraining includes BraTS 2025, i.e. UCSF patients), UPENN.",
    13: "Van Puyvelde et al., BrainG3N, arXiv 2606.19651: frozen ridge on a pooled multi-cohort set.",
    14: "Bhagwat et al., J Psychiatry Neurosci 2019, Table 4: ADNI baseline, ROI features, r = 0.55 (R2 ~ 0.30).",
    15: "Kaczmarek et al., 3D-Neuro-SimCLR, arXiv 2509.10620, Table 2: IXI linear probe MAE 5.90, fine-tuned 4.35 (R2 estimated from MAE and IXI's age SD 16.7).",
    16: "Tak et al., BrainIAC, Nature Neuroscience 2026: linear probe / fine-tuned; UCSF IDH LP 0.69, FT 0.79; UPENN 12-month survival LP 0.68, FT 0.72.",
    17: "Byeon et al., GlioMT, npj Digital Medicine 2025, Table 2: 4 sequences + age / sex, trained elsewhere, UCSF external; grade is a 3-class macro AUROC.",
    18: "Kim et al., Cancers 2022: RSNA-MICCAI 2021 MGMT winner (T1c only) 0.621 private, 0.562 on 400 external patients; Saeed et al. MedIA 2023: T1c 0.63.",
    19: "Christodoulou et al., Front Oncol 2026; Astaraki et al., Front Oncol 2025/26: radiomics MGMT 0.62-0.67 (pooled UCSF + UPENN / CV).",
    20: "Ruffle et al., Brain Communications 2023: dedicated nnU-Net per sequence set, BraTS 2021 CV; whole tumour from T1c alone 0.908.",
    21: "Missing-modality models, T1c-only input (BraTS 2018): mmFormer (MICCAI 2022) ET 0.726 / TC 0.754; M3AE (AAAI 2023) ET 0.737 / TC 0.829.",
    22: "Munk et al., AMAES, ADSMI @ MICCAI 2024 (arXiv 2408.00640), Table 4: SSL model fine-tuned on 20 cases; WMH FLAIR-only 0.69-0.72, ISLES'22 DWI 0.54-0.57.",
    23: "Kuijf et al., IEEE TMI 2019 (WMH challenge): winner Dice 0.80 (FLAIR + T1, official test); Park et al. NeuroImage 2021: 0.81.",
    24: "Bashiri Bawil et al., MS3SEG (Scientific Data 2026; official repository): 2D U-Net 5-fold CV, 4-class: MS lesion 0.667, ventricles 0.890.",
    25: "de la Rosa et al., ISLES'22 / DeepISLES (arXiv 2403.19425, Nat Commun 2025): winner median Dice 0.82 (DWI + ADC, hidden test).",
    26: "Sudre et al., VALDO, Medical Image Analysis 2024: EPVS count Spearman rho 0.70-0.75 (ensembles); lesion F1 ~0.39; inter-rater F1 ~0.20.",
    27: "Wald et al., OpenMind, arXiv 2412.17041: the one frozen-encoder segmentation result on our datasets (ATLAS): frozen MAE ViT 0.576 vs 0.615 fine-tuned.",
}

# (task, metric, ours R14-F, best earlier BQ "R13-EMA 0.961" or None, comparable (value, label, ref) or None,
#  ceiling (value, label, ref) or None, note)
G = [
 ("CT · CQ500 (external, 471 scans; frozen encoder)", [
  ("Any bleed", "AUC", .897, "R14 0.898", None, (.942, "supervised, 290k scans", 1), "Wang (own labels): 0.985 [2]"),
  ("Intraparenchymal", "AUC", .878, None, None, (.954, "supervised, 290k scans", 1), ""),
  ("Intraventricular", "AUC", .957, "R10 0.959", None, (.931, "supervised, 290k scans", 1), ""),
  ("Subdural", "AUC", .856, "R10 avg 0.876", None, (.952, "supervised, 290k scans", 1), "largest CQ500 gap: chronic SDH, site shift"),
  ("Subarachnoid", "AUC", .869, "R13 avg 0.877", None, (.957, "supervised, 290k scans", 1), ""),
  ("Epidural", "AUC", .847, "R14 avg 0.856", None, (.973, "supervised, 290k scans", 1), ""),
  ("Calvarial fracture", "AUC", .759, "R13 0.774", None, (.962, "supervised, 290k scans", 1), ""),
  ("Mass effect", "AUC", .933, "R13 0.935", None, (.922, "supervised, 290k scans", 1), ""),
  ("Midline shift", "AUC", .940, "R10 0.948", None, (.970, "supervised, 290k scans", 1), ""),
  ("Bleed side (unilateral)", "AUC", .907, None, None, None, "no published result"),
  ("Other fracture", "AUC", .796, "R13 0.847", None, None, "no published result"),
  ("All 8 findings, macro", "AUC", .886, None, (.863, "FM-HCT fine-tuned, 10-task mean", 3), None,
   "FM-HCT tasks differ slightly; HLIP's frozen probe beats FM-HCT (absolutes only in a figure)"),
 ]),
 ("CT · CQ500 read by our own RSNA heads (zero-shot: no CQ500 labels; all 471 scans)", [
  ("Any bleed", "AUC", .917, None, None, (.942, "supervised, 290k scans", 1), "frozen readout 0.897"),
  ("Intraparenchymal", "AUC", .903, None, None, (.954, "supervised, 290k scans", 1), "frozen readout 0.878"),
  ("Intraventricular", "AUC", .981, None, None, (.931, "supervised, 290k scans", 1), "frozen readout 0.957"),
  ("Subdural", "AUC", .903, None, None, (.952, "supervised, 290k scans", 1), "frozen readout 0.856"),
  ("Subarachnoid", "AUC", .935, None, None, (.957, "supervised, 290k scans", 1), "frozen readout 0.869"),
  ("Epidural (13 positives)", "AUC", .882, None, None, (.973, "supervised, 290k scans", 1),
   "frozen readout 0.847; about the head's RSNA in-domain 0.878"),
 ]),
 ("CT · RSNA ICH (own trained heads, our held-out split)", [
  ("Any bleed", "AUC", .960, "R13 avg 0.964", (.967, "Pillar-0 fine-tuned", 4), None, "Teneggi 0.961 [4]"),
  ("5-class macro (any, IPH, IVH, SAH, SDH)", "AUC", .947, None, (.923, "FM-HCT fine-tuned", 3), None, ""),
  ("Epidural", "AUC", .878, None, None, None, "no comparable study-level result"),
 ]),
 ("CT · segmentation and stroke", [
  ("Bleed segmentation (own head)", "Dice", .569, "R14 0.589", None, (.790, "nnU-Net, HemSeg-200 IPH/IVH", 6),
   "BHSD binary 0.45-0.50 [7]"),
  ("INSTANCE bleed (frozen + decoder)", "Dice", .522, None, None, (.693, "challenge winner, hidden test", 5), "R14 encoder"),
  ("AISD infarct (frozen + decoder)", "Dice", None, None, None, (.578, "SEAN, supervised", 8), "our run diverged (NaN), open"),
  ("AISD old infarct present", "AUC", .726, "R14 0.729", None, None, "no published result"),
  ("ISLES'24 LVO side (CTA)", "AUC", .955, None, None, None, "no published result; LVO detection 0.74-0.89 elsewhere"),
  ("ISLES'24 poor outcome, 90 d (NCCT)", "AUC", .732, "R12 0.737", None, (.85, "imaging-only, other cohorts", 9),
   "untrained encoder 0.730: task barely separates models"),
  ("ISLES'24 poor outcome, discharge", "AUC", .695, None, None, None, "untrained encoder 0.739"),
  ("RSNA-ICA aneurysm present, CTA", "AUC", .751, "R14 0.765", None, None, "challenge score 0.867 is a blended metric [10]"),
  ("RSNA-ICA aneurysm present, MR", "AUC", .680, "R14 0.686", None, None, "as above"),
 ]),
 ("MRI · neurodegeneration and age (frozen encoder)", [
  ("ADNI CN vs AD", "AUC", .918, None, (.922, "frozen 3D MAE probe", 11), (.945, "same, fine-tuned", 11), "BrainDINO frozen 0.850 [12]"),
  ("ADNI CN vs MCI", "AUC", .757, "R10 0.765", (.757, "frozen 3D MAE probe", 11), (.780, "same, fine-tuned", 11), "BrainDINO 0.701 [12]"),
  ("ADNI CN / MCI / AD", "macro AUC", .775, None, (.69, "BrainDINO frozen", 12), None, "only published result"),
  ("ADNI MMSE", "R²", .375, "R13 avg 0.385", (.427, "BrainG3N frozen, pooled cohorts", 13), None, "ADNI-only ROI model ~0.30 [14]"),
  ("PPMI Parkinson's vs control", "AUC", .514, "R10 0.614", (.610, "BrainDINO frozen", 12), None,
   "BrainFMBench reports chance for all FMs (unverified); untrained encoder 0.684"),
  ("IXI age", "R²", .827, "R14 avg 0.840", (.80, "3D SimCLR linear probe (from MAE)", 15), (.89, "same, fine-tuned", 15), ""),
  ("Pixar age (children + adults)", "R²", .831, "R14 avg 0.879", None, None, "no published result"),
  ("MS3SEG age (MS patients)", "R²", .438, None, None, None, "no published result; MS brain-age gap caps R² ~0.5-0.65"),
  ("OASIS-1 / -2 CDR, ABIDE autism", "AUC", None, None, (.83, "OASIS-1 CDR: BrainDINO frozen", 12), (.88, "BrainIAC fine-tuned", 16),
   "trained on by BrainQuery (seen): no transfer number; ABIDE T1-only 0.58-0.65 [11]"),
 ]),
 ("MRI · glioma (frozen encoder, one T1c scan)", [
  ("UCSF IDH mutation", "AUC", .915, "R13 avg 0.961", (.901, "BrainDINO frozen, T1c + FLAIR", 12), (.981, "GlioMT, 4 seq + age / sex", 17),
   "BrainIAC linear probe 0.69 [16]"),
  ("UCSF grade 2-3 vs 4", "AUC", .968, None, None, (.960, "GlioMT, 3-class macro", 17), "no binary result published"),
  ("UCSF MGMT (IDH-wt GBM)", "AUC", .477, "R10 0.520", (.621, "RSNA-MICCAI winner, T1c only", 18), None,
   "0.562 on external data; MRI predicts MGMT barely above chance"),
  ("UCSF survival 12 mo", "AUC", .742, "R13 0.820", None, None, "no 12-month result published"),
  ("UPENN survival 12 mo (own head)", "AUC", .701, "R14 0.782", (.683, "BrainDINO frozen", 12), (.72, "BrainIAC fine-tuned", 16),
   "BrainIAC linear probe 0.68"),
  ("UPENN MGMT (own head)", "AUC", .406, "R14 0.452", None, (.67, "radiomics, pooled cohorts", 19), ""),
 ]),
 ("MRI · tumour masks (own heads, single sequence)", [
  ("Whole tumour", "Dice", .617, "R14 0.638", None, (.908, "nnU-Net from T1c alone", 20),
   "0.652 with co-registered sequences (R14-F)"),
  ("Tumour core", "Dice", .378, None, None, (.829, "missing-modality model, T1c only", 21),
   "ours asked on every sequence (labels v11); T1c-only R16 pending"),
  ("Enhancing tumour", "Dice", .335, None, None, (.737, "missing-modality model, T1c only", 21), "as above"),
  ("MSD enhancing-tumour burden", "AUC", .757, None, None, None, "no published result"),
 ]),
 ("MRI · lesions (frozen encoder + decoder, single sequence)", [
  ("WMH (FLAIR)", "Dice", .578, None, (.70, "AMAES fine-tuned, 20 cases, FLAIR", 22), (.80, "challenge winner, FLAIR + T1", 23),
   "untrained encoder 0.596"),
  ("MS3SEG MS lesion", "Dice", .543, None, None, (.667, "2D U-Net, 4-class", 24), "ventricles 0.825 vs 0.890"),
  ("ISLES'22 infarct", "Dice", .528, None, (.55, "AMAES fine-tuned, 20 cases, DWI", 22), (.82, "challenge winner (median)", 25), ""),
  ("ATLAS lesion volume", "R²", .777, "R10 avg 0.792", None, None, "no published result; frozen ATLAS segmentation 0.58 [27]"),
  ("ATLAS time since stroke", "R²", .304, "R10 0.419", None, None, "BrainDINO reports MAE only"),
  ("MS3SEG lesion load", "R²", .751, "R13 avg 0.778", None, None, "no published result"),
  ("VALDO EPVS count", "R²", .066, "R12 0.173", None, None, "count Spearman 0.70-0.75 [26] (rho² ~0.5, not the same metric)"),
 ]),
]


def fmt(x):
    return "–" if x is None else (f"{x:.2f}" if abs(x) >= 0.995 or x in (.80, .70, .85, .55, .89, .69, .83, .88, .67, .72) else f"{x:.3f}")


def gap(ours, ref):
    if ours is None or ref is None:
        return '<td class="num quiet">–</td>'
    d = ours - ref[0]
    cls = "up" if d >= 0 else "near" if d > -0.05 else "down"
    return f'<td class="num gap {cls}">{d:+.3f}</td>'


def ref_cell(ref):
    if ref is None:
        return '<td class="quiet">–</td>'
    v, label, n = ref
    return f'<td><span class="v">{fmt(v)}</span> <span class="lab">{label}</span> <sup>[{n}]</sup></td>'


rows = []
for title, items in G:
    rows.append(f'<tr class="group"><td colspan="8">{title}</td></tr>')
    for task, metric, ours, earlier, comp, ceil, note in items:
        rows.append(
            f'<tr><td>{task}</td><td>{metric}</td>'
            f'<td class="num ours">{fmt(ours)}</td><td class="quiet small">{earlier or ""}</td>'
            f'{ref_cell(comp)}{gap(ours, comp)}{ref_cell(ceil)}{gap(ours, ceil)}</tr>'
            + (f'<tr class="note"><td></td><td colspan="7">{note}</td></tr>' if note else ""))

refs = "".join(f"<li value=\"{n}\">{t}</li>" for n, t in REFS.items())
css = (HERE / "style.css").read_text() + """
td.ours { font-weight: 600; }
td.small, .lab { font-size: 13px; color: var(--soft); }
td .v { font-variant-numeric: tabular-nums; }
td.gap.up { background: var(--best); }
td.gap.near { background: var(--tie); }
td.gap.down { background: #f6dcd6; }
tr.note td { font-size: 13px; color: var(--soft); padding-top: 0; }
table.lit { min-width: 860px; }
ol.refs { font-size: 13px; color: var(--soft); max-width: 90ch; padding-left: 26px; }
ol.refs li + li { margin-top: 3px; }
"""
page = f'''<title>Literature Comparison</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
{css}
</style>

<div class="page">
  <h1>BrainQuery against the published state of the art, task by task</h1>
  <p class="date">7 November 2026 · literature checked in the cited papers</p>
  <p class="lede">Ours: R14-F, the newest scored run (frozen encoder with a linear / MLP / small conv readout on the
    task's training split, one scan; own trained heads where marked). Comparable: the best published result under a
    similar protocol (frozen or linear-probe foundation models, same task unit). Best published: any protocol, mostly
    end-to-end supervised or fine-tuned, often on far more labelled data or several sequences. Gap = ours minus the
    reference: green at or above it, light green within 0.05, red further below.</p>
  <div class="tablebox"><table class="lit"><thead><tr><th>Task</th><th>Metric</th><th class="num">Ours</th>
    <th>Best earlier BQ</th><th>Comparable published</th><th class="num">Gap</th><th>Best published (any protocol)</th>
    <th class="num">Gap</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>

  <h2>Reading it</h2>
  <ol class="notes">
    <li>Where comparable results exist, ours is at or above them: ADNI (CN vs AD 0.918 vs 0.922 for a frozen 3D MAE),
      the ADNI 3-class task, IXI age, UCSF IDH and grade, RSNA ICH and the CQ500 macro.</li>
    <li>Against fully supervised references the gap is ~0.02-0.10 AUC on CQ500 (largest on subdural, epidural and
      skull fracture) with three findings at or above Chilamkurthy et al., whose model trained on ~290k labelled head CTs.</li>
    <li>Read by the model's own RSNA heads, with no CQ500 labels, every bleed finding beats the CQ500-fitted frozen
      readout (+0.02 to +0.07 AUC): IVH 0.981 is above the supervised reference, SAH and any bleed within 0.03.
      Refitting those heads on CQ500's labels never improved them; started from the head, one or two positives
      match what a from-scratch probe needs eight or more for.</li>
    <li>Segmentation is the clear gap: frozen tokens with a token-grid decoder reach 0.52-0.58 Dice against 0.69-0.82
      for supervised models, and below the 20-case fine-tuned AMAES on WMH. Our tumour masks are far below
      single-sequence references.</li>
    <li>Several tasks are near chance for everyone (MGMT, PPMI, ISLES'24 outcome, EPVS count); an untrained encoder
      matches trained ones there, so they say little about representation quality.</li>
    <li>Many of our tasks have no published counterpart (bleed side, other fracture, LVO side, ATLAS lesion volume, MS
      lesion load, Pixar / MS3SEG age, aneurysm presence per modality).</li>
    <li>Caveats: published numbers use their own splits; test sets of 40-130 cases carry ±0.05-0.10 AUC intervals;
      several foundation models saw our evaluation cohorts in pretraining (noted per reference); "best earlier BQ"
      is the best of eight checkpoints, chosen with hindsight.</li>
  </ol>

  <h2>References</h2>
  <ol class="refs">{refs}</ol>
</div>
'''
if __name__ == "__main__":
    Path(sys.argv[1]).write_text(page)
    print(len(page))
