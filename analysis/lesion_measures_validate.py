"""Validate the mask-derived lesion labels (``brain.multitask.lesion_measures``) before they train (2026-11-12).

* side: Seg-CQ500 bleed masks vs the three readers' BleedLocation-Left / -Right (majority), over unilateral
  thresholds; the store's x axis convention is checked by the same comparison;
* compartments: BHSD masks by subtype (1 EDH, 2 IPH, 3 IVH, 4 SAH, 5 SDH): share of each subtype's voxels in each
  prior compartment (IVH should land intraventricular, EDH / SDH extra-axial);
* midline shift (CT): CQ500 radiologist MidlineShift (majority) vs the septum measure, AUC + coverage; healthy-CT
  (Pedi-Cranial-CT) and MRI (SynthSeg: IXI healthy, BraTS) distributions;
* components: per family distribution.

Aggregates only (no case ids). Writes <out>/{side.csv, compartments.csv, midline.csv, components.csv, summary.txt}.

    python analysis/lesion_measures_validate.py --out ~/mask_labels_work/validate --workers 8
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.downstream.tasks import DEFAULT_VOLUMES, label_file  # noqa: E402
from brain.multitask import lesion_measures as LM  # noqa: E402
from brain.prep import store  # noqa: E402

ROOT = DEFAULT_VOLUMES.parent
PRIORS: dict = {}
BAD_SEG_CQ500 = {"CQ500-CT-108", "CQ500-CT-65", "CQ500-CT-61", "CQ500-CT-173", "CQ500-CT-75"}


def _init(prior_path: str) -> None:
    PRIORS.update(LM.compartment_priors(Path(prior_path)))


def _mask(ref: str, ids: list[int] | None) -> np.ndarray:
    m = np.asarray(store.read_image(ROOT / ref, "mask"))
    return np.isin(m, ids) if ids else m > 0


def job_lesion(args) -> dict:
    key, image_path, ref, ids, per_id = args
    brain = np.asarray(store.read(ROOT / image_path, "brain")) > 0
    brain = brain[:, 0] if brain.ndim == 4 else brain
    m = np.asarray(store.read_image(ROOT / ref, "mask"))
    out = {"key": key, **LM.lesion_stats(np.isin(m, ids) if ids else m > 0, brain, PRIORS)}
    if per_id:
        for i in per_id:
            s = LM.lesion_stats(m == i, brain, PRIORS)
            for c in LM.LOCATION_CLASSES + ("ml",):
                kk = f"{c}_ml" if c != "ml" else "ml"
                out[f"id{i}_{kk}"] = s[kk]
    return out


def job_midline(args) -> dict:
    key, image_path, modality, synthseg = args
    a = store.read_many(ROOT / image_path, ("image", "brain"))
    brain = store.squeeze_channel(a["brain"]) > 0
    if modality == "ct":
        vent = LM.ventricles_ct(store.squeeze_channel(a["image"]), brain)
    else:
        vent = LM.ventricles_synthseg(store.read_image(ROOT / synthseg, "mask"))
    from brain.multitask.measures import head_shell, symmetry_plane

    out = {"key": key}
    plane = symmetry_plane(head_shell(store.squeeze_channel(a["image"]).astype(np.float32), modality))
    refs = [("grid", LM.GRID_MID_X), ("cent", LM.midline_x(brain))] + ([("plane", plane)] if plane else [])
    for tag, x0 in refs:
        r = LM.midline_shift(vent, PRIORS, x0)
        out.update({f"{k}_{tag}": v for k, v in r.items()})
    return out


def auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    ok = ~np.isnan(s)
    return float(roc_auc_score(y[ok], s[ok])) if len(np.unique(y[ok])) == 2 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--priors", type=Path, default=None)
    ap.add_argument("--n-sample", type=int, default=150)
    ap.add_argument("--parts", default="side,compartment,lesion,midline")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pri = a.priors or a.out / "ixi_synthseg_compartments.npz"
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
    if not Path(pri).exists():
        ixi = v[v["dataset"] == "ixi_t1"].sample(60, random_state=0)
        LM.compartment_priors(pri, [store.path_for(ROOT, f, c, "synthseg") for f, c in zip(ixi.folder, ixi.case_id)])
    log = open(a.out / f"summary_{a.parts.replace(',', '_')}.txt", "w")

    def say(*x):
        print(*x, flush=True)
        print(*x, file=log, flush=True)

    ss = lambda r: store.path_for(ROOT, r.folder, str(r.case_id), "synthseg")      # noqa: E731
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(pri),)) as ex:
        reads = pd.read_csv(label_file("cq500_meta", "reads.csv"))
        vote = pd.DataFrame({"name": reads["name"].astype(str)})
        for lab in ("BleedLocation-Left", "BleedLocation-Right", "MidlineShift", "ICH", "SDH", "MassEffect"):
            cols = [c for c in reads.columns if c.endswith(":" + lab)]
            vote[lab] = reads[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        # ---------------- side on Seg-CQ500
        if "side" in a.parts:
            seg = v[(v["dataset"] == "seg_cq500") & v["mask_ich"].notna()].copy()
            seg["name"] = seg["volume_id"].astype(str).str.split(":", n=1).str[1]
            seg = seg[~seg["name"].isin(BAD_SEG_CQ500)].merge(vote, on="name")
            jobs = [(r.name, r.image_path, r.mask_ich, None, None) for r in seg.itertuples()]
            side = pd.DataFrame(list(ex.map(job_lesion, jobs))).merge(seg[["name", "BleedLocation-Left",
                                                                             "BleedLocation-Right"]],
                                                                       left_on="key", right_on="name")
            side.drop(columns=["key", "name"]).to_csv(a.out / "side.csv", index=False)
            L, R = side["BleedLocation-Left"] >= 0.5, side["BleedLocation-Right"] >= 0.5
            truth = np.where(L & ~R, 0, np.where(R & ~L, 1, np.where(L & R, 2, -1)))
            lf = side["left_ml"] / (side["left_ml"] + side["right_ml"]).clip(lower=1e-9)
            say(f"[side] Seg-CQ500 masks n={len(side)}; readers: left {int((truth == 0).sum())} right "
                f"{int((truth == 1).sum())} both {int((truth == 2).sum())} none {int((truth == -1).sum())}")
            say(f"[side] unilateral reads: median left_frac when read LEFT {lf[truth == 0].median():.2f}, when read RIGHT "
                f"{lf[truth == 1].median():.2f} (convention check: x < midline = patient left)")
            for t in (0.6, 0.7, 0.8, 0.9, 0.95):
                pred = np.array([LM.side_class(l, r, t) for l, r in zip(side.left_ml, side.right_ml)])
                k, k2 = truth >= 0, (truth >= 0) & (truth < 2)
                say(f"[side] unilateral={t}: 3-class agreement {np.mean(pred[k] == truth[k]):.3f} (n={int(k.sum())}); "
                    f"on unilateral reads {np.mean(pred[k2] == truth[k2]):.3f} (n={int(k2.sum())}); on bilateral reads "
                    f"{np.mean(pred[truth == 2] == 2) if (truth == 2).any() else float('nan'):.3f}")
        # ---------------- compartments on BHSD by subtype
        if "compartment" in a.parts:
            bh = v[(v["dataset"] == "bhsd") & v["mask_ich"].notna()]
            jobs = [(str(i), r.image_path, r.mask_ich, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) for i, r in enumerate(bh.itertuples())]
            comp = pd.DataFrame(list(ex.map(job_lesion, jobs)))
            comp.to_csv(a.out / "compartments.csv", index=False)
            names = {1: "EDH", 2: "IPH", 3: "IVH", 4: "SAH", 5: "SDH"}
            for i, nm in names.items():
                tot = comp[f"id{i}_ml"].sum()
                if tot <= 0:
                    continue
                shares = {c: comp[f"id{i}_{c}_ml"].sum() / tot for c in LM.LOCATION_CLASSES}
                dom = [LM.location_class({f"{c}_ml": r[f"id{i}_{c}_ml"] for c in LM.LOCATION_CLASSES})
                       for _, r in comp[comp[f"id{i}_ml"] > 0].iterrows()]
                say(f"[compartment] BHSD {nm} (n={int((comp[f'id{i}_ml'] > 0).sum())} scans, {tot:.0f} ml): voxel share "
                    + " ".join(f"{c[:6]}={s:.2f}" for c, s in shares.items())
                    + " | dominant " + str(pd.Series([LM.LOCATION_CLASSES[d] for d in dom]).value_counts().to_dict()))
        # ---------------- components
        if "lesion" in a.parts:
            rows = []
            for fam, ref_col, ids in (("bhsd", "mask_ich", [1, 2, 3, 4, 5]), ("hemseg", "mask_ich", [1])):
                f = v[(v["dataset_family"] == fam) & v[ref_col].notna()]
                rows += [(fam, r.image_path, getattr(r, ref_col), ids, None) for r in f.itertuples()]
            bt = v[v["dataset"].isin(["brats2023_gli_t1n", "brats2023_men_t1n", "brats2023_ped_t1n"])]
            bt = bt.sample(min(a.n_sample, len(bt)), random_state=0)
            for r in bt.itertuples():
                mp = f"{r.folder}/{str(r.case_id).rsplit('-', 1)[0]}.seg.zip"
                if (ROOT / mp).exists():
                    rows.append((r.dataset_family + ":core", r.image_path, mp, [1, 3], None))
                    rows.append((r.dataset_family + ":whole", r.image_path, mp, [1, 2, 3], None))
            rs = v[v["dataset_family"] == "rsna_ich"].sample(3000, random_state=0)
            n_rs = 0
            for r in rs.itertuples():
                mp = store.path_for(ROOT, r.folder, str(r.case_id), "bleedpl")
                if mp.exists() and n_rs < a.n_sample:
                    rows.append(("rsna_bleedpl", r.image_path, str(mp.relative_to(ROOT)), [1], None))
                    n_rs += 1
            res = pd.DataFrame(list(ex.map(job_lesion, rows, chunksize=2)))
            res["family"] = [r[0] for r in rows]
            res.to_csv(a.out / "components.csv", index=False)
            for fam, g in res.groupby("family"):
                sc = [LM.side_class(l, r) for l, r in zip(g.left_ml, g.right_ml)]
                lc = [LM.location_class(r) for r in g.to_dict("records")]
                say(f"[lesion] {fam} n={len(g)} ml median {g.ml.median():.1f}; components "
                    + str(g.components.clip(upper=5).value_counts().sort_index().to_dict())
                    + "; side " + str(pd.Series([LM.SIDE_CLASSES[s] if s >= 0 else "none" for s in sc]).value_counts().to_dict())
                    + "; location " + str(pd.Series([LM.LOCATION_CLASSES[c] if c >= 0 else "none" for c in lc]).value_counts().to_dict()))
        # ---------------- midline shift: CQ500 (CT) + controls
        if "midline" in a.parts:
            cq = v[v["dataset"] == "cq500"].copy()
            cq["name"] = cq["volume_id"].astype(str).str.split(":", n=1).str[1]
            cq = cq.merge(vote, on="name")
            jobs = [("cq500|" + r.name, r.image_path, "ct", None) for r in cq.itertuples()]
            ped = v[v["dataset_family"] == "pedi_cranial_ct"]
            jobs += [("pedi|" + str(i), r.image_path, "ct", None) for i, r in enumerate(ped.itertuples())]
            rsn = v[v["dataset_family"] == "rsna_ich"].sample(a.n_sample, random_state=1)
            jobs += [("rsna|" + str(r.volume_id), r.image_path, "ct", None) for r in rsn.itertuples()]
            for ds in ("ixi_t1", "brats2023_gli_t1n", "brats2023_men_t1n"):
                g = v[v["dataset"] == ds].sample(min(a.n_sample, (v["dataset"] == ds).sum()), random_state=2)
                jobs += [(ds + "|" + str(r.volume_id), r.image_path, "mri", str(ss(r).relative_to(ROOT)))
                         for r in g.itertuples() if ss(r).exists()]
            mid = pd.DataFrame(list(ex.map(job_midline, jobs, chunksize=2)))
            mid["set"] = mid["key"].str.split("|").str[0]
            cqm = mid[mid["set"] == "cq500"].assign(name=lambda d: d["key"].str.split("|").str[1]).merge(vote, on="name")
            mid.drop(columns=["key"]).to_csv(a.out / "midline.csv", index=False)
            cqm.drop(columns=["key", "name"]).to_csv(a.out / "midline_cq500.csv", index=False)
            y = (cqm["MidlineShift"] >= 0.5).astype(int).to_numpy()
            say(f"[midline] CQ500 n={len(cqm)} MidlineShift majority positives {int(y.sum())}")
            for tag in ("grid", "cent", "plane"):
                for m in ("mls_mm", "mls_p90"):
                    if f"{m}_{tag}" not in cqm:
                        continue
                    s = cqm[f"{m}_{tag}"].to_numpy(float)
                    cov_pos, cov_neg = np.mean(~np.isnan(s[y == 1])), np.mean(~np.isnan(s[y == 0]))
                    filled = np.where(np.isnan(s), np.nanmedian(s), s)
                    say(f"[midline] {tag} {m}: AUC measured {auc(y, s):.3f} (coverage pos {cov_pos:.2f} neg {cov_neg:.2f}); "
                        f"AUC NaN->median {auc(y, filled):.3f}; median mm pos {np.nanmedian(s[y == 1]):.1f} "
                        f"neg {np.nanmedian(s[y == 0]):.1f}")
                    for thr in (3, 5, 8):
                        p = (s >= thr)
                        k = ~np.isnan(s)
                        tp = int((p & (y == 1) & k).sum())
                        say(f"[midline]    >= {thr} mm: sens {tp / max(int(((y == 1) & k).sum()), 1):.2f} spec "
                            f"{int((~p & (y == 0) & k).sum()) / max(int(((y == 0) & k).sum()), 1):.2f}")
            for st, g in mid.groupby("set"):
                if st == "cq500":
                    continue
                for tag in ("grid", "plane"):
                    s = g.get(f"mls_mm_{tag}", pd.Series(dtype=float))
                    say(f"[midline] {st} {tag} n={len(g)} coverage {s.notna().mean():.2f} median {s.median():.2f} p95 "
                        f"{s.quantile(0.95):.2f} share >= 5 mm {(s >= 5).mean():.3f}")
            # agreement with the existing SynthSeg measure (brain.multitask.measures, labels/synthseg/measures.csv)
            mpath = ROOT / "labels" / "synthseg" / "measures.csv"
            if mpath.exists():
                meas = pd.read_csv(mpath, usecols=["volume_id", "shift_mm"])
                other = mid[mid["set"] != "cq500"].assign(volume_id=lambda d: d["key"].str.split("|", n=1).str[1])
                j = other.merge(meas, on="volume_id").dropna(subset=["shift_mm"])
                for st, g in j.groupby("set"):
                    for tag in ("grid", "plane"):
                        c = f"mls_mm_{tag}"
                        if c in g and g[c].notna().sum() >= 10:
                            k = g[c].notna()
                            rho = pd.Series(g.loc[k, c]).corr(g.loc[k, "shift_mm"], method="spearman")
                            say(f"[midline] vs measures.csv shift_mm {st} {tag}: n={int(k.sum())} spearman {rho:.2f} "
                                f"median |diff| {np.median(np.abs(g.loc[k, c] - g.loc[k, 'shift_mm'])):.2f} mm")
    log.close()


if __name__ == "__main__":
    main()
