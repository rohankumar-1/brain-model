"""QC of the stored Pedi-Cranial-CT-Healthy volumes (2026-11-05): HU sanity (brain median, air, bone), brain mask
and registration flags per scan, against a few stored adult CTs (SynthRAD CT, CQ500), and a montage (one scan per
age year: axial / coronal / sagittal through the brain centre, 0-80 HU, brain-mask outline).

    python runs/diagnostics/pedi_ct_qc.py --store STORE \
        --participants STORE/labels/sources/pedi_cranial_ct/participants.tsv --out runs/diagnostics/pedi_ct_qc
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brain.prep import store


def stats(zp: Path) -> dict:
    a = store.read_many(zp, ("image", "brain"))
    img = store.squeeze_channel(a["image"]).astype(np.float32)
    out = {"air_frac": float((img < -900).mean()), "bone_frac": float((img > 300).mean()),
           "dense_bone_frac": float((img > 1000).mean()), "max_hu": float(img.max())}
    if "brain" in a:
        b = store.squeeze_channel(a["brain"]).astype(bool)
        x = img[b]
        out |= {"brain_vox": int(b.sum()), "brain_median_hu": float(np.median(x)) if x.size else np.nan,
                "brain_p05_hu": float(np.percentile(x, 5)) if x.size else np.nan,
                "brain_p95_hu": float(np.percentile(x, 95)) if x.size else np.nan,
                "brain_bone_frac": float((x > 300).mean()) if x.size else np.nan}
    return out


def montage(rows: pd.DataFrame, root: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.5 * len(rows)))
    for i, (_, r) in enumerate(rows.iterrows()):
        a = store.read_many(root / "pedi_cranial_ct" / f"{r.case_id}.zip", ("image", "brain"))
        img = store.squeeze_channel(a["image"]).astype(np.float32)
        b = store.squeeze_channel(a["brain"]).astype(bool) if "brain" in a else np.zeros(img.shape, bool)
        zc, yc, xc = (np.round(np.argwhere(b).mean(0)).astype(int) if b.any() else np.array(img.shape) // 2)
        for j, (sl, msk) in enumerate(((img[zc], b[zc]), (img[:, yc][::-1], b[:, yc][::-1]),
                                       (img[:, :, xc][::-1], b[:, :, xc][::-1]))):
            ax = axes[i, j]
            ax.imshow(sl, cmap="gray", vmin=0, vmax=80)
            if msk.any():
                ax.contour(msk, levels=[0.5], colors="r", linewidths=0.6)
            ax.set_axis_off()
        axes[i, 0].set_title(f"{r.case_id} {r.age:.1f} y {r.sex} brain med {r.brain_median_hu:.0f} HU "
                             f"ok={r.brain_ok}", fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--participants", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root = a.store
    man = pd.concat([pd.read_json(f, lines=True) for f in sorted(root.glob("manifest_v6_pedi_cranial_ct_*.jsonl"))])
    man = man.drop_duplicates(["dataset", "case_id"], keep="last")
    man = man[man["dataset"] == "pedi_cranial_ct"]
    p = pd.read_csv(a.participants, sep="\t")
    p["case_id"] = p["participant_id"] + "_" + p["session_id"]
    cols = [c for c in ("case_id", "status", "reg_ok", "reg_scale", "reg_mi", "brain_method", "brain_ok", "brain_frac",
                        "brain_dice_tpl", "brain_cover", "soft_median_hu", "fg_frac", "qc_flag", "qc_reason",
                        "tilt_deg", "offset_repaired", "spacing_mm", "error") if c in man]
    df = p[["case_id", "age", "sex"]].merge(man[cols], on="case_id", how="left")
    rec = []
    for cid in df["case_id"]:
        zp = root / "pedi_cranial_ct" / f"{cid}.zip"
        rec.append({"case_id": cid, "stored": zp.exists()} | (stats(zp) if zp.exists() else {}))
    df = df.merge(pd.DataFrame(rec), on="case_id", how="left")
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "pedi_ct_qc.csv", index=False)
    ref = []
    for folder in ("synthrad_ct", "cq500"):
        for zp in sorted((root / folder).glob("*.zip"))[:200:20]:
            if zp.name.count(".") == 1:
                ref.append({"folder": folder, "file": zp.name} | stats(zp))
    ref = pd.DataFrame(ref)
    ref.to_csv(a.out / "reference_ct_qc.csv", index=False)
    s = df[df["stored"]]
    num = ["brain_median_hu", "brain_p05_hu", "brain_p95_hu", "air_frac", "bone_frac", "dense_bone_frac", "brain_frac",
           "brain_dice_tpl", "reg_scale", "brain_bone_frac"]
    print("[qc] status", df["status"].value_counts(dropna=False).to_dict(), "stored", int(df["stored"].sum()))
    for c in ("reg_ok", "brain_ok", "brain_method", "qc_flag", "offset_repaired"):
        if c in df:
            print(f"[qc] {c}", df[c].value_counts(dropna=False).to_dict())
    print("[qc] pedi_cranial_ct\n", s[[c for c in num if c in s]].describe().round(3).to_string())
    print("[qc] reference adult CT (median of 10 per folder)\n",
          ref.groupby("folder")[[c for c in num if c in ref]].median().round(3).to_string())
    ok = s["brain_ok"].astype(str).str.lower().isin(["true", "1", "1.0"])
    bad = s[~s["brain_median_hu"].between(15, 50) | (s["air_frac"] < 0.05) | (s["bone_frac"] < 0.01) | ~ok]
    print(f"[qc] outside brain median 15-50 HU / no air / no bone / brain_ok False: {len(bad)}")
    if len(bad):
        print(bad[[c for c in ("case_id", "age", "brain_median_hu", "air_frac", "bone_frac", "brain_ok",
                               "brain_dice_tpl", "reg_scale", "qc_reason") if c in bad]].to_string())
    print("[qc] by age (years)\n", s.groupby(pd.cut(s["age"], [-0.01, 1, 2, 4, 6, 8, 10]), observed=True)[
        ["brain_median_hu", "brain_frac", "reg_scale", "brain_dice_tpl"]].median().round(3).to_string())
    srt = s.sort_values("age")
    montage(srt.groupby(srt["age"].astype(int)).head(1), root, a.out / "pedi_ct_montage.png")
    print(f"[qc] wrote {a.out}/pedi_ct_qc.csv, reference_ct_qc.csv, pedi_ct_montage.png")


if __name__ == "__main__":
    main()
