"""Does registration hold up across every dataset, and does defacing bias it? (2026-10-29)

Skull-on scans are fitted to the whole-head MNI template, face included, by a metric over the whole image, so a scan
whose face was removed could be pulled (pitched / shifted / scaled). Per sampled scan this measures, on the stored grid:
* face_frac: the share of a face region (in front of and below the brain where >= 90% of IXI heads, which are not
  defaced, have tissue) that holds signal in this scan: ~1 intact, ~0 defaced;
* the brain mask's centroid offset (mm, z = superior, y = anterior) and extent ratio against the mean IXI brain, and its
  Dice with it (systematic pitch / shift / scale show up here);
then compares defaced with intact scans per cohort kind. Also renders, per dataset, 2 random + 2 lowest-template-Dice
scans: sagittal (the face) and axial, with the mean IXI brain outline (green) and the scan's own brain mask (red).

    python runs/diagnostics/registration_qc.py [--per 15]
Writes runs/diagnostics/registration_qc.csv and registration_qc_<page>.png."""
import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "/home/roku/brain-model")
from brain.downstream.tasks import DEFAULT_VOLUMES  # noqa: E402
from brain.prep import store  # noqa: E402

ROOT = DEFAULT_VOLUMES.parent
OUT = "/home/roku/brain-model/runs/diagnostics/registration_qc"
ROWS_PER_PAGE = 26


def load(path):
    a = store.read_many(ROOT / path, ("image", "brain"))
    img = store.squeeze_channel(np.asarray(a["image"])).astype(np.float32)
    return img, (np.asarray(a["brain"]) > 0) if "brain" in a else np.zeros(img.shape, bool)


def signal(img, ct):
    return img > (-500 if ct else -5.0)                   # above the CT air / MRI floor


def reference(v):
    ixi = v[v.dataset == "ixi_t1"].sample(20, random_state=0)
    heads, brains = [], []
    for p in ixi.image_path:
        img, b = load(p)
        heads.append(signal(img, False))
        brains.append(b)
    head, brain = np.mean(heads, 0), np.mean(brains, 0)
    zc, yc, _ = (np.array(np.nonzero(brain > 0.5)).mean(1)).astype(int)
    z, y, _ = np.indices(brain.shape)
    face = (head > 0.9) & (brain < 0.01) & (y > yc + 45) & (z < zc)
    print(f"[ref] face region {int(face.sum())} voxels; brain centroid z {zc} y {yc}", flush=True)
    return brain > 0.5, face


def measure(img, b, ct, ref_brain, face):
    out = {"face_frac": float(signal(img, ct)[face].mean())}
    if b.any():
        c, cr = np.array(np.nonzero(b)).mean(1), np.array(np.nonzero(ref_brain)).mean(1)
        out |= {"dz_mm": c[0] - cr[0], "dy_mm": c[1] - cr[1], "dx_mm": c[2] - cr[2],
                "dice_ixi": 2 * (b & ref_brain).sum() / (b.sum() + ref_brain.sum())}
        for ax, name in ((0, "z"), (1, "y")):
            ext = lambda m: np.ptp(np.nonzero(m.any(axis=tuple(i for i in range(3) if i != ax)))[0])  # noqa: E731
            out[f"ext_{name}"] = ext(b) / ext(ref_brain)
    return out


def panel(ax, img2d, ref2d, b2d, ct):
    lo, hi = (-20, 100) if ct else np.percentile(img2d[img2d > -5.9], [1, 99.5]) if (img2d > -5.9).any() else (0, 1)
    ax.imshow(img2d, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
    ax.contour(ref2d, [0.5], colors="#00e5a0", linewidths=0.5)
    if b2d.any():
        ax.contour(b2d, [0.5], colors="#ff3b30", linewidths=0.4)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=15, help="scans measured per dataset")
    a = ap.parse_args()
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False,
                    usecols=["volume_id", "dataset", "dataset_family", "modality", "image_path", "case_id",
                             "stripped_input", "reg_method", "reg_scale", "reg_rot_deg", "brain_dice_tpl"])
    v = v.drop_duplicates("volume_id")
    ref_brain, face = reference(v)
    zc = int(np.nonzero(ref_brain)[0].mean())
    xc = ref_brain.shape[2] // 2
    rows, pages = [], []
    for ds, g in sorted(v.groupby("dataset"), key=lambda kv: kv[0]):
        rng = np.random.default_rng(0)
        pick = g.sample(min(a.per, len(g)), random_state=0)
        worst = g.nsmallest(2, "brain_dice_tpl")
        show = list(pick.index[:2]) + [i for i in worst.index if i not in pick.index[:2]][:2]
        todo = pick.index.union(pd.Index(show))
        shown = []
        for i in todo:
            r = v.loc[i]
            ct = r.modality == "ct"
            try:
                img, b = load(r.image_path)
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {ds} {r.case_id}: {type(e).__name__}", flush=True)
                continue
            m = measure(img, b, ct, ref_brain, face)
            if i in pick.index:
                rows.append({"dataset": ds, "family": r.dataset_family, "modality": r.modality,
                             "stripped": bool(r.stripped_input == 1), "reg_method": r.reg_method,
                             "reg_scale": r.reg_scale, "reg_rot_deg": r.reg_rot_deg,
                             "brain_dice_tpl": r.brain_dice_tpl, "case_id": r.case_id, **m})
            if i in show:
                shown.append((r.case_id, img[:, :, xc], b[:, :, xc], img[zc], b[zc], ct, m, r.brain_dice_tpl,
                              i not in pick.index[:2]))
        pages.append((ds, shown))
        del rng
        print(f"[ds] {ds} {len(g)} volumes, measured {len(pick)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT + ".csv", index=False)
    for p in range(0, len(pages), ROWS_PER_PAGE):
        chunk = pages[p:p + ROWS_PER_PAGE]
        fig, axes = plt.subplots(len(chunk), 8, figsize=(13, 1.75 * len(chunk)), facecolor="black", squeeze=False)
        for i, (ds, shown) in enumerate(chunk):
            for j in range(4):
                axes[i, 2 * j].axis("off")
                axes[i, 2 * j + 1].axis("off")
                if j >= len(shown):
                    continue
                cid, sag, bsag, axl, baxl, ct, m, dice, worst = shown[j]
                panel(axes[i, 2 * j], sag, ref_brain[:, :, xc], bsag, ct)
                panel(axes[i, 2 * j + 1], axl, ref_brain[zc], baxl, ct)
                tag = "low Dice" if worst else "random"
                axes[i, 2 * j].set_title(f"{str(cid)[-12:]} {tag} D={dice:.2f} face={m['face_frac']:.2f}",
                                         color="orange" if worst else "white", fontsize=5.5)
            axes[i, 0].text(-0.06, 0.5, ds, transform=axes[i, 0].transAxes, color="white", fontsize=7, ha="right",
                            va="center")
        fig.suptitle("Registration QC: sagittal (midline) + axial; green = mean IXI brain, red = scan brain mask; "
                     "2 random + 2 lowest template-Dice scans", color="white", fontsize=8)
        fig.subplots_adjust(left=0.13, right=0.995, top=0.975, bottom=0.005, wspace=0.03, hspace=0.3)
        name = f"{OUT}_{p // ROWS_PER_PAGE + 1}.png"
        fig.savefig(name, dpi=85, facecolor="black")
        plt.close(fig)
        print("wrote", name, flush=True)

    mri = d[(d.modality == "mri") & ~d.stripped].copy()
    mri["defaced"] = mri.face_frac < 0.3
    mri["intact"] = mri.face_frac > 0.7
    print("\nskull-on MRI: share defaced (face_frac < 0.3) per dataset")
    print(mri.groupby("dataset").face_frac.agg(["median", lambda s: (s < 0.3).mean()]).round(2).to_string())
    cols = ["brain_dice_tpl", "dice_ixi", "dz_mm", "dy_mm", "dx_mm", "ext_z", "ext_y", "reg_scale", "reg_rot_deg"]
    grp = np.where(mri.defaced, "defaced", np.where(mri.intact, "intact", "partial"))
    print("\nskull-on MRI, defaced vs intact (median; IQR of the offsets)")
    print(mri.groupby(grp)[cols].median().round(3).to_string())
    print(mri.groupby(grp)[["dz_mm", "dy_mm"]].quantile([0.1, 0.9]).round(2).to_string())


if __name__ == "__main__":
    main()
