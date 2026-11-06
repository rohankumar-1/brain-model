"""Visual QC of the stored segmentation masks across datasets (2026-10-29).

lesions: every dataset x mask type with lesion / tumour / hemorrhage / vessel masks (index columns mask_*, plus
UPENN-GBM from its manifest until it is indexed); four random cases each, axial and coronal through the slice with the
most mask, labels filled (1 red, 2 green, 3 blue, 4 magenta, 5 orange) with their outline. Split into MRI and CT pages.
synthseg: SynthSeg anatomy (FreeSurfer colours; stored masks index synthseg.LABELS) on a spread of cohorts (SYNTHSEG_FAMILIES:
CT, adult / child / infant / tumour / low-field MRI), two QC-passed scans and one QC-failed scan per cohort where one exists.

    python runs/diagnostics/mask_gallery.py
Writes runs/diagnostics/mask_gallery_{mri,ct,synthseg}.png."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, "/home/roku/brain-model")
from brain.downstream.tasks import DEFAULT_VOLUMES  # noqa: E402
from brain.prep import store  # noqa: E402
from brain.prep.index import folder_of  # noqa: E402

ROOT = DEFAULT_VOLUMES.parent
OUT = "/home/roku/brain-model/runs/diagnostics/mask_gallery_{}.png"
MASKS = ("mask_seg", "mask_ich", "mask_lesion", "mask_lvo", "mask_cow")
N_CASES = 4
SYNTHSEG_FAMILIES = ("rsna_ich", "ixi", "oasis1", "abide1", "fomo_hbn", "fomo_infant", "fomo_m4raw", "fomo_nigerian",
                     "fomo_acpi", "fomo_yale_highres", "fomo_yale_mets", "brats_gli", "brats_men", "brats_ped")
LABEL_RGB = np.array([[0, 0, 0], [255, 59, 48], [52, 199, 89], [10, 132, 255], [255, 45, 210], [255, 159, 10],
                      [255, 214, 10], [100, 210, 255]]) / 255
FS_RGB = {2: (245, 245, 245), 3: (205, 62, 78), 4: (120, 18, 134), 5: (196, 58, 250), 7: (220, 248, 164),
          8: (230, 148, 34), 10: (0, 118, 14), 11: (122, 186, 220), 12: (236, 13, 176), 13: (12, 48, 255),
          14: (204, 182, 142), 15: (42, 204, 164), 16: (119, 159, 176), 17: (220, 216, 20), 18: (103, 255, 255),
          24: (60, 60, 60), 26: (255, 165, 0), 28: (165, 42, 42)}
FS_RGB |= {r: FS_RGB[l] for l, r in ((2, 41), (3, 42), (4, 43), (5, 44), (7, 46), (8, 47), (10, 49), (11, 50),
                                     (12, 51), (13, 52), (17, 53), (18, 54), (26, 58), (28, 60))}


def window(img: np.ndarray, modality: str) -> tuple[float, float]:
    if modality == "ct":
        return -20.0, 100.0
    lo, hi = np.percentile(img[img > -5.9], [1, 99.5])
    return float(lo), float(hi)


def overlay(ax, img2d, lab2d, lim, rgb):
    ax.imshow(img2d, cmap="gray", vmin=lim[0], vmax=lim[1], origin="lower", interpolation="nearest")
    lab = np.clip(lab2d.astype(int), 0, len(rgb) - 1)
    rgba = np.zeros(lab.shape + (4,))
    rgba[..., :3] = rgb[lab]
    rgba[..., 3] = (lab > 0) * 0.45
    ax.imshow(rgba, origin="lower", interpolation="nearest")
    if lab.any():
        ax.contour(lab > 0, [0.5], colors="white", linewidths=0.4)
    ax.set_xticks([])
    ax.set_yticks([])


def lesion_rows() -> list[dict]:
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False, usecols=["volume_id", "dataset", "dataset_family", "modality",
                                                                  "case_id", "image_path", *MASKS])
    rows = []
    for col in MASKS:
        has = v[v[col].notna() & (v[col].astype(str) != "")]
        for ds, g in has.groupby("dataset"):
            g = g.sample(min(N_CASES, len(g)), random_state=0)
            rows.append({"name": f"{ds} / {col.removeprefix('mask_')}", "modality": g.modality.iloc[0],
                         "cases": [(r.case_id, ROOT / r.image_path, ROOT / getattr(r, col)) for r in g.itertuples()]})
    m = pd.DataFrame([json.loads(line) for line in open(ROOT / "manifest_v6_upenn_gbm_0of1.jsonl")])
    m = m[(m.dataset == "upenn_gbm_t1n") & (m.status == "ok")].sample(N_CASES, random_state=0)
    folder = folder_of("upenn_gbm_t1n")
    rows.append({"name": "upenn_gbm_t1n / seg (not indexed yet)", "modality": "mri",
                 "cases": [(c, store.path_for(ROOT, folder, c), store.path_for(ROOT, folder, c, "seg")) for c in m.case_id]})
    return sorted(rows, key=lambda r: r["name"])


def lesion_page(rows: list[dict], tag: str) -> None:
    fig, axes = plt.subplots(len(rows), 2 * N_CASES, figsize=(2 * N_CASES * 1.6, len(rows) * 1.9), facecolor="black",
                             squeeze=False)
    for i, row in enumerate(rows):
        labels = set()
        for j in range(N_CASES):
            for k in (0, 1):
                axes[i, 2 * j + k].axis("off")
            if j >= len(row["cases"]):
                continue
            cid, ip, mp = row["cases"][j]
            try:
                img = np.asarray(store.read_image(ip), np.float32)
                lab = np.asarray(store.read_image(mp, "mask"))
            except Exception as e:  # noqa: BLE001
                axes[i, 2 * j].set_title(f"read failed: {type(e).__name__}", color="orange", fontsize=6)
                continue
            labels |= set(np.unique(lab).tolist()) - {0}
            lim = window(img, row["modality"])
            if lab.any():
                z = int(lab.reshape(lab.shape[0], -1).sum(1).argmax())
                y = int(np.round(np.nonzero(lab)[1].mean()))
            else:
                z, y = lab.shape[0] // 2, lab.shape[1] // 2
            overlay(axes[i, 2 * j], img[z], lab[z], lim, LABEL_RGB)
            overlay(axes[i, 2 * j + 1], img[:, y], lab[:, y], lim, LABEL_RGB)
            vox = int((lab > 0).sum())
            axes[i, 2 * j].set_title(f"{str(cid)[-14:]}  {vox / 1000:.1f} ml", color="white", fontsize=6)
            if not lab.any():
                axes[i, 2 * j].set_title(f"{str(cid)[-14:]}  EMPTY", color="orange", fontsize=6)
        axes[i, 0].text(-0.08, 0.5, f"{row['name']}\nlabels {sorted(labels)}", transform=axes[i, 0].transAxes,
                        color="white", fontsize=7, ha="right", va="center")
    fig.suptitle(f"Stored masks, {tag}: 4 random cases per row, axial + coronal at the largest mask slice "
                 "(1 red, 2 green, 3 blue, 4 magenta, 5 orange)", color="white", fontsize=9)
    fig.subplots_adjust(left=0.2, right=0.995, top=0.97, bottom=0.005, wspace=0.03, hspace=0.25)
    fig.savefig(OUT.format(tag), dpi=90, facecolor="black")
    plt.close(fig)
    print("wrote", OUT.format(tag), flush=True)


def synthseg_page() -> None:
    from brain.multitask.labels import synthseg_volumes
    from brain.prep.synthseg import LABELS

    ss = synthseg_volumes()[["qc_ok"]]
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False, usecols=["volume_id", "dataset", "dataset_family", "modality",
                                                                  "case_id", "image_path"]).drop_duplicates("volume_id")
    v = v.merge(ss, left_on="volume_id", right_index=True)
    fams = (v.groupby(["modality", "dataset_family"]).size().rename("n").reset_index()
            .set_index("dataset_family").loc[[f for f in SYNTHSEG_FAMILIES if f in set(v.dataset_family)]].reset_index())
    rgb = np.zeros((61, 3))
    for k, c in FS_RGB.items():
        rgb[k] = np.array(c) / 255
    cols = 3
    fig, axes = plt.subplots(len(fams), 2 * cols, figsize=(2 * cols * 1.6, len(fams) * 1.9), facecolor="black",
                             squeeze=False)
    for i, f in enumerate(fams.itertuples()):
        g = v[v.dataset_family == f.dataset_family]
        ok, bad = g[g.qc_ok], g[~g.qc_ok]
        pick = [*ok.sample(min(2, len(ok)), random_state=0).itertuples()]
        pick += [*bad.sample(min(1, len(bad)), random_state=0).itertuples()] or [*ok.sample(min(3, len(ok)), random_state=1).itertuples()][2:]
        for j in range(cols):
            axes[i, 2 * j].axis("off")
            axes[i, 2 * j + 1].axis("off")
            if j >= len(pick):
                continue
            r = pick[j]
            img = np.asarray(store.read_image(ROOT / r.image_path), np.float32)
            lab = LABELS[np.asarray(store.read_image(store.path_for(ROOT, folder_of(r.dataset), r.case_id, "synthseg"),
                                                     "mask")).astype(int)]
            lim = window(img, r.modality)
            zs = np.nonzero((lab == 17) | (lab == 53))[0]                  # through the hippocampi
            z = int(np.median(zs)) if len(zs) else lab.shape[0] // 2
            ys = np.nonzero(lab > 0)[1]
            y = int(np.median(ys)) if len(ys) else lab.shape[1] // 2
            overlay(axes[i, 2 * j], img[z], lab[z], lim, rgb)
            overlay(axes[i, 2 * j + 1], img[:, y], lab[:, y], lim, rgb)
            axes[i, 2 * j].set_title(f"{str(r.case_id)[-14:]}  QC {'pass' if r.qc_ok else 'FAIL'}",
                                     color="white" if r.qc_ok else "orange", fontsize=6)
        axes[i, 0].text(-0.08, 0.5, f"{f.dataset_family} ({f.modality})\n{len(g)} labelled, QC fail {(~g.qc_ok).mean():.0%}",
                        transform=axes[i, 0].transAxes, color="white", fontsize=7, ha="right", va="center")
    fig.suptitle("SynthSeg anatomy (FreeSurfer colours): 2 QC-passed + 1 QC-failed scan per cohort, axial through the "
                 "hippocampi + coronal", color="white", fontsize=9)
    fig.subplots_adjust(left=0.2, right=0.995, top=0.97, bottom=0.005, wspace=0.03, hspace=0.25)
    fig.savefig(OUT.format("synthseg"), dpi=90, facecolor="black")
    plt.close(fig)
    print("wrote", OUT.format("synthseg"), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["synthseg"]:
        synthseg_page()
        sys.exit()
    rows = lesion_rows()
    lesion_page([r for r in rows if r["modality"] == "mri"], "mri")
    lesion_page([r for r in rows if r["modality"] == "ct"], "ct")
    synthseg_page()
