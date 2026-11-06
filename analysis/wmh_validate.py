"""WMH-SynthSeg against expert masks (2026-11-01): our WMH labels are WMH-SynthSeg volumes (brain.prep.wmh_run), never
checked against ground truth. The WMH Segmentation Challenge FLAIRs in the store (dataset wmh_flair, 170 scans, three
sites) carry manual masks (label 1 WMH, label 2 other pathology, left out here as in the challenge). Per scan: manual
ml, tool soft / hard ml, Dice of the tool's argmax mask, and agreement summaries per site.

    python runs/diagnostics/wmh_validate.py   -> runs/diagnostics/wmh_validate.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, "/home/roku/brain-model")
from brain.prep import store  # noqa: E402
from brain.prep.wmh_run import CROP, LABELS, VFLIP, WMH, load_model  # noqa: E402

S = Path("/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr")
OUT = Path(__file__).with_suffix(".csv")


def icc(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1), absolute agreement."""
    x = np.c_[a, b]
    n, k = x.shape
    m = x.mean()
    msr = k * ((x.mean(1) - m) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - m) ** 2).sum() / (k - 1)
    mse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + m) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


@torch.no_grad()
def main() -> None:
    v = pd.read_csv(S / "volumes.csv", low_memory=False)
    v = v[(v["dataset"] == "wmh_flair") & v["mask_lesion"].notna()].drop_duplicates("volume_id")
    model = load_model()
    rows = []
    for r in v.itertuples():
        img = store.read_image(S / str(r.image_path)).astype(np.float32)
        man = store.squeeze_channel(store.read_image(S / str(r.mask_lesion), "mask"))
        x = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 1, 0))[CROP])).cuda()
        x = x - x.min()
        x = x / x.max().clamp_min(1e-6)
        p = model(x[None, None])[0, : len(LABELS)].softmax(0)
        p += torch.flip(model(torch.flip(x, [0])[None, None]), [2])[0, VFLIP].softmax(0)
        p *= 0.5
        hard = (p.argmax(0) == WMH).cpu().numpy()
        m = np.transpose(man, (2, 1, 0))[CROP]                     # manual mask on the tool's (X, Y, Z) crop
        valid = m != 2
        gt = (m == 1) & valid
        pr = hard & valid
        dice = 2 * (gt & pr).sum() / max(gt.sum() + pr.sum(), 1)
        site = str(r.case_id).split("_")[1] if "_" in str(r.case_id) else "?"
        rows.append({"volume_id": r.volume_id, "site": site, "manual_ml": float((man == 1).sum() / 1000),
                     "tool_soft_ml": float(p[WMH].sum() / 1000), "tool_hard_ml": float(hard.sum() / 1000),
                     "dice": float(dice)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    res = []
    for name, g in [("all", d), *d.groupby("site")]:
        res.append({"site": name, "n": len(g), "manual_med_ml": g.manual_ml.median(), "tool_med_ml": g.tool_hard_ml.median(),
                    "spearman": stats.spearmanr(g.manual_ml, g.tool_hard_ml).correlation,
                    "icc_log": icc(np.log1p(g.manual_ml.to_numpy()), np.log1p(g.tool_hard_ml.to_numpy())),
                    "ratio_med": float(np.median((g.tool_hard_ml + 0.1) / (g.manual_ml + 0.1))),
                    "dice_med": g.dice.median(),
                    "grade_agree": float(np.mean(np.digitize(g.manual_ml, [5, 15]) == np.digitize(g.tool_hard_ml, [5, 15])))})
    print(pd.DataFrame(res).round(3).to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
