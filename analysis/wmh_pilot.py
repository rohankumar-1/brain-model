"""WMH-SynthSeg pilot on store volumes: speed, Dice vs manual WMH masks (held-out WMH challenge), WMH volume vs age and
dementia. Reproduces FreeSurfer's mri_WMHsynthseg inference (max-normalised input, original + left-right-flipped pass
averaged, argmax); store volumes are on a 1 mm RAS grid already, so resampling / centring reduce to a fixed crop."""
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/roku/brain-model")
sys.path.insert(0, "/home/roku/orcd/scratch/brain-model/tools/wmh_synthseg/WMHSynthSeg")
from unet3d.model import UNet3D  # noqa: E402

from brain.downstream.tasks import DEFAULT_VOLUMES, load_task  # noqa: E402
from brain.multitask import labels as L  # noqa: E402
from brain.multitask.registry import load_registry  # noqa: E402
from brain.prep import store  # noqa: E402

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
LABELS = [0, 14, 15, 16, 24, 77, 85, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 43, 46,
          47, 49, 50, 51, 52, 53, 54, 58, 60]
N_NEUTRAL, WMH = 7, LABELS.index(77)
dev = torch.device("cuda")
model = UNet3D(1, len(LABELS) + 6, final_sigmoid=False, f_maps=64, layer_order="gcl", num_groups=8, num_levels=5,
               is_segmentation=False, is3d=True).to(dev).eval()
# safe (weights-only) load: the checkpoint also holds training bookkeeping with numpy scalars / dtypes
SAFE = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, np.dtypes.Float64DType,
        np.dtypes.Float32DType, np.dtypes.Int64DType]
with torch.serialization.safe_globals(SAFE):
    model.load_state_dict(torch.load("/home/roku/orcd/scratch/brain-model/tools/wmh_synthseg/WMH-SynthSeg_v10_231110.pth",
                                     map_location=dev, weights_only=True)["model_state_dict"])
nlat = (len(LABELS) - N_NEUTRAL) // 2
VFLIP = np.r_[np.arange(N_NEUTRAL), np.arange(N_NEUTRAL + nlat, len(LABELS)), np.arange(N_NEUTRAL, N_NEUTRAL + nlat)]
CROP = (slice(32, 224), slice(16, 240), slice(0, 160))          # (X, Y, Z) of the 256 x 256 x 160 grid


@torch.no_grad()
def wmh(img_zyx: np.ndarray) -> tuple[np.ndarray, float]:
    """Store volume (Z, Y, X) -> WMH probability on the grid (Z, Y, X) and soft WMH volume (ml)."""
    x = torch.from_numpy(np.transpose(img_zyx.astype(np.float32), (2, 1, 0))[CROP]).to(dev)
    x = x - x.min()
    x = x / x.max().clamp_min(1e-6)
    # softmax over the segmentation channels only (the other 6 outputs are image / bias reconstructions)
    p1 = model(x[None, None])[0, : len(LABELS)].softmax(0)
    p2 = torch.flip(model(torch.flip(x, [0])[None, None]), [2])[0, VFLIP].softmax(0)
    p = 0.5 * (p1 + p2)
    wp = p[WMH]
    full = torch.zeros(256, 256, 160, device=dev)
    full[CROP] = (p.argmax(0) == WMH).float()                    # hard WMH mask (argmax), on the grid
    return np.transpose(full.cpu().numpy(), (2, 1, 0)), float(wp.sum() / 1000.0)


v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
# 1) Dice vs manual masks, held-out WMH challenge
b = load_task("wmh", v)
rows, times = [], []
for vid in b.volume_ids[:40]:
    img = store.read_image(L.STORE / b.image_paths[vid]).astype(np.float32)
    man = store.read_image(L.STORE / b.mask_paths[vid], "mask") > 0
    torch.cuda.synchronize(); t0 = time.time()
    pred, vol = wmh(img)
    torch.cuda.synchronize(); times.append(time.time() - t0)
    pr = pred > 0.5
    d = 2 * (pr & man).sum() / max(pr.sum() + man.sum(), 1)
    rows.append((vid, d, man.sum() / 1000, pr.sum() / 1000))
t = pd.DataFrame(rows, columns=["vid", "dice", "manual_ml", "pred_ml"])
print(f"[speed] {np.mean(times[1:]):.2f} s/scan (2 passes), peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
print(f"[wmh challenge] n={len(t)} Dice mean {t.dice.mean():.3f} median {t.dice.median():.3f} | volume r="
      f"{np.corrcoef(t.manual_ml, t.pred_ml)[0, 1]:.2f} | manual {t.manual_ml.median():.1f} ml pred {t.pred_ml.median():.1f} ml")
# 2) WMH volume vs age (NKI lifespan, T1w / FLAIR) and dementia (BrainLat)
reg = load_registry()
age = L.src_fomo_participants(v, reg["age"], reg["age"].sources[0]).set_index("volume_id").value
dx = L.src_fomo_participants(v, reg["neuro_diagnosis"], reg["neuro_diagnosis"].sources[0]).set_index("volume_id").value
vi = v.set_index("volume_id")
out = []
for fam, n in (("fomo_nki", 80), ("fomo_brainlat", 80)):
    ids = [i for i in vi.index[(vi.dataset_family == fam)] if i in age.index][:400]
    ids = pd.Series(ids).sample(min(n, len(ids)), random_state=0)
    for vid in ids:
        img = store.read_image(L.STORE / vi.loc[vid, "image_path"]).astype(np.float32)
        _, vol = wmh(img)
        out.append((vid, fam, vi.loc[vid, "contrast"], age.get(vid), dx.get(vid), vol))
o = pd.DataFrame(out, columns=["vid", "fam", "contrast", "age", "dx", "wmh_ml"])
for (fam, con), g in o.groupby(["fam", "contrast"]):
    if len(g) >= 8:
        print(f"[age] {fam} {con}: n={len(g)} Spearman(WMH, age)={g.wmh_ml.corr(g.age, method='spearman'):.2f} "
              f"median {g.wmh_ml.median():.2f} ml")
bl = o[o.fam == "fomo_brainlat"].dropna(subset=["dx"])
cls = dict(enumerate(reg["neuro_diagnosis"].classes))
print("[dementia] BrainLat median WMH ml by diagnosis:",
      bl.assign(dx=bl.dx.astype(int).map(cls)).groupby("dx").wmh_ml.agg(["median", "size"]).round(2).to_dict("index"))
