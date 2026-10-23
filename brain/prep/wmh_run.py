"""White-matter hyperintensity burden from WMH-SynthSeg (FreeSurfer ``mri_WMHsynthseg``, research licence) over the
non-tumour FLAIR / T2 scans of the pretraining pool, one GPU per shard. Volumes only, no masks.

    python -m brain.prep.wmh_run --shard 0 --n-shards 2       # skips volumes already in labels/wmh/volumes_*.csv

Inference reproduces ``mri_WMHsynthseg`` (validated in runs/diagnostics/wmh_pilot.py): max-normalised input, original
and left-right-flipped passes averaged, softmax over the segmentation channels. Store volumes are already on a 1 mm
RAS grid, so its resampling / centring reduce to a fixed crop. Per volume: ``wmh_ml`` (soft, summed probability) and
``wmh_ml_hard`` (argmax voxels), template-space ml.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from brain.prep import store
from brain.prep.synthseg_run import ROOT

OUT = ROOT / "labels" / "wmh"
TOOLS = ROOT.parents[2] / "tools" / "wmh_synthseg"
WEIGHTS = TOOLS / "WMH-SynthSeg_v10_231110.pth"
CONTRASTS = ("flair", "t2f", "t2w", "t2")
TUMOUR = "brats|mets"
LABELS = [0, 14, 15, 16, 24, 77, 85, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 43, 46,
          47, 49, 50, 51, 52, 53, 54, 58, 60]
N_NEUTRAL, WMH = 7, LABELS.index(77)
_NLAT = (len(LABELS) - N_NEUTRAL) // 2
VFLIP = np.r_[np.arange(N_NEUTRAL), np.arange(N_NEUTRAL + _NLAT, len(LABELS)), np.arange(N_NEUTRAL, N_NEUTRAL + _NLAT)]
CROP = (slice(32, 224), slice(16, 240), slice(0, 160))          # (X, Y, Z) of the 256 x 256 x 160 grid


def load_model(device: str = "cuda") -> torch.nn.Module:
    sys.path.insert(0, str(TOOLS / "WMHSynthSeg"))
    from unet3d.model import UNet3D

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = UNet3D(1, len(LABELS) + 6, final_sigmoid=False, f_maps=64, layer_order="gcl", num_groups=8,
                   num_levels=5, is_segmentation=False, is3d=True).to(device).eval()
    # weights-only load: the checkpoint also holds training bookkeeping with numpy scalars / dtypes
    safe = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, np.dtypes.Float64DType,
            np.dtypes.Float32DType, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(safe):
        state = torch.load(WEIGHTS, map_location=device, weights_only=True)["model_state_dict"]
    model.load_state_dict(state)
    return model


@torch.no_grad()
def wmh_volume(model: torch.nn.Module, img_zyx: np.ndarray, device: str = "cuda") -> tuple[float, float]:
    """Store volume (Z, Y, X) -> (soft, hard) WMH volume in ml."""
    x = torch.from_numpy(np.ascontiguousarray(np.transpose(img_zyx, (2, 1, 0))[CROP])).to(device)
    x = x - x.min()
    x = x / x.max().clamp_min(1e-6)
    # the 6 outputs after the segmentation channels are image / bias reconstructions
    p = model(x[None, None])[0, : len(LABELS)].softmax(0)
    p += torch.flip(model(torch.flip(x, [0])[None, None]), [2])[0, VFLIP].softmax(0)
    p *= 0.5
    return float(p[WMH].sum() / 1000.0), float((p.argmax(0) == WMH).sum() / 1000.0)


def selection() -> pd.DataFrame:
    """Train / val pretraining-pool FLAIR and T2 volumes outside the tumour cohorts (WMH-SynthSeg is unreliable on
    T1, and calls peritumoral oedema WMH)."""
    from brain.downstream.tasks import DEFAULT_VOLUMES

    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
    v = v[v["pretrain"].astype(bool) & ~v["held_out"].astype(bool) & v["split"].isin(["train", "val"])]
    v = v[v["contrast"].astype(str).str.lower().isin(CONTRASTS) & ~v["dataset_family"].str.contains(TUMOUR)]
    return v[["volume_id", "image_path", "contrast", "dataset_family"]].reset_index(drop=True)


def run_shard(shard: int, n_shards: int) -> None:
    sel = selection()
    done_ids = set()
    for f in OUT.glob("volumes_*.csv"):
        done_ids |= set(pd.read_csv(f, usecols=["volume_id"])["volume_id"].astype(str))
    rows = [r for i, r in enumerate(sel.itertuples()) if i % n_shards == shard and str(r.volume_id) not in done_ids]
    print(f"[wmh] shard {shard}/{n_shards}: {len(rows)} to do", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / f"volumes_{shard:03d}.csv"
    model = load_model()

    def load(r):
        return store.read_image(ROOT / str(r.image_path)).astype(np.float32)

    t0, recs = time.time(), []
    with ThreadPoolExecutor(4) as io:
        loads = [io.submit(load, r) for r in rows[:4]]
        for i, r in enumerate(rows):
            img = loads[i].result()
            loads[i] = None                           # drop the future: it would keep the volume alive
            if i + 4 < len(rows):
                loads.append(io.submit(load, rows[i + 4]))
            recs.append([r.volume_id, r.contrast, *np.round(wmh_volume(model, img), 4)])
            if len(recs) == 50 or i == len(rows) - 1:
                pd.DataFrame(recs, columns=["volume_id", "contrast", "wmh_ml", "wmh_ml_hard"]).to_csv(
                    csv, mode="a", header=not csv.exists(), index=False)
                recs = []
                el = time.time() - t0
                print(f"[wmh] {i + 1}/{len(rows)} {el / (i + 1):.2f}s/vol eta {(len(rows) - i - 1) * el / (i + 1) / 3600:.1f}h",
                      flush=True)
    print(f"[wmh] shard {shard} done in {(time.time() - t0) / 3600:.2f}h", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--count", action="store_true", help="print the selection by family / contrast and exit")
    a = ap.parse_args()
    if a.count:
        sel = selection()
        print(sel.groupby(["dataset_family", "contrast"]).size().to_string(), f"\ntotal {len(sel)}")
        return
    run_shard(a.shard, a.n_shards)


if __name__ == "__main__":
    main()
