"""Every translation-fallback volume: brain-mask Dice with a reference mask as stored vs flipped in y (the old nibabel
fallback reader stored such images rotated 180 deg about the superior axis). Writes runs/diagnostics/mirrored.csv."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from brain.prep import store

S = "/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr"


def mask(path):
    return np.asarray(store.read(f"{S}/{path}", "brain"))[::2, ::2, ::2] > 0


def score(args):
    vid, path, ref = args
    try:
        m = mask(path)
    except Exception:                                    # noqa: BLE001
        return vid, np.nan, np.nan
    d = lambda a: 2 * (a & ref).sum() / max(a.sum() + ref.sum(), 1)  # noqa: E731
    return vid, d(m), d(m[:, ::-1, ::-1])


if __name__ == "__main__":
    v = pd.read_csv(f"{S}/volumes.csv", low_memory=False,
                    usecols=["volume_id", "dataset", "dataset_family", "reg_method", "image_path", "pretrain", "held_out"])
    ref = np.mean([mask(p) for p in v[(v.reg_method == "affine") & (v.dataset_family == "fomo_nki")]
                   .image_path.sample(8, random_state=0)], 0) > 0.5
    t = v[v.reg_method == "translation"]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(score, [(a, b, ref) for a, b in zip(t.volume_id, t.image_path)], chunksize=8))
    r = pd.DataFrame(res, columns=["volume_id", "dice", "dice_flipped"]).merge(t, on="volume_id")
    r["mirrored"] = r.dice_flipped > r.dice + 0.03
    r.to_csv("runs/diagnostics/mirrored.csv", index=False)
    print(r.groupby("dataset_family").agg(n=("mirrored", "size"), mirrored=("mirrored", "sum"))
          .sort_values("mirrored", ascending=False).head(20).to_string())
    print("total mirrored:", int(r.mirrored.sum()), "of", len(r), "; in pretrain:", int((r.mirrored & r.pretrain.astype(bool)).sum()),
          "; held_out:", int((r.mirrored & r.held_out.astype(bool)).sum()))
