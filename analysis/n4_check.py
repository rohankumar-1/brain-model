"""Is bias-field correction (N4) applied consistently? For a sample of stored MRI scans per dataset family, run N4 on
the stored (registered, robust-z) image inside its brain mask and measure the spread of the estimated log bias field:
an already corrected scan gives an almost flat field. Writes runs/diagnostics/n4_check.csv."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import SimpleITK as sitk

from brain.prep import store
from brain.prep.pipeline import needs_n4

S = "/home/roku/orcd/scratch/brain-model/data/processed/slices_zarr"


def bias_spread(path: str) -> float:
    try:
        img = np.asarray(store.read_image(f"{S}/{path}"), np.float32)[::2, ::2, ::2]
        m = np.asarray(store.read(f"{S}/{path}", "brain"))[::2, ::2, ::2] > 0
    except Exception:                                    # noqa: BLE001
        return float("nan")
    if m.sum() < 2000:
        return float("nan")
    x = sitk.GetImageFromArray(img - img.min() + 1.0)
    mk = sitk.GetImageFromArray(m.astype(np.uint8))
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations([50, 50, 50])
    small, small_m = sitk.Shrink(x, [2, 2, 2]), sitk.Shrink(mk, [2, 2, 2])
    n4.Execute(small, small_m)
    log_field = sitk.GetArrayFromImage(n4.GetLogBiasFieldAsImage(x))
    return float(np.std(log_field[m]))


if __name__ == "__main__":
    v = pd.read_csv(f"{S}/volumes.csv", low_memory=False,
                    usecols=["volume_id", "dataset", "dataset_family", "modality", "contrast", "image_path", "brain_ok"])
    v = v[v.modality.str.lower().eq("mri") & v.brain_ok.astype(str).str.lower().isin(["true", "1", "1.0"])]
    v = v[~v.dataset.str.contains("adc|dwi", regex=True)]                 # not bias-field images
    fams = v.dataset_family.value_counts()
    sample = pd.concat([g.sample(min(8, len(g)), random_state=0) for f, g in v.groupby("dataset_family")
                        if fams[f] >= 5])
    with ProcessPoolExecutor(14) as ex:
        sample["bias_spread"] = list(ex.map(bias_spread, sample.image_path, chunksize=2))
    sample["n4_in_prep"] = sample.dataset.map(needs_n4)
    sample.to_csv("runs/diagnostics/n4_check.csv", index=False)
    t = sample.groupby("dataset_family").agg(n4_in_prep=("n4_in_prep", "mean"), n=("bias_spread", "count"),
                                             median_spread=("bias_spread", "median")).sort_values("median_spread")
    print(t.round(3).to_string())
    print(sample.groupby("n4_in_prep").bias_spread.describe().round(3).to_string())
