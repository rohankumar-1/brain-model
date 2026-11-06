"""SynthSeg-robust port vs the original Keras model on store volumes of several cohorts / modalities.

Step 1 (TF venv): ``brain.prep.synthseg --save-ref`` writes Keras posteriors. Step 2 (torch, GPU): the port's
posteriors on the same inputs; reports max |diff|, voxel label agreement and per-structure Dice after post-processing.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/roku/brain-model")
from brain.downstream.tasks import DEFAULT_VOLUMES
from brain.prep import store
from brain.prep.synthseg import LABELS, NAMES, Segmenter, postprocess

ROOT = DEFAULT_VOLUMES.parent
OUT = Path("/home/roku/brain-model/runs/diagnostics/synthseg_ref.npz")
v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
pick = []
for fam, con in [("fomo_hbn", "t1w"), ("ixi", "t2"), ("fomo_nki", "flair"), ("rsna_ich", None), ("brats_gli", "t1c"),
                 ("oasis1", None)]:
    s = v[v.dataset_family == fam]
    if con:
        s = s[s.contrast.astype(str).str.lower() == con]
    r = s.sample(1, random_state=0).iloc[0]
    pick.append((str(ROOT / r.image_path), str(r.modality), f"{fam}/{con or r.modality}"))
from brain.prep.synthseg import normalise

INP = OUT.with_name("synthseg_inputs.npz")
if not OUT.exists():
    arrs = {}
    for i, (path, mod, _) in enumerate(pick):
        img = store.read(Path(path), "image")
        img = np.asarray(img[:, 0] if img.ndim == 4 else img, np.float32)
        arrs[str(i)] = normalise(np.transpose(img, (2, 1, 0)), mod)
    np.savez(INP, **arrs)
    t0 = time.time()
    subprocess.run(["/home/roku/orcd/scratch/brain-model/tools/tfref/bin/python",
                    "/home/roku/brain-model/runs/diagnostics/synthseg_keras_ref.py", str(INP), str(OUT)],
                   check=True, env={"TF_USE_LEGACY_KERAS": "1", "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "12",
                                    "TF_CPP_MIN_LOG_LEVEL": "2", "CUDA_VISIBLE_DEVICES": ""})
    print(f"[keras] {len(pick)} volumes in {time.time() - t0:.0f}s (CPU)", flush=True)
ref = np.load(OUT)
seg = Segmenter("cuda")
for i, (path, mod, tag) in enumerate(pick):
    img = store.read(Path(path), "image")
    img = np.asarray(img[:, 0] if img.ndim == 4 else img, np.float32)
    torch.cuda.synchronize()
    t0 = time.time()
    p = seg.posteriors([np.transpose(img, (2, 1, 0))], [mod])[0]
    torch.cuda.synchronize()
    dt = time.time() - t0
    k = ref[str(i)]
    s_t, vol_t = postprocess(p)
    s_k, vol_k = postprocess(k)
    agree = float((s_t == s_k).mean())
    fg = (s_k > 0) | (s_t > 0)
    dice = {}
    for lab in LABELS[1:]:
        a, b = s_t == lab, s_k == lab
        den = a.sum() + b.sum()
        if den:
            dice[NAMES[lab]] = 2 * (a & b).sum() / den
    worst = sorted(dice.items(), key=lambda kv: kv[1])[:3]
    print(f"{tag:22s} max|dpost|={np.abs(p - k).max():.2e} label agree all={agree:.5f} fg={float((s_t == s_k)[fg].mean()):.5f} "
          f"min Dice={min(dice.values()):.4f} ({', '.join(f'{n} {d:.3f}' for n, d in worst)}) "
          f"brain ml={vol_t.sum():.0f} vs {vol_k.sum():.0f} | gpu {dt:.2f}s", flush=True)
sys.exit(0)  # speed measured in the first run
# throughput, batched bf16 vs fp32 on one volume
img = store.read(Path(pick[0][0]), "image")
img = np.transpose(np.asarray(img[:, 0] if img.ndim == 4 else img, np.float32), (2, 1, 0))
for amp in (False, True):
    s2 = Segmenter("cuda", amp=amp)
    for b in (1, 2, 4):
        try:
            s2.posteriors([img] * b, ["mri"] * b)
            torch.cuda.synchronize()
            t0 = time.time()
            s2.posteriors([img] * b, ["mri"] * b)
            torch.cuda.synchronize()
            print(f"[speed] amp={amp} batch={b}: {(time.time() - t0) / b:.2f}s/vol, peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[speed] amp={amp} batch={b}: OOM")
            torch.cuda.empty_cache()
    pb = s2.posteriors([img], ["mri"])[0]
    print(f"[amp={amp}] label agreement with fp32 Keras ref: {(postprocess(pb)[0] == postprocess(ref['0'])[0]).mean():.5f}")
