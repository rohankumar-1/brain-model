"""SynthSeg 2.0 automated QC (Billot et al., PNAS 2023; ``mri_synthseg --qc``) in PyTorch, on stored SynthSeg masks.

The QC network reads only the final segmentation, so it scores the ``<case>.synthseg.zip`` masks ``brain.prep.synthseg``
already wrote, without re-running SynthSeg: argmax indices -> ``MakeShape(224)`` (crop the bounding box of every index
but 0 and 24 to 224^3 around its centre, zero-padding where needed) -> FreeSurfer labels -> 9 QC groups (one-hot) ->
a neurite ``conv_enc`` (4 levels, 24 features x2 per level, 5^3 convolutions, residual, batch norm) -> max pool -> two
5^3 convolutions -> spatial mean. One score in [0, 1] per group (background dropped): a predicted Dice of that
structure group against a manual segmentation; the paper flags < 0.65 as a likely failure.

Weights ``synthseg_qc_2.0.h5`` and the label tables come from FreeSurfer's model annex (SHA-256 fb92e37d...), in
``tools/synthseg``.

    python -m brain.prep.synthseg_qc --shard 0 --n-shards 4      -> labels/synthseg/qc_000.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.prep import store
from brain.prep.synthseg import BN_EPS, LABELS, TOOLS
from brain.prep.synthseg_run import ROOT

WEIGHTS = TOOLS / "synthseg_qc_2.0.h5"
SEG_TABLE = TOOLS / "synthseg_segmentation_labels_2.0.npy"
QC_TABLE = TOOLS / "synthseg_qc_labels_2.0.npy"
QC_NAMES = TOOLS / "synthseg_qc_names_2.0.npy"
OUT = ROOT / "labels" / "synthseg"
SHAPE = 224


def qc_groups() -> tuple[np.ndarray, list[str]]:
    """Per index of ``LABELS`` its QC group (0 = background), and the group names (index 0 = background)."""
    seg, qc, names = np.load(SEG_TABLE), np.load(QC_TABLE), np.load(QC_NAMES)
    lut = dict(zip(seg.tolist(), qc.tolist()))
    groups = np.array([lut[int(lab)] for lab in LABELS], dtype=np.int64)
    uniq, first = np.unique(qc, return_index=True)
    return groups, [str(names[i]) for i in first[np.argsort(uniq)]]


def make_shape(idx_xyz: np.ndarray, size: int = SHAPE) -> np.ndarray:
    """``MakeShape``: crop / zero-pad an index map to ``size``^3 around the bounding box of indices other than 0 and 24
    (an index, as in mri_synthseg, not the CSF label)."""
    shape = np.array(idx_xyz.shape)
    nz = np.argwhere((idx_xyz != 0) & (idx_xyz != 24))
    lo = nz.min(0) if len(nz) else np.zeros(3, int)
    hi = nz.max(0) + 1 if len(nz) else np.minimum(shape, size)
    ext = hi - lo
    lo = lo - np.ceil((size - ext) / 2).astype(int)
    hi = hi + np.floor((size - ext) / 2).astype(int)
    a, b = np.maximum(lo, 0), np.minimum(hi, shape)
    x = idx_xyz[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    pad = [(int(-min(l, 0)), int(max(h - s, 0))) for l, h, s in zip(lo, hi, shape)]
    return np.pad(x, pad) if any(p for pp in pad for p in pp) else x


class QCNet(nn.Module):
    def __init__(self, n_groups: int = 9, feats: int = 24, levels: int = 4, k: int = 5):
        super().__init__()
        f = [feats * 2 ** lv for lv in range(levels)]
        self.down, self.expand, self.bn = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = n_groups
        for lv in range(levels):
            self.down.append(nn.ModuleList([nn.Conv3d(c, f[lv], k, padding=k // 2),
                                            nn.Conv3d(f[lv], f[lv], k, padding=k // 2)]))
            self.expand.append(nn.Conv3d(c, f[lv], k, padding=k // 2))
            self.bn.append(nn.BatchNorm3d(f[lv], eps=BN_EPS))
            c = f[lv]
        self.final = nn.ModuleList([nn.Conv3d(c, n_groups, k, padding=k // 2),
                                    nn.Conv3d(n_groups, n_groups, k, padding=k // 2)])
        self.levels = levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 9, X, Y, Z) one-hot QC groups -> (B, 9) scores (index 0 = background)."""
        for lv in range(self.levels):
            first = x
            h = self.down[lv][1](F.relu(self.down[lv][0](x)))          # last conv of a residual level: no activation
            x = F.relu(F.relu(self.expand[lv](first)) + h)             # expand_down_merge carries the activation
            x = self.bn[lv](x)
            if lv < self.levels - 1:
                x = F.max_pool3d(x, 2, ceil_mode=True)
        x = F.max_pool3d(x, 2, ceil_mode=True)                         # qc_maxpool_3
        for conv in self.final:
            x = F.relu(conv(x))
        return x.mean(dim=(2, 3, 4))


def load_qc(path: Path = WEIGHTS, device: str = "cuda") -> QCNet:
    import h5py

    net = QCNet()
    with h5py.File(path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f

        def ds(layer, var):
            return torch.from_numpy(np.asarray(g[layer][layer][f"{var}:0"]))

        def conv(m, name):
            m.weight.data.copy_(ds(name, "kernel").permute(4, 3, 0, 1, 2))   # (kx,ky,kz,in,out)->(out,in,kx,ky,kz)
            m.bias.data.copy_(ds(name, "bias"))

        for lv in range(net.levels):
            for c in range(2):
                conv(net.down[lv][c], f"qc_conv_downarm_{lv}_{c}")
            conv(net.expand[lv], f"qc_expand_down_merge_{lv}")
            b = net.bn[lv]
            b.weight.data.copy_(ds(f"qc_bn_down_{lv}", "gamma"))
            b.bias.data.copy_(ds(f"qc_bn_down_{lv}", "beta"))
            b.running_mean.copy_(ds(f"qc_bn_down_{lv}", "moving_mean"))
            b.running_var.copy_(ds(f"qc_bn_down_{lv}", "moving_variance"))
        for i, m in enumerate(net.final):
            conv(m, f"qc_final_conv_{i}")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return net.to(device).eval()


def one_hot_input(idx_zyx: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Stored mask (Z, Y, X; indices into ``LABELS``) -> (X, Y, Z) group map after ``make_shape``, as mri_synthseg
    feeds its network (axes as SynthSeg saw the volume: the port's (X, Y, Z))."""
    idx = make_shape(np.transpose(np.asarray(idx_zyx), (2, 1, 0)))
    return groups[idx]


@torch.no_grad()
def qc_scores(net: QCNet, idx_zyx: np.ndarray, groups: np.ndarray, device: str = "cuda") -> np.ndarray:
    g = torch.from_numpy(one_hot_input(idx_zyx, groups)).to(device)
    x = F.one_hot(g, 9).permute(3, 0, 1, 2)[None].float()
    return np.clip(net(x)[0, 1:].cpu().numpy(), 0, 1)


def run_shard(shard: int, n_shards: int) -> None:
    import glob

    groups, names = qc_groups()
    cols = [n.replace(" ", "_").replace("+", "_") for n in names[1:]]
    vols = pd.concat([pd.read_csv(f, usecols=["volume_id"]) for f in sorted(glob.glob(str(OUT / "volumes_*.csv")))])
    v = pd.read_csv(ROOT / "volumes.csv", low_memory=False, usecols=["volume_id", "folder", "case_id"])
    v = v.drop_duplicates("volume_id")
    todo = v[v["volume_id"].isin(set(vols["volume_id"]))].sort_values("volume_id").iloc[shard::n_shards]
    csv = OUT / f"qc_{shard:03d}.csv"
    done = set(pd.read_csv(csv, usecols=["volume_id"])["volume_id"]) if csv.exists() else set()
    todo = todo[~todo["volume_id"].isin(done)]
    print(f"[ssqc] shard {shard}/{n_shards}: {len(todo)} masks", flush=True)
    net = load_qc()
    recs, t0 = [], time.time()
    for i, r in enumerate(todo.itertuples()):
        p = store.path_for(ROOT, r.folder, str(r.case_id), "synthseg")
        try:
            s = qc_scores(net, store.squeeze_channel(store.read_image(p, "mask")), groups)
        except Exception as e:  # noqa: BLE001
            print(f"[ssqc] {r.volume_id}: {str(e)[:100]}", flush=True)
            continue
        recs.append([r.volume_id, *np.round(s, 4)])
        if len(recs) == 200 or i == len(todo) - 1:
            pd.DataFrame(recs, columns=["volume_id", *cols]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
            recs = []
            print(f"[ssqc] {i + 1}/{len(todo)} {(time.time() - t0) / (i + 1):.2f}s/mask", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()
    run_shard(a.shard, a.n_shards)


if __name__ == "__main__":
    main()
