"""Does a token depend on how much context the encoder sees? Whole-volume vs training-size-window encoding.

Training only shows the encoder crops up to 128 x 128 x 64 mm (<= 512 tokens); frozen evaluation encodes the whole
256 x 256 x 160 volume (5120 tokens) in one pass. For a sample of validation volumes this encodes both ways and
reports, per encoder, the cosine similarity of each foreground token between the two encodings, and how far apart
(in units of the between-volume spread) the pooled embeddings land.

    python -m brain.downstream.context_check --ckpt A.pt --ckpt B.pt --n 24
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brain.common.windowing import windows_for
from brain.downstream.embed3d import read_volume_xyz
from brain.downstream.embeddings import load_frozen_encoder, resolve_path
from brain.downstream.tasks import DEFAULT_VOLUMES

TILE = {"window_xyz": (128, 128, 64), "stride_xyz": (64, 64, 32)}


def sample_volumes(n: int, seed: int = 0) -> pd.DataFrame:
    v = pd.read_csv(DEFAULT_VOLUMES, low_memory=False)
    v = v[(v.split == "val") & v.pretrain.astype(bool)]
    ct, mri = v[v.modality == "ct"], v[v.modality != "ct"]
    return pd.concat([ct.sample(n // 3, random_state=seed), mri.sample(n - n // 3, random_state=seed)])


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True, type=Path)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = sample_volumes(a.n)
    root = DEFAULT_VOLUMES.parent
    for ck in a.ckpt:
        enc = load_frozen_encoder(ck, dev)
        sims, pooled_whole, pooled_tile = [], [], []
        for r in rows.itertuples():
            vol = torch.from_numpy(np.ascontiguousarray(read_volume_xyz(resolve_path(root, r.image_path)))).to(dev)
            mod = "ct" if r.modality == "ct" else "mri"
            win = windows_for("canonical", mod)
            whole, _, fg = enc.embed_windows(vol, win, modality=mod)
            tiled, _, fg2 = enc.embed_windows(vol, win, modality=mod, **TILE)
            keep = (fg.bool() & fg2.bool()).reshape(-1)
            w, t = whole[0].float()[keep], tiled[0].float()[keep]
            sims.append(torch.nn.functional.cosine_similarity(w, t, dim=-1).cpu().numpy())
            pooled_whole.append(w.mean(0).cpu().numpy())
            pooled_tile.append(t.mean(0).cpu().numpy())
        s = np.concatenate(sims)
        pw, pt = np.stack(pooled_whole), np.stack(pooled_tile)
        spread = np.linalg.norm(pw - pw.mean(0), axis=1).mean()
        shift = np.linalg.norm(pw - pt, axis=1).mean()
        # does the pooled embedding still identify its own volume across the two encodings?
        d = np.linalg.norm(pw[:, None] - pt[None], axis=-1)
        match = float((d.argmin(1) == np.arange(len(pw))).mean())
        print(f"[context] {ck.parent.name}/{ck.stem}: token cosine whole-vs-tiled mean={s.mean():.3f} "
              f"median={np.median(s):.3f} p10={np.percentile(s, 10):.3f} | pooled shift / between-volume spread="
              f"{shift / spread:.2f} | pooled nearest-neighbour match={match:.2f} (n={len(rows)})", flush=True)


if __name__ == "__main__":
    main()
