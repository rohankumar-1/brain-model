"""What a checkpoint's learned window bank does to real scans: for a few validation scans (CT and MRI), the canonical
window next to each of the K learned windows (axial slice through the brain), plus each window's transfer curve in
native units (HU / robust z). No volume ids on the figure.

    python analysis/window_bank_view.py --ckpt runs/multitask/r18/last.pt --out runs/diagnostics/window_bank_r18"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAMES = ("narrow tissue", "tissue", "wide", "dark", "bright", "very bright")      # the contract6 init order


def load_bank(ckpt: Path, model_cfg: dict):
    from brain.models.window_bank import build_bank

    state = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    sd = state["model"]
    bank = build_bank(model_cfg)
    sub = {k.split("window_bank.", 1)[1]: v for k, v in sd.items() if "window_bank." in k}
    bank.load_state_dict(sub)
    return bank.eval()


def pick(volumes: pd.DataFrame, rng) -> list[tuple[str, pd.Series]]:
    v = volumes[(volumes["split"] == "val") & volumes["brain_ok"].astype(str).isin(["True", "true", "1"])]
    out = []
    for title, m in (("CT (RSNA head CT)", v["dataset_family"].eq("rsna_ich")),
                     ("MRI T1 (BraTS glioma)", v["dataset"].astype(str).str.endswith("t1n") & v["dataset_family"].eq("brats_gli")),
                     ("MRI FLAIR (BraTS glioma)", v["dataset"].astype(str).str.endswith("t2f") & v["dataset_family"].eq("brats_gli"))):
        c = v[m]
        if len(c):
            out.append((title, c.iloc[int(rng.integers(len(c)))]))
    return out


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from brain.common.config import load_config
    from brain.common.paths import resolve
    from brain.models.window_bank import CALIBRATION, MOD_CT, MOD_MRI, init_windows
    from brain.multitask.data import canonical_window
    from brain.prep import store

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None, help="default: config.yaml next to the checkpoint")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    cfg = load_config(a.config or a.ckpt.parent / "config.yaml")
    bank = load_bank(a.ckpt, cfg["model"])
    nat = bank.native_windows()                       # {"ct": [(centre, width) ...], "mri": [...]} in native units
    k = bank.k
    names = list(NAMES[:k]) if k == len(NAMES) else [f"window {i}" for i in range(k)]
    st = resolve(cfg["data"]["store_root"])
    volumes = pd.read_csv(st / cfg["data"].get("index_name", "volumes.csv"), low_memory=False)
    scans = pick(volumes, np.random.default_rng(a.seed))
    a.out.mkdir(parents=True, exist_ok=True)

    # ---- images: canonical window + the K learned windows, one row per scan
    fig, axes = plt.subplots(len(scans), k + 1, figsize=(2.2 * (k + 1), 2.5 * len(scans)), dpi=130, squeeze=False,
                             gridspec_kw=dict(wspace=0.03, hspace=0.12))
    for r, (title, row) in enumerate(scans):
        img = np.asarray(store.squeeze_channel(store.read(st / row["image_path"], "image")), dtype=np.float32)
        brain = np.asarray(store.read(st / row["image_path"], "brain")) > 0
        z = int(np.argmax(brain.sum((1, 2)))) if brain.any() else img.shape[0] // 2
        sl = img[z][::-1]
        mod = MOD_CT if str(row["modality"]).lower().startswith("ct") else MOD_MRI
        with torch.no_grad():
            y = bank(torch.from_numpy(img[None, None, z:z + 1].copy()), torch.tensor([mod]))[0, :, 0].numpy()
        canon = canonical_window(torch.from_numpy(sl.copy())[None, None], torch.tensor([mod]))[0, 0].numpy()
        lo0, hi0 = np.percentile(canon, [0.5, 99.5])
        axes[r, 0].imshow(canon, cmap="gray", vmin=lo0, vmax=hi0)
        axes[r, 0].set_ylabel(title, fontsize=9)
        for c in range(k):
            ch = y[c][::-1]
            lo, hi = np.percentile(ch, [0.5, 99.5])
            axes[r, c + 1].imshow(ch, cmap="gray", vmin=lo, vmax=hi)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    unit = {"ct": "HU", "mri": "z"}
    for c in range(k + 1):
        if c == 0:
            axes[0, 0].set_title("canonical window\n(fixed reference)", fontsize=9)
        else:
            (cc, wc), (cm, wm) = nat["ct"][c - 1], nat["mri"][c - 1]
            axes[0, c].set_title(f"{names[c - 1]}\nCT {cc:.0f}±{wc:.0f} HU · MRI {cm:.1f}±{wm:.1f} z", fontsize=8)
    fig.savefig(a.out / "windows_on_scans.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- transfer curves: native intensity -> each window's output, learned (solid) vs initial (dashed)
    init = init_windows(cfg["model"]["window_bank"].get("init", "contract6"), k)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=130)
    colors = plt.cm.tab10(np.arange(k))
    for ax, (mod, key, xr) in zip(axes, ((MOD_CT, "ct", (-1100, 1500)), (MOD_MRI, "mri", (-6, 22)))):
        m, s = CALIBRATION[mod]
        x = np.linspace(*xr, 800)
        u = (x - m) / s
        for c in range(k):
            ctr, wid = nat[key][c]
            ax.plot(x, np.arcsinh((x - ctr) / wid), color=colors[c], lw=1.8, label=names[c])
            ci, wi = init[c]
            ax.plot(x, np.arcsinh((u - ci) / wi), color=colors[c], lw=1, ls="--", alpha=0.6)
        ax.set_title(f"{key.upper()}: learned (solid) vs initial (dashed)", fontsize=10)
        ax.set_xlabel(unit[key])
        ax.set_ylabel("window output (asinh)")
        ax.axhline(0, color="0.8", lw=0.6)
        if key == "ct":
            for hu, lab in ((-1000, "air"), (0, "water/CSF"), (40, "brain"), (70, "blood"), (1000, "bone")):
                ax.axvline(hu, color="0.85", lw=0.6)
                ax.text(hu, ax.get_ylim()[1] * 0.92, lab, fontsize=7, color="0.45", ha="center")
    axes[1].legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(a.out / "window_curves.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    rows = [{"window": names[c], "ct_centre_hu": nat["ct"][c][0], "ct_width_hu": nat["ct"][c][1],
             "mri_centre_z": nat["mri"][c][0], "mri_width_z": nat["mri"][c][1]} for c in range(k)]
    pd.DataFrame(rows).round(3).to_csv(a.out / "windows.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
