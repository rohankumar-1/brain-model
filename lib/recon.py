"""Reconstruction metrics and visualization for the ConvVAE."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from lib.data import ID_TO_MODALITY
from models.vae import ConvVAE


@torch.no_grad()
def evaluate_recon_mse(
    model: ConvVAE,
    loader: DataLoader,
    device: torch.device,
    progress: str | None = None,
) -> dict[str, float]:
    model.eval()
    sum_sq = {"total": 0.0, "mri": 0.0, "ct": 0.0}
    count = {"total": 0, "mri": 0, "ct": 0}
    steps = loader
    if progress:
        from lib.progress import EpochBar

        steps = EpochBar(loader, desc=progress, total=len(loader))
    for x, mod_ids in steps:
        x = x.to(device, non_blocking=True)
        recon, _, _ = model(x)
        per = torch.mean((recon - x) ** 2, dim=(1, 2, 3))
        for i in range(x.size(0)):
            mse = float(per[i].item())
            sum_sq["total"] += mse
            count["total"] += 1
            mid = int(mod_ids[i].item())
            name = ID_TO_MODALITY.get(mid)
            if name is not None:
                sum_sq[name] += mse
                count[name] += 1
    out: dict[str, float] = {}
    for k in sum_sq:
        out[k] = sum_sq[k] / max(count[k], 1)
        out[f"{k}_n"] = float(count[k])
    return out


@torch.no_grad()
def save_recon_grid(
    model: ConvVAE,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    n: int = 8,
    title: str = "VAE reconstructions (mixed CT/MRI)",
) -> None:
    model.eval()
    xs, _mods = next(iter(loader))
    xs = xs[:n].to(device, non_blocking=True)
    recon, _, _ = model(xs)
    xs_np = xs.cpu().numpy()
    recon_np = recon.cpu().numpy()
    n = xs_np.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        axes[0, i].imshow(xs_np[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_np[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("input")
    axes[1, 0].set_ylabel("recon")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def format_recon_report(name: str, metrics: dict[str, float]) -> str:
    lines = [
        f"=== {name} reconstruction MSE ===",
        f"total: {metrics['total']:.6f}  (n={int(metrics['total_n'])})",
        f"mri:   {metrics['mri']:.6f}  (n={int(metrics['mri_n'])})",
        f"ct:    {metrics['ct']:.6f}  (n={int(metrics['ct_n'])})",
    ]
    return "\n".join(lines)
