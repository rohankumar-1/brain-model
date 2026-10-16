"""Train and score a skip-connected segmentation decoder on a frozen ViT.

Slices are sampled from the pixel cache (foreground-weighted), augmented on the
GPU, encoded by the frozen ViT in bf16, and decoded by :class:`SegDecoder`.
Checkpoint selection uses val Dice; the report is per-volume test Dice with a
bootstrap 95% CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from brain.downstream.embeddings import load_frozen_encoder
from brain.downstream.metrics import bootstrap_ci
from brain.downstream.results import finalize_run, new_run_dir, write_json
from brain.downstream.seg_data import PixelCache, augment, load_pixel_cache, train_batches
from brain.downstream.seg_model import SegDecoder, region_seg_loss
from brain.downstream.tasks import TaskBundle
from brain.common.optim import lr_at
from brain.common.windowing import asinh_window, canonical_c_w


def n_classes_for(regions: dict[str, tuple[int, ...]]) -> int:
    return max(i for ids in regions.values() for i in ids) + 1


def build_cache(task: TaskBundle, slices_root: Path, cache_dir: Path, max_slices: int,
                all_sequences: bool = True) -> PixelCache:
    return load_pixel_cache(task.volume_ids, task.image_paths, task.mask_paths, slices_root,
                            cache_dir, task.name, max_slices, all_sequences)


def split_idx(cache: PixelCache, task: TaskBundle, split: str) -> np.ndarray:
    return np.array([i for i, v in enumerate(cache.volume_ids) if task.splits[v] == split], dtype=np.int64)


class Segmenter:
    """Frozen ViT + trainable decoder; maps (B,C,H,W) images to logits.

    Each of the C sequences is encoded separately (optionally upscaled by ``scale``)
    and their tokens are concatenated channel-wise for the decoder.
    """

    def __init__(self, vit: torch.nn.Module, decoder: SegDecoder, device: torch.device,
                 scale: int = 1, modality: str = "mri", last_only: bool = False):
        self.vit, self.decoder, self.device, self.scale = vit, decoder, device, scale
        self.modality = modality
        # last_only: every decoder input is the final block (what a 3D decoder on final tokens gets)
        self.n_taps = decoder.n_stages + 1
        self.taps = (vit.depth - 1,) if last_only else vit.default_taps(self.n_taps)
        self.amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
        self.cw = canonical_c_w(modality)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        x = x.reshape(b * c, 1, *x.shape[2:])
        x = asinh_window(x, *self.cw)
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        with torch.autocast(**self.amp):
            with torch.no_grad():
                tokens = self.vit.forward_features(x, taps=self.taps)[2]
                tokens = tokens * self.n_taps if len(tokens) == 1 else tokens
            tokens = [t.reshape(b, c, *t.shape[1:]).permute(0, 2, 1, 3).reshape(b, t.shape[1], -1) for t in tokens]
            return self.decoder(tokens)


@torch.no_grad()
def score_volumes(seg: Segmenter, cache: PixelCache, vols: np.ndarray, regions: dict) -> tuple[dict, pd.DataFrame]:
    """Per-volume Dice over the cached slices. A region counts only where the ground truth has it."""
    seg.decoder.eval()
    ids = {name: torch.tensor(r, device=seg.device) for name, r in regions.items()}
    rows = []
    for vi in vols:
        x = torch.from_numpy(cache.images[vi]).to(seg.device).float()
        pred = seg(x).argmax(1)
        gt = torch.from_numpy(cache.masks[vi]).to(seg.device)
        row = {"volume_id": cache.volume_ids[vi]}
        for name, r in ids.items():
            g, p = torch.isin(gt, r), torch.isin(pred, r)
            row[name] = float(2 * (g & p).sum() / (g.sum() + p.sum())) if g.any() else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df["dice_macro"] = df[list(regions)].mean(axis=1)
    per = {name: float(df[name].mean()) for name in regions}
    out = {"task_type": "segmentation", "dice_per_class": per, "dice_macro": float(np.nanmean(list(per.values()))),
           "n_volumes": len(df), "n_positive": {n: int(df[n].notna().sum()) for n in regions}}
    out.update({f"dice_{n}": v for n, v in per.items()})
    return out, df


def train_decoder(seg: Segmenter, cache: PixelCache, task: TaskBundle, *, steps: int, batch_size: int,
                  lr: float, seed: int, eval_every: int, max_val: int = 48) -> list[dict]:
    train, val = split_idx(cache, task, "train"), split_idx(cache, task, "val")[:max_val]
    if len(train) == 0:
        raise SystemExit("segmentation task has no train volumes")
    opt = torch.optim.AdamW(seg.decoder.parameters(), lr=lr, weight_decay=0.01)
    torch.manual_seed(seed)
    history, best, best_state = [], -1.0, None
    seg.decoder.train()
    run_loss = 0.0
    for step, (x, y) in enumerate(train_batches(cache, train, batch_size, steps, seed), 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, steps, lr, warmup=min(300, steps // 10))
        x, y = augment(x.to(seg.device).float(), y.to(seg.device), modality=task.modality_filter or "mri")
        loss = region_seg_loss(seg(x), y, task.regions)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(seg.decoder.parameters(), 1.0)
        opt.step()
        run_loss += float(loss)
        if step % eval_every == 0 or step == steps:
            row = {"step": step, "train_loss": run_loss / min(eval_every, step)}
            run_loss = 0.0
            if len(val):
                m, _ = score_volumes(seg, cache, val, task.regions)
                row["val_dice_macro"] = m["dice_macro"]
                if m["dice_macro"] > best:
                    best = m["dice_macro"]
                    best_state = {k: v.detach().clone() for k, v in seg.decoder.state_dict().items()}
            print(f"[seg] step {step}/{steps} " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "step"))
            history.append(row)
            seg.decoder.train()
    if best_state is not None:
        seg.decoder.load_state_dict(best_state)
        print(f"[seg] restored best val dice {best:.4f}")
    return history


def run_latent_segmentation(task: TaskBundle, *, ckpt: Path, slices_root: Path, out_root: Path, cache_dir: Path,
                            max_slices: int, seed: int, run_name: str | None, steps: int, batch_size: int,
                            lr: float, eval_every: int = 500, all_sequences: bool = True, scale: int = 1,
                            window: str = "canonical", last_only: bool = False) -> Path:
    if not task.regions:
        raise SystemExit(f"{task.name} has no segmentation regions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = load_frozen_encoder(ckpt, device)
    if enc.kind != "vit":
        raise SystemExit("segmentation needs a ViT checkpoint")
    cache = build_cache(task, slices_root, cache_dir, max_slices, all_sequences)
    n_seq = cache.images.shape[2]
    decoder = SegDecoder(enc.model.embed_dim * n_seq, n_classes_for(task.regions), final_up=scale == 1).to(device)
    seg = Segmenter(enc.model, decoder, device, scale, modality=task.modality_filter or "mri", last_only=last_only)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"[seg] decoder {n_params / 1e6:.2f}M params  sequences={n_seq} scale={scale}x steps={steps} "
          f"batch={batch_size} lr={lr} taps={seg.taps} window={window} modality={seg.modality}")

    history = train_decoder(seg, cache, task, steps=steps, batch_size=batch_size, lr=lr,
                            seed=seed, eval_every=eval_every)
    test = split_idx(cache, task, "test")
    metrics, per_volume = score_volumes(seg, cache, test, task.regions)
    lo, hi = bootstrap_ci(lambda i: float(per_volume["dice_macro"].to_numpy()[i].mean()), len(per_volume))
    metrics.update(head="unetr_decoder", dice_macro_ci=[lo, hi], history=history, n_test=len(test),
                   n_train=len(split_idx(cache, task, "train")), n_val=len(split_idx(cache, task, "val")))
    print(f"[seg] test dice_macro={metrics['dice_macro']:.4f} [{lo:.4f}, {hi:.4f}] per_class={metrics['dice_per_class']}")

    run_dir = new_run_dir(out_root, task.name, ckpt, run_name)
    write_json(run_dir / "heads" / "unetr_decoder.json", metrics)
    per_volume.to_csv(run_dir / "predictions" / "unetr_decoder_volumes.csv", index=False)
    torch.save({"model": decoder.state_dict(), "regions": {k: list(v) for k, v in task.regions.items()}},
               run_dir / "seg_decoder.pt")
    config = {"task": task.name, "task_type": "segmentation", "head": "unetr_decoder", "ckpt": str(ckpt),
              "regions": {k: list(v) for k, v in task.regions.items()}, "label_names": task.label_names,
              "sequences": n_seq, "scale": scale, "steps": steps, "batch_size": batch_size, "lr": lr, "seed": seed,
              "max_slices": max_slices, "window": window, "last_only": last_only,
              "n_train": metrics["n_train"], "n_test": metrics["n_test"]}
    print(f"[seg] summary → {finalize_run(run_dir, config, {'unetr_decoder': metrics})}")
    return run_dir
