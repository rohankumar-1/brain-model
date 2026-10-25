"""Per-task metrics for the multitask model on a fixed set of volumes.

Each volume is asked every labelled task it has, each task on its own (``alone``: no slot sees another, as at
deployment; the scan is encoded once). Metrics per task: binary AUROC, regression R2 and MAE (original units),
multiclass balanced accuracy, segmentation mean Dice (decoder stride, threshold 0.5). ``loss`` is the unweighted mean
over tasks of the per-task mean loss; ``score`` the mean over task groups of the tasks' skill (``brain.eval.skill``:
0 = chance, 1 = perfect), with ``group_<g>`` per group.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score

from brain.eval.skill import skill, summarize
from brain.multitask.data import prepare
from brain.multitask.losses import seg_target, slot_losses


def dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p, t = (logits > 0).float(), (target > 0.5).float()
    inter = (p * t).sum(dim=(1, 2, 3, 4))
    den = p.sum(dim=(1, 2, 3, 4)) + t.sum(dim=(1, 2, 3, 4))
    return torch.where(den > 0, 2 * inter / den.clamp_min(1), torch.ones_like(den))


@torch.no_grad()
def run_eval(model, loader, cfg: dict, reg, stats: dict, device, patch_xyz, amp: dict, return_preds: bool = False,
             alone: bool = True):
    """Per-task metrics over ``loader`` (the model's own heads). ``return_preds`` also returns the (target, output)
    rows per task index, in loader order (paired comparisons across checkpoints). ``alone=False`` asks a volume's
    tasks together in one query (slots attend to each other), as before 2026-10-25."""
    was = model.training
    model.eval()
    preds: dict[int, list] = defaultdict(list)
    losses: dict[int, list] = defaultdict(list)
    task_type = model.task_type
    for raw in loader:
        b = prepare(raw, cfg, device, patch_xyz, train=False)
        with torch.autocast(**amp):
            out = model(b["x"], b["keep"], b["task"], b["seg_pair"], alone=alone)
        loss, has = slot_losses(out, b["task"], b["y"], task_type, b["seg"], b["seg_pair"])
        task, y = b["task"], b["y"]
        for i, j in has.nonzero().tolist():
            losses[int(task[i, j])].append(float(loss[i, j]))
        for i, j in (task >= 0).nonzero().tolist():
            t = int(task[i, j])
            kind = reg[t].type
            if kind == "binary":
                preds[t].append((float(y[i, j]), float(out["binary"][i, j])))
            elif kind == "regression":
                preds[t].append((float(y[i, j]), float(out["regression"][i, j])))
            elif kind == "multiclass":
                preds[t].append((float(y[i, j]), int(out["multiclass"][i, j].float().argmax())))
        if out.get("seg") is not None:
            d = dice(out["seg"].float(), seg_target(b["seg"]))
            for (i, j), v in zip(b["seg_pair"].tolist(), d.tolist()):
                preds[int(task[i, j])].append((1.0, v))
    model.train(was)
    res: dict[str, float] = {}
    skills = []
    for t, rows in sorted(preds.items()):
        task = reg[t]
        y, p = np.asarray([r[0] for r in rows]), np.asarray([r[1] for r in rows])
        ok = np.isfinite(y)
        y, p = y[ok], p[ok]
        if len(y) < 2:
            continue
        if task.type == "binary" and len(np.unique(y)) == 2:
            res[f"{task.id}_auc"] = float(roc_auc_score(y, p))
            skills.append((task.group, skill("auc", res[f"{task.id}_auc"])))
        elif task.type == "regression":
            s = stats.get(task.id, {})
            sd = float(s.get("std", 1.0))
            res[f"{task.id}_r2"] = float(r2_score(y, p))
            res[f"{task.id}_mae"] = float(np.abs(y - p).mean() * sd)
            skills.append((task.group, skill("r2", res[f"{task.id}_r2"])))
        elif task.type == "multiclass":
            res[f"{task.id}_bacc"] = float(balanced_accuracy_score(y.astype(int), p.astype(int)))
            skills.append((task.group, skill("bacc", res[f"{task.id}_bacc"], len(task.classes))))
        elif task.type == "segmentation":
            res[f"{task.id}_dice"] = float(p.mean())
            skills.append((task.group, res[f"{task.id}_dice"]))
    per_task = [np.mean(v) for v in losses.values() if v]
    res["loss"] = float(np.mean(per_task)) if per_task else float("nan")
    res.update(summarize(skills))
    return (res, dict(preds)) if return_preds else res
