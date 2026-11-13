"""Per-task metrics for BrainQuery on a fixed set of volumes.

Each volume is asked every labelled task it has, each task on its own (``alone``: no slot sees another, as at
deployment; the scan is encoded once). Metrics per task: binary AUROC, regression R2 and MAE (original units),
multiclass balanced accuracy, segmentation mean Dice (decoder stride, threshold 0.5), dense (synthesis) mean absolute
error in window units over the target's foreground (``<task>_mae``; not part of ``score``, which stays comparable with
runs without self-supervised tasks). ``loss`` is the unweighted mean over tasks of the per-task mean loss; ``score``
the mean over task groups of the tasks' skill (``brain.eval.skill``: 0 = chance, 1 = perfect), with ``group_<g>`` per
group.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import r2_score, roc_auc_score

from brain.downstream.metrics import balanced_accuracy
from brain.eval.skill import skill, summarize
from brain.multitask.data import prepare
from brain.multitask.losses import emphasis_table, seg_target, slot_losses


def dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p, t = (logits > 0).float(), (target > 0.5).float()
    inter = (p * t).sum(dim=(1, 2, 3, 4))
    den = p.sum(dim=(1, 2, 3, 4)) + t.sum(dim=(1, 2, 3, 4))
    return torch.where(den > 0, 2 * inter / den.clamp_min(1), torch.ones_like(den))


@torch.no_grad()
def run_eval(model, loader, cfg: dict, reg, stats: dict, device, patch_xyz, amp: dict, return_preds: bool = False):
    """Per-task metrics over ``loader`` (the model's own heads), every task asked on its own. ``return_preds`` also
    returns the (target, output, volume_id) rows per task index, in loader order (paired comparisons across
    checkpoints pair by volume_id)."""
    was = model.training
    model.eval()
    preds: dict[int, list] = defaultdict(list)
    losses: dict[int, list] = defaultdict(list)
    neg_ids = set((cfg.get("loss") or {}).get("negative_tasks") or [])   # their training loss, so val loss matches
    negatives = torch.tensor([t.id in neg_ids for t in reg.tasks], device=device) if neg_ids else None
    emphasis = emphasis_table(cfg.get("loss"), reg, device)          # val loss = training loss for synthesis too
    task_type = model.task_type
    for raw in loader:
        b = prepare(raw, cfg, device, patch_xyz, train=False)
        with torch.autocast(**amp):
            out = model(b["x"], b["keep"], b["task"], b["seg_pair"], alone=True, meta=b.get("meta"),
                        dense_pair=b.get("dense_pair"), study=b.get("study"), modality=b["modality"])
        loss, has = slot_losses(out, b["task"], b["y"], task_type, b["seg"], b["seg_pair"], dense=b,
                                negatives=negatives, emphasis=emphasis)
        task, y = b["task"], b["y"]
        for i, j in has.nonzero().tolist():
            losses[int(task[i, j])].append(float(loss[i, j]))
        for i, j in (task >= 0).nonzero().tolist():
            t = int(task[i, j])
            kind = reg[t].type
            vid = str(raw["volume_id"][i])
            if kind == "binary":
                preds[t].append((float(y[i, j]), float(out["binary"][i, j]), vid))
            elif kind == "regression":
                preds[t].append((float(y[i, j]), float(out["regression"][i, j]), vid))
            elif kind == "multiclass":
                preds[t].append((float(y[i, j]), int(out["multiclass"][i, j].float().argmax()), vid))
        if out.get("seg") is not None:
            t = seg_target(b["seg"])
            d = dice(out["seg"].float().masked_fill(t[:, -1:] < 0.5, -1.0) if t.shape[1] == 2 else out["seg"].float(),
                     t[:, :1])                                    # ignored voxels count as predicted background
            # y 1: a target Dice can see (> half an output voxel, dice's threshold); 0: empty (a true negative,
            # labels v11). A target too small for the output grid is neither and is left out
            flat = t[:, :1].flatten(1)
            has_t, empty_t = (flat > 0.5).any(1), ~(flat > 0).any(1)
            for (i, j), v, h, e in zip(b["seg_pair"].tolist(), d.tolist(), has_t.tolist(), empty_t.tolist()):
                if h or e:
                    preds[int(task[i, j])].append((float(h), v, str(raw["volume_id"][i])))
        if out.get("dense") is not None:
            w = b["dense_w"]
            mae = ((out["dense"].float() - b["dense_tgt"]).abs() * w).sum(dim=(1, 2, 3, 4)) \
                / w.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
            for (i, j), v in zip(b["dense_pair"].tolist(), mae.tolist()):
                preds[int(task[i, j])].append((1.0, v, str(raw["volume_id"][i])))
    model.train(was)
    res, skills = _metrics(preds, reg, stats)
    per_task = [np.mean(v) for v in losses.values() if v]
    res["loss"] = float(np.mean(per_task)) if per_task else float("nan")
    res.update(summarize(skills))
    ref = getattr(getattr(loader, "dataset", None), "ref", None)
    if ref:                       # the reference run's tasks on its scans only (``train.ref_score``): comparable score
        res.update(ref_scores(preds, reg, stats, ref))
    return (res, dict(preds)) if return_preds else res


def ref_scores(preds: dict, reg, stats: dict, ref: dict) -> dict[str, float]:
    """``ref_score`` / ``ref_group_<g>``: the skill summary over ``ref["tasks"]`` (ids) on ``ref["volumes"]`` only."""
    sub = {t: [r for r in rows if r[2] in ref["volumes"]] for t, rows in preds.items() if reg[t].id in ref["tasks"]}
    return {f"ref_{k}": v for k, v in summarize(_metrics(sub, reg, stats)[1]).items()}


def _metrics(preds: dict, reg, stats: dict) -> tuple[dict[str, float], list]:
    """Per-task metrics and (group, skill) pairs from ``run_eval``'s prediction rows."""
    res: dict[str, float] = {}
    skills = []
    for t, rows in sorted(preds.items()):
        task = reg[t]
        y, p = np.asarray([r[0] for r in rows]), np.asarray([r[1] for r in rows])
        ok = np.isfinite(y)
        y, p = y[ok], p[ok]
        if len(y) < 2:
            continue
        if task.type == "binary":
            y = (y > 0.5).astype(float)          # soft labels (MR-RATE "not reported" = 0.05) score as negatives
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
            res[f"{task.id}_bacc"] = balanced_accuracy(y.astype(int), p.astype(int))
            skills.append((task.group, skill("bacc", res[f"{task.id}_bacc"], len(task.classes))))
        elif task.type == "segmentation":
            # Dice on scans with a target; on empty ones (Dice 1 = nothing predicted) the false-positive rate
            if (y > 0).any():
                res[f"{task.id}_dice"] = float(p[y > 0].mean())
                skills.append((task.group, res[f"{task.id}_dice"]))
            if (y == 0).any():                  # specificity counts in the score, so model selection sees it
                res[f"{task.id}_fp"] = float(1.0 - p[y == 0].mean())
                skills.append((task.group, 1.0 - res[f"{task.id}_fp"]))
        elif task.type == "dense":
            res[f"{task.id}_mae"] = float(p.mean())
    return res, skills


METRICS = ("auc", "r2", "mae", "bacc", "dice", "fp")


def format_eval(vals: dict[str, float], reg, width: int = 150) -> list[str]:
    """``run_eval`` results (keys with or without the trainer's ``val_`` prefix) as log lines: ``score`` and ``loss``,
    then one line per task group (registry order): the group's skill (``brain.eval.skill``: mean over its tasks of the
    metric rescaled to 0 = chance, 1 = perfect, e.g. AUC 0.85 -> 0.70), then each task's raw metric, named once after
    the bar when the whole group shares it. The group's shared prefix is dropped from task names
    (``hemorrhage_subdural`` -> ``subdural``); long groups wrap under their first task."""
    v = {k.removeprefix("val_"): x for k, x in vals.items()}
    head = "  ".join(f"{k} {v[k]:.4f}" for k in ("score", "ref_score", "loss") if k in v)
    per_task: dict[str, list[str]] = {}
    used = {"score", "ref_score", "loss"} | {k for k in v if k.startswith("ref_group_")}
    for t in reg.tasks:
        cells = [f"{m} {v[f'{t.id}_{m}']:.{2 if m == 'mae' else 3}f}" for m in METRICS if f"{t.id}_{m}" in v]
        if cells:
            used |= {f"{t.id}_{m}" for m in METRICS}
            per_task.setdefault(t.group, []).append((t.id, cells))
    lines = [head]
    gw = max((len(g) for g in per_task), default=0)
    for g, tasks in per_task.items():
        common = os.path.commonprefix([tid for tid, _ in tasks]) if len(tasks) > 1 else ""
        stem = common[: common.rfind("_") + 1]                      # e.g. hemorrhage_, synth_, seg_
        kinds = {m.split(" ")[0] for _, ms in tasks for m in ms}
        metric = next(iter(kinds)) if len(kinds) == 1 else ""      # one metric: named once, in the group's lead
        cells = []
        for tid, ms in tasks:
            name = tid.removeprefix(f"{g}_").removeprefix(stem).removeprefix("seg_").removeprefix("vol_") or tid
            cells.append(f"{name} {ms[0].split(' ', 1)[1] if metric else ' '.join(ms)}")
        skill = v.get(f"group_{g}")
        lead = f"  {g:<{gw}}  skill " + (f"{skill:6.3f}" if skill is not None else "     -") + " | " \
            + (f"{metric:<4} " if metric else "")
        line = lead
        for c in cells:
            if len(line) + len(c) + 2 > width and line != lead:
                lines.append(line.rstrip())
                line = " " * len(lead)
            line += c + "  "
        lines.append(line.rstrip())
    rest = {k: x for k, x in v.items() if k not in used and not k.startswith("group_")}
    if rest:
        lines.append("  other  " + "  ".join(f"{k} {x:.4f}" for k, x in rest.items()))
    return lines
