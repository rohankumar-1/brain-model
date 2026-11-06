"""Can a model's aneurysm_location map find aneurysms (2026-11-01)? On the RSNA-ICA test split: each scan's peak
predicted logit as an aneurysm-present score (AUC against the label, beside the model's own aneurysm_present head on the
same scans), and for positive scans whether the peak voxel lies inside an annotated 4 mm ball (hit rate), by
sequence. The go / no-go for a find -> refine pass: refining helps only if the finder puts the aneurysm in its top cells.

    python runs/diagnostics/aneurysm_finder.py --ckpt runs/multitask/r10/last.pt   -> runs/diagnostics/aneurysm_finder_<run>.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/roku/brain-model")
from brain.common.paths import resolve  # noqa: E402
from brain.multitask.data import load_labels, prepare  # noqa: E402
from brain.multitask.fewshot import _loader, load_model  # noqa: E402
from brain.multitask.losses import seg_target  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    model, reg, cfg = load_model(Path(a.ckpt), dev)
    store = resolve(cfg["data"]["store_root"])
    labels, stats = load_labels(resolve(cfg["data"]["labels_dir"]))
    volumes = pd.read_csv(store / cfg["data"].get("index_name", "volumes.csv"), low_memory=False)
    test = labels[(labels["split"] == "test") & labels["task"].isin(["aneurysm_location", "aneurysm_present"])]
    ids = sorted(set(test.loc[test.task == "aneurysm_location", "volume_id"]) &
                 set(test.loc[test.task == "aneurysm_present", "volume_id"]))
    present = test[test.task == "aneurysm_present"].set_index("volume_id")["value"]
    patch = model.backbone.patch_xyz
    rows = {}
    for task in ("aneurysm_location", "aneurysm_present"):
        loader, _ = _loader(store, labels, volumes, reg, stats, task, "test", ids=ids, use="train", workers=a.workers)
        for raw in loader:
            b = prepare(raw, cfg, dev, patch, train=False)
            with torch.no_grad(), torch.autocast(**amp):
                out = model(b["x"], b["keep"], b["task"], b["seg_pair"], alone=True, meta=b.get("meta"))
            vids = [str(v) for v in raw["volume_id"]]
            if task == "aneurysm_present":
                for (i, j) in (b["task"] >= 0).nonzero().tolist():
                    rows.setdefault(vids[i], {})["head"] = float(out["binary"][i, j])
                continue
            logit = out["seg"].float()
            tgt = seg_target(b["seg"])[:, :1].float()
            if tgt.shape[2:] != logit.shape[2:]:
                tgt = F.interpolate(tgt, size=logit.shape[2:], mode="nearest")
            for k, (i, _) in enumerate(b["seg_pair"].tolist()):
                flat = logit[k, 0].flatten()
                peak = int(flat.argmax())
                r = rows.setdefault(vids[i], {})
                r["peak"] = float(flat[peak])
                r["hit"] = bool(tgt[k, 0].flatten()[peak] > 0.5) if tgt[k].any() else np.nan
                r["has_ball"] = bool(tgt[k].any())
    d = pd.DataFrame.from_dict(rows, orient="index").rename_axis("volume_id").reset_index()
    d["present"] = d["volume_id"].map(present)
    d["dataset"] = d["volume_id"].map(volumes.drop_duplicates("volume_id").set_index("volume_id")["dataset"])
    out = Path(__file__).with_name(f"aneurysm_finder_{Path(a.ckpt).parent.name}_{Path(a.ckpt).stem}.csv")
    d.to_csv(out, index=False)

    def auc(g, col):
        g = g.dropna(subset=[col, "present"])
        return roc_auc_score(g["present"], g[col]) if g["present"].nunique() == 2 else np.nan

    res = []
    for name, g in [("all", d), *d.groupby("dataset")]:
        res.append({"subset": name, "n": len(g), "pos": int(g["present"].sum()), "auc_peak": auc(g, "peak"),
                    "auc_head": auc(g, "head"), "hit_rate": g.loc[g["has_ball"] == True, "hit"].mean()})  # noqa: E712
    print(pd.DataFrame(res).round(3).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
