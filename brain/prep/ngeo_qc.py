"""Spot checks of ngeo output: overlay PNGs (image + brain + label masks) and a QC table from the manifests.

    python -m brain.prep.ngeo_qc --out-root runs/diagnostics/ngeo/pilot [--n 50]
writes ``<out-root>/qc/<dataset>__<case>.png`` (three slices: inferior quarter, the slice with most lesion voxels or
the middle, superior quarter; brain window; brain mask red, label masks yellow / cyan) and ``<out-root>/qc/qc.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brain.prep import store
from brain.prep.ngeo_run import QC_RULES, load_manifests

COLORS = ("yellow", "cyan", "lime", "magenta")


def overlay(out_root: Path, row: dict, dst: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = store.read_many(out_root / row["path"], ("image", "brain"))
    img = store.squeeze_channel(arr["image"]).astype(np.float32)
    brain = arr.get("brain")
    masks = {n: store.read(out_root / p, "mask") for n, p in (row.get("masks") or {}).items()}
    S = img.shape[0]
    lesion = next((m for n, m in masks.items() if n not in ("synthseg",)), None)
    mid = int(np.argmax((lesion > 0).sum((1, 2)))) if lesion is not None and lesion.any() else S // 2
    picks = [max(0, S // 4), mid, min(S - 1, 3 * S // 4)]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    for a, s in zip(ax, picks):
        a.imshow(np.clip(img[s], -20, 100), cmap="gray", vmin=-20, vmax=100)
        if brain is not None and brain[s].any():
            a.contour(brain[s], [0.5], colors="red", linewidths=0.6)
        for (name, m), col in zip(masks.items(), COLORS):
            if name != "synthseg" and m[s].any():
                a.contour(m[s] > 0, [0.5], colors=col, linewidths=0.8)
        a.set_title(f"s={s}/{S}", fontsize=8)
        a.axis("off")
    q = {k: row.get(k) for k in ("qc_brain_tissue", "qc_corr_1mm", "tpl_resid_mm", "asm_tilt_deg")}
    fig.suptitle(f"{row['dataset']} {row['case_id'][:40]}  D={row.get('slice_mm', 0):.2f} synth={row.get('synth_thick')} "
                 + " ".join(f"{k.replace('qc_', '')}={v:.2f}" for k, v in q.items() if isinstance(v, float))
                 + f"  masks={list(masks)}", fontsize=8)
    plt.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dst, dpi=60)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50, help="overlays to draw (first n ok rows)")
    a = ap.parse_args()
    m = load_manifests(a.out_root)
    ok = m[m["status"] == "ok"].sort_values("written_ts").drop_duplicates(["dataset", "case_id"], keep="last")
    rows = ok.to_dict("records")
    for r in rows:
        r["qc_flags"] = ",".join(name for name, rule in QC_RULES if rule(r))
    for r in rows[: a.n]:
        overlay(a.out_root, r, a.out_root / "qc" / f"{r['dataset']}__{str(r['case_id'])[:60]}.png")
    cols = [c for c in ("dataset", "case_id", "S", "slice_mm", "thick_mm", "synth_thick", "n_synth", "series_kind",
                        "asm_tilt_deg", "asm_n_dup_removed", "asm_n_gaps", "asm_uneven", "store_frame",
                        "tpl_resid_mm", "qc_brain_tissue", "qc_brain_air", "qc_brain_bone", "qc_corr_1mm",
                        "qc_edge_tissue", "seg_cq500_corr", "bytes", "t_read", "t_build", "t_write", "qc_flags")
            if c in pd.DataFrame(rows).columns]
    extra = sorted(c for c in pd.DataFrame(rows).columns if c.startswith("qc_") and c.endswith(("_tissue", "_vox"))
                   and c not in cols)
    t = pd.DataFrame(rows)[cols + extra]
    t.to_csv(a.out_root / "qc" / "qc.csv", index=False)
    err = m[m["status"] != "ok"]
    print(f"[ngeo-qc] {len(t)} ok, {len(err)} error rows; overlays in {a.out_root / 'qc'}")
    if len(err):
        print(err[["dataset", "case_id", "error"]].drop_duplicates(["dataset", "case_id"]).to_string(max_colwidth=120))


if __name__ == "__main__":
    main()
