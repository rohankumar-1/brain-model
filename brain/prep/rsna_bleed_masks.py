"""Voxel bleed masks for RSNA ICH studies from two independent segmenters, checked against RSNA's slice labels
(2026-11-03). We hold ~700 manual bleed masks (BHSD, HemSeg) against ~7k bleed-positive RSNA training studies.

  ts       TotalSegmentator ``cerebral_bleed`` (nnU-Net Dataset150, Basel ICH data outside our sources) run as a batch:
           the nnU-Net predictor is loaded once and volumes stream through ``predict_from_list_of_npy_arrays``
           (preprocessing and export in worker processes around the GPU). The tool's own wrapper also runs its
           whole-body model on every scan just to find the brain; we crop to our stored brain mask instead. The store
           is RAS voxel order on a 1 mm grid, which is what the model's reorienting NIfTI reader hands nnU-Net, so
           stored (Z, Y, X) arrays go in as they are (``ts-verify`` checks this against the tool's own API).
           -> ``<case>.tsbleed.zip`` (uint8 0 / 1) next to the image.
  model    our haemorrhage model's ``hemorrhage_segmentation`` head through the evaluation input path (the dataset,
           collate and ``prepare`` of training), upsampled from the decoder's stride to the grid.
           -> ``<case>.mtbleed.zip`` (uint8 probability x 255).
  validate the 144 RSNA studies that BHSD / HemSeg annotated in their val / test splits (the internal model never
           trained on them): merging rules scored against the manual masks (both on the registered stored grid).
  fuse     over the training studies (rule from ``validate``, 144 studies, 2026-11-03): the target is the union of the
           two masks, accepted when they agree (Dice >= 0.5) and at most 30% of it lies in slices RSNA read bleed-free
           (71% accepted at Dice 0.66 against the manual masks; the rejected scored 0.39). Removing voxels in negative
           slices lowered Dice, and slice coverage was no signal. Both tools segment intraparenchymal / intraventricular
           blood well and subdural / subarachnoid / epidural poorly (model Dice 0.34 / 0.44 / 0.27), so on a study
           RSNA also reads one of those, the voxels of its positive slices outside the union are left out of the loss
           (value 2): unsegmented blood is not taught as background, and its negative slices still are.
           -> ``<case>.bleedpl.zip`` (uint8: 1 bleed, 2 ignore) and ``labels/rsna_bleed/fuse.parquet``.

    python -m brain.prep.rsna_bleed_masks ts --set train --shard 0/8
    python -m brain.prep.rsna_bleed_masks model --set train --ckpt runs/single/hemorrhage_r13/last.pt
    python -m brain.prep.rsna_bleed_masks validate
    python -m brain.prep.rsna_bleed_masks fuse --set train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brain.common.paths import PROCESSED_ROOT
from brain.prep import store
from brain.prep.morph import dice

ROOT = PROCESSED_ROOT
TS_MODEL = Path.home() / ".totalsegmentator/nnunet/results/Dataset150_icb_v0/nnUNetTrainer__nnUNetPlans__3d_fullres"
OUT = ROOT / "labels" / "rsna_bleed"
MARGIN = 20      # voxels around the brain mask kept in the crop: extra-axial blood (subdural, epidural) lies outside
                 # it (6 cut a 45 ml bleed to 33 ml against the tool's own crop, 2026-11-03)


def targets(which: str) -> pd.DataFrame:
    """``train``: bleed-positive RSNA training studies; ``val``: the RSNA twins of BHSD / HemSeg val / test masks (with
    ``manual`` / ``manual_ids`` / ``manual_image``)."""
    v = pd.read_csv(ROOT / "volumes.csv", low_memory=False)
    r = v[v["dataset_family"] == "rsna_ich"].drop_duplicates("volume_id")
    if which == "val":
        b = v[v["dataset_family"].isin(["bhsd", "hemseg"]) & v["mask_ich"].notna() & v["split"].isin(["val", "test"])]
        b = b.assign(manual=b["mask_ich"], manual_image=b["image_path"],
                     manual_ids=np.where(b["dataset_family"] == "bhsd", "1,2,3,4,5", "1"))
        return r.merge(b[["case_id", "manual", "manual_ids", "manual_image"]], on="case_id")
    from brain.downstream.tasks import build_rsna_volume_labels

    lab = build_rsna_volume_labels()
    pos = set(lab.loc[lab["any"] == 1, "volume_id"].astype(str))
    # studies BHSD / HemSeg already annotated by hand (the internal model trained on them): not pseudo-labelled
    manual = set(v.loc[v["dataset_family"].isin(["bhsd", "hemseg"]) & v["mask_ich"].notna(), "case_id"].astype(str))
    return r[r["volume_id"].astype(str).isin(pos) & (r["split"] == which) & ~r["case_id"].astype(str).isin(manual)]


def out_path(r, kind: str) -> Path:
    return store.path_for(ROOT, str(r.folder), str(r.case_id), kind)


def _crop(brain: np.ndarray) -> tuple[slice, ...]:
    idx = np.nonzero(brain)
    return tuple(slice(max(0, int(i.min()) - MARGIN), min(n, int(i.max()) + MARGIN + 1))
                 for i, n in zip(idx, brain.shape))


# ------------------------------------------------------------------------------------------------ TotalSegmentator
def ts_predictor(device: str = "cuda"):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,      # the tool runs it tta=False
                        perform_everything_on_device=True, device=torch.device(device), verbose=False,
                        verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(str(TS_MODEL), use_folds=[0], checkpoint_name="checkpoint_final.pth")
    return p


def run_ts(rows: pd.DataFrame, chunk: int = 12, workers: int = 4, overwrite: bool = False) -> None:
    pred = ts_predictor()
    rows = rows if overwrite else rows[[not out_path(r, "tsbleed").exists() for r in rows.itertuples()]]
    for i in range(0, len(rows), chunk):
        part, imgs, props, crops, shapes = [], [], [], [], []
        for r in rows.iloc[i:i + chunk].itertuples():
            a = store.read_many(ROOT / str(r.image_path), ("image", "brain"))
            img = store.squeeze_channel(a["image"]).astype(np.float32)
            brain = store.squeeze_channel(a["brain"]) > 0 if "brain" in a else img > -500
            if not brain.any():
                continue
            c = _crop(brain)
            part.append(r), crops.append(c), shapes.append(img.shape)
            imgs.append(img[c][None])
            props.append({"spacing": [1.0, 1.0, 1.0]})
        if not imgs:
            continue
        segs = pred.predict_from_list_of_npy_arrays(imgs, None, props, None, num_processes=workers,
                                                    num_processes_segmentation_export=workers)
        for r, c, shape, seg in zip(part, crops, shapes, segs):
            full = np.zeros(shape, dtype=np.uint8)
            full[c] = (np.asarray(seg) > 0).astype(np.uint8)
            store.write(out_path(r, "tsbleed"), "mask", full, {"source": "totalsegmentator cerebral_bleed (Dataset150)"})
        print(f"[ts] {min(i + chunk, len(rows))}/{len(rows)}", flush=True)


def ts_verify(rows: pd.DataFrame, n: int = 3) -> None:
    """Our batched call against TotalSegmentator's own API on ``n`` scans (orientation, cropping)."""
    import tempfile

    import SimpleITK as sitk
    from totalsegmentator.python_api import totalsegmentator

    from brain.prep import native

    rows = rows.head(n)
    run_ts(rows, overwrite=True)
    with tempfile.TemporaryDirectory() as d:
        for r in rows.itertuples():
            attrs = store.read_attrs(ROOT / str(r.image_path), "image")
            arr = store.squeeze_channel(store.read_image(ROOT / str(r.image_path))).astype(np.float32)
            inp, out = Path(d) / "ct.nii.gz", Path(d) / "seg.nii.gz"
            sitk.WriteImage(native._image(arr, attrs["grid_origin"], attrs["grid_spacing"], attrs["grid_direction"]),
                            str(inp))
            totalsegmentator(str(inp), str(out), task="cerebral_bleed", ml=True, device="gpu", quiet=True)
            ref = sitk.GetArrayFromImage(sitk.ReadImage(str(out))) > 0
            ours = store.read_image(out_path(r, "tsbleed"), "mask") > 0
            s = ref.sum() + ours.sum()
            print(f"[ts-verify] {r.volume_id}: tool {ref.sum() / 1000:.1f} ml, batched {ours.sum() / 1000:.1f} ml, "
                  f"Dice {2 * (ref & ours).sum() / s if s else 1.0:.3f}", flush=True)


# ------------------------------------------------------------------------------------------------ internal model
def run_model(rows: pd.DataFrame, ckpt: Path, batch: int = 2, workers: int = 6) -> None:
    import torch
    import torch.nn.functional as F

    from brain.multitask.data import MultitaskDataset, checkpoint_stats, load_labels, make_loader, prepare
    from brain.multitask.fewshot import load_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, reg, cfg = load_model(ckpt, dev)
    labels_dir = Path(cfg["data"]["labels_dir"])
    _, stats = load_labels(labels_dir)
    stats = checkpoint_stats(model.label_stats, labels_dir, stats)
    volumes = pd.read_csv(ROOT / "volumes.csv", low_memory=False)
    rows = rows[[not out_path(r, "mtbleed").exists() for r in rows.itertuples()]]
    lab = pd.DataFrame({"volume_id": rows["volume_id"].astype(str), "task": "hemorrhage_segmentation",
                        "value": np.nan, "ref": "@empty", "split": "infer", "use": "train"})
    ds = MultitaskDataset(ROOT, lab, volumes, reg, stats, split="infer", tasks=["hemorrhage_segmentation"],
                          train=False, use="train", max_seg=1)
    loader = make_loader(ds, batch_size=batch, shuffle=False, seed=0, workers=workers, drop_last=False)
    by_id = {str(r.volume_id): r for r in rows.itertuples()}
    patch = tuple(model.backbone.patch_xyz)
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    done = 0
    with torch.no_grad():
        for raw in loader:
            b = prepare(raw, cfg, dev, patch, train=False)
            with torch.autocast(**amp):
                out = model(b["x"], b["keep"], b["task"], b["seg_pair"], alone=True, meta=b.get("meta"),
                            dense_pair=b.get("dense_pair"), modality=b["modality"])
            prob = torch.sigmoid(F.interpolate(out["seg"].float(), size=tuple(raw["image"].shape[1:]),
                                               mode="trilinear", align_corners=False))
            for (i, _), p in zip(b["seg_pair"].tolist(), prob):
                r = by_id[str(raw["volume_id"][i])]
                store.write(out_path(r, "mtbleed"), "mask", (p[0] * 255).round().byte().cpu().numpy(),
                            {"source": f"hemorrhage_segmentation head of {ckpt}"})
            done += len(raw["volume_id"])
            if done % 200 < batch:
                print(f"[model] {done}/{len(rows)}", flush=True)


# ------------------------------------------------------------------------------------------------ slice labels
def slice_state(r, slices: pd.DataFrame, shape) -> np.ndarray:
    """Per stored voxel: 1 inside a slice RSNA read positive for any bleed, 0 inside one read negative, -1 unlabelled
    (between slices or outside the series)."""
    from brain.downstream.tasks import RSNA_LABELS

    s = slices.loc[str(r.volume_id)]
    pos, bits = np.asarray(s.s, dtype=np.float64), np.asarray(s.bits).astype(np.int64)
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float32) for n in shape), indexing="ij", sparse=True)
    sv = s.g_z * z + s.g_y * y + s.g_x * x + s.c
    k = np.clip(np.searchsorted(pos, sv), 1, len(pos) - 1)
    near = np.where(np.abs(sv - pos[k - 1]) <= np.abs(sv - pos[k]), k - 1, k)
    ok = np.abs(sv - pos[near]) <= float(s.spacing) / 2
    b = bits[near]
    any_bit = 1 << RSNA_LABELS.index("any")
    return np.where(ok & (b != 255), (b & any_bit > 0).astype(np.int8), -1).astype(np.int8)


def candidates(r, slices: pd.DataFrame) -> dict[str, np.ndarray] | None:
    ts_p, mt_p = out_path(r, "tsbleed"), out_path(r, "mtbleed")
    if not (ts_p.exists() and mt_p.exists()):
        return None
    ts = store.read_image(ts_p, "mask") > 0
    mt = store.read_image(mt_p, "mask") >= 128
    st = slice_state(r, slices, ts.shape)
    allowed = st != 0
    return {"ts": ts, "model": mt, "union": ts | mt, "both": ts & mt,
            "ts_sl": ts & allowed, "model_sl": mt & allowed, "union_sl": (ts | mt) & allowed,
            "_state": st}


def slice_qc(mask: np.ndarray, state: np.ndarray) -> tuple[float, float]:
    """(share of positive-slice z-levels the mask reaches, share of the mask's voxels in negative slices)."""
    pos_z = np.flatnonzero((state == 1).any(axis=(1, 2)))
    hit = np.flatnonzero(mask.any(axis=(1, 2)))
    recall = float(np.isin(pos_z, hit).mean()) if pos_z.size else float("nan")
    spill = float((mask & (state == 0)).sum() / max(mask.sum(), 1))
    return recall, spill


def validate(rows: pd.DataFrame) -> None:
    slices = pd.read_parquet(ROOT / "labels/rsna_slices/rsna_slices.parquet").set_index("volume_id")
    res = []
    for r in rows.itertuples():
        c = candidates(r, slices) if str(r.volume_id) in slices.index else None
        if c is None:
            continue
        st = c.pop("_state")
        man = np.isin(store.read_image(ROOT / str(r.manual), "mask"), [int(i) for i in r.manual_ids.split(",")])
        if man.shape != st.shape:
            continue
        row = {"volume_id": r.volume_id, "manual_ml": man.sum() / 1000}
        for k, m in c.items():
            row[f"dice_{k}"] = dice(m, man, empty=1.0)
            row[f"ml_{k}"] = m.sum() / 1000
        row["recall_union"], row["spill_union"] = slice_qc(c["union"], st)
        row["dice_ts_model"] = dice(c["ts"], c["model"], empty=1.0)
        res.append(row)
    df = pd.DataFrame(res)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "validate.parquet")
    keys = [k for k in df.columns if k.startswith("dice_")]
    print(f"[validate] {len(df)} studies with a manual mask; mean / median Dice against it:")
    print(df[keys].agg(["mean", "median"]).T.round(3).to_string())


def fuse(rows: pd.DataFrame, min_agree: float, max_spill: float) -> None:
    from brain.downstream.tasks import build_rsna_volume_labels

    slices = pd.read_parquet(ROOT / "labels/rsna_slices/rsna_slices.parquet").set_index("volume_id")
    lab = build_rsna_volume_labels().set_index("volume_id")
    others = [t for t in ("subdural", "subarachnoid", "epidural") if t in lab]
    res = []
    for r in rows.itertuples():
        c = candidates(r, slices) if str(r.volume_id) in slices.index else None
        if c is None:
            res.append({"volume_id": r.volume_id, "status": "missing"})
            continue
        st, m = c["_state"], c["union"]
        recall, spill = slice_qc(m, st)
        agree = dice(c["ts"], c["model"], empty=1.0)
        mixed = bool(str(r.volume_id) in lab.index and (lab.loc[str(r.volume_id), others] > 0).any())
        ok = agree >= min_agree and spill <= max_spill and m.sum() >= 100
        if ok:
            out = m.astype(np.uint8)
            if mixed:                              # blood the tools may have missed: out of the loss, not background
                out[(st == 1) & ~m] = 2
            store.write(out_path(r, "bleedpl"), "mask", out, {"agree": agree, "spill": spill, "mixed": mixed})
        res.append({"volume_id": r.volume_id, "status": "ok" if ok else "rejected", "mixed": mixed,
                    "recall": recall, "spill": spill, "ml": m.sum() / 1000, "agree": agree})
    df = pd.DataFrame(res)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "fuse.parquet")
    print(f"[fuse] {df.groupby('mixed').status.value_counts().to_dict()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["ts", "ts-verify", "model", "validate", "fuse"])
    ap.add_argument("--set", default="train", choices=["train", "val"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ckpt", type=Path, default=Path("runs/single/hemorrhage_r13/last.pt"))
    ap.add_argument("--min-agree", type=float, default=0.5)
    ap.add_argument("--max-spill", type=float, default=0.3)
    ap.add_argument("--overwrite", action="store_true", help="ts: redo masks already written")
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    rows = targets("val" if a.step == "validate" else a.set).iloc[k::n]
    if a.step == "ts":
        run_ts(rows, overwrite=a.overwrite)
    elif a.step == "ts-verify":
        ts_verify(rows)
    elif a.step == "model":
        run_model(rows, a.ckpt)
    elif a.step == "validate":
        validate(rows)
    else:
        fuse(rows, a.min_agree, a.max_spill)


if __name__ == "__main__":
    main()
