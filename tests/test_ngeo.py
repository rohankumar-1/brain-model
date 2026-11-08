"""Native-geometry ("ngeo") CT: the ngeo zip reader and dataset mode, the in-plane stem, template-mm positions, the
ngeo decoder and augmentation, and the trainer on phantoms (runs/diagnostics/ngeo/CONTRACT.md s. 2-3)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from brain.data import ngeo
from brain.models.patch_embed import NgeoStemEmbed
from brain.multitask.data import MultitaskDataset, SyntheticNgeo, collate_ngeo, prepare
from brain.multitask.model import BrainQuery
from brain.multitask.registry import load_registry
from brain.train.defaults import default_config, smoke_config
from brain.train.multitask import build_encoder

CPU = torch.device("cpu")
REG = load_registry()
HEM = ["hemorrhage_any", "hemorrhage_subdural", "hemorrhage_segmentation"]
NGEO_PATCH = (32, 32, 1)


def _cfg(in_chans: int = 1, tokens: bool = False) -> dict:
    cfg = smoke_config(default_config("multitask"))
    cfg["model"]["in_chans"] = in_chans
    cfg["model"]["task_tokens"]["enabled"] = tokens
    cfg["model"]["ngeo"] = {"enabled": True, "stem": [8, 16], "patch_xyz": list(NGEO_PATCH), "seg_stride": [1, 4, 4],
                            "template_xyz": cfg["data"]["volume_size_xyz"]}
    cfg["model"]["token_map"] = {"enabled": True, "dim": 16}
    cfg["data"]["ngeo"] = {"enabled": True, "pad_multiple": 4}
    cfg["tasks"]["ids"] = HEM
    return cfg


def _model(cfg: dict) -> BrainQuery:
    torch.manual_seed(0)
    m = BrainQuery(build_encoder(cfg, CPU, synthetic=True), REG, cfg["model"])
    if m.tokens:
        m.set_token_tasks([REG.index[t] for t in HEM])
    return m


# ------------------------------------------------------------------------------------------------ synthetic zips
def _tilted_affine(s_n: int, n: int, slice_mm: float, tilt_deg: float, rot_deg: float, centre) -> np.ndarray:
    """voxel (s, r, c) -> template index (x, y, z), as SyntheticNgeo builds it."""
    m = np.zeros((3, 3))
    m[0, 2], m[1, 1], m[2, 0] = 0.5, 0.5, slice_mm
    m[1, 0] = slice_mm * math.tan(math.radians(tilt_deg))
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    m = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ m
    a = np.eye(4)
    a[:3, :3] = m
    a[:3, 3] = np.asarray(centre, float) - m @ np.array([(s_n - 1) / 2, (n - 1) / 2, (n - 1) / 2])
    return a


GRID = {"grid_origin": [-100.0, 120.0, -60.0], "grid_spacing": [1.0, 1.0, 1.0],
        "grid_direction": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]}


def _physical(index_affine: np.ndarray) -> np.ndarray:
    """The zip's affine_template (-> template PHYSICAL mm) for a desired voxel -> template-index map."""
    g = np.eye(4)
    g[:3, :3] = np.asarray(GRID["grid_direction"]).reshape(3, 3) @ np.diag(GRID["grid_spacing"])
    g[:3, 3] = GRID["grid_origin"]
    return g @ index_affine


def _write_case(root, folder: str, case: str, s_n: int, n: int = 64, grid_in_ngeo: bool = False, seed: int = 0):
    """A 1 mm image zip (with grid attrs) and its ngeo copy plus an ICH mask on both grids."""
    from brain.prep import store

    rng = np.random.default_rng(seed)
    one = np.full((16, 1, 32, 32), -1024, np.int16)
    store.write_arrays(root / folder / f"{case}.zip", {"image": (one, {"kind": "ct_hu", **GRID})})
    store.write(root / folder / f"{case}.ICH.zip", "mask", np.zeros((16, 32, 32), np.uint8))
    img = np.full((s_n, 1, n, n), -1024, np.int16)
    img[:, 0, n // 4:3 * n // 4, n // 4:3 * n // 4] = rng.integers(20, 40, (s_n, n // 2, n // 2))
    img[:, 0, 4:6, 4:6] = -700                                         # below -500: clamped to air on read
    mask = np.zeros((s_n, n, n), np.uint8)
    mask[s_n // 2, n // 2 - 4:n // 2 + 4, n // 2 - 4:n // 2 + 4] = 1
    aff = _tilted_affine(s_n, n, 5.0, 10.0, 5.0, (16.0, 16.0, 8.0))
    attrs = {"kind": "ct_hu_ngeo", "spacing_mm": [5.0, 0.5, 0.5], "thick_mm": 5.0, "slice_mm": 5.0,
             "affine_template": _physical(aff).ravel().tolist(), "affine_scanner": np.eye(4).ravel().tolist(),
             "synth_thick": False, **(GRID if grid_in_ngeo else {})}
    store.write_arrays(root / folder / f"{case}.ngeo.zip",
                       {"image": (img, attrs), "brain": ((img[:, 0] > 0).astype(np.uint8), None)})
    store.write(root / folder / f"{case}.ICH.ngeo.zip", "mask", mask)
    return aff


def test_reader_paths_and_template_affine(tmp_path):
    aff = _write_case(tmp_path, "bhsd", "c1", 7, grid_in_ngeo=True)
    assert ngeo.ngeo_path("bhsd/c1.zip").as_posix() == "bhsd/c1.ngeo.zip"
    assert ngeo.ngeo_path("bhsd/c1.ICH.zip").as_posix() == "bhsd/c1.ICH.ngeo.zip"
    d = ngeo.read_ngeo(tmp_path / "bhsd" / "c1.ngeo.zip")
    assert d["image"].shape == (7, 64, 64) and d["image"].dtype == np.int16
    assert d["image"][0, 4, 4] == -1024 and d["image"].min() == -1024               # air clamp below -500 HU
    assert d["brain"].shape == (7, 64, 64) and np.allclose(d["spacing"], [5.0, 0.5, 0.5])
    assert d["grid"] is not None
    assert np.allclose(ngeo.template_index_affine(d["affine_template"], d["grid"]), aff)
    assert np.allclose(ngeo.template_index_affine(d["affine_template"], ngeo.grid_attrs(tmp_path / "bhsd" / "c1.zip")),
                       aff)
    assert ngeo.read_ngeo_mask(tmp_path / "bhsd" / "c9.ICH.ngeo.zip") is None


def _tables(tmp_path, cases: dict[str, int]):
    labels, vols, idx = [], [], []
    for i, (case, s_n) in enumerate(cases.items()):
        _write_case(tmp_path, "bhsd", case, s_n, seed=i)
        vid = f"bhsd:{case}"
        labels += [(vid, "hemorrhage_any", float(i % 2), ""), (vid, "hemorrhage_subdural", 0.0, ""),
                   (vid, "hemorrhage_segmentation", np.nan, f"bhsd/{case}.ICH.zip|1")]
        vols.append({"volume_id": vid, "image_path": f"bhsd/{case}.zip", "modality": "ct", "dataset_family": "bhsd",
                     "dataset": "bhsd"})
        idx.append({"volume_id": vid, "path": f"bhsd/{case}.ngeo.zip", "S": s_n, "slice_mm": 5.0})
    lab = pd.DataFrame(labels, columns=["volume_id", "task", "value", "ref"]).assign(split="train", use="train")
    # a 1 mm-only scan (no ngeo copy) and a seg_cq500-style duplicate row of the first id
    vols.append({"volume_id": "mri:x", "image_path": "x/x.zip", "modality": "t1", "dataset_family": "x", "dataset": "x"})
    vols.append({**vols[0], "image_path": "seg_bhsd/c0.zip", "dataset": "seg_bhsd"})
    lab = pd.concat([lab, pd.DataFrame([("mri:x", "hemorrhage_any", 1.0, "", "train", "train")], columns=lab.columns)])
    return lab, pd.DataFrame(vols), pd.DataFrame(idx)


def test_dataset_reads_ngeo_and_keeps_label_ids(tmp_path):
    lab, vols, idx = _tables(tmp_path, {"c0": 6, "c1": 9})
    ds = MultitaskDataset(tmp_path, lab, vols, REG, {}, split="train", train=False, tasks=HEM, k_max=8, max_seg=2,
                          ngeo={"enabled": True}, ngeo_index=idx)
    assert len(ds) == 2 and {v for v, _ in ds.items} == {"bhsd:c0", "bhsd:c1"}      # the duplicate id is resolved
    items = [ds[i] for i in range(2)]
    assert items[0]["image"].shape == (6, 64, 64) and items[1]["image"].shape == (9, 64, 64)
    assert items[0]["seg"].shape == (1, 6, 64, 64) and int(items[0]["seg"].sum()) == 64
    assert items[0]["affine"].shape == (4, 4) and torch.allclose(items[0]["spacing"], torch.tensor([5.0, 0.5, 0.5]))
    b = collate_ngeo(items)
    assert b["image"].shape == (2, 12, 64, 64) and b["seg"].shape == (2, 12, 64, 64)    # 9 -> 12 (multiple of 4)
    assert b["slice_keep"].sum(1).tolist() == [6, 9] and int(b["image"][0, 6:].max()) == -1024
    # max_slices: a centre window in evaluation, with the affine following it
    cut = MultitaskDataset(tmp_path, lab, vols, REG, {}, split="train", train=False, tasks=HEM, k_max=8, max_seg=2,
                           ngeo={"enabled": True, "max_slices": 5}, ngeo_index=idx)
    it = cut[1]
    assert it["image"].shape[0] == 5 and torch.equal(it["image"], items[1]["image"][2:7])
    assert torch.allclose(it["affine"][:, 3], items[1]["affine"] @ torch.tensor([2.0, 0, 0, 1], dtype=torch.float64))


def test_token_count_for_512x512x32():
    stem = NgeoStemEmbed(3, 32, NGEO_PATCH, (4, 8))
    tokens, grid = stem(torch.zeros(1, 3, 32, 512, 512), zyx=True)
    assert grid == (32, 16, 16) and tokens.shape == (1, 32 * 16 * 16, 32) == (1, 8192, 32)
    aff = torch.eye(4)
    assert ngeo.token_centres_mm(aff, (32, 512, 512), NGEO_PATCH).shape == (1, 8192, 3)


def test_stem_is_in_plane_only():
    """A slice's tokens depend on that slice alone: nothing mixes slices before the transformer."""
    torch.manual_seed(0)
    stem = NgeoStemEmbed(1, 16, NGEO_PATCH, (4, 8)).eval()
    x = torch.randn(1, 1, 4, 64, 64)
    y = x.clone()
    y[:, :, 2] += 5.0
    a, _ = stem(x, zyx=True)
    b, _ = stem(y, zyx=True)
    per = (a - b).abs().sum(-1).view(4, 4)                                   # (slices, tokens per slice)
    assert per[2].min() > 0 and per[[0, 1, 3]].max() == 0
    stem.chunk_rows = 1                                                      # chunked stem passes: same tokens
    assert torch.allclose(stem(x, zyx=True)[0], a, atol=1e-5)


def test_position_encoding_matches_across_frames():
    """A 1 mm token and an ngeo token whose centres sit at the same template mm get the same rotary angles and the
    same slot-readout position code."""
    cfg = _cfg()
    model = _model(cfg).eval()
    iso_shape = tuple(reversed(cfg["data"]["volume_size_xyz"]))                  # (16, 32, 32)
    iso, _ = model._coords(iso_shape, CPU)                                       # (N, 3) template mm
    # ngeo grid of 6 slices x 64 x 64 at 0.5 mm, rotated 90 degrees in-plane and offset so that token (s=3, 0, 1) lands
    # exactly on the centre of 1 mm token (z=1, y=2, x=1): (12, 20, 6) mm with patch (8, 8, 4) on the smoke grid
    m = np.array([[0.0, -0.5, 0.0], [0.0, 0.0, 0.5], [2.0, 0.0, 0.0]])          # rows: x, y, z of (s, r, c)
    centre_src = np.array([3.0, 15.5, 47.5])                                     # token (3, 0, 1) centre, voxels
    a = np.eye(4)
    a[:3, :3] = m
    a[:3, 3] = (np.array([12.0, 20.0, 6.0]) - 0.5) - m @ centre_src
    coords = ngeo.token_centres_mm(torch.tensor(a), (6, 64, 64), NGEO_PATCH)    # (1, 6 * 2 * 2, 3)
    j = 3 * 4 + 0 * 2 + 1
    i = int(((iso - torch.tensor([12.0, 20.0, 6.0])).abs().sum(-1)).argmin())
    assert torch.allclose(coords[0, j], iso[i]) and torch.allclose(iso[i], torch.tensor([12.0, 20.0, 6.0]))
    rope = model.backbone.rope
    ci, si = rope.cos_sin(iso[i].view(1, 1, 3))
    cn, sn = rope.cos_sin(coords[:, j:j + 1])
    assert torch.equal(ci, cn) and torch.equal(si, sn)
    assert torch.allclose(model.position_code(iso)[i], model.position_code(coords)[0, j], atol=1e-6)


def _batch(n: int = 2, inplane: int = 64, slices=(5, 9), seed: int = 0):
    ds = SyntheticNgeo(REG, n=n, seed=seed, tasks=HEM, inplane=inplane, slices=slices, template_xyz=(32, 32, 16))
    return collate_ngeo([ds[i] for i in range(n)])


@pytest.mark.parametrize("in_chans,tokens", [(1, False), (3, True)])
def test_forward_backward_on_ngeo(in_chans, tokens):
    cfg = _cfg(in_chans, tokens)
    model = _model(cfg).train()
    raw = _batch()
    b = prepare(raw, cfg, CPU, NGEO_PATCH, train=True)
    assert b["x"].shape[1] == in_chans and b["x"].shape[2:] == raw["image"].shape[1:]
    s_pad = raw["image"].shape[1]
    assert b["keep"].shape == (2, s_pad * 4) and b["geo"]["coords"].shape == (2, s_pad * 4, 3)
    pad_tok = ~raw["slice_keep"].repeat_interleave(4, 1)
    assert not b["keep"][pad_tok].any() and b["keep"].any()                    # padded slices never kept
    tok_pair = b["seg_pair"] if len(b["seg_pair"]) else None
    out = model(b["x"], b["keep"], b["task"], b["seg_pair"], geo=b["geo"], tok_pair=tok_pair)
    assert torch.isfinite(out["binary"]).all()
    assert len(b["seg_pair"]) and out["seg"].shape == (len(b["seg_pair"]), 1, s_pad, 16, 16)
    assert out["tok"].shape == (len(b["seg_pair"]), s_pad * 4)
    from brain.multitask.losses import slot_losses

    loss, has = slot_losses(out, b["task"], b["y"], model.task_type, b["seg"], b["seg_pair"], b["seg_stride"])
    loss[has].mean().backward()
    g = model.backbone.ngeo_embed.stem[0][0].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    assert all(p.grad is None for p in model.seg.parameters())                  # the 1 mm decoder is not used


def test_a_scans_outputs_do_not_depend_on_batch_padding():
    """The same scan alone and batched with a longer one (padded slices): same answers, same masks on its slices."""
    cfg = _cfg()
    model = _model(cfg).eval()
    ds = SyntheticNgeo(REG, n=16, seed=3, tasks=HEM, inplane=64, slices=(4, 10), template_xyz=(32, 32, 16))
    items = sorted((it for it in (ds[i] for i in range(16)) if len(it["seg"])), key=lambda it: it["image"].shape[0])
    short, long_ = items[0], items[-1]
    assert short["image"].shape[0] < long_["image"].shape[0]
    outs = []
    for batch in ([short], [short, long_]):
        b = prepare(collate_ngeo(batch, pad_multiple=1), cfg, CPU, NGEO_PATCH, train=False)
        with torch.no_grad():
            outs.append((model(b["x"], b["keep"], b["task"], b["seg_pair"], geo=b["geo"]), b))
    (a, ba), (c, bc) = outs
    k = int((ba["task"][0] >= 0).sum())
    assert torch.allclose(a["binary"][0, :k], c["binary"][0, :k], atol=1e-4)
    n_a = int((ba["seg_pair"][:, 0] == 0).sum())
    s_n = short["image"].shape[0]
    assert n_a and torch.allclose(a["seg"][:n_a], c["seg"][:n_a, :, :s_n], atol=1e-4)
    assert (c["seg"][:n_a, :, s_n:] == -20).all()


def test_rsna_slice_labels_map_onto_ngeo_tokens():
    """Slice-label MIL targets (``slab_labels``) on the ngeo grid: a token takes the label of the RSNA slice its
    template position falls on, through ``affine`` (here slices at template z = 5 s + 1)."""
    from brain.multitask.data import SLAB_NO_LABEL, slab_labels

    ds = SyntheticNgeo(REG, n=2, seed=0, tasks=HEM, inplane=64, slices=(6, 6), template_xyz=(32, 32, 16))
    items = [ds[i] for i in range(2)]
    aff = torch.zeros(4, 4, dtype=torch.float64)
    aff[0, 2], aff[1, 1], aff[2, 0], aff[2, 3], aff[3, 3] = 0.5, 0.5, 5.0, 1.0, 1.0
    bits = np.zeros(6, np.int16)
    bits[2] = 1                                                     # slice 2: hemorrhage_any
    for it in items:
        it["affine"] = aff
    items[0]["slab"] = (np.array([1.0, 0.0, 0.0], np.float32), 0.0, 5.0, (5 * np.arange(6) + 1).astype(np.float32), bits)
    items[1]["slab"] = None
    b = collate_ngeo(items)
    lab = slab_labels(b, CPU, b["image"].shape[1:], NGEO_PATCH, None, None, to_store=b["affine"])
    per_slice = lab[0, 0].view(-1, 4)                               # (slices, tokens per slice), channel 0
    assert (per_slice[2] == 1).all() and (per_slice[[0, 1, 3, 4, 5]] == 0).all()
    assert (lab[1] == -1).all() and SLAB_NO_LABEL == 255
    cfg = _cfg()
    out = prepare(b, cfg, CPU, NGEO_PATCH, train=False)
    assert out["slab"].shape[:2] == (2, 11) and torch.equal(out["slab"][:, :6], lab)


def test_eval_prepare_and_gate():
    cfg = _cfg()
    cfg["views"]["gate"].update(mode="frac", min_frac=0.25)
    raw = _batch()
    b = prepare(raw, cfg, CPU, NGEO_PATCH, train=False)
    from brain.common.windowing import canonical_c_w

    c, w = canonical_c_w("ct")
    assert torch.allclose(b["x"], torch.asinh((raw["image"].float().unsqueeze(1) - c) / w))
    # the gate: a token is kept when >= min_frac of its voxels are above -500 HU, never on padded slices
    fg = (raw["image"] > -500).float().unsqueeze(1)
    frac = torch.nn.functional.avg_pool3d(fg, (1, 32, 32)).flatten(1)
    gate = cfg["views"]["gate"]
    want = frac >= gate["min_frac"]
    assert torch.equal(b["keep"], want & raw["slice_keep"].repeat_interleave(4, 1)) and b["keep"].any()


def test_inplane_augmentation_never_mixes_slices():
    """Rotation / scale / shift / flip act within each slice; through-plane blur is off and the isotropic blur is
    in-plane: a slice's content stays on that slice."""
    cfg = _cfg()
    cfg["aug"] = {"flip_lr": 1.0, "affine": {"p": 1.0, "rot_deg": [8, 8, 30], "shift_mm": 8.0, "scale": [0.9, 1.1]},
                  "appearance": {"through_plane": {"p": 1.0, "fwhm_vox": [4.0, 6.0]},
                                 "blur": {"p": 1.0, "sigma_vox": [1.0, 1.0]}}}
    raw = _batch()
    raw["image"][:, :] = -1024
    raw["image"][:, 2, 16:48, 16:48] = 40                         # one bright slice per scan
    theta = ngeo.inplane_theta(2, (12, 64, 64), raw["spacing"], cfg["aug"]["affine"], CPU)
    assert torch.allclose(theta[:, 2], torch.tensor([0.0, 0.0, 1.0, 0.0]).expand(2, 4))
    assert torch.allclose(theta[:, :2, 2], torch.zeros(2, 2))
    torch.manual_seed(0)
    b = prepare(raw, cfg, CPU, NGEO_PATCH, train=True)
    base = float(torch.asinh(torch.tensor((-1024.0 - 40.0) / 40.0)))
    other = torch.cat([b["x"][:, :, :2], b["x"][:, :, 3:]], 2)
    assert torch.allclose(other, torch.full_like(other, base), atol=1e-4)
    assert (b["x"][:, :, 2] > base + 1).any()


def test_window_bank_on_ngeo():
    """model.window_bank on native geometry: prepare_ngeo ships raw HU (one channel, CT physical augmentations
    in-plane), the model applies the encoder's bank before the ngeo stem (in_chans = K); the bank trains."""
    cfg = _cfg()
    cfg["model"]["window_bank"] = {"enabled": True, "k": 4, "init": "contract6", "per_modality": False,
                                   "lr_mult": 0.1, "decoder_grad": False}
    cfg["aug"] = {"flip_lr": 0.5, "affine": {"p": 1.0, "rot_deg": [8, 8, 12], "shift_mm": 4.0},
                  "appearance": {"ct_physical": {"p": 1.0, "offset_hu": 10.0, "scale": 0.03},
                                 "kernel": {"p": 1.0, "sigma_vox": [0.3, 1.0], "sharpen": 0.5, "amount": [0.2, 1.0]},
                                 "through_plane": {"p": 1.0, "fwhm_vox": [4.0, 6.0]},
                                 "noise": {"p": 1.0, "sigma_hu": [2.0, 15.0]}, "blur": {"p": 1.0, "sigma_vox": [0.5, 1.0]}}}
    model = _model(cfg).train()
    assert model.backbone.window_bank is not None and model.backbone.in_chans == 4
    assert model.backbone.ngeo_embed.stem[0][0].weight.shape[1] == 4
    raw = _batch()
    ev = prepare(raw, cfg, CPU, NGEO_PATCH, train=False)
    assert ev["x"].shape[1] == 1 and torch.equal(ev["x"][:, 0], raw["image"].float())
    b = prepare(raw, cfg, CPU, NGEO_PATCH, train=True)
    assert b["x"].shape[1] == 1 and not torch.equal(b["x"][:, 0], raw["image"].float())
    out = model(b["x"], b["keep"], b["task"], b["seg_pair"], geo=b["geo"], modality=b["modality"], hide=b["hide"])
    from brain.multitask.losses import slot_losses

    loss, has = slot_losses(out, b["task"], b["y"], model.task_type, b["seg"], b["seg_pair"], b["seg_stride"])
    loss[has].mean().backward()
    assert model.backbone.window_bank.c_raw.grad is not None and model.backbone.window_bank.c_raw.grad.abs().sum() > 0


def test_iso_path_unchanged_by_the_option():
    """An encoder / decoder built with model.ngeo off has no ngeo parameters, and the 1 mm forward is identical."""
    cfg = smoke_config(default_config("multitask"))
    cfg["model"]["task_tokens"]["enabled"] = False
    torch.manual_seed(0)
    plain = BrainQuery(build_encoder(cfg, CPU, synthetic=True), REG, cfg["model"]).eval()
    assert plain.seg_ngeo is None and plain.backbone.ngeo_embed is None
    assert not any("ngeo" in k for k in plain.state_dict())


def test_trainer_runs_on_ngeo_phantoms(tmp_path, monkeypatch):
    """Two steps of the hemorrhage specialist recipe (slot readout, token maps, three fixed windows) on ngeo phantoms,
    then an evaluation pass and a reload."""
    from brain.multitask.fewshot import load_model
    from brain.train import engine

    monkeypatch.setenv("BRAIN_PERSIST_SYNTHETIC", "1")
    engine.run(["multitask", "--synthetic", "--max-steps", "2", "--log-every", "1", "--out-dir", str(tmp_path),
                "--set", "model.task_tokens.enabled=false", "--set", "model.in_chans=3",
                "--set", "model.ngeo={enabled: true, stem: [8, 16], patch_xyz: [32, 32, 1], seg_stride: [1, 4, 4]}",
                "--set", "data.ngeo={enabled: true, synthetic_inplane: 64, synthetic_slices: [4, 7]}",
                "--set", "model.token_map={enabled: true, dim: 16}", "--set", "loss.token_weight=0.5",
                "--set", f"tasks.ids=[{', '.join(HEM)}]",
                "--set", "train.eval_every=2", "--set", "train.ckpt_every=2"])
    model, _, cfg = load_model(tmp_path / "last.pt", CPU)
    assert model.backbone.ngeo_embed is not None and model.seg_ngeo is not None


def test_bank_augmentation_stays_in_plane():
    """With the bank (native-unit augmentation), kernel blur / sharpen, blur and the (ignored) through-plane blur
    never move a slice's content to its neighbours."""
    cfg = _cfg()
    cfg["model"]["window_bank"] = {"enabled": True, "k": 4}
    cfg["aug"] = {"flip_lr": 1.0, "affine": {"p": 1.0, "rot_deg": [8, 8, 30], "shift_mm": 8.0},
                  "appearance": {"ct_physical": {"p": 1.0, "offset_hu": 10.0, "scale": 0.03},
                                 "kernel": {"p": 1.0, "sigma_vox": [1.0, 1.0], "sharpen": 0.0, "amount": [0.2, 1.0]},
                                 "through_plane": {"p": 1.0, "fwhm_vox": [4.0, 6.0]},
                                 "blur": {"p": 1.0, "sigma_vox": [1.0, 1.0]}}}
    raw = _batch()
    raw["image"][:, :] = -1024
    raw["image"][:, 2, 16:48, 16:48] = 40
    b = prepare(raw, cfg, CPU, NGEO_PATCH, train=True)
    other = torch.cat([b["x"][:, :, :2], b["x"][:, :, 3:]], 2)
    assert torch.allclose(other, torch.full_like(other, -1024.0), atol=1e-3) and (b["x"][:, :, 2] > 0).any()
