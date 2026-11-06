"""R14-F real-data check: one BraTS scan with its partners through dataset -> collate -> prepare -> BrainQuery (R14 +
fusion). The fused forward runs, and the same scan alone gives R14's unfused outputs exactly."""
import numpy as np, pandas as pd, torch, yaml
from brain.multitask.data import MultitaskDataset, collate, load_labels, prepare, study_partners
from brain.multitask.registry import load_registry
from brain.multitask.model import BrainQuery
from brain.train.multitask import build_encoder

cfg = yaml.safe_load(open("configs/multitask/r14f.yaml"))
from brain.train.defaults import default_config, _merge
full = _merge(default_config("multitask"), cfg)
full["model"]["init_from"] = "runs/multitask/r14/last.pt"
STORE = full["data"]["store_root"]
lab, stats = load_labels(full["data"]["labels_dir"])
vol = pd.read_csv(f"{STORE}/volumes.csv", low_memory=False)
reg = load_registry(None)
part = study_partners(full["data"]["labels_dir"], vol, set(lab.volume_id.astype(str)))
anchor = next(v for v in part if "BraTS-GLI" in v and v.endswith("t2f"))
split = str(vol.drop_duplicates("volume_id").set_index("volume_id").at[anchor, "split"])
ds = MultitaskDataset(STORE, lab, vol, reg, stats, split=split, train=False, tasks=["whole_tumor", "tumor_type"], k_max=4,
                      max_seg=2, volume_ids=[anchor], use=None, study={"eval_p": 1.0, "max_scans": 3}, partners=part)
item = ds[0]
print("anchor", anchor.split(":")[0], "partners loaded", 0 if item["study_img"] is None else len(item["study_img"]))
b = prepare(collate([item]), full, torch.device("cpu"), (16, 16, 8), train=False)
print("study x", tuple(b["study"]["x"].shape), "seq", b["study"]["seq"].tolist(), "pseq", b["study"]["pseq"].tolist())
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = build_encoder(full, dev, False)
m = BrainQuery(enc, reg, full["model"]).to(dev).eval()
ck = torch.load("runs/multitask/r14/last.pt", map_location="cpu", weights_only=False)
from brain.train.multitask import remap_tasks
sd = remap_tasks(ck["multitask"], list(ck["tasks"]), [t.id for t in reg.tasks], {k: v.cpu() for k, v in m.state_dict().items()})
missing, unexpected = m.load_state_dict(sd, strict=False)
print("missing beyond fusion/new tasks:", [k for k in missing if "fusion" not in k][:5], "unexpected:", unexpected[:3])
m.set_token_tasks([reg.index[t] for t in reg.train_ids(full["tasks"])])
mv = lambda d: {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
st = mv(b["study"])
with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
    fused = m(b["x"].to(dev), b["keep"].to(dev), b["task"].to(dev), b["seg_pair"].to(dev), study=st)
    alone = m(b["x"].to(dev), b["keep"].to(dev), b["task"].to(dev), b["seg_pair"].to(dev))
k = [x for x in fused if x not in ("seg", "dense", "tok") and torch.is_tensor(fused[x])][0]
print("fused vs alone differ (fusion o-proj is zero at init, mean of scans):", not torch.allclose(fused[k], alone[k]))
print("seg map", tuple(fused["seg"].shape), "finite", bool(torch.isfinite(fused["seg"]).all()))
