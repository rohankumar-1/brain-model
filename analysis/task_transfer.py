"""Task-to-task usable information: condition the encoder on task A (its task token's output z_A), fit a probe from
z_A to task B's label, and score it on B's held-out scans. The probe's held-out cross-entropy bounds the mutual
information from below, I(z_A; Y_B) >= H(Y_B) - CE (usable information, Xu et al. 2020), so the A -> B matrix says how
much of B the representation conditioned on A carries.

    collect (GPU): every task token's output on up to --n-train train + --n-test test scans of each scalar task, in two
        encoder modes: "write" (as trained: patch tokens attend every task token, so information can route A ->
        patches -> B) and "readonly" (no task token writes into the patches: the encoder-only pathway); plus the
        mean-pooled patch tokens (a ceiling: everything the encoder has).
    probe (CPU): per (source, target) a linear probe on 64 principal components (fitted on train), L2 strength by
        3-fold CV on train; usable bits on test. Also conditional bits: what [z_A, z_B] adds over z_B alone.
    plot: heat maps, sources grouped by task group.

    python analysis/task_transfer.py collect --ckpt runs/multitask/full_iso_bank/last.pt --out runs/diagnostics/transfer_iso
    python analysis/task_transfer.py probe --out runs/diagnostics/transfer_iso
    python analysis/task_transfer.py plot --out runs/diagnostics/transfer_iso

Regression bits use the Gaussian channel, -0.5 log2(1 - R^2) (R^2 <= 0 -> 0 bits)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCALAR = ("binary", "regression", "multiclass")


def collect(a) -> None:
    import pandas as pd

    from brain.common.paths import resolve
    from brain.multitask.data import (MultitaskDataset, checkpoint_stats, load_labels, make_loader, mixed_parts,
                                      ngeo_setup, prepare)
    from brain.multitask.fewshot import load_model

    dev = torch.device("cuda")
    amp = dict(device_type="cuda", dtype=torch.bfloat16)
    model, reg, cfg = load_model(Path(a.ckpt), dev)
    model.eval()
    if not model.tokens:
        raise SystemExit("needs a task-token model")
    d = cfg["data"]
    store = resolve(d["store_root"])
    labels_dir = resolve(d["labels_dir"])
    labels, stats = load_labels(labels_dir)
    stats = checkpoint_stats(model.label_stats, labels_dir, stats)
    volumes = pd.read_csv(store / d.get("index_name", "volumes.csv"), low_memory=False)
    from brain.multitask.data import drop_datasets, drop_volumes
    labels = drop_volumes(drop_datasets(labels, volumes, d.get("exclude_datasets")), d.get("exclude_volumes"))

    trained = set(reg.train_ids(cfg["tasks"]))
    targets = [t.id for t in reg.tasks if t.id in trained and t.type in SCALAR
               and not any(str(s.get("source", "")).startswith("ssl") for s in t.sources)]
    tok_of = model.tok_of.tolist()
    tok_ids = model.tt_write.nonzero()[:, 0].tolist()
    rep = [next(i for i in range(len(reg)) if tok_of[i] == t) for t in tok_ids]     # one task asks each token
    members = {reg[r].id: [reg[i].id for i in range(len(reg)) if tok_of[i] == t and reg[i].id in trained]
               for t, r in zip(tok_ids, rep)}
    lab = labels[labels["use"].eq("train") & labels["task"].isin(targets)]
    rng = np.random.default_rng(0)
    picked = {}
    for split, n in (("train", a.n_train), ("test", a.n_test)):
        ids = set()
        for t, g in lab[lab.split == split].groupby("task"):
            u = g["volume_id"].unique()
            ids.update(u[rng.permutation(len(u))[:n]].tolist())
        picked[split] = sorted(ids)
        print(f"[transfer] {split}: {len(ids)} scans for {len(targets)} targets", flush=True)

    ngeo_kw, cf = ngeo_setup(cfg, store)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write0 = model.tt_write.clone()
    for split, ids in picked.items():
        ds = MultitaskDataset(store, labels, volumes, reg, stats, split=split, tasks=targets, train=False,
                              use="train", volume_ids=ids, max_seg=0, k_max=128, **ngeo_kw)
        loader = make_loader(ds, batch_size=a.batch, shuffle=False, seed=0, workers=a.workers, drop_last=False,
                             collate_fn=cf)
        feats = {"write": [], "readonly": []}
        pooled, vids = [], []
        for raw in (p for full in loader for p in mixed_parts(full)):
            ngeo = "affine" in raw
            patch = model.backbone.ngeo_patch_xyz if ngeo else model.backbone.patch_xyz
            b = prepare(raw, cfg, dev, patch, train=False)
            B = b["x"].shape[0]
            task = torch.tensor([rep] * B, device=dev)
            geo = {"geo": b["geo"]} if b.get("geo") is not None else {}
            for mode in ("write", "readonly"):
                model.tt_write.copy_(write0 if mode == "write" else torch.zeros_like(write0))
                with torch.no_grad(), torch.autocast(**amp):
                    enc = model.encode(b["x"], b["keep"], b.get("meta"), task, modality=b["modality"], **geo)
                    z = model.token_slots(enc, task)                                   # (B, T, width)
                feats[mode].append(z.float().cpu().half())
                if mode == "readonly":                       # patch tokens do not depend on read-only task tokens
                    k = b["keep"].unsqueeze(-1).float()
                    pooled.append(((enc["tokens"].float() * k).sum(1) / k.sum(1).clamp_min(1)).cpu().half())
            model.tt_write.copy_(write0)
            vids += [str(v) for v in raw["volume_id"]]
            if len(vids) % 500 < B:
                print(f"[transfer] {split}: {len(vids)}/{len(ids)}", flush=True)
        torch.save({"vids": vids, "write": torch.cat(feats["write"]), "readonly": torch.cat(feats["readonly"]),
                    "pooled": torch.cat(pooled)}, out / f"feats_{split}.pt")
    y = lab[lab.volume_id.isin(set(picked["train"]) | set(picked["test"]))][["volume_id", "task", "split", "value"]]
    y.to_parquet(out / "targets.parquet")
    meta = {"ckpt": str(a.ckpt), "tokens": [reg[r].id for r in rep], "members": members,
            "groups": [reg[r].group for r in rep], "targets": targets,
            "target_type": {t: reg[t].type for t in targets}, "target_group": {t: reg[t].group for t in targets}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out, flush=True)


# ------------------------------------------------------------------------------------------------------- probe
def _bits_classifier(xtr, ytr, xte, yte, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import log_loss

    classes = np.unique(ytr)
    if len(classes) < 2:
        return np.nan, np.nan
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    g = GridSearchCV(LogisticRegression(max_iter=2000), {"C": np.logspace(-4, 1, 8)}, cv=cv, scoring="neg_log_loss")
    g.fit(xtr, ytr)
    keep = np.isin(yte, classes)
    p = g.predict_proba(xte[keep])
    prior = np.array([(ytr == c).mean() for c in classes])
    h = log_loss(yte[keep], np.tile(prior, (keep.sum(), 1)), labels=classes) / np.log(2)
    ce = log_loss(yte[keep], p, labels=classes) / np.log(2)
    if len(classes) == 2:
        from sklearn.metrics import roc_auc_score
        perf = roc_auc_score(yte[keep], p[:, 1]) if len(np.unique(yte[keep])) == 2 else np.nan
    else:
        from sklearn.metrics import balanced_accuracy_score
        perf = balanced_accuracy_score(yte[keep], classes[p.argmax(1)])
    return max(h - ce, 0.0), perf


def _bits_regression(xtr, ytr, xte, yte, seed):
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score

    m = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(xtr, ytr)
    r2 = r2_score(yte, m.predict(xte))
    return (-0.5 * np.log2(1 - r2) if r2 > 0 else 0.0), r2


def probe(a) -> None:
    import pandas as pd
    from joblib import Parallel, delayed
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    o = Path(a.out)
    meta = json.loads((o / "meta.json").read_text())
    F = {s: torch.load(o / f"feats_{s}.pt", weights_only=False) for s in ("train", "test")}
    y = pd.read_parquet(o / "targets.parquet")
    tokens = meta["tokens"]
    own_tok = {t: next(i for i, (k, m) in enumerate(meta["members"].items()) if t in m) for t in meta["targets"]}

    def red(mat_tr, mat_te):                         # standardise, then 64 PCs fitted on train
        s = StandardScaler().fit(mat_tr)
        p = PCA(min(a.pcs, mat_tr.shape[1], len(mat_tr) - 1), random_state=0).fit(s.transform(mat_tr))
        return p.transform(s.transform(mat_tr)), p.transform(s.transform(mat_te))

    def job(t):
        kind = meta["target_type"][t]
        rows = []
        idx = {}
        for s in ("train", "test"):
            g = y[(y.task == t) & (y.split == s)].drop_duplicates("volume_id").set_index("volume_id")["value"]
            pos = {v: i for i, v in enumerate(F[s]["vids"])}
            v = [x for x in g.index if x in pos]
            idx[s] = (np.array([pos[x] for x in v]), g.loc[v].to_numpy(float))
        (itr, ytr), (ite, yte) = idx["train"], idx["test"]
        rs = np.random.default_rng(1)                  # same probe budget for every target (shared scans add more)
        if len(itr) > a.cap:
            k = rs.permutation(len(itr))[:a.cap]
            itr, ytr = itr[k], ytr[k]
        if len(ite) > a.cap:
            k = rs.permutation(len(ite))[:a.cap]
            ite, yte = ite[k], yte[k]
        if len(itr) < 30 or len(ite) < 20:
            return rows
        fn = _bits_regression if kind == "regression" else _bits_classifier
        if kind != "regression":
            ytr, yte = ytr.astype(int), yte.astype(int)
        cache = {}
        for mode in ("write", "readonly"):
            for j, src in enumerate(tokens):
                xtr, xte = red(F["train"][mode][itr, j].float().numpy(), F["test"][mode][ite, j].float().numpy())
                cache[(mode, j)] = (xtr, xte)
                bits, perf = fn(xtr, ytr, xte, yte, 0)
                rows.append({"target": t, "source": src, "mode": mode, "bits": bits, "perf": perf,
                             "own": j == own_tok[t], "n_train": len(itr), "n_test": len(ite)})
            # conditional: what source A adds to the target's own token
            jo = own_tok[t]
            for j, src in enumerate(tokens):
                if j == jo:
                    continue
                xtr = np.hstack([cache[(mode, jo)][0], cache[(mode, j)][0]])
                xte = np.hstack([cache[(mode, jo)][1], cache[(mode, j)][1]])
                bits, perf = fn(xtr, ytr, xte, yte, 0)
                rows.append({"target": t, "source": src, "mode": mode + "+own", "bits": bits, "perf": perf,
                             "own": False, "n_train": len(itr), "n_test": len(ite)})
        xtr, xte = red(F["train"]["pooled"][itr].float().numpy(), F["test"]["pooled"][ite].float().numpy())
        bits, perf = fn(xtr, ytr, xte, yte, 0)
        rows.append({"target": t, "source": "_patches", "mode": "pooled", "bits": bits, "perf": perf, "own": False,
                     "n_train": len(itr), "n_test": len(ite)})
        print(f"[probe] {t}: own {[r['bits'] for r in rows if r['own'] and r['mode'] == 'write'][0]:.3f} bits, "
              f"patches {bits:.3f}", flush=True)
        return rows

    res = Parallel(n_jobs=a.jobs, prefer="threads")(delayed(job)(t) for t in meta["targets"])
    pd.DataFrame([r for rs in res for r in rs]).to_csv(o / "transfer.csv", index=False)
    print("wrote", o / "transfer.csv")


# -------------------------------------------------------------------------------------------------------- plot
def plot(a) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    o = Path(a.out)
    meta = json.loads((o / "meta.json").read_text())
    d = pd.read_csv(o / "transfer.csv")
    gorder = list(dict.fromkeys(meta["groups"]))
    src = sorted(meta["tokens"], key=lambda s: (gorder.index(meta["groups"][meta["tokens"].index(s)]), s))
    tgt = sorted(meta["targets"], key=lambda t: (gorder.index(meta["target_group"][t])
                                                 if meta["target_group"][t] in gorder else 99, t))
    tgt = [t for t in tgt if t in set(d.target)]
    lab = lambda s: s.replace("hemorrhage_", "hem. ").replace("_", " ")          # noqa: E731

    def mat(mode, norm):
        x = d[d["mode"] == mode].pivot_table(index="source", columns="target", values="bits").reindex(src)[tgt]
        if norm:
            own = d[(d["mode"] == "write") & d.own].set_index("target")["bits"].reindex(tgt)
            x = x / own.where(own > 0.02).to_numpy()[None, :]
        return x

    for mode, norm, name, cl, title in (
            ("write", True, "transfer_write", (0, 1.2), "Share of task B's own usable information carried by task A's "
             "output (as trained)"),
            ("readonly", True, "transfer_readonly", (0, 1.2), "Same, encoder-only pathway (task tokens do not write "
             "into the patches)"),
            ("write+own", False, "transfer_conditional", (0, None), "Bits task A's output adds to task B's own output")):
        x = mat(mode, norm)
        fig, ax = plt.subplots(figsize=(16, 11), dpi=170)
        im = ax.imshow(x.to_numpy(float), cmap="magma", vmin=cl[0], vmax=cl[1] or np.nanpercentile(x, 98),
                       aspect="auto")
        ax.set_yticks(range(len(src)), [lab(s) for s in src], fontsize=6.5)
        ax.set_xticks(range(len(tgt)), [lab(t) for t in tgt], fontsize=6.5, rotation=90)
        for k in range(1, len(src)):
            if meta["groups"][meta["tokens"].index(src[k])] != meta["groups"][meta["tokens"].index(src[k - 1])]:
                ax.axhline(k - 0.5, color="white", lw=0.8)
        for k in range(1, len(tgt)):
            if meta["target_group"][tgt[k]] != meta["target_group"][tgt[k - 1]]:
                ax.axvline(k - 0.5, color="white", lw=0.8)
        ax.set_ylabel("source: task A's token output (shared tokens named by their first task)")
        ax.set_xlabel("target: task B's label")
        fig.colorbar(im, ax=ax, fraction=0.02)
        ax.set_title(title, fontsize=12, loc="left")
        fig.tight_layout()
        fig.savefig(o / f"{name}.png", facecolor="white")
        plt.close(fig)
    w, r = mat("write", True), mat("readonly", True)
    owner = [next(s for s, m in meta["members"].items() if t in m) for t in tgt]
    off = np.array([[s != ow for ow in owner] for s in src])
    print(f"median off-diagonal share: write {np.nanmedian(w.to_numpy()[off]):.2f}, "
          f"readonly {np.nanmedian(r.to_numpy()[off]):.2f}")
    print("wrote", o)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--n-train", type=int, default=500)
    c.add_argument("--n-test", type=int, default=300)
    c.add_argument("--batch", type=int, default=4)
    c.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("probe")
    p.add_argument("--out", required=True)
    p.add_argument("--pcs", type=int, default=64)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--cap", type=int, default=500, help="max train / test scans per target")
    q = sub.add_parser("plot")
    q.add_argument("--out", required=True)
    a = ap.parse_args()
    {"collect": collect, "probe": probe, "plot": plot}[a.cmd](a)


if __name__ == "__main__":
    main()
