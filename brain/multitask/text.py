"""Frozen text embeddings of task descriptions and class names.

    python -m brain.multitask.text                 # embed every registry string (run where the HF hub is reachable)

The cache (``runs/multitask/text/<encoder>.pt``) maps each string to a unit vector; training only reads it, so compute
nodes need neither ``transformers`` nor network access. Strings missing from the cache are an error naming the command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from brain.common.paths import resolve
from brain.multitask.registry import Registry, load_registry

CACHE_DIR = Path("runs/multitask/text")


def cache_path(encoder: str) -> Path:
    return resolve(CACHE_DIR) / f"{encoder.replace('/', '__')}.pt"


@torch.no_grad()
def embed_texts(texts: list[str], encoder: str, batch: int = 32) -> torch.Tensor:
    """Mean-pooled last hidden state over real tokens, L2-normalised (sentence-transformers' pooling)."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1))
    return torch.cat(out)


def update_cache(reg: Registry, encoder: str | None = None) -> Path:
    encoder = encoder or reg.text_encoder
    path = cache_path(encoder)
    cache = torch.load(path) if path.exists() else {}
    todo = [t for t in reg.texts() if t not in cache]
    if todo:
        cache.update(zip(todo, embed_texts(todo, encoder)))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, path)
    print(f"[text] {encoder}: {len(todo)} new, {len(cache)} cached -> {path}")
    return path


def load_embeddings(texts: list[str], encoder: str) -> torch.Tensor:
    path = cache_path(encoder)
    cache = torch.load(path) if path.exists() else {}
    missing = [t for t in texts if t not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} task strings are not embedded (e.g. {missing[0]!r}); "
                         f"run: python -m brain.multitask.text")
    return torch.stack([cache[t].float() for t in texts])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--encoder", default=None, help="HF model id (default: the registry's text_encoder)")
    a = ap.parse_args()
    update_cache(load_registry(a.registry), a.encoder)


if __name__ == "__main__":
    main()
