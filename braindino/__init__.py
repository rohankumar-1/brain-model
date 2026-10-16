"""Slice-wise DINOv2 + iBOT (BrainDINO-style) comparison model.

Teacher-student self-distillation on axial slabs: DINO on the CLS token, iBOT on
masked patch tokens, KoLeo on student CLS. The backbone is the existing 2D ViT
so frozen downstream loaders consume ``ckpt['model']`` unchanged.
"""
