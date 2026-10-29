"""Brain CT/MRI self-supervised learning.

Subpackages: ``prep`` (raw archives -> Zarr volumes), ``data`` (loaders, crops, augmentation),
``models`` (ViT backbones and heads), ``objectives`` (SSL losses), ``train`` (one engine, three methods),
``downstream`` (frozen-encoder evaluation), ``common`` (config, distributed, checkpoint, windowing).
"""
