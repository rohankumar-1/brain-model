"""The stored volume grid and the encoder's token grids: one place for their sizes, shared by prep, training defaults
and downstream code.

1 mm isotropic, axis-aligned in template (MNI152) space, centered on the template brain's centroid. 160 mm of depth
holds the whole 154 mm MNI brain (the old 128 mm slab cut ~18 mm of cerebellum/brainstem and ~8 mm of vertex) and is
a multiple of both token depths (8 -> 20 tokens, 4 -> 40).
"""

GRID_XYZ = (256, 256, 160)  # (x, y, z) voxels = mm
DEPTH = GRID_XYZ[2]
PATCH_XYZ = (16, 16, 8)       # the 3D encoder's native patch
FINE_PATCH_XYZ = (8, 8, 4)    # tiled half-size patch


def token_grid(patch_xyz=PATCH_XYZ, size_xyz=GRID_XYZ) -> tuple[int, int, int]:
    """(Z, Y, X) token grid of a full volume; tokens are flattened in this order (X fastest)."""
    px, py, pz = (int(v) for v in patch_xyz)
    sx, sy, sz = (int(v) for v in size_xyz)
    return sz // pz, sy // py, sx // px


TOKEN_GRID = token_grid()                   # (20, 16, 16) = 5120 tokens
FINE_TOKEN_GRID = token_grid(FINE_PATCH_XYZ)  # (40, 32, 32) = 40960 tokens
