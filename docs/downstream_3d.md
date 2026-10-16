# Downstream on the 3D BrainModel

The existing suite (`brain/downstream/run_suite.py`) stays the same tasks, splits, heads, and windows. A 3D checkpoint is detected automatically. Classification and regression reuse the frozen heads as-is. Segmentation is `brain/downstream/seg3d.py`. Set `BRAIN_FINE_POOL=1` to cache half-size tokens averaged onto the 16³ grid.

## Why not encode 2D slices?

`BrainEncoder` is a 3D ViT with anisotropic patches (default global `16×16×8`) and RoPE in millimetres. A single axial slice is thinner than one z-patch, so pushing 256×256×1 through the 3D stem is undefined. Downstream has to stay volumetric.

## Classification / regression (3D tokens)

1. Load the native v4 volume `(X, Y, Z)` (256×256×128).
2. Apply the same asinh window(s) as the 2D suite (`canonical` / `multi` / `shifted`).
3. Center-crop to a multiple of the train global patch (16×16×8) and tokenize **in place**. No resize onto the 128×128×64 train canvas. A 256×256×128 volume is 16×16×16 = 4096 tokens, each 16×16×8 mm. The encoder has no CLS token.
4. Cache `(N, T, D)` tokens and `(N, T, 3)` centers normalized to `[0, 1]`. Multi-window concatenates on `D`.
5. Linear / tree heads mean+max pool the token set `token_attn_mlp` scores each token with its `(x, y, z)`.

Cache files are tagged `v5_3d_tok`. The older `v4_3d` caches packed 8 axial planes into a fake 32/64-slice sequence and are not read.

```bash
python -m brain.downstream.run_suite --task oasis1_age --ckpt runs/lejepa3d/last.pt --window multi
sbatch --export=ALL,CKPT=runs/lejepa3d/last.pt,TASKS=oasis1_age:oasis1_cdr:cq500 jobs/downstream_vit.sbatch
```

## Segmentation (`brain/downstream/seg3d.py`)

The encoder runs once per volume; its 16×16×16 token map is cached (`runs/downstream/_cache/_seg3d/<task>/`). A small 3D conv decoder (`TokenDecoder3D`: 1×1 stem, two 3×3×3 convs, three 2× steps, one in-plane 2× step, 1×1 head) trained on the cached tokens upsamples to one logit per voxel of the 128×256×256 grid. Loss = CE + soft Dice per region (same as 2D). Training volumes also get three encoded copies (a shift off the patch grid, a left-right flip, both), the mask moving with them; input dropout 0.1. Test Dice is scored on slices 1, 3, 5, … (the slices the 2D reference scores, `dice_*`) and on the whole volume (`dice_full_*`). BraTS encodes each of the four sequences separately and concatenates tokens.

```bash
sbatch --export=ALL,TASKS="wmh seg_cq500",CKPTS="runs/snap/predict_8k.pt" jobs/seg3d.sbatch
sbatch --export=ALL,FINE=1,TASKS="wmh",CKPTS="runs/snap/predict_8k.pt" jobs/seg3d.sbatch   # 8x8x4 tiled + blocks 3 and 7
```

`--fine` is 8×8×4 patches with tiled 128×128×64 windows (in-plane match to the 2D 16×16 patch). It stores one copy of the final tokens (no mid-depth taps, no extra encoded copies). Training flips/rolls that token grid instead. Do not put commas in `EXTRA_ARGS`: Slurm `--export` splits on them (use `:` or `FINE=1`).

Compare against 2D with `--seg-last-only` (decoder reads only the final block, like the 3D decoder): `run_suite --task wmh --ckpt <2D> --seg-last-only --run-name last_only`. The default 2D decoder reads four depths, which adds 0.01–0.08 Dice.

Test Dice (BrainModel tiny, 3000 decoder steps). The pretraining objective (predict vs control vs 28.5k) does not move any of these; patch size does.

| Setup | WMH | BraTS gli | CQ500 ICH |
|---|---|---|---|
| 3D 16×16×8, final tokens | 0.46 | 0.78 | 0.18–0.20 |
| 3D 16×16×8, taps 3+7 | 0.48–0.49 | 0.79–0.80 | 0.24–0.26 |
| 3D 8×8×4 tiled, taps 3+7 | **0.62** | running | running |
| 2D last block only (tiny / small) | 0.56 / 0.61 | 0.83 / 0.86 | 0.25 / 0.25 |
| 2D all taps (tiny / small) | 0.65 / 0.65 | 0.88 / 0.88 | 0.26 / 0.28 |

Lesion-probe lift at the coarse grid also jumps when each coarse token is the mean of its 8 half-size tokens (`probe3d --fine`): control 4.97 → 5.69, predict 5.24 → 5.91, vs 2D small 6.15.

## Memory / fidelity knobs (config later)

| Knob | Default | Notes |
|---|---|---|
| Eval grid | native volume, global patch | 4096 tokens at 16×16×8 mm on v4 |
| Train crops | native-grid crops of the volume (global 128×128×64 mm = 512 tokens) | same 16×16×8 mm patch as eval; no resizing |
| Overlapping tiles | stitch by `coords_mm` | train-small / infer-big check from the spec |

## Comparability

Numbers are **not** apples-to-apples with Tiny 2D: the 3D model sees 16×16×8 mm patches on the native grid; 2D sees 32–64 full-res slices. Same labels, splits, heads, and windows. Read deltas as “3D encoder + same probes”, not as a drop-in replacement of the 2D feature.
