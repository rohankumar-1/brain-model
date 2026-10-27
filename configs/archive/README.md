# Archived configs

Superseded experiment configs, kept for provenance (each run also stores its full config in its checkpoints).

- `lejepa3d/ablate_prefix/`: CLS / register prefix-token ablation (its CLS arm became v3, runs/v3/cls_s42).
- `lejepa3d/tiny_*_s43`, `tiny_predict_fine`: seed repeats and the fine-grid prediction variant of the tiny recipe.
- `lejepa3d/small_harder`, `small_multiscale`: augmentation and multi-scale variants of the small recipe.
- `multitask/`: the multitask runs up to 2026-10-26 (runs 1-6: `v3_init` ... `v7_nosmall`) and the task-token
  comparison (`omni_a_slots` / `omni_b_tokens` / `omni_c_tokens_moe`). Task tokens (arm B) became the default model;
  these configs use options retired on 2026-10-27 (`slot.mode` / `embedding` / `width`, `model.general`,
  `loss.general`), so they document runs rather than rerun them.
- `joint/`: the joint pilot (`small`, its `ssl_only` control) and A1 (`a1_warm_v3`, the slot-readout joint run);
  `configs/joint/tokens_warm_v3.yaml` is A1 with task tokens.
