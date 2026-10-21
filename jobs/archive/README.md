# Archived jobs

Finished one-off jobs, kept for provenance (how a dataset was fetched, staged or repaired, and how past experiments
were run). They still work from here: each sources `jobs/_env.sh` through `$BRAIN_REPO`.

| Job | What it did |
|---|---|
| `fetch_ms3seg.sbatch`, `fetch_valdo.sbatch` | downloaded MS3SEG (figshare, multi-volume RAR5) and VALDO (Zenodo) |
| `prep_v6_new.sbatch` | staged + prepped MS3SEG and VALDO into the format-6 store |
| `prep_oasis1_raw.sbatch` | re-prepped OASIS-1 from raw MPRAGE after the Analyze orientation fix (2026-10-21) |
| `cleanup_extracts_v6.sbatch` | removed extracted archives after the v6 prep |
| `rawpack_pool.sbatch` | packed raw downloads into pool archives |
| `select_and_train.sbatch`, `submit_ablation_prefix.sh` | LeJEPA prefix-token ablation (CLS / registers) and arm selection |
| `context_diag.sbatch`, `layer_diag.sbatch` | context-mismatch and early-layer diagnostics of v3 |
| `rsna_reader_pilot.sbatch` | dcm2niix vs SimpleITK RSNA assembly pilot (negative: no study rescued) |
