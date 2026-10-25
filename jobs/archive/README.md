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
| `submit_ablation_prefix.sh` | LeJEPA prefix-token ablation (CLS / registers); its arm selection and context diagnostic modules were removed (2026-10-24, in git history) |
| `layer_diag.sbatch` | early-layer diagnostic of v3 |
| `rsna_reader_pilot.sbatch` | dcm2niix vs SimpleITK RSNA assembly pilot (negative: no study rescued) |
| `store_cleanup_v6.sbatch`, `verify_v6.sbatch` | dropped non-format-6 files after the v6 rebuild; verified every indexed zip |
| `transcode_quick.sbatch`, `transcode_merge.sbatch`, `transcode_list.txt` | transcoded pool raw archives to JPEG-LS / JPEG XL |
| `prep_bhsd.sbatch`, `prep_hemseg.sbatch` | prepped the BHSD and HemSeg-500 hemorrhage cohorts (2026-10-22) |
| `download_rsna_ica.sbatch` | downloaded the RSNA Intracranial Aneurysm archive to pool (presigned link, expired) |
