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
| `transcode_quick.sbatch`, `transcode_merge.sbatch`, `transcode_list.txt` | transcoded pool raw archives to JPEG-LS / JPEG XL (`brain.prep.transcode` removed 2026-10-29, in git history) |
| `prep_bhsd.sbatch`, `prep_hemseg.sbatch` | prepped the BHSD and HemSeg-500 hemorrhage cohorts (2026-10-22) |
| `download_rsna_ica.sbatch` | downloaded the RSNA Intracranial Aneurysm archive to pool (presigned link, expired) |
| `archive_fomo_small.sbatch` | packed the 16 small FOMO cohorts' extracts to pool (JPEG-LS) and dropped the verified scratch copies (2026-10-25) |
| `download_cq500.sbatch` | re-downloaded CQ500 as the original DICOM (one plain soft-tissue series per study, 2026-10-26) |
| `n4_store.sbatch` | N4-corrected the stored MRI that prep had not corrected (BraTS, MSD, VALDO, MS3SEG; 2026-10-25) |
| `download_ucsf_pdgm.sbatch`, `download_upenn_gbm.sbatch` | fetched UCSF-PDGM and UPENN-GBM from TCIA over Aspera (2026-10-28 / 29) |
| `rsna_ica_pilot.sbatch` | 30-series-per-modality RSNA-ICA prep pilot (dcm2niix reader, rescale-tag issue) |
