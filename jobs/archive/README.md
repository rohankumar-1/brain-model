# Archived jobs

Finished one-off jobs, kept for provenance: how a dataset was fetched, staged or prepped, and how past experiments were
run. Each sources `jobs/_env.sh` through `$BRAIN_REPO`, but they are not maintained: a job may call options or modules
that changed since (check before re-running). Jobs whose module was removed are deleted; they are in git history
(transcode, store_cleanup_v6, n4_store, the FOMO archive packing and extract cleanup, 2026-11-06). The rebuild
recipe lives in `jobs/` (`prep_v6*.sbatch`, `index_append.sbatch`) and `docs/prep.md`.

| Job | What it did |
|---|---|
| `fetch_ms3seg.sbatch`, `fetch_valdo.sbatch` | downloaded MS3SEG (figshare, multi-volume RAR5) and VALDO (Zenodo) |
| `prep_v6_new.sbatch` | staged + prepped MS3SEG and VALDO into the format-6 store |
| `prep_oasis1_raw.sbatch` | re-prepped OASIS-1 from raw MPRAGE after the Analyze orientation fix (2026-10-21) |
| `rawpack_pool.sbatch` | packed raw downloads into pool archives |
| `submit_ablation_prefix.sh` | LeJEPA prefix-token ablation (CLS / registers); its arm selection and context diagnostic modules were removed (2026-10-24, in git history) |
| `layer_diag.sbatch` | early-layer diagnostic of v3 |
| `rsna_reader_pilot.sbatch` | dcm2niix vs SimpleITK RSNA assembly pilot (negative: no study rescued) |
| `verify_v6.sbatch` | verified every indexed zip after the v6 rebuild (`brain.prep.verify_store`, still the post-rebuild check) |
| `prep_bhsd.sbatch`, `prep_hemseg.sbatch` | prepped the BHSD and HemSeg-500 hemorrhage cohorts (2026-10-22) |
| `download_rsna_ica.sbatch` | downloaded the RSNA Intracranial Aneurysm archive to pool (presigned link, expired) |
| `download_cq500.sbatch` | re-downloaded CQ500 as the original DICOM (one plain soft-tissue series per study, 2026-10-26) |
| `download_ucsf_pdgm.sbatch`, `download_upenn_gbm.sbatch` | fetched UCSF-PDGM and UPENN-GBM from TCIA over Aspera (2026-10-28 / 29) |
| `rsna_ica_pilot.sbatch` | 30-series-per-modality RSNA-ICA prep pilot (dcm2niix reader, rescale-tag issue) |
| `index_v6.sbatch` | (deleted 2026-11-06, in git history) the v6 index rebuild; now `jobs/index_append.sbatch FULL=1` |
| `download_adni.sbatch` | fetched ADNI / PPMI image collections from LONI IDA (Advanced Download links, 2026-10-29 / 30) |
| `prep_ida.sbatch` | prepped the ADNI / PPMI baseline picks from the IDA zips (dcm2niix reader; 2026-10-30) |
| `fetch_fastmri.sbatch` | streamed the 1,001 fastMRI+ annotated brain scans out of the fastMRI tar.xz batches (2026-11-01) |
| `prep_fastmri.sbatch` | staged and prepped the fastMRI+ scans; wrote `labels/fastmri_patients.csv` (2026-11-02) |
| `prep_synthrad.sbatch` | prepped SynthRAD2023 brain (CT + MR, CT + CBCT; 2026-10-30) |
| `prep_ucsf_pdgm.sbatch`, `prep_upenn_gbm.sbatch` | prepped UCSF-PDGM and UPENN-GBM onto their T1's registration (2026-10-28 / 29) |
| `prep_rsna_ica.sbatch` | prepped the RSNA Intracranial Aneurysm series from the pool zip (2026-10-30) |
| `content_audit.sbatch` | report-only content audit before the 2026-10-29 index rebuild (`runs/diagnostics/content_audit.py`, removed; the glioma twin search is now `brain.prep.subjects glioma-twins`) |
| `fetch_pedi_ct.sbatch`, `prep_pedi_ct.sbatch` | fetched (TCIA Aspera) and prepped Pedi-Cranial-CT-Healthy (2026-11-06) |
