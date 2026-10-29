"""The benchmark: one versioned protocol (``configs/bench/benchmark.yaml``) for any model the suite can encode with.

    python -m brain.bench list [--model BQ-A1]                   # the benchmark tasks (those a model is scored on)
    python -m brain.bench run --model BQ-T1 --full --submit      # every evaluation that applies, as Slurm jobs
    python -m brain.bench run --model regional_stats             # a models.yaml entry, run here
    python -m brain.bench card --model BQ-A1                     # every number recorded for one model
    python -m brain.bench board                                  # leaderboard -> runs/bench/board_v<version>.md / .csv

A model is one of our BrainQuery (BQ) checkpoints, by name (``BQ-A1``) or path, or a published encoder or baseline in
``configs/bench/models.yaml`` (``brain.downstream.models``). It runs the frozen transfer tasks of the modalities it
takes (``--full``: BrainQuery checkpoints also segmentation transfer, token probes, own heads and few-shot); the
leaderboard pairs every model with the spec's reference on the
tasks both have (paired subject bootstrap, ``brain.eval.report``), per modality, and groups the models by training
class. Tasks whose labels a model trained on are left out of its comparisons; results computed on another store index
than the spec pins are left out and counted.
"""
