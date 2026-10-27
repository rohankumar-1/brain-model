"""The benchmark: one versioned protocol (``configs/bench/benchmark.yaml``) for any model the suite can encode with.

    python -m brain.bench list                                   # the benchmark tasks, by group and modality
    python -m brain.bench run --model runs/multitask/tokens_s42/best.pt --submit      # one Slurm job, its tasks
    python -m brain.bench run --model regional_stats             # a models.yaml entry, run here
    python -m brain.bench board                                  # leaderboard -> runs/bench/board_v<version>.md / .csv

A model is a checkpoint of ours or a ``configs/bench/models.yaml`` entry (``brain.downstream.models``). It runs the
frozen transfer tasks of the modalities it takes; the leaderboard pairs every model with the spec's reference on the
tasks both have (paired subject bootstrap, ``brain.eval.report``), per modality, and groups the models by training
class. Tasks whose labels a model trained on are left out of its comparisons; results computed on another store index
than the spec pins are left out and counted.
"""
