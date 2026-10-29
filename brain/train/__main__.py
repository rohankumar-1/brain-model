"""``python -m brain.train <lejepa3d|mae3d|multitask|joint> [--config cfg.yaml]``."""

from brain.train.engine import run

if __name__ == "__main__":
    run()
