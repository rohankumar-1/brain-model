"""Epoch progress bars that stay readable in Slurm logs.

tqdm turns itself off when stdout is not a terminal. These bars stay on and
write a new line every ``mininterval`` seconds, including batches per second.
"""

from __future__ import annotations

import sys

from tqdm import tqdm


class EpochBar(tqdm):
    def __init__(self, iterable, *args, **kwargs):
        kwargs.setdefault("disable", False)
        kwargs.setdefault("mininterval", 10)
        kwargs.setdefault("ascii", True)
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("file", sys.stdout)
        kwargs.setdefault("leave", True)
        super().__init__(iterable, *args, **kwargs)

    def display(self, msg=None, pos=None):
        # refresh() calls display() with msg left empty. The bar text is __str__().
        if msg is None:
            msg = self.__str__()
        if not msg:
            return
        self.fp.write(str(msg) + "\n")
        self.fp.flush()
