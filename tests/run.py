"""Minimal test runner for environments without pytest: ``python tests/run.py tests/test_x.py [...]``.

Runs every ``test_*`` function with the fixtures it names: ``tmp_path`` (a temporary directory) and ``monkeypatch``
(``setattr`` / ``setenv`` / ``delenv``, undone after the test). Tests needing any other fixture are skipped.
"""

import importlib
import importlib.util
import inspect
import os
import sys
import tempfile
import traceback
from pathlib import Path

_MISSING = object()


class MonkeyPatch:
    """The subset of pytest's ``monkeypatch`` the suite uses."""

    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value=_MISSING, raising=True):
        if value is _MISSING:                       # setattr("pkg.module.attr", value)
            value = name
            module, _, name = str(target).rpartition(".")
            target = importlib.import_module(module)
        old = getattr(target, name, _MISSING)
        if old is _MISSING and raising:
            raise AttributeError(f"{target!r} has no attribute {name!r}")
        self._undo.append(lambda: delattr(target, name) if old is _MISSING else setattr(target, name, old))
        setattr(target, name, value)

    def setenv(self, name, value):
        old = os.environ.get(name)
        self._undo.append(lambda: os.environ.pop(name, None) if old is None else os.environ.__setitem__(name, old))
        os.environ[name] = str(value)

    def delenv(self, name, raising=True):
        if name not in os.environ:
            if raising:
                raise KeyError(name)
            return
        old = os.environ.pop(name)
        self._undo.append(lambda: os.environ.__setitem__(name, old))

    def undo(self):
        while self._undo:
            self._undo.pop()()


def run_test(fn) -> None:
    params = inspect.signature(fn).parameters
    mp = MonkeyPatch()
    try:
        with tempfile.TemporaryDirectory() as d:
            kw = {"tmp_path": Path(d), "monkeypatch": mp}
            fn(**{k: kw[k] for k in params})
    finally:
        mp.undo()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    fails = total = 0
    for f in sys.argv[1:]:
        spec = importlib.util.spec_from_file_location(Path(f).stem, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__:
                continue
            unknown = set(inspect.signature(fn).parameters) - {"tmp_path", "monkeypatch"}
            if unknown:
                print("SKIP", f, name, sorted(unknown))
                continue
            total += 1
            try:
                run_test(fn)
            except Exception:
                fails += 1
                print("FAIL", f, name)
                traceback.print_exc(limit=3)
    print(f"{total - fails}/{total} passed")
    sys.exit(1 if fails else 0)
