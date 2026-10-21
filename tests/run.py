"""Minimal test runner for environments without pytest: ``python tests/run.py tests/test_x.py [...]``.

Runs every ``test_*`` function; supplies ``tmp_path`` (a temporary directory); skips tests needing other fixtures.
"""

import importlib.util
import inspect
import sys
import tempfile
import traceback
from pathlib import Path

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
            params = inspect.signature(fn).parameters
            if set(params) - {"tmp_path"}:
                print("SKIP", f, name, list(params))
                continue
            total += 1
            try:
                with tempfile.TemporaryDirectory() as d:
                    fn(**({"tmp_path": Path(d)} if "tmp_path" in params else {}))
            except Exception:
                fails += 1
                print("FAIL", f, name)
                traceback.print_exc(limit=3)
    print(f"{total - fails}/{total} passed")
    sys.exit(1 if fails else 0)
