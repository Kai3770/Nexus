#!/usr/bin/env python3
"""
Dependency-free test runner: imports every ``tests/test_*.py`` module and calls
each ``test_*`` function.  ``python tests/run_all.py [name filter]``.
"""
import importlib
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    passed, failed, skipped = 0, [], 0
    started = time.perf_counter()
    for filename in sorted(os.listdir(HERE)):
        if not (filename.startswith("test_") and filename.endswith(".py")):
            continue
        module = importlib.import_module(filename[:-3])
        for name in sorted(n for n in dir(module) if n.startswith("test_")):
            label = "%s::%s" % (filename, name)
            if pattern and pattern not in label:
                continue
            t0 = time.perf_counter()
            try:
                result = getattr(module, name)()
            except Exception:
                failed.append(label)
                print("FAIL  %s\n%s" % (label, traceback.format_exc()))
                continue
            if result == "skipped":
                skipped += 1
                print("skip  %s" % label)
            else:
                passed += 1
                print("ok    %-74s %6.2fs" % (label, time.perf_counter() - t0))
    print("\n%d passed, %d failed, %d skipped in %.1fs" % (passed, len(failed), skipped, time.perf_counter() - started))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
