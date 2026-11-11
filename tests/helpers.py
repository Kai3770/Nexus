"""Tiny test helpers so the suite runs under pytest and under ``python tests/run_all.py`` alike."""
import os
import sys
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@contextmanager
def raises(expected):
    """Context manager asserting that the body raises ``expected``."""
    try:
        yield
    except expected:
        return
    raise AssertionError("expected %s to be raised" % getattr(expected, "__name__", expected))
