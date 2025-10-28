#!/usr/bin/env python3
"""Convenience runner for the deterministic end-to-end test suite."""

from __future__ import annotations

import importlib
import sys
import unittest


def main() -> int:
    try:
        importlib.import_module("typer.testing")
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        missing = exc.name or "typer"
        print(
            f"Missing dependency '{missing}'. Install project requirements first "
            "(e.g. `python -m pip install -e .`).",
            file=sys.stderr,
        )
        return 1

    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_e2e")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
