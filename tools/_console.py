"""
CLI stdout/stderr helpers.

Operator tools write human-readable output here instead of bare ``print()`` so
static analysis can distinguish CLI surfaces from application logging.
"""

from __future__ import annotations

import sys


def out(message: str = "") -> None:
    """Write a line to stdout (operator-facing CLI output)."""
    sys.stdout.write(message + "\n")


def err(message: str = "") -> None:
    """Write a line to stderr (operator-facing errors / notes)."""
    sys.stderr.write(message + "\n")
