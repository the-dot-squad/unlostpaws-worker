"""Shared path helpers for operator CLI modules."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_import_path() -> Path:
    """Put the repository root on sys.path so `app` imports work from any subcommand."""
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return ROOT
