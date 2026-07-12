"""
End-to-end smoke tests — load real models and run the full pipeline.

Skipped in default CI (`pytest` excludes the integration marker).
Run locally after `pip install -e ".[dev]"`:

    pytest -m integration -v
"""

from __future__ import annotations

import asyncio

import pytest

from tools.smoke import run_smoke


@pytest.mark.integration
@pytest.mark.parametrize(
    "profile",
    [
        "dedup-only",
        "quality",
    ],
)
def test_profile_smoke(profile: str) -> None:
    """Warmup + single-image pipeline for supported Tier-1 profiles."""
    result = asyncio.run(run_smoke(profile))
    assert result.get("ok") is True, result
