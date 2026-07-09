"""Unit tests: Healthcheck CLI."""

import time
import pytest
from unittest.mock import patch

from app.healthcheck import main as healthcheck_main

# ==============================================================================
# 7. Healthcheck CLI tests
# ==============================================================================


def test_healthcheck_missing_heartbeat():
    # If heartbeat file doesn't exist, health check raises SystemExit(1)
    with (
        patch("app.healthcheck.os.path.exists", return_value=False),
        pytest.raises(SystemExit) as exc_info,
    ):
        healthcheck_main()
    assert exc_info.value.code == 1


def test_healthcheck_stale_heartbeat():
    # If heartbeat exists but is too old (e.g. 120 seconds ago), raises SystemExit(1)
    mtime = time.time() - 120
    with (
        patch("app.healthcheck.os.path.exists", return_value=True),
        patch("app.healthcheck.os.path.getmtime", return_value=mtime),
        pytest.raises(SystemExit) as exc_info,
    ):
        healthcheck_main()
    assert exc_info.value.code == 1


def test_healthcheck_fresh_heartbeat():
    # If heartbeat exists and is fresh, raises SystemExit(0)
    mtime = time.time() - 10
    with (
        patch("app.healthcheck.os.path.exists", return_value=True),
        patch("app.healthcheck.os.path.getmtime", return_value=mtime),
        pytest.raises(SystemExit) as exc_info,
    ):
        healthcheck_main()
    assert exc_info.value.code == 0
