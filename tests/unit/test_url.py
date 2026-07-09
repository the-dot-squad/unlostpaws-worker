"""Unit tests: URL rewrite utilities."""

import os
import pytest
from unittest.mock import patch

from app.utils.url import rewrite_local_url

# ==============================================================================
# 1. URL Rewrite Utilities tests
# ==============================================================================


@pytest.mark.parametrize(
    "url, in_docker, env_var, expected",
    [
        (
            "http://localhost:3000/api",
            True,
            "false",
            "http://host.docker.internal:3000/api",
        ),
        (
            "http://127.0.0.1:3000/api",
            True,
            "false",
            "http://host.docker.internal:3000/api",
        ),
        (
            "http://localhost:3000/api",
            False,
            "true",
            "http://host.docker.internal:3000/api",
        ),
        (
            "http://127.0.0.1:3000/api",
            False,
            "true",
            "http://host.docker.internal:3000/api",
        ),
        ("http://localhost:3000/api", False, "false", "http://localhost:3000/api"),
        ("http://example.com/api", True, "true", "http://example.com/api"),
    ],
)
def test_rewrite_local_url(url, in_docker, env_var, expected):
    # Patch specifically in the app.utils.url module
    with (
        patch("app.utils.url.os.path.exists", return_value=in_docker),
        patch.dict(os.environ, {"RUNNING_IN_DOCKER": env_var}),
    ):
        res = rewrite_local_url(url)
        assert res == expected
