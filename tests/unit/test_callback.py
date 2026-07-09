"""Unit tests: HTTP webhook callback client."""

import pytest
from unittest.mock import MagicMock, patch

from tests.unit.conftest import AsyncMock
import httpx

from app.callback.client import send_callback, send_failure_callback
from app.schemas.result import CallbackPayload

# ==============================================================================
# 3. HTTP Webhook Callback Client tests
# ==============================================================================


@pytest.mark.asyncio
async def test_send_callback_success():
    payload = CallbackPayload(
        jobType="listing",
        workerVersion="0.2.0",
        matchModel="test",
        safetyModel="test",
        embeddingModel="test",
        images=[],
    )

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status.return_value = None
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await send_callback("http://example.com/callback", payload)

        mock_client.post.assert_called_once()
        called_args = mock_client.post.call_args[0]
        called_kwargs = mock_client.post.call_args[1]
        assert called_args[0] == "http://example.com/callback"
        assert called_kwargs["json"]["jobType"] == "listing"


@pytest.mark.asyncio
async def test_send_failure_callback():
    job = {"jobType": "listing", "listingId": "listing_abc", "ownedPetId": "pet_123"}

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status.return_value = None
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await send_failure_callback("http://example.com/fail", job, "Model load failed")

        mock_client.post.assert_called_once()
        called_kwargs = mock_client.post.call_args[1]
        assert called_kwargs["json"]["jobType"] == "listing"
        assert called_kwargs["json"]["listingId"] == "listing_abc"
        assert called_kwargs["json"]["ownedPetId"] == "pet_123"
        assert called_kwargs["json"]["error"] == "Model load failed"
