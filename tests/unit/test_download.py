"""Unit tests: Image download and decoding."""

import pytest
from unittest.mock import MagicMock, patch

from tests.unit.conftest import AsyncMock
import httpx
from PIL import Image

from app.pipeline.download import (
    download_bytes,
    decode_image,
    download_all,
    DecodedImage,
)

# ==============================================================================
# 2. Image Download and Decoding Module tests
# ==============================================================================


@pytest.mark.asyncio
async def test_download_bytes_success():
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.content = b"fake-image-bytes"

    # We need to make client.get an AsyncMock since it's awaited
    mock_client.get = AsyncMock(return_value=mock_response)

    res_bytes = await download_bytes(mock_client, "http://example.com/test.png")
    assert res_bytes == b"fake-image-bytes"
    mock_client.get.assert_called_once_with("http://example.com/test.png")


@pytest.mark.asyncio
async def test_download_bytes_http_error():
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="404 Client Error", request=MagicMock(), response=mock_response
    )
    mock_client.get = AsyncMock(return_value=mock_response)

    with pytest.raises(httpx.HTTPStatusError):
        await download_bytes(mock_client, "http://example.com/missing.png")


@pytest.mark.asyncio
async def test_download_bytes_retry_transient_then_success():
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_response_fail = MagicMock(spec=httpx.Response)
    mock_response_fail.status_code = 503
    mock_response_fail.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="503 Service Unavailable",
        request=MagicMock(),
        response=mock_response_fail,
    )

    mock_response_ok = MagicMock(spec=httpx.Response)
    mock_response_ok.status_code = 200
    mock_response_ok.content = b"success-bytes"
    mock_response_ok.raise_for_status.return_value = None

    # First call raises 503, second call succeeds
    mock_client.get = AsyncMock(side_effect=[mock_response_fail, mock_response_ok])

    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        res = await download_bytes(mock_client, "http://example.com/retry.png")
        assert res == b"success-bytes"
        assert mock_client.get.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1


@pytest.mark.asyncio
async def test_download_bytes_retry_transient_exhausted():
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_response_fail = MagicMock(spec=httpx.Response)
    mock_response_fail.status_code = 503
    mock_response_fail.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="503 Service Unavailable",
        request=MagicMock(),
        response=mock_response_fail,
    )

    mock_client.get = AsyncMock(return_value=mock_response_fail)

    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        with pytest.raises(httpx.HTTPStatusError):
            await download_bytes(mock_client, "http://example.com/fail.png")
        assert mock_client.get.call_count == 3  # 1 initial + 2 retries
        assert mock_sleep.call_count == 2


def test_decode_image():
    from io import BytesIO

    img = Image.new("RGB", (1, 1), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    decoded = decode_image("http://example.com/red.png", png_bytes)
    assert isinstance(decoded, DecodedImage)
    assert decoded.url == "http://example.com/red.png"
    assert decoded.raw_bytes == png_bytes
    assert decoded.image.size == (1, 1)


@pytest.mark.asyncio
async def test_download_all_mixed_results():
    from io import BytesIO

    img = Image.new("RGB", (10, 10), color="blue")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    # Create mock response
    mock_response_ok = MagicMock(spec=httpx.Response)
    mock_response_ok.content = jpeg_bytes
    mock_response_ok.raise_for_status.return_value = None

    # Mock client and its async context manager methods
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    def side_effect(url, *args, **kwargs):
        if "good" in url:
            return mock_response_ok
        raise Exception("Connection timed out")

    mock_client.get = AsyncMock(side_effect=side_effect)

    with patch("httpx.AsyncClient", return_value=mock_client):
        decoded, errors = await download_all(
            ["http://example.com/good.jpg", "http://example.com/bad.jpg"]
        )

        assert len(decoded) == 1
        assert decoded[0].url == "http://example.com/good.jpg"
        assert decoded[0].image.size == (10, 10)

        assert len(errors) == 1
        assert errors[0]["url"] == "http://example.com/bad.jpg"
        assert "Connection timed out" in errors[0]["error"]
