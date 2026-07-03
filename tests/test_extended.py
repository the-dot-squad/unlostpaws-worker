import os
import time
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open
import httpx
from PIL import Image

# Import utility code under test
from app.utils.url import rewrite_local_url
from app.pipeline.download import (
    download_bytes,
    decode_image,
    download_all,
    DecodedImage,
)
from app.callback.client import send_callback, send_failure_callback
from app.pipeline.orchestrator import run_pipeline, resolve_stages
from app.queue.consumer import (
    ensure_consumer_group,
    requeue_job,
    send_to_dlq,
    update_heartbeat,
    handle_job,
    reclaim_pending_messages,
)
from app.models.registry import resolve_torch_device, health_models, warmup
from app.healthcheck import main as healthcheck_main
from app.schemas.result import CallbackPayload, JobResult, ProcessedImageResult


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


# ==============================================================================
# 3. HTTP Webhook Callback Client tests
# ==============================================================================


@pytest.mark.asyncio
async def test_send_callback_success():
    payload = CallbackPayload(
        jobType="listing",
        workerVersion="0.1.3",
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


# ==============================================================================
# 4. Pipeline Orchestration tests
# ==============================================================================


def test_resolve_stages_custom_override():
    from app.config.settings import settings

    job = {"pipeline": ["quality"]}
    stages = resolve_stages(job, settings)
    assert stages == ("quality",)

    stages_default = resolve_stages({}, settings)
    assert stages_default == settings.profile.stages


@pytest.mark.asyncio
async def test_run_pipeline_success_mocked():
    job = {
        "jobType": "listing",
        "listingId": "listing_123",
        "imageUrls": ["http://example.com/img1.jpg"],
        "pipeline": ["quality", "fingerprint"],
    }

    from io import BytesIO

    img = Image.new("RGB", (200, 200), color="blue")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    decoded_img = DecodedImage(
        url="http://example.com/img1.jpg", raw_bytes=raw_bytes, image=img
    )

    with patch(
        "app.pipeline.orchestrator.download_all", return_value=([decoded_img], [])
    ):
        result = await run_pipeline(job)

        assert isinstance(result, JobResult)
        assert result.listing_id == "listing_123"
        assert len(result.images) == 1
        assert result.images[0].url == "http://example.com/img1.jpg"
        assert len(result.images[0].md5) == 32
        assert len(result.images[0].phash) > 0


@pytest.mark.asyncio
async def test_run_pipeline_all_downloads_fail():
    job = {
        "jobType": "listing",
        "imageUrls": ["http://example.com/img1.jpg"],
        "pipeline": ["quality"],
    }

    errors = [{"url": "http://example.com/img1.jpg", "error": "Connection timed out"}]

    with patch("app.pipeline.orchestrator.download_all", return_value=([], errors)):
        result = await run_pipeline(job)

        assert isinstance(result, JobResult)
        assert len(result.images) == 0
        assert len(result.errors) == 1
        assert result.errors[0].url == "http://example.com/img1.jpg"
        assert result.errors[0].error == "Connection timed out"


# ==============================================================================
# 5. Redis Streams Consumer tests
# ==============================================================================


@pytest.mark.asyncio
async def test_ensure_consumer_group():
    mock_redis = MagicMock()
    mock_redis.xgroup_create = AsyncMock()

    await ensure_consumer_group(mock_redis)
    mock_redis.xgroup_create.assert_called_once()

    from redis.exceptions import ResponseError

    mock_redis.xgroup_create.side_effect = ResponseError(
        "BUSYGROUP Consumer Group name already exists"
    )
    await ensure_consumer_group(mock_redis)  # should not raise exception

    mock_redis.xgroup_create.side_effect = ResponseError(
        "NOAUTH Authentication required"
    )
    with pytest.raises(ResponseError):
        await ensure_consumer_group(mock_redis)


@pytest.mark.asyncio
async def test_requeue_job():
    mock_redis = MagicMock()
    mock_redis.xadd = AsyncMock()
    job = {"listingId": "abc"}

    await requeue_job(mock_redis, job, attempt=2)

    mock_redis.xadd.assert_called_once()
    # Fields is passed as a positional argument (the second one)
    called_args = mock_redis.xadd.call_args[0]
    payload = json.loads(called_args[1]["payload"])
    assert payload["attempt"] == 2


@pytest.mark.asyncio
async def test_send_to_dlq():
    mock_redis = MagicMock()
    mock_redis.xadd = AsyncMock()
    job = {"listingId": "abc"}

    await send_to_dlq(mock_redis, job, "Network timeout")

    mock_redis.xadd.assert_called_once()
    called_args = mock_redis.xadd.call_args[0]
    assert called_args[1]["error"] == "Network timeout"


@pytest.mark.asyncio
async def test_reclaim_pending_messages():
    mock_redis = MagicMock()

    # Mock xautoclaim response: (next_start_id, [(message_id, {fields})])
    # First call returns a claimed message, second call returns no messages to break loop
    claimed_payload = json.dumps(
        {"jobType": "listing", "webhookUrl": "http://test.com", "imageUrls": ["x"]}
    )
    mock_redis.xautoclaim = AsyncMock(
        side_effect=[
            ("0-0", [("msg_123", {"payload": claimed_payload})]),
            ("0-0", []),
        ]
    )
    mock_redis.xack = AsyncMock()

    with patch("app.queue.consumer.handle_job", AsyncMock()) as mock_handle:
        await reclaim_pending_messages(mock_redis)
        mock_handle.assert_called_once_with(
            mock_redis,
            {"jobType": "listing", "webhookUrl": "http://test.com", "imageUrls": ["x"]},
        )
        mock_redis.xack.assert_called_once_with(
            "unlostpaws:stream:vision-processing",
            "vision-worker",
            "msg_123",
        )


def test_update_heartbeat():
    m = mock_open()
    with patch("builtins.open", m):
        update_heartbeat()
        m.assert_called_once_with("/tmp/worker-heartbeat", "w")


@pytest.mark.asyncio
async def test_handle_job_success():
    mock_redis = MagicMock()
    job = {
        "jobType": "listing",
        "webhookUrl": "http://example.com/webhook",
        "imageUrls": ["http://example.com/img.jpg"],
    }

    mock_result = JobResult(
        job_type="listing",
        images=[
            ProcessedImageResult(
                url="http://example.com/img.jpg",
                md5="hash123",
                phash="phash123",
            )
        ],
    )

    with (
        patch(
            "app.queue.consumer.run_pipeline", return_value=mock_result
        ) as mock_run_pip,
        patch("app.queue.consumer.send_callback") as mock_send_cb,
    ):
        await handle_job(mock_redis, job)

        mock_run_pip.assert_called_once_with(job)
        mock_send_cb.assert_called_once()


@pytest.mark.asyncio
async def test_handle_job_missing_webhook():
    mock_redis = MagicMock()
    job = {"jobType": "listing"}
    with pytest.raises(ValueError, match="missing webhookUrl"):
        await handle_job(mock_redis, job)


@pytest.mark.asyncio
async def test_handle_job_failure_retry_then_dlq():
    from app.config.settings import settings

    mock_redis = MagicMock()
    mock_redis.xadd = AsyncMock()

    job = {
        "jobType": "listing",
        "webhookUrl": "http://example.com/webhook",
        "imageUrls": ["http://example.com/img.jpg"],
        "attempt": 0,
    }

    # Backup original max_attempts value
    orig_max_attempts = settings.max_attempts

    # Simulate pipeline failure
    with (
        patch(
            "app.queue.consumer.run_pipeline",
            side_effect=RuntimeError("Pipeline crashed"),
        ),
        patch("app.queue.consumer.send_failure_callback") as mock_fail_cb,
        patch("asyncio.sleep") as mock_sleep,
    ):
        # 1. Under max attempts limit: Requeues
        object.__setattr__(settings, "max_attempts", 3)
        try:
            with pytest.raises(RuntimeError):
                await handle_job(mock_redis, job)

            mock_sleep.assert_called_once_with(2)
            mock_redis.xadd.assert_called_once()
        finally:
            object.__setattr__(settings, "max_attempts", orig_max_attempts)

        mock_redis.xadd.reset_mock()

        # 2. Exceeds max attempts limit: Sends to DLQ and calls failure callback
        job["attempt"] = 2
        object.__setattr__(settings, "max_attempts", 3)
        try:
            with pytest.raises(RuntimeError):
                await handle_job(mock_redis, job)

            mock_fail_cb.assert_called_once()
            mock_redis.xadd.assert_called_once()
        finally:
            object.__setattr__(settings, "max_attempts", orig_max_attempts)


# ==============================================================================
# 6. Model Registry tests
# ==============================================================================


def test_resolve_torch_device():
    import torch

    # Since torch is mocked, we can set return_value directly on its cuda mock
    torch.cuda.is_available.return_value = False
    assert resolve_torch_device("cuda") == "cpu"
    assert resolve_torch_device("cpu") == "cpu"

    torch.cuda.is_available.return_value = True
    assert resolve_torch_device("cuda") == "cuda"
    assert resolve_torch_device("cpu") == "cpu"


def test_health_models():
    from app.config.settings import settings

    res = health_models(settings)
    assert "device" in res
    assert "matchModel" in res
    assert "safetyLoaded" in res


@pytest.mark.asyncio
async def test_warmup_none_enabled():
    from app.config.settings import settings

    orig_embed_enabled = settings.embed_enabled
    orig_safety_enabled = settings.safety_enabled

    object.__setattr__(settings, "embed_enabled", False)
    object.__setattr__(settings, "safety_enabled", False)
    try:
        await warmup(settings)
    finally:
        object.__setattr__(settings, "embed_enabled", orig_embed_enabled)
        object.__setattr__(settings, "safety_enabled", orig_safety_enabled)


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


# ==============================================================================
# Helper Mock Class for Async Redis methods
# ==============================================================================


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)
