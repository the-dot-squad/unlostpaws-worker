"""Unit tests: Redis Streams consumer."""

import json
import pytest
from unittest.mock import MagicMock, patch, mock_open

from tests.unit.conftest import AsyncMock

from app.queue.consumer import (
    ensure_consumer_group,
    requeue_job,
    send_to_dlq,
    update_heartbeat,
    handle_job,
    reclaim_pending_messages,
)
from app.schemas.result import JobResult, ProcessedImageResult

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

        mock_run_pip.assert_called_once()
        called_job = mock_run_pip.call_args[0][0]
        assert called_job["webhookUrl"] == job["webhookUrl"]
        assert called_job["imageUrls"] == job["imageUrls"]
        mock_send_cb.assert_called_once()


@pytest.mark.asyncio
async def test_handle_job_missing_webhook():
    mock_redis = MagicMock()
    # Valid job shape but no callback URL — consumer rejects before pipeline.
    job = {
        "jobType": "listing",
        "imageUrls": ["http://example.com/img.jpg"],
    }
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


@pytest.mark.asyncio
async def test_handle_job_validation_error():
    mock_redis = MagicMock()
    mock_redis.xadd = AsyncMock()

    # Missing imageUrls (fundamentally invalid job format)
    job = {
        "jobType": "listing",
        "webhookUrl": "http://example.com/webhook",
    }

    with patch("app.queue.consumer.send_failure_callback") as mock_fail_cb:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            await handle_job(mock_redis, job)

        mock_redis.xadd.assert_called_once()
        called_args = mock_redis.xadd.call_args[0]
        assert called_args[0] == "unlostpaws:stream:vision-processing:dlq"
        assert "Validation error" in called_args[1]["error"]

        mock_fail_cb.assert_called_once()
        args = mock_fail_cb.call_args[0]
        assert args[0] == "http://example.com/webhook"
        assert args[1] == job
        assert "Validation error" in args[2]
