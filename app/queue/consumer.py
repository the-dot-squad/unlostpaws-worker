"""
Redis Streams Consumer — reliable background job processing daemon.

This module houses the core message consumer loop. It reads processing requests
from a Redis Stream via Consumer Groups, manages job retry attempts with exponential
backoff, dispatches items to the vision execution pipeline, and coordinates webhook
responses (success callbacks and DLQ error alerts).
"""

import asyncio
import json
import logging
import time

import redis.asyncio as aioredis

from app.callback.client import send_callback, send_failure_callback
from app.config.settings import settings
from app.pipeline.orchestrator import build_callback_payload, run_pipeline

logger = logging.getLogger(__name__)


def create_redis_client() -> aioredis.Redis:
    """
    Instantiates an asynchronous Redis connection client.
    Configures SSL parameters if the target connection starts with 'rediss://'.
    """
    url = settings.redis_url
    kwargs: dict = {"decode_responses": True}
    if url.startswith("rediss://"):
        # Disable certificate requirements for compatibility with serverless/managed
        # Redis brokers like Upstash over SSL.
        kwargs["ssl_cert_reqs"] = None
    return aioredis.from_url(url, **kwargs)


async def ensure_consumer_group(redis_client: aioredis.Redis) -> None:
    """
    Registers the consumer group on the Redis stream.
    Creates both the stream and consumer group if they do not already exist.
    """
    try:
        await redis_client.xgroup_create(
            settings.stream_key,
            settings.consumer_group,
            id="0",  # Start reading from the beginning of the stream ('0')
            mkstream=True,  # Automatically generate the stream key if missing
        )
        logger.info("Created consumer group %s", settings.consumer_group)
    except aioredis.ResponseError as exc:
        # Catch expected 'BUSYGROUP' errors which indicate that the group already exists.
        if "BUSYGROUP" not in str(exc):
            raise


async def requeue_job(redis_client: aioredis.Redis, job: dict, attempt: int) -> None:
    """
    Re-adds a failed job back to the end of the Redis stream.
    Increments the attempt counter to govern the retry limit.
    """
    job["attempt"] = attempt
    await redis_client.xadd(settings.stream_key, {"payload": json.dumps(job)})


async def send_to_dlq(redis_client: aioredis.Redis, job: dict, error: str) -> None:
    """
    Routes a permanently failed job payload and its traceback to the Dead Letter Queue stream.
    """
    await redis_client.xadd(
        settings.dlq_stream_key,
        {"payload": json.dumps(job), "error": error},
    )


def update_heartbeat() -> None:
    """
    Writes the current epoch timestamp to the local heartbeat check file.
    Used by app/healthcheck.py to confirm the consumer loop is actively running.
    """
    try:
        with open("/tmp/worker-heartbeat", "w") as f:
            f.write(str(time.time()))
    except Exception:
        # Ignore filesystem errors during heartbeat write (e.g. read-only volume mounts)
        pass


async def handle_job(redis_client: aioredis.Redis, job: dict) -> None:
    """
    Orchestrates the processing of a single job.
    Executes the pipeline, maps payloads, fires callbacks, and manages retries/DLQ on failure.
    """
    job_type = job.get("jobType", "listing")
    listing_id = job.get("listingId", "")
    attempt = int(job.get("attempt", 0))
    webhook_url = job.get("webhookUrl")

    logger.info(
        "Processing job type=%s listing=%s attempt=%d (%d images)",
        job_type,
        listing_id,
        attempt,
        len(job.get("imageUrls", [])),
    )

    # All jobs must provide a webhook URL so results can be returned to Next.js.
    if not webhook_url:
        raise ValueError("Job payload is missing webhookUrl")

    try:
        # 1. Dispatch image URLs to the multi-stage vision pipeline
        result = await run_pipeline(job)

        # 2. Structure results into the Callback payload format
        payload = build_callback_payload(result)

        # 3. If no images were successfully processed and errors occurred, raise error to trigger retry
        if not payload.images and payload.errors:
            raise RuntimeError("; ".join(e["error"] for e in payload.errors[:3]))

        # 4. POST the results back to the Next.js API endpoint
        await send_callback(webhook_url, payload)

        if result.errors:
            logger.warning(
                "Job finished with %d partial image errors",
                len(result.errors),
            )
    except Exception as exc:
        logger.exception("Job failed")
        next_attempt = attempt + 1

        # Evaluate retry limits
        if next_attempt < settings.max_attempts:
            # Exponential backoff delay calculation: 2s, 4s, 8s, up to 30s.
            delay = min(2**next_attempt, 30)
            logger.info("Requeueing job in %ds (attempt %d)", delay, next_attempt)
            await asyncio.sleep(delay)
            await requeue_job(redis_client, job, next_attempt)
        else:
            # Persistent failures are moved to the DLQ stream and reported via failure webhook
            logger.error("Moving job to DLQ after %d attempts", next_attempt)
            await send_to_dlq(redis_client, job, str(exc))
            try:
                await send_failure_callback(webhook_url, job, str(exc))
            except Exception:
                logger.exception("Failure callback failed for job type=%s", job_type)
        raise


async def reclaim_pending_messages(redis_client: aioredis.Redis) -> None:
    """
    Scans the Pending Entries List (PEL) for messages that have been delivered
    but never acknowledged (e.g. due to worker container crash/SIGKILL).
    Reclaims and processes them to guarantee no lost jobs.
    """
    logger.info("Checking for unacknowledged pending messages on startup...")
    # Consider messages pending for more than 5 minutes (300,000 ms) as timed out / crashed
    min_idle_time = 300000
    start_id = "0-0"

    while True:
        try:
            # xautoclaim atomically reassigns ownership of timed-out pending messages to our consumer name.
            # Returns (next_start_id, [(message_id, {field: value})])
            res = await redis_client.xautoclaim(
                name=settings.stream_key,
                groupname=settings.consumer_group,
                consumername=settings.consumer_name,
                min_idle_time=min_idle_time,
                start_id=start_id,
                count=10,
            )
            next_start_id, claimed_messages = res[:2]

            if not claimed_messages:
                break

            logger.info(
                "Claimed %d pending messages that timed out", len(claimed_messages)
            )
            for message_id, fields in claimed_messages:
                raw = fields.get("payload", "{}")
                job = json.loads(raw)
                try:
                    await handle_job(redis_client, job)
                    await redis_client.xack(
                        settings.stream_key,
                        settings.consumer_group,
                        message_id,
                    )
                except Exception:
                    # Acknowledge the message even on failure to avoid stuck loops (retry handles requeueing)
                    await redis_client.xack(
                        settings.stream_key,
                        settings.consumer_group,
                        message_id,
                    )

            start_id = next_start_id
            if start_id == "0-0":
                break
        except Exception as exc:
            logger.error("Failed to auto-claim pending messages: %s", exc)
            break


async def run_consumer(redis_client: aioredis.Redis) -> None:
    """
    Main asynchronous loop. Registers the consumer, polls for new messages,
    processes them, and issues acknowledgments.
    """
    await ensure_consumer_group(redis_client)
    logger.info(
        "Listening on stream %s group %s",
        settings.stream_key,
        settings.consumer_group,
    )

    # Reclaim unacknowledged messages from crashed workers on startup
    await reclaim_pending_messages(redis_client)

    backoff_delay = 1
    while True:
        try:
            # Poll new messages from Redis Streams.
            # - streams={settings.stream_key: ">"}: read messages that have not been delivered
            #   to any other consumer in the consumer group.
            # - count=1: process one job at a time to prevent CPU saturation.
            # - block=5000: block and wait up to 5 seconds if the stream is empty.
            messages = await redis_client.xreadgroup(
                groupname=settings.consumer_group,
                consumername=settings.consumer_name,
                streams={settings.stream_key: ">"},
                count=1,
                block=5000,
            )
            # Update the container heartbeat timestamp
            update_heartbeat()

            # If we successfully communicated with Redis, reset backoff delay
            backoff_delay = 1

            if not messages:
                continue

            # Process returned entries
            for _stream, entries in messages:
                for message_id, fields in entries:
                    raw = fields.get("payload", "{}")
                    job = json.loads(raw)

                    try:
                        await handle_job(redis_client, job)
                        # Acknowledge completion to remove the message from the pending list
                        await redis_client.xack(
                            settings.stream_key,
                            settings.consumer_group,
                            message_id,
                        )
                    except Exception:
                        # Even if processing fails, acknowledge the stream message to prevent
                        # infinite automatic redelivery, since the error retry logic handles
                        # requeueing separately (via requeue_job/DLQ).
                        await redis_client.xack(
                            settings.stream_key,
                            settings.consumer_group,
                            message_id,
                        )
        except asyncio.CancelledError:
            logger.info("Queue consumer stopped")
            raise
        except Exception:
            logger.exception("Consumer loop error — retrying in %ds", backoff_delay)
            await asyncio.sleep(backoff_delay)
            backoff_delay = min(backoff_delay * 2, 30)
