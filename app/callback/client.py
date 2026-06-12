"""
HTTP Webhook Callback Client — posts results to the Next.js frontend app.

Handles HTTP POST callback notifications. Sends success payloads (safety scores,
embeddings, quality metrics) or failure notifications (download timeouts, pipeline
exceptions) back to the Next.js API handler.
"""

import logging

import httpx

from app.config.settings import settings
from app.schemas.result import CallbackPayload
from app.utils.url import rewrite_local_url

logger = logging.getLogger(__name__)


async def send_callback(webhook_url: str, payload: CallbackPayload) -> None:
    """
    POSTs a success job results payload to the designated Next.js callback URL.

    Verifies that the target returns a successful HTTP status code (2xx),
    otherwise raises an exception to trigger job requeueing/retry.
    """
    body = payload.model_dump(exclude_none=True)
    request_url = rewrite_local_url(webhook_url)

    # Use HTTP client connection with configured timeouts
    async with httpx.AsyncClient(timeout=settings.callback_timeout) as client:
        response = await client.post(
            request_url,
            json=body,
        )
        response.raise_for_status()
        logger.info(
            "Callback POST success for jobType=%s. Images: %d",
            payload.jobType,
            len(payload.images),
        )


async def send_failure_callback(webhook_url: str, job: dict, error: str) -> None:
    """
    POSTs a failure alert payload to the Next.js callback URL.

    Triggered when a job exceeds the maximum number of retries and is moved
    to the Dead Letter Queue. Helps the frontend update database records
    and notify users of upload processing errors.
    """
    payload: dict = {
        "jobType": job.get("jobType", "listing"),
        "error": error,
    }
    # Echo back identifiers so the frontend can locate matching records
    if job.get("listingId"):
        payload["listingId"] = job["listingId"]
    if job.get("ownedPetId"):
        payload["ownedPetId"] = job["ownedPetId"]
    if job.get("searchSessionId"):
        payload["searchSessionId"] = job["searchSessionId"]

    request_url = rewrite_local_url(webhook_url)
    async with httpx.AsyncClient(timeout=settings.callback_timeout) as client:
        response = await client.post(
            request_url,
            json=payload,
        )
        response.raise_for_status()
        logger.info("Failure callback POST success for jobType=%s", payload["jobType"])
