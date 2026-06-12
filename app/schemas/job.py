"""
Pydantic Schemas for incoming queue jobs and HTTP request payloads.

This module defines the validation schemas for jobs fetched from Redis Streams.
It mirrors the Next.js frontend schema for job dispatch.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Supported job categorization types:
# - 'listing': Processing images for pet listings (lost/found pets).
# - 'owned-pet': Processing images for user registered pets.
# - 'search': Processing a single image to find similar visual matches.
JobType = Literal["listing", "owned-pet", "search"]


class JobPayload(BaseModel):
    """
    Validation schema representing jobs enqueued by Next.js enqueueImageJob.
    """

    # The classification category of the processing job
    jobType: JobType = "listing"

    # Target database record identifier for lost/found pet listings (optional)
    listingId: str | None = None

    # Target database record identifier for registered owned pets (optional)
    ownedPetId: str | None = None

    # Session identifier when executing visual search queries (optional)
    searchSessionId: str | None = None

    # List of public URLs to download and process. Must contain at least 1 URL.
    imageUrls: list[str] = Field(min_length=1)

    # HTTP callback URL where the worker posts completion/error payloads
    webhookUrl: str | None = None

    # Category metadata specifying pet listings (lost, found, etc.)
    listingType: str = ""

    # Target species (e.g. 'dog', 'cat') used to evaluate pet relevance
    petType: str = ""

    # Number of execution attempts. Kept in payload to manage exponential retry backoff
    attempt: int = 0

    # Optional list of stages to override default profile sequence (e.g., ["quality", "embed"])
    pipeline: list[str] | None = None
