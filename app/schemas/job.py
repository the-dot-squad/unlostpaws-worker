"""
Pydantic schemas for jobs consumed from Redis Streams.

Jobs are validated at the consumer boundary so malformed payloads fail fast
with a clear error instead of crashing mid-pipeline.
"""

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.models.relevance import PET_KEYS

JobType = Literal["listing", "owned-pet", "search"]


class JobPayload(BaseModel):
    """Mirrors the Next.js enqueueImageJob payload."""

    jobType: JobType = "listing"
    listingId: str | None = None
    ownedPetId: str | None = None
    searchSessionId: str | None = None
    imageUrls: list[str] = Field(min_length=1)
    webhookUrl: str | None = None
    listingType: str = ""
    petType: str = Field(
        default="",
        description=(
            "Optional species hint from the listing (dog, cat, bird, etc.). "
            "Empty string = zero-shot. Unknown values are ignored."
        ),
    )
    attempt: int = 0
    pipeline: list[str] | None = None

    @field_validator("petType")
    @classmethod
    def normalize_pet_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            return ""
        if normalized not in PET_KEYS:
            return ""
        return normalized


def parse_job(raw: dict) -> JobPayload:
    """Validate and parse a Redis job dict. Raises ValidationError on bad input."""
    return JobPayload.model_validate(raw)


__all__ = ["JobPayload", "JobType", "parse_job", "ValidationError"]
