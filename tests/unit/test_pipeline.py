"""Unit tests: Pipeline orchestration."""

import pytest
from unittest.mock import patch
from PIL import Image

from app.pipeline.download import (
    DecodedImage,
)
from app.pipeline.orchestrator import run_pipeline, resolve_stages
from app.schemas.result import JobResult

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
