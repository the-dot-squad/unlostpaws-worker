import pytest
from PIL import Image

from app.config.profiles import PRESETS, SIGLIP2_BASE, SIGLIP2_QUALITY, get_preset
from app.pipeline.quality import assess_quality, laplacian_blur_score
from app.pipeline.stages.fingerprint import compute_md5, fingerprint_image
from app.schemas.result import (
    CallbackPayload,
    JobResult,
    ProcessedImageResult,
    SafetyResult,
)


def _solid_image(size=(100, 100), color=(128, 128, 128)) -> Image.Image:
    img = Image.new("RGB", size, color)
    return img


def test_profiles_known():
    assert set(PRESETS.keys()) == {"dedup-only", "standard", "quality"}
    standard = get_preset("standard")
    assert standard.match_model == SIGLIP2_BASE
    assert standard.relevance_enabled is True
    assert "relevance" in standard.stages

    quality = get_preset("quality")
    assert quality.match_model == SIGLIP2_QUALITY
    assert quality.relevance_enabled is True


def test_profiles_unknown_raises():
    with pytest.raises(ValueError):
        get_preset("nonexistent")


def test_fingerprint_stable():
    img = _solid_image()
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG")
    data = buf.getvalue()
    md5_a = compute_md5(data)
    md5_b = compute_md5(data)
    assert md5_a == md5_b
    md5, phash = fingerprint_image(data)
    assert len(md5) == 32
    assert len(phash) > 0


def test_quality_dimensions():
    img = _solid_image((800, 600))
    q = assess_quality(img, min_width=400, min_height=400)
    assert q["width"] == 800
    assert q["height"] == 600
    assert q["ok"] is True


def test_blur_score_range():
    sharp = _solid_image()
    score = laplacian_blur_score(sharp)
    assert 0.0 <= score <= 1.0


def test_callback_payload_schema():
    result = JobResult(
        job_type="listing",
        listing_id="abc",
        images=[
            ProcessedImageResult(
                url="/api/media/x.jpg",
                md5="abc",
                phash="def",
                safety=SafetyResult(nsfwScore=0.01, label="normal", model="test"),
            )
        ],
    )
    from app.pipeline.orchestrator import build_callback_payload

    payload = build_callback_payload(result)
    assert isinstance(payload, CallbackPayload)
    dumped = payload.model_dump(exclude_none=True)
    assert dumped["jobType"] == "listing"
    assert dumped["images"][0]["safety"]["nsfwScore"] == 0.01
